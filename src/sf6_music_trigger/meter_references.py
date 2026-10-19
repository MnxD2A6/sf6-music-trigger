"""Canonical local meter references and allocation-safe NPZ persistence."""

from collections.abc import Sequence
from dataclasses import dataclass
import io
import math
from numbers import Real
import os
from pathlib import Path
import struct
import tempfile
import zipfile
import zlib

import cv2
import numpy as np

from .domain import Side
from .hud_profile import resolve_contained_path
from .meter_color_features import MeterColorModel


REFERENCE_SCHEMA_VERSION = 1
COLOR_REFERENCE_SCHEMA_VERSION = 2
PREPROCESSING_VERSION = 1
METER_OUTPUT_SHAPE = (32, 64)
MIN_REFERENCE_DISTANCE = 0.08
MAX_REFERENCE_ARCHIVE_BYTES = 1 * 1024 * 1024
MAX_NPY_MEMBER_BYTES = 16 * 1024
MAX_NPZ_UNCOMPRESSED_BYTES = 64 * 1024

_LEGACY_EXPECTED_MEMBERS = {
    "schema_version.npy": ((), np.dtype(np.uint16)),
    "preprocessing_version.npy": ((), np.dtype(np.uint16)),
    "side.npy": ((), np.dtype("<U2")),
    "shape_hw.npy": ((2,), np.dtype(np.uint16)),
    "full_gray.npy": (METER_OUTPUT_SHAPE, np.dtype(np.uint8)),
    "spent_gray.npy": (METER_OUTPUT_SHAPE, np.dtype(np.uint8)),
}
_COLOR_EXPECTED_MEMBERS = {
    **_LEGACY_EXPECTED_MEMBERS,
    "classifier.npy": ((), np.dtype("<U19")),
    "active_mask.npy": (METER_OUTPUT_SHAPE, np.dtype(np.bool_)),
    "full_hue.npy": ((), np.dtype(np.uint8)),
    "hue_tolerance.npy": ((), np.dtype(np.uint8)),
    "saturation_floor.npy": ((), np.dtype(np.uint8)),
    "value_floor.npy": ((), np.dtype(np.uint8)),
    "full_fill.npy": ((), np.dtype(np.float64)),
    "spent_fill.npy": ((), np.dtype(np.float64)),
}
_EXPECTED_MEMBERS = _LEGACY_EXPECTED_MEMBERS
_COLOR_CLASSIFIER = "masked_hsv_segments"
_LOCAL_FILE_HEADER = struct.Struct("<4sHHHHHIIIHH")
_LOCAL_FILE_HEADER_SIGNATURE = b"PK\x03\x04"
_ZIP_ENCRYPTION_FLAGS = 0x0001 | 0x0040 | 0x2000


def _require_nonnegative_finite_real(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _require_actual_int(name: str, value: object, expected: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value != expected:
        raise ValueError(f"unsupported {name}: {value}")


def _require_array(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return value


def _owned_readonly_copy(value: np.ndarray) -> np.ndarray:
    owned = np.array(value, dtype=value.dtype, order="C", copy=True)
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True, eq=False)
class PreprocessedMeter:
    normalized_gray: np.ndarray
    mean_luminance: float
    std_luminance: float
    hud_visible: bool

    def __post_init__(self) -> None:
        array = _require_array(
            "normalized_gray",
            self.normalized_gray,
            shape=METER_OUTPUT_SHAPE,
            dtype=np.dtype(np.uint8),
        )
        _require_nonnegative_finite_real("mean_luminance", self.mean_luminance)
        _require_nonnegative_finite_real("std_luminance", self.std_luminance)
        if type(self.hud_visible) is not bool:
            raise TypeError("hud_visible must be a boolean")
        object.__setattr__(self, "normalized_gray", _owned_readonly_copy(array))


def _reference_distance(full_gray: np.ndarray, spent_gray: np.ndarray) -> float:
    difference = np.abs(
        full_gray.astype(np.int16, copy=False)
        - spent_gray.astype(np.int16, copy=False)
    )
    return float(np.mean(difference, dtype=np.float64) / 255.0)


def _validate_meter_references(value: object) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, MeterReferences):
        raise TypeError("references must be MeterReferences")
    if not isinstance(value.side, Side):
        raise TypeError("side must be Side")
    if type(value.schema_version) is not int:
        raise TypeError("schema_version must be an integer")
    allowed_schemas = (
        {REFERENCE_SCHEMA_VERSION, COLOR_REFERENCE_SCHEMA_VERSION}
        if value.color_model is not None
        else {REFERENCE_SCHEMA_VERSION}
    )
    if value.schema_version not in allowed_schemas:
        raise ValueError(f"unsupported schema_version: {value.schema_version}")
    _require_actual_int(
        "preprocessing_version",
        value.preprocessing_version,
        PREPROCESSING_VERSION,
    )
    full = _require_array(
        "full_gray",
        value.full_gray,
        shape=METER_OUTPUT_SHAPE,
        dtype=np.dtype(np.uint8),
    )
    spent = _require_array(
        "spent_gray",
        value.spent_gray,
        shape=METER_OUTPUT_SHAPE,
        dtype=np.dtype(np.uint8),
    )
    if _reference_distance(full, spent) < MIN_REFERENCE_DISTANCE:
        raise ValueError(
            f"full/spent reference distance must be at least {MIN_REFERENCE_DISTANCE}"
        )
    if value.color_model is not None:
        if not isinstance(value.color_model, MeterColorModel):
            raise TypeError("color_model must be MeterColorModel or None")
        if value.color_model.side is not value.side:
            raise ValueError("color_model side must match references side")
    return full, spent


@dataclass(frozen=True, eq=False)
class MeterReferences:
    side: Side
    full_gray: np.ndarray
    spent_gray: np.ndarray
    schema_version: int = REFERENCE_SCHEMA_VERSION
    preprocessing_version: int = PREPROCESSING_VERSION
    color_model: MeterColorModel | None = None

    def __post_init__(self) -> None:
        full, spent = _validate_meter_references(self)
        object.__setattr__(self, "full_gray", _owned_readonly_copy(full))
        object.__setattr__(self, "spent_gray", _owned_readonly_copy(spent))
        if self.color_model is not None:
            object.__setattr__(self, "schema_version", COLOR_REFERENCE_SCHEMA_VERSION)


def _validate_bgr(crop_bgr: object) -> np.ndarray:
    if not isinstance(crop_bgr, np.ndarray):
        raise TypeError("crop_bgr must be a NumPy array")
    if crop_bgr.dtype != np.dtype(np.uint8):
        raise TypeError("crop_bgr must have uint8 dtype")
    if (
        crop_bgr.ndim != 3
        or crop_bgr.shape[0] == 0
        or crop_bgr.shape[1] == 0
        or crop_bgr.shape[2] != 3
    ):
        raise ValueError("crop_bgr must be a nonempty H x W x 3 BGR array")
    return crop_bgr


def preprocess_meter(crop_bgr: np.ndarray) -> PreprocessedMeter:
    """Produce the one canonical observation representation."""

    crop = _validate_bgr(crop_bgr)
    resized_bgr = cv2.resize(
        crop,
        (METER_OUTPUT_SHAPE[1], METER_OUTPUT_SHAPE[0]),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    mean_luminance = float(np.mean(gray, dtype=np.float64))
    std_luminance = float(np.std(gray, dtype=np.float64, ddof=0))
    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        sigmaX=0,
        sigmaY=0,
        borderType=cv2.BORDER_REFLECT_101,
    )
    p05, p95 = np.percentile(blurred, (5.0, 95.0), method="linear")
    if p95 <= p05:
        normalized = np.zeros(METER_OUTPUT_SHAPE, dtype=np.uint8)
        visible = False
    else:
        gray32 = blurred.astype(np.float32)
        p05_32 = np.float32(p05)
        p95_32 = np.float32(p95)
        scaled = (
            np.clip(
                (gray32 - p05_32) / (p95_32 - p05_32),
                np.float32(0.0),
                np.float32(1.0),
            )
            * np.float32(255.0)
        )
        normalized = np.rint(scaled).astype(np.uint8)
        visible = (
            std_luminance >= 6.0
            and 4.0 <= mean_luminance <= 251.0
        )
    return PreprocessedMeter(
        normalized,
        mean_luminance,
        std_luminance,
        visible,
    )


def _require_five_crops(name: str, value: object) -> Sequence[np.ndarray]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(value) != 5:
        raise ValueError(f"{name} must contain exactly five samples")
    return value


def build_meter_references(
    side: Side,
    full_crops: Sequence[np.ndarray],
    spent_crops: Sequence[np.ndarray],
) -> MeterReferences:
    if not isinstance(side, Side):
        raise TypeError("side must be Side")
    full_values = _require_five_crops("full_crops", full_crops)
    spent_values = _require_five_crops("spent_crops", spent_crops)
    full_samples = [preprocess_meter(crop) for crop in full_values]
    spent_samples = [preprocess_meter(crop) for crop in spent_values]
    if not all(
        sample.hud_visible for sample in (*full_samples, *spent_samples)
    ):
        raise ValueError("all calibration samples must have a visible HUD")
    full_gray = np.median(
        np.stack([sample.normalized_gray for sample in full_samples], axis=0),
        axis=0,
    ).astype(np.uint8)
    spent_gray = np.median(
        np.stack([sample.normalized_gray for sample in spent_samples], axis=0),
        axis=0,
    ).astype(np.uint8)
    return MeterReferences(side, full_gray, spent_gray)


def _reference_target(
    user_data_root: str | os.PathLike[str],
    profile_path: str,
    reference_path: str,
) -> Path:
    profile_target = resolve_contained_path(
        user_data_root, profile_path, field_name="profile_path"
    )
    return resolve_contained_path(
        profile_target.parent,
        reference_path,
        field_name="reference_path",
        required_suffix=".npz",
    )


def _validate_npy_member(
    member_name: str,
    member_bytes: bytes,
    expected_members: dict[str, tuple[tuple[int, ...], np.dtype]],
) -> None:
    expected_shape, expected_dtype = expected_members[member_name]
    stream = io.BytesIO(member_bytes)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                stream
            )
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                stream
            )
        else:
            raise ValueError(f"unsupported NPY header version: {version}")
    except (EOFError, TypeError, ValueError, UnicodeError) as error:
        raise ValueError(f"invalid NPY header for {member_name}") from error
    if dtype.hasobject or dtype.fields is not None:
        raise ValueError(f"object and structured dtypes are not allowed: {member_name}")
    if fortran_order:
        raise ValueError(f"Fortran-order NPY members are not allowed: {member_name}")
    if shape != expected_shape or dtype != expected_dtype:
        raise ValueError(f"invalid NPY dtype or shape for {member_name}")
    header_end = stream.tell()
    expected_payload_bytes = math.prod(shape) * dtype.itemsize
    if len(member_bytes) - header_end != expected_payload_bytes:
        raise ValueError(f"invalid NPY payload length for {member_name}")


def _local_general_purpose_flags(
    archive_bytes: bytes, info: zipfile.ZipInfo
) -> int:
    offset = info.header_offset
    if offset < 0 or offset + _LOCAL_FILE_HEADER.size > len(archive_bytes):
        raise ValueError(f"invalid local ZIP header for {info.filename}")
    header = _LOCAL_FILE_HEADER.unpack_from(archive_bytes, offset)
    if header[0] != _LOCAL_FILE_HEADER_SIGNATURE:
        raise ValueError(f"invalid local ZIP header for {info.filename}")
    return header[2]


def _schema_from_bounded_members(bounded_members: dict[str, bytes]) -> int:
    schema_member = "schema_version.npy"
    if schema_member not in bounded_members:
        raise ValueError("reference archive is missing schema_version")
    _validate_npy_member(
        schema_member, bounded_members[schema_member], _LEGACY_EXPECTED_MEMBERS
    )
    try:
        stream = io.BytesIO(bounded_members[schema_member])
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("unsupported NPY header version")
        schema = int(
            np.frombuffer(
                bounded_members[schema_member], dtype=np.dtype(np.uint16),
                count=1, offset=stream.tell()
            )[0]
        )
    except (EOFError, IndexError, TypeError, ValueError, UnicodeError) as error:
        raise ValueError("invalid schema_version member") from error
    if schema not in {REFERENCE_SCHEMA_VERSION, COLOR_REFERENCE_SCHEMA_VERSION}:
        raise ValueError("unsupported reference schema_version")
    return schema


def _preflight_archive(archive_bytes: bytes) -> int:
    if len(archive_bytes) > MAX_REFERENCE_ARCHIVE_BYTES:
        raise ValueError("reference archive exceeds 1 MiB")
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(set(names)) != len(names):
                raise ValueError("reference archive members must be distinct")
            if any(info.is_dir() for info in infos):
                raise ValueError("reference archive directories are not allowed")
            for info in infos:
                local_flags = _local_general_purpose_flags(archive_bytes, info)
                central_flags = info.flag_bits
                if (local_flags | central_flags) & _ZIP_ENCRYPTION_FLAGS:
                    raise ValueError("encrypted reference archive members are not allowed")
                if local_flags != central_flags:
                    raise ValueError("local and central ZIP flags do not match")
            if any(info.compress_type != zipfile.ZIP_DEFLATED for info in infos):
                raise ValueError("reference archive members must use ZIP_DEFLATED")
            if any(info.file_size > MAX_NPY_MEMBER_BYTES for info in infos):
                raise ValueError("reference archive member exceeds 16 KiB")
            if sum(info.file_size for info in infos) > MAX_NPZ_UNCOMPRESSED_BYTES:
                raise ValueError("reference archive exceeds 64 KiB uncompressed")

            bounded_members: dict[str, bytes] = {}
            for info in infos:
                with archive.open(info, mode="r") as member:
                    payload = member.read(MAX_NPY_MEMBER_BYTES + 1)
                if len(payload) > MAX_NPY_MEMBER_BYTES:
                    raise ValueError("reference archive member exceeds 16 KiB")
                if len(payload) != info.file_size:
                    raise ValueError("reference archive member length mismatch")
                bounded_members[info.filename] = payload
    except (zipfile.BadZipFile, EOFError, OSError, RuntimeError, zlib.error) as error:
        raise ValueError("invalid reference ZIP archive") from error

    schema = _schema_from_bounded_members(bounded_members)
    expected_members = (
        _COLOR_EXPECTED_MEMBERS
        if schema == COLOR_REFERENCE_SCHEMA_VERSION
        else _LEGACY_EXPECTED_MEMBERS
    )
    if len(bounded_members) != len(expected_members) or set(bounded_members) != set(
        expected_members
    ):
        raise ValueError("reference archive has invalid members")
    for member_name, payload in bounded_members.items():
        _validate_npy_member(member_name, payload, expected_members)
    return schema


def _load_validated_arrays(
    archive_bytes: bytes,
    expected_members: dict[str, tuple[tuple[int, ...], np.dtype]],
) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as loader:
            if len(loader.files) != len(expected_members) or set(loader.files) != {
                name.removesuffix(".npy") for name in expected_members
            }:
                raise ValueError("invalid reference archive keys")
            return {
                name: np.array(loader[name], copy=True, order="C")
                for name in loader.files
            }
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("invalid reference NumPy archive") from error


def load_references(
    user_data_root: str | os.PathLike[str],
    profile_path: str,
    reference_path: str,
    *,
    expected_side: Side,
) -> MeterReferences:
    if not isinstance(expected_side, Side):
        raise TypeError("expected_side must be Side")
    target = _reference_target(user_data_root, profile_path, reference_path)
    with target.open("rb") as stream:
        archive_bytes = stream.read(MAX_REFERENCE_ARCHIVE_BYTES + 1)
    if len(archive_bytes) > MAX_REFERENCE_ARCHIVE_BYTES:
        raise ValueError("reference archive exceeds 1 MiB")
    schema = _preflight_archive(archive_bytes)
    expected_members = (
        _COLOR_EXPECTED_MEMBERS
        if schema == COLOR_REFERENCE_SCHEMA_VERSION
        else _LEGACY_EXPECTED_MEMBERS
    )
    arrays = _load_validated_arrays(archive_bytes, expected_members)

    if arrays["schema_version"].item() != schema:
        raise ValueError("unsupported reference schema_version")
    if arrays["preprocessing_version"].item() != PREPROCESSING_VERSION:
        raise ValueError("unsupported preprocessing_version")
    stored_side_text = arrays["side"].item()
    if stored_side_text not in {Side.P1.value, Side.P2.value}:
        raise ValueError("invalid stored side")
    stored_side = Side(stored_side_text)
    if stored_side is not expected_side:
        raise ValueError("stored side does not match expected side")
    if not np.array_equal(
        arrays["shape_hw"], np.array(METER_OUTPUT_SHAPE, dtype=np.uint16)
    ):
        raise ValueError("stored output shape does not match canonical shape")
    color_model = None
    if schema == COLOR_REFERENCE_SCHEMA_VERSION:
        if arrays["classifier"].item() != _COLOR_CLASSIFIER:
            raise ValueError("unsupported colour classifier")
        color_model = MeterColorModel(
            stored_side,
            arrays["active_mask"],
            int(arrays["full_hue"].item()),
            int(arrays["hue_tolerance"].item()),
            int(arrays["saturation_floor"].item()),
            int(arrays["value_floor"].item()),
            float(arrays["full_fill"].item()),
            float(arrays["spent_fill"].item()),
        )
    return MeterReferences(
        stored_side,
        arrays["full_gray"],
        arrays["spent_gray"],
        schema_version=int(arrays["schema_version"].item()),
        preprocessing_version=int(arrays["preprocessing_version"].item()),
        color_model=color_model,
    )


def _serialize_references(references: MeterReferences) -> bytes:
    full, spent = _validate_meter_references(references)
    schema = (
        COLOR_REFERENCE_SCHEMA_VERSION
        if references.color_model is not None
        else REFERENCE_SCHEMA_VERSION
    )
    values: dict[str, np.ndarray] = {
        "schema_version": np.array(schema, dtype=np.uint16),
        "preprocessing_version": np.array(PREPROCESSING_VERSION, dtype=np.uint16),
        "side": np.array(references.side.value, dtype="<U2"),
        "shape_hw": np.array(METER_OUTPUT_SHAPE, dtype=np.uint16),
        "full_gray": np.array(full, dtype=np.uint8, order="C", copy=True),
        "spent_gray": np.array(spent, dtype=np.uint8, order="C", copy=True),
    }
    if references.color_model is not None:
        model = references.color_model
        values.update(
            classifier=np.array(_COLOR_CLASSIFIER, dtype="<U19"),
            active_mask=np.array(model.active_mask, dtype=np.bool_, order="C", copy=True),
            full_hue=np.array(model.full_hue, dtype=np.uint8),
            hue_tolerance=np.array(model.hue_tolerance, dtype=np.uint8),
            saturation_floor=np.array(model.saturation_floor, dtype=np.uint8),
            value_floor=np.array(model.value_floor, dtype=np.uint8),
            full_fill=np.array(model.full_fill, dtype=np.float64),
            spent_fill=np.array(model.spent_fill, dtype=np.float64),
        )
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    payload = stream.getvalue()
    _preflight_archive(payload)
    return payload


def _atomic_write(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def save_references(
    user_data_root: str | os.PathLike[str],
    profile_path: str,
    reference_path: str,
    references: MeterReferences,
) -> None:
    _validate_meter_references(references)
    target = _reference_target(user_data_root, profile_path, reference_path)
    payload = _serialize_references(references)
    _atomic_write(target, payload)
