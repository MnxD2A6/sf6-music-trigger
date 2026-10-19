from __future__ import annotations

from dataclasses import FrozenInstanceError
import io
import json
import os
from pathlib import Path
import struct
import warnings
import zipfile

import numpy as np
import pytest

import sf6_music_trigger.meter_references as meter_references_module

from sf6_music_trigger.domain import CapturedFrame, ClientGeometry, Side
from sf6_music_trigger.meter_color_features import MeterColorModel
from sf6_music_trigger.meter_references import (
    MAX_NPY_MEMBER_BYTES,
    MAX_NPZ_UNCOMPRESSED_BYTES,
    MAX_REFERENCE_ARCHIVE_BYTES,
    METER_OUTPUT_SHAPE,
    MIN_REFERENCE_DISTANCE,
    PREPROCESSING_VERSION,
    REFERENCE_SCHEMA_VERSION,
    MeterReferences,
    PreprocessedMeter,
    build_meter_references,
    load_references,
    preprocess_meter,
    save_references,
)
from sf6_music_trigger.hud_profile import load_profile


EXPECTED_MEMBER_NAMES = (
    "schema_version.npy",
    "preprocessing_version.npy",
    "side.npy",
    "shape_hw.npy",
    "full_gray.npy",
    "spent_gray.npy",
)


def bgr(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[:, :, None], 3, axis=2)


def calibration_crops() -> tuple[list[np.ndarray], list[np.ndarray]]:
    full = np.tile(np.arange(0, 256, 4, dtype=np.uint8), (32, 1))
    spent = np.flip(full, axis=1).copy()
    return [bgr(full.copy()) for _ in range(5)], [bgr(spent.copy()) for _ in range(5)]


def reference_arrays(side: str = "P1") -> dict[str, np.ndarray]:
    return {
        "schema_version": np.array(REFERENCE_SCHEMA_VERSION, dtype=np.uint16),
        "preprocessing_version": np.array(PREPROCESSING_VERSION, dtype=np.uint16),
        "side": np.array(side, dtype="<U2"),
        "shape_hw": np.array(METER_OUTPUT_SHAPE, dtype=np.uint16),
        "full_gray": np.zeros(METER_OUTPUT_SHAPE, dtype=np.uint8),
        "spent_gray": np.full(METER_OUTPUT_SHAPE, 255, dtype=np.uint8),
    }


def npy_bytes(
    array: np.ndarray, *, version: tuple[int, int] = (1, 0), allow_pickle: bool = False
) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream, array, version=version, allow_pickle=allow_pickle
    )
    return stream.getvalue()


def header_only_npy(
    *, shape: tuple[int, ...], dtype: np.dtype = np.dtype(np.uint8), fortran: bool = False
) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream,
        {"descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": fortran, "shape": shape},
    )
    return stream.getvalue()


class ArchiveFactory:
    def __init__(self) -> None:
        self._arrays = reference_arrays()

    def valid_member_bytes(self) -> dict[str, bytes]:
        return {
            f"{name}.npy": npy_bytes(array)
            for name, array in self._arrays.items()
        }

    def archive(
        self,
        *,
        replacements: dict[str, bytes] | None = None,
        missing: set[str] | None = None,
        extra: tuple[str, bytes] | None = None,
        duplicate: str | None = None,
        compression_overrides: dict[str, int] | None = None,
        raw_members: dict[str, bytes] | None = None,
    ) -> bytes:
        members = self.valid_member_bytes() if raw_members is None else dict(raw_members)
        members.update(replacements or {})
        for name in missing or set():
            members.pop(name, None)

        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in members.items():
                archive.writestr(
                    name,
                    payload,
                    compress_type=(compression_overrides or {}).get(
                        name, zipfile.ZIP_DEFLATED
                    ),
                )
            if extra is not None:
                archive.writestr(extra[0], extra[1], compress_type=zipfile.ZIP_DEFLATED)
            if duplicate is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    archive.writestr(
                        duplicate,
                        members[duplicate],
                        compress_type=zipfile.ZIP_DEFLATED,
                    )
        return output.getvalue()

    def duplicate_member(self, member_name: str) -> bytes:
        return self.archive(duplicate=f"{member_name}.npy" if not member_name.endswith(".npy") else member_name)

    def npy_header(self, member_name: str, *, shape: tuple[int, ...]) -> bytes:
        name = f"{member_name}.npy" if not member_name.endswith(".npy") else member_name
        return self.archive(replacements={name: header_only_npy(shape=shape)})

    def write_reference(self, user_data_root: Path, name: str, payload: bytes) -> Path:
        target = user_data_root / "profiles" / "refs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target


@pytest.fixture
def archive_factory() -> ArchiveFactory:
    return ArchiveFactory()


SMALL_GOLDEN_ROWS = (
    "61616161616161616161616580a5b4b4b4b4b4b4b4b4b4b4a56a280d0d0d0d0d0d0d0d0d0d142333373737373737373737375493bfc6c6c6c6c6c6c6c6c6c6c6",
    "61616161616161616161616580a5b4b4b4b4b4b4b4b4b4b4a56a280d0d0d0d0d0d0d0d0d0d142333373737373737373737375493bfc6c6c6c6c6c6c6c6c6c6c6",
    "61616161616161616161616580a5b4b4b4b4b4b4b4b4b4b4a56a280d0d0d0d0d0d0d0d0d0d142333373737373737373737375493bfc6c6c6c6c6c6c6c6c6c6c6",
    "61616161616161616161616580a5b4b4b4b4b4b4b4b4b4b4a56a280d0d0d0d0d0d0d0d0d0d142333373737373737373737375493bfc6c6c6c6c6c6c6c6c6c6c6",
    "61616161616161616161616580a5b4b4b4b4b4b4b4b4b4b4a56a280d0d0d0d0d0d0d0d0d0d142333373737373737373737375493bfc6c6c6c6c6c6c6c6c6c6c6",
    "61616161616161616161616580a5b4b4b4b4b4b4b4b4b4b4a56a280d0d0d0d0d0d0d0d0d0d142333373737373737373737375493bfc6c6c6c6c6c6c6c6c6c6c6",
    "61616161616161616161616580a5b4b4b4b4b4b4b4b4b4b4a56a280d0d0d0d0d0d0d0d0d0d142333373737373737373737375493bfc6c6c6c6c6c6c6c6c6c6c6",
    "61616161616161616161616580a5b4b4b4b4b4b4b4b4b4b4a56a280d0d0d0d0d0d0d0d0d0d142333373737373737373737375493bfc6c6c6c6c6c6c6c6c6c6c6",
    "61616161616161616161616580a5b4b4b4b4b4b4b4b4b4b4a56a280d0d0d0d0d0d0d0d0d0d142333373737373737373737375493bfc6c6c6c6c6c6c6c6c6c6c6",
    "58585858585858585858585c79a5b9b9b9b9b9b9b9b9b9b9a9773923232323232323232323282e3739393939393939393939548fb6bdbdbdbdbdbdbdbdbdbdbd",
    "30303030303030303030303765a9c8c8c8c8c8c8c8c8c8c8c1a78a80808080808080808080755c46404040404040404040405179939898989898989898989898",
    "00000000000000000000000b4db0dcdcdcdcdcdcdcdcdcdcdee5edf0f0f0f0f0f0f0f0f0f0d7955a4b4b4b4b4b4b4b4b4b4b515f676a6a6a6a6a6a6a6a6a6a6a",
    "00000000000000000000000042b2e5e5e5e5e5e5e5e5e5e5ebffffffffffffffffffffffffffab634f4f4f4f4f4f4f4f4f4f5154565656565656565656565656",
    "00000000000000000000000042b2e5e5e5e5e5e5e5e5e5e5ebffffffffffffffffffffffffffab634f4f4f4f4f4f4f4f4f4f5154565656565656565656565656",
    "00000000000000000000000042b2e5e5e5e5e5e5e5e5e5e5ebffffffffffffffffffffffffffab634f4f4f4f4f4f4f4f4f4f5154565656565656565656565656",
    "00000000000000000000000042b2e5e5e5e5e5e5e5e5e5e5ebffffffffffffffffffffffffffab634f4f4f4f4f4f4f4f4f4f5154565656565656565656565656",
    "00000000000000000000000042b2e5e5e5e5e5e5e5e5e5e5ebffffffffffffffffffffffffffab634f4f4f4f4f4f4f4f4f4f5154565656565656565656565656",
    "00000000000000000000000042b2e5e5e5e5e5e5e5e5e5e5ebffffffffffffffffffffffffffab634f4f4f4f4f4f4f4f4f4f5154565656565656565656565656",
    "00000000000000000000000042b2e5e5e5e5e5e5e5e5e5e5ebffffffffffffffffffffffffffab634f4f4f4f4f4f4f4f4f4f5154565656565656565656565656",
    "00000000000000000000000042b2e5e5e5e5e5e5e5e5e5e5ebffffffffffffffffffffffffffab634f4f4f4f4f4f4f4f4f4f5154565656565656565656565656",
    "0909090909090909090909124ba0c8c8c8c8c8c8c8c8c8c8cfe5fffffffffffffffffffffff0ae72616161616161616161616365676767676767676767676767",
    "51515151515151515151515461757d7d7d7d7d7d7d7d7d7d86a7ccdadadadadadadadadadacfb0958d8d8d8d8d8d8d8d8d8d8f91939393939393939393939393",
    "8d8d8d8d8d8d8d8d8d8d8d887251424242424242424242424d75a3b2b2b2b2b2b2b2b2b2b2b2b2b0b0b0b0b0b0b0b0b0b0b0b2b4b6b6b6b6b6b6b6b6b6b6b6b6",
    "9a9a9a9a9a9a9a9a9a9a9a95774933333333333333333333406a98a9a9a9a9a9a9a9a9a9a9abb2b6b9b9b9b9b9b9b9b9b9b9bbbdbfbfbfbfbfbfbfbfbfbfbfbf",
    "9a9a9a9a9a9a9a9a9a9a9a95774933333333333333333333406a98a9a9a9a9a9a9a9a9a9a9abb2b6b9b9b9b9b9b9b9b9b9b9bbbdbfbfbfbfbfbfbfbfbfbfbfbf",
    "9a9a9a9a9a9a9a9a9a9a9a95774933333333333333333333406a98a9a9a9a9a9a9a9a9a9a9abb2b6b9b9b9b9b9b9b9b9b9b9bbbdbfbfbfbfbfbfbfbfbfbfbfbf",
    "9a9a9a9a9a9a9a9a9a9a9a95774933333333333333333333406a98a9a9a9a9a9a9a9a9a9a9abb2b6b9b9b9b9b9b9b9b9b9b9bbbdbfbfbfbfbfbfbfbfbfbfbfbf",
    "9a9a9a9a9a9a9a9a9a9a9a95774933333333333333333333406a98a9a9a9a9a9a9a9a9a9a9abb2b6b9b9b9b9b9b9b9b9b9b9bbbdbfbfbfbfbfbfbfbfbfbfbfbf",
    "9a9a9a9a9a9a9a9a9a9a9a95774933333333333333333333406a98a9a9a9a9a9a9a9a9a9a9abb2b6b9b9b9b9b9b9b9b9b9b9bbbdbfbfbfbfbfbfbfbfbfbfbfbf",
    "9a9a9a9a9a9a9a9a9a9a9a95774933333333333333333333406a98a9a9a9a9a9a9a9a9a9a9abb2b6b9b9b9b9b9b9b9b9b9b9bbbdbfbfbfbfbfbfbfbfbfbfbfbf",
    "9a9a9a9a9a9a9a9a9a9a9a95774933333333333333333333406a98a9a9a9a9a9a9a9a9a9a9abb2b6b9b9b9b9b9b9b9b9b9b9bbbdbfbfbfbfbfbfbfbfbfbfbfbf",
    "9a9a9a9a9a9a9a9a9a9a9a95774933333333333333333333406a98a9a9a9a9a9a9a9a9a9a9abb2b6b9b9b9b9b9b9b9b9b9b9bbbdbfbfbfbfbfbfbfbfbfbfbfbf",
)


LARGE_GOLDEN_ROWS = (
    "0006142332414f5e5c39110b1c2a37465564738392a1ac925e4e5c6b7a8998a4aeb1b3b5bcac7c6067768594a3b1c0d1deedfbf9d4aeaab9c8d6e5f4ffffffff",
    "070f1e2c3b4a5967643f181425344351606d7c8c9baab1986657667483909da4acaeb3bbc2b183697180909facbbcbdae9f6ffffdab5b3c2d1e0ebf2fbfffffb",
    "1c23303f4f5e6d7667412529374655647381909faebcb9966f697887929b9fa3a8b0bbc8cdb1897a8394a3b1c0cfdeedfbffffffdec2c6d1d6dcdac4b9c4b7a1",
    "2e36445362718083663d303b4a5967768594a3b1c0cdbc8c747c8b949b9b9da4b0bccddcd4ac8b8b98a6b5c4d3e1f0ffffffffffcbb3bbbca6988b67535b574e",
    "414857667483908c69433f4e5c6d7c8998a6b5c4d3dcc08e808990969b9da4b1c0cfe0eee3b3969baab9c8d6e5f4ffffffffffde92676b694e39342c272a2e30",
    "535b697887969f8e674e53606f808e9dacb9c8d6e5e1b78e858b8e969daab7c6d4e3f2fdedc0a6aebccbdaebf8f9f6f4e3c6ae80370f111818181c1f23272e34",
    "676f7c8b99a8ac90665966748392a1b0becddce9f6e0a4838389929facbbc9d8e7f6ffffe9c2b7c0cfd3cbcdd8c8a6998b6b481c00000009141c1e1e252e3b43",
    "7a81909faebcbe9d7169788796a4b3c2d1e0edf6fbda997e8794a1b0c0cfdeebf9ffffffe7bba4a1aca17a6b766d554f554c290000000911161a1f2532414f57",
    "8c94a3b1c0cdc8a4807c8b99a8b7c6d4e1ebf2f4f2cb988998a6b5c4d3e1f0fffffffff2b380513d4441292330393d48534e2a0600060b0f141e293644536269",
    "a1a8b5c4d3dac69d89909facbbc9d8e5ebebeef6e7b9989baab9c8d6e3f2f9f0e7eed88536140000040f141f2e3d4c5b624c1e02000209121e2a39485766747c",
    "b3bbc9d8e7e9cb9f96a3b1c0cfdae3e9ebebf4ffedb7a1aebbc6cdc8bbbebe9d7c7a661f000000000614253443515e6b6b440f0000071221303f4e5c6b7a8990",
    "c8cfdeedf8f0cdaaa8b7c4cfd8dee1e5edf8ffffe9b3a6b1ac9990764f484f463d3d2a000000000b1a29374655646b73693d09000918273643516071808e999f",
    "dae1f0ffffeec4b1b9c6cdd3d6dce3edf8fdfff9b974666f5c34251f1418273644431f0000000f1f2e3b4a59666f6f6b5b300d0d1c2a39485766748392a1a39f",
    "edf4fffffff2c6b9c2c8cbd1d8dee1e7e3cdc2ac5b0b00070700000f1c2a39485751270406142332414e5b64696d6f6d5127161f2e3d4c5b69788796a4b3a694",
    "ffffffffffebc4bbc0c0bebcc6c6b0a192715c410200000000041221303d4c5c695e3416182736434f595e62676f7876552a2332414f5e6d7c8b99a8b7c4ac92",
    "ffffffffffd3aa9b9d9b877374735743413b2e0900000000071625344351606f745c36232a37444c51575c626d7a857a53363744536473808e9dacbbcbc9a68c",
    "ffffffffe59862433f432e1a1c252325303f34040000000b1a29374655647381805b37303941464a4f59626f7e8c967c4f3f4a5967768594a3b1c0cfded1a185",
    "fbfdfbd47a2c06000000000009182736444e370700000f1e2c3b4a5967768590855c3b363b3f444e59667483929f9f80554e5c6b7a8998a6b5c4d3e1f0e0aa8e",
    "81838359070000000000000d1c2a394857532e09061423323f4f5e6d7c8b999d8153363439434e5b69788796a4b0a1785c606f7e8c9baab9c8d6e5f4fde5b5a1",
    "2a303616000000000002111f2e3f4c5b695c2c1118273644536271808e9daaa476443237434f5e6d7c8b99a8b7c0a67667748392a1aebecddcebf8ffffdab7b0",
    "252c2c0b00000000071625344351606f765e321f2a394857667483929faab1a16f413946536271808e9dacbbc9d1b585788796a4b3c2d1e0eefdffffffdac0c0",
    "393d2c000000000b1c2a3946556473817c5332303f4e5c6b7a87949fa6aaa89467484a5967768594a3b1c0cfdcdcb9928c9baab9c8d6e5f2ffffffffffdecbcb",
    "4e4f36040002111f2e3d4c5b6978879689573b435160717e8b94999fa3a8a6875e535c6d7c8b98a6b5c4d4e3eedcae989faebccbdae9f8fffffffffff2c6bbbb",
    "6060441406142332414f5e6d7c8b99a696674e55647380898e94989da8b3ae89646271808e9dacbbc9d8e7f6fde0b0a3b1c0cfdeedfbffffffffffffa1666266",
    "7167431c18273644536271808e9dacb1966d5e69768083898c929daab9c0b18b71748392a1b0becddcebf9ffffdeb9b7c4d3e3f2fdffffffffedcb8721000000",
    "816939232c3b485967768594a3b1c0b98e6d6d7880818189929faebccbcdb0877c8796a4b3c4d1e0eefdffffffd6c0c9d4d8dae3e5cfb7b1a47e4a0700000000",
    "896d3d303f4e5c6b7a8998a6b5c4cfbe9073747a80818994a3b1c0cfdcd6b18e8c9baab7c8d6e5f4ffffffffffdcc0bcbbaa8e8b87694a4141371c0000000000",
    "8166434351606f7e8c9baab9c8d6d6b387737378808b99a8b7c4d3e1e9d6b0999faebccbdae9f8ffffffffffedb98e6f624f3427271e181e29321f0000000000",
    "7c624a57647381909fb0becbd8e3d4a1766f76818e9dacbbc9d8e7f6f8d8b0a6b1c0cfdeedfbfff6f6fff4b5714e341c1211090407121e2c3b3b1a000000060d",
    "7a665b69788796a4b3c2cfdae1e5cf9974768392a1b0becddcebf9ffffe7bcb7c6d3e1e7dcd8cba6949d965911000000000002091421303f4e441a00000b1a21",
    "6d696d7c8b99a8b7c4cfd6dadedec698808796a4b3c2d1e0eefdffffffedcbc0b7b0b5ae8b7167534e5959320000000000000b18273644515b481e070f1e2c34",
    "666b768594a3b1c0cdd3d4d6dcdabe9687909faebccbdae9f8ffffffffe9cfc2a18180784f36363b434f4e270000000000041221303f4e5c60441e0f1827363d",
)


def golden_array(rows: tuple[str, ...]) -> np.ndarray:
    assert len(rows) == METER_OUTPUT_SHAPE[0]
    assert all(len(row) == METER_OUTPUT_SHAPE[1] * 2 for row in rows)
    return np.frombuffer(b"".join(bytes.fromhex(row) for row in rows), dtype=np.uint8).reshape(
        METER_OUTPUT_SHAPE
    )


def test_preprocess_upscale_matches_literal_bgr_golden_vector():
    crop = np.array(
        [
            [[0, 32, 255], [17, 220, 5], [91, 7, 143], [240, 81, 3], [12, 199, 77]],
            [[201, 4, 55], [33, 144, 222], [8, 250, 111], [123, 19, 207], [77, 88, 99]],
            [[14, 173, 61], [255, 9, 134], [42, 201, 18], [166, 73, 245], [5, 117, 230]],
        ],
        dtype=np.uint8,
    )
    original = crop.copy()
    result = preprocess_meter(crop)
    assert np.array_equal(result.normalized_gray, golden_array(SMALL_GOLDEN_ROWS))
    assert result.mean_luminance == pytest.approx(110.18115234375, rel=0, abs=1e-12)
    assert result.std_luminance == pytest.approx(36.510752593203335, rel=0, abs=1e-12)
    assert result.hud_visible
    assert np.array_equal(crop, original) and crop.flags.writeable


def test_preprocess_downscale_matches_literal_bgr_golden_vector():
    yy, xx = np.indices((35, 67))
    crop = np.stack(
        (
            (xx * 7 + yy * 13) % 256,
            (xx * 3 + yy * 11 + 37) % 256,
            (xx * 17 + yy * 5 + 91) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    original = crop.copy()
    result = preprocess_meter(crop)
    assert np.array_equal(result.normalized_gray, golden_array(LARGE_GOLDEN_ROWS))
    assert result.mean_luminance == pytest.approx(130.6591796875, rel=0, abs=1e-12)
    assert result.std_luminance == pytest.approx(46.53619279606022, rel=0, abs=1e-12)
    assert result.hud_visible
    assert np.array_equal(crop, original) and crop.flags.writeable


@pytest.mark.parametrize("level", [0, 255])
def test_black_and_white_crops_are_safe_zero_and_hidden(level: int):
    result = preprocess_meter(np.full((40, 80, 3), level, dtype=np.uint8))
    assert np.array_equal(result.normalized_gray, np.zeros(METER_OUTPUT_SHAPE, np.uint8))
    assert result.mean_luminance == level
    assert result.std_luminance == 0
    assert not result.hud_visible


def test_percentile_degenerate_crop_is_zero_and_hidden():
    crop = np.full((32, 64, 3), 120, dtype=np.uint8)
    crop[0, 0] = 0
    crop[-1, -1] = 255
    sample = preprocess_meter(crop)
    assert np.array_equal(sample.normalized_gray, np.zeros(METER_OUTPUT_SHAPE, np.uint8))
    assert not sample.hud_visible


def test_low_contrast_nonconstant_crop_is_hidden_without_numeric_instability():
    yy, xx = np.indices(METER_OUTPUT_SHAPE)
    gray = (120 + ((xx + yy) % 2)).astype(np.uint8)
    sample = preprocess_meter(bgr(gray))
    assert np.isfinite(sample.mean_luminance)
    assert np.isfinite(sample.std_luminance)
    assert sample.std_luminance < 6.0
    assert not sample.hud_visible


@pytest.mark.parametrize(
    ("gray", "expected_visible"),
    [
        (np.pad(np.full((32, 16), 16, np.uint8), ((0, 0), (0, 48))), True),
        (np.pad(np.full((32, 16), 239, np.uint8), ((0, 0), (0, 48)), constant_values=255), True),
        (np.pad(np.full((32, 6), 30, np.uint8), ((0, 0), (0, 58))), False),
        (np.pad(np.full((32, 6), 225, np.uint8), ((0, 0), (0, 58)), constant_values=255), False),
    ],
    ids=["mean-4-inclusive", "mean-251-inclusive", "below-4", "above-251"],
)
def test_hud_visibility_mean_boundaries_are_inclusive(
    gray: np.ndarray, expected_visible: bool
):
    result = preprocess_meter(bgr(gray))
    assert result.std_luminance >= 6.0
    assert result.hud_visible is expected_visible


@pytest.mark.parametrize(
    "crop",
    [
        [0],
        np.empty((0, 2, 3), dtype=np.uint8),
        np.empty((2, 0, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 1), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((2, 2, 3), dtype=np.uint16),
    ],
)
def test_preprocess_rejects_malformed_bgr(crop: object):
    with pytest.raises((TypeError, ValueError)):
        preprocess_meter(crop)


def test_preprocess_keeps_input_and_captured_frame_image_writable_and_unchanged():
    image = np.arange(40 * 80 * 3, dtype=np.uint8).reshape(40, 80, 3)
    original = image.copy()
    frame = CapturedFrame(1, image, 1.0, ClientGeometry(0, 0, 80, 40))
    result = preprocess_meter(frame.image)
    assert frame.image is image
    assert frame.image.flags.writeable
    assert np.array_equal(frame.image, original)
    assert result.normalized_gray.flags.owndata
    assert not result.normalized_gray.flags.writeable


def test_preprocessed_meter_constructor_copies_and_freezes_only_its_owned_array():
    source = np.arange(np.prod(METER_OUTPUT_SHAPE), dtype=np.uint8).reshape(
        METER_OUTPUT_SHAPE
    )
    before = source.copy()
    value = PreprocessedMeter(source, 120.0, 7.0, True)
    assert value.normalized_gray.flags.owndata
    assert not value.normalized_gray.flags.writeable
    assert not np.shares_memory(value.normalized_gray, source)
    source[:] = 0
    assert np.array_equal(value.normalized_gray, before)
    assert source.flags.writeable
    with pytest.raises(ValueError):
        value.normalized_gray[0, 0] = 0
    with pytest.raises(FrozenInstanceError):
        value.mean_luminance = 0
    assert value != PreprocessedMeter(before, 120.0, 7.0, True)
    with pytest.raises(TypeError):
        value < value


@pytest.mark.parametrize(
    "array",
    [
        [[0]],
        np.zeros((32, 64), dtype=np.float32),
        np.zeros((32, 64), dtype=np.uint16),
        np.zeros((31, 64), dtype=np.uint8),
        np.zeros((32, 63), dtype=np.uint8),
        np.zeros((32, 64, 1), dtype=np.uint8),
    ],
)
def test_preprocessed_meter_rejects_wrong_array_type_dtype_and_shape(array: object):
    with pytest.raises((TypeError, ValueError)):
        PreprocessedMeter(array, 120.0, 7.0, True)


@pytest.mark.parametrize(
    ("mean", "std", "visible"),
    [
        (-1, 1, True),
        (float("nan"), 1, True),
        (float("inf"), 1, True),
        (True, 1, True),
        ("120", 1, True),
        (1, -1, True),
        (1, float("nan"), True),
        (1, float("inf"), True),
        (1, False, True),
        (1, "7", True),
        (1, 7, 1),
        (1, 7, "true"),
    ],
)
def test_preprocessed_meter_rejects_invalid_stats_and_visibility(
    mean: object, std: object, visible: object
):
    with pytest.raises((TypeError, ValueError)):
        PreprocessedMeter(
            np.zeros(METER_OUTPUT_SHAPE, dtype=np.uint8), mean, std, visible
        )


def test_meter_references_constructor_copies_freezes_and_separates_arrays():
    full = np.zeros(METER_OUTPUT_SHAPE, dtype=np.uint8)
    spent = np.full(METER_OUTPUT_SHAPE, 255, dtype=np.uint8)
    value = MeterReferences(Side.P1, full, spent)
    assert value.side is Side.P1
    assert value.full_gray.flags.owndata and not value.full_gray.flags.writeable
    assert value.spent_gray.flags.owndata and not value.spent_gray.flags.writeable
    assert not np.shares_memory(value.full_gray, full)
    assert not np.shares_memory(value.spent_gray, spent)
    assert not np.shares_memory(value.full_gray, value.spent_gray)
    full[:] = 127
    spent[:] = 127
    assert np.all(value.full_gray == 0)
    assert np.all(value.spent_gray == 255)
    assert full.flags.writeable and spent.flags.writeable
    with pytest.raises(ValueError):
        value.full_gray[0, 0] = 1
    with pytest.raises(ValueError):
        value.spent_gray[0, 0] = 1
    with pytest.raises(FrozenInstanceError):
        value.side = Side.P2
    assert value != MeterReferences(
        Side.P1,
        np.zeros(METER_OUTPUT_SHAPE, dtype=np.uint8),
        np.full(METER_OUTPUT_SHAPE, 255, dtype=np.uint8),
    )
    with pytest.raises(TypeError):
        value < value


@pytest.mark.parametrize(
    ("side", "full", "spent", "schema", "preprocessing"),
    [
        ("P1", np.zeros(METER_OUTPUT_SHAPE, np.uint8), np.full(METER_OUTPUT_SHAPE, 255, np.uint8), 1, 1),
        (Side.P1, [[0]], np.full(METER_OUTPUT_SHAPE, 255, np.uint8), 1, 1),
        (Side.P1, np.zeros((31, 64), np.uint8), np.full(METER_OUTPUT_SHAPE, 255, np.uint8), 1, 1),
        (Side.P1, np.zeros(METER_OUTPUT_SHAPE, np.float32), np.full(METER_OUTPUT_SHAPE, 255, np.uint8), 1, 1),
        (Side.P1, np.zeros(METER_OUTPUT_SHAPE, np.uint8), np.ones((32, 63), np.uint8), 1, 1),
        (Side.P1, np.zeros(METER_OUTPUT_SHAPE, np.uint8), np.ones(METER_OUTPUT_SHAPE, np.uint16), 1, 1),
        (Side.P1, np.zeros(METER_OUTPUT_SHAPE, np.uint8), np.full(METER_OUTPUT_SHAPE, 255, np.uint8), True, 1),
        (Side.P1, np.zeros(METER_OUTPUT_SHAPE, np.uint8), np.full(METER_OUTPUT_SHAPE, 255, np.uint8), np.uint16(1), 1),
        (Side.P1, np.zeros(METER_OUTPUT_SHAPE, np.uint8), np.full(METER_OUTPUT_SHAPE, 255, np.uint8), 2, 1),
        (Side.P1, np.zeros(METER_OUTPUT_SHAPE, np.uint8), np.full(METER_OUTPUT_SHAPE, 255, np.uint8), 1, False),
        (Side.P1, np.zeros(METER_OUTPUT_SHAPE, np.uint8), np.full(METER_OUTPUT_SHAPE, 255, np.uint8), 1, 1.0),
        (Side.P1, np.zeros(METER_OUTPUT_SHAPE, np.uint8), np.full(METER_OUTPUT_SHAPE, 255, np.uint8), 1, 2),
    ],
)
def test_meter_references_rejects_wrong_types_shapes_dtypes_and_versions(
    side: object,
    full: object,
    spent: object,
    schema: object,
    preprocessing: object,
):
    with pytest.raises((TypeError, ValueError)):
        MeterReferences(side, full, spent, schema, preprocessing)


def test_meter_references_rejects_distance_below_exact_threshold():
    full = np.zeros(METER_OUTPUT_SHAPE, dtype=np.uint8)
    below = np.full(
        METER_OUTPUT_SHAPE,
        int(np.floor(MIN_REFERENCE_DISTANCE * 255)),
        dtype=np.uint8,
    )
    assert np.mean(np.abs(below.astype(np.int16))) / 255 < MIN_REFERENCE_DISTANCE
    with pytest.raises(ValueError):
        MeterReferences(Side.P1, full, below)


@pytest.mark.parametrize(("full_count", "spent_count"), [(4, 5), (6, 5), (5, 4), (5, 6)])
def test_build_references_requires_exactly_five_samples(
    full_count: int, spent_count: int
):
    full, spent = calibration_crops()
    with pytest.raises(ValueError):
        build_meter_references(
            Side.P1,
            (full * 2)[:full_count],
            (spent * 2)[:spent_count],
        )


def test_build_references_rejects_wrong_side():
    full, spent = calibration_crops()
    with pytest.raises(TypeError):
        build_meter_references("P1", full, spent)


@pytest.mark.parametrize("hidden_collection", ["full", "spent"])
def test_build_references_rejects_hidden_sample_on_either_side(
    hidden_collection: str,
):
    full, spent = calibration_crops()
    crops = full if hidden_collection == "full" else spent
    crops[2] = np.full_like(crops[2], 120)
    with pytest.raises(ValueError):
        build_meter_references(Side.P1, full, spent)


def test_build_references_rejects_insufficient_full_spent_distance():
    full, _ = calibration_crops()
    with pytest.raises(ValueError):
        build_meter_references(Side.P1, full, [crop.copy() for crop in full])


def test_build_references_owns_outputs_without_mutating_or_sharing_inputs():
    full, spent = calibration_crops()
    originals = [crop.copy() for crop in full + spent]
    first = build_meter_references(Side.P1, full, spent)
    second = build_meter_references(Side.P2, full, spent)
    assert all(crop.flags.writeable for crop in full + spent)
    assert all(np.array_equal(crop, before) for crop, before in zip(full + spent, originals))
    assert not np.shares_memory(first.full_gray, first.spent_gray)
    assert not np.shares_memory(first.full_gray, second.full_gray)
    assert not np.shares_memory(first.spent_gray, second.spent_gray)
    assert not any(
        np.shares_memory(output, crop)
        for output in (first.full_gray, first.spent_gray, second.full_gray, second.spent_gray)
        for crop in full + spent
    )


def test_build_references_takes_the_per_pixel_median_of_distinct_samples(
    monkeypatch: pytest.MonkeyPatch,
):
    full_canonical = [
        np.full(METER_OUTPUT_SHAPE, value, dtype=np.uint8)
        for value in (10, 20, 30, 40, 50)
    ]
    spent_canonical = [
        np.full(METER_OUTPUT_SHAPE, value, dtype=np.uint8)
        for value in (180, 190, 200, 210, 220)
    ]
    for sample, value in zip(full_canonical, (200, 60, 90, 70, 80)):
        sample[0, 1] = value
    for sample, value in zip(spent_canonical, (255, 230, 240, 250, 245)):
        sample[0, 1] = value

    input_crops = [
        np.full((1, 1, 3), index, dtype=np.uint8) for index in range(10)
    ]
    canonical = [*full_canonical, *spent_canonical]
    calls: list[np.ndarray] = []

    def fake_preprocess(crop: np.ndarray) -> PreprocessedMeter:
        index = len(calls)
        assert crop is input_crops[index]
        calls.append(crop)
        return PreprocessedMeter(canonical[index], 120.0, 10.0, True)

    monkeypatch.setattr(
        meter_references_module, "preprocess_meter", fake_preprocess
    )

    references = build_meter_references(
        Side.P1, input_crops[:5], input_crops[5:]
    )

    expected_full = np.full(METER_OUTPUT_SHAPE, 30, dtype=np.uint8)
    expected_full[0, 1] = 80
    expected_spent = np.full(METER_OUTPUT_SHAPE, 200, dtype=np.uint8)
    expected_spent[0, 1] = 245
    assert len(calls) == len(input_crops)
    assert all(actual is expected for actual, expected in zip(calls, input_crops))
    assert np.array_equal(references.full_gray, expected_full)
    assert np.array_equal(references.spent_gray, expected_spent)


def reference_value(side: Side = Side.P1) -> MeterReferences:
    return MeterReferences(
        side,
        np.zeros(METER_OUTPUT_SHAPE, dtype=np.uint8),
        np.full(METER_OUTPUT_SHAPE, 255, dtype=np.uint8),
    )


def colour_model(side: Side = Side.P1) -> MeterColorModel:
    mask = np.zeros(METER_OUTPUT_SHAPE, dtype=np.bool_)
    mask[8:24, 8:56] = True
    return MeterColorModel(
        side,
        mask,
        full_hue=170,
        hue_tolerance=7,
        saturation_floor=120,
        value_floor=100,
        full_fill=0.90,
        spent_fill=0.05,
    )


def write_and_load_archive(
    tmp_path: Path,
    archive_factory: ArchiveFactory,
    name: str,
    payload: bytes,
    *,
    expected_side: Side = Side.P1,
) -> MeterReferences:
    archive_factory.write_reference(tmp_path, name, payload)
    return load_references(
        tmp_path,
        "profiles/default",
        f"refs/{name}",
        expected_side=expected_side,
    )


def corrupt_compressed_member(payload: bytes, member_name: str) -> bytes:
    data = bytearray(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        info = archive.getinfo(member_name)
    name_length, extra_length = struct.unpack_from("<HH", data, info.header_offset + 26)
    compressed_start = info.header_offset + 30 + name_length + extra_length
    assert info.compress_size > 2
    data[compressed_start + info.compress_size // 2] ^= 0x20
    return bytes(data)


def truncate_last_compressed_member(payload: bytes, member_name: str) -> bytes:
    data = bytearray(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = archive.infolist()
        info = archive.getinfo(member_name)
    assert info.header_offset == max(item.header_offset for item in infos)

    name_length, extra_length = struct.unpack_from("<HH", data, info.header_offset + 26)
    compressed_start = info.header_offset + 30 + name_length + extra_length
    removed_bytes = 2
    assert info.compress_size > removed_bytes
    truncated_size = info.compress_size - removed_bytes
    struct.pack_into("<I", data, info.header_offset + 18, truncated_size)
    del data[compressed_start + truncated_size : compressed_start + info.compress_size]

    cursor = data.find(b"PK\x01\x02")
    encoded_name = member_name.encode("utf-8")
    while cursor >= 0:
        central_name_length, central_extra_length, comment_length = struct.unpack_from(
            "<HHH", data, cursor + 28
        )
        name_start = cursor + 46
        if bytes(data[name_start : name_start + central_name_length]) == encoded_name:
            struct.pack_into("<I", data, cursor + 20, truncated_size)
            break
        cursor += 46 + central_name_length + central_extra_length + comment_length
        cursor = data.find(b"PK\x01\x02", cursor)
    else:
        raise AssertionError(f"central directory entry not found: {member_name}")

    eocd = data.rfind(b"PK\x05\x06")
    assert eocd >= 0
    central_offset = struct.unpack_from("<I", data, eocd + 16)[0]
    struct.pack_into("<I", data, eocd + 16, central_offset - removed_bytes)
    return bytes(data)


def add_member_general_purpose_flag(
    payload: bytes,
    member_name: str,
    flag: int,
    *,
    local: bool,
    central: bool,
) -> bytes:
    assert local or central
    data = bytearray(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        info = archive.getinfo(member_name)

    if local:
        local_flags = struct.unpack_from("<H", data, info.header_offset + 6)[0]
        struct.pack_into("<H", data, info.header_offset + 6, local_flags | flag)

    cursor = data.find(b"PK\x01\x02")
    encoded_name = member_name.encode("utf-8")
    while cursor >= 0:
        name_length, extra_length, comment_length = struct.unpack_from(
            "<HHH", data, cursor + 28
        )
        name_start = cursor + 46
        if bytes(data[name_start : name_start + name_length]) == encoded_name:
            if central:
                central_flags = struct.unpack_from("<H", data, cursor + 8)[0]
                struct.pack_into("<H", data, cursor + 8, central_flags | flag)
            return bytes(data)
        cursor += 46 + name_length + extra_length + comment_length
        cursor = data.find(b"PK\x01\x02", cursor)
    raise AssertionError(f"central directory entry not found: {member_name}")


def mark_member_encrypted(payload: bytes, member_name: str) -> bytes:
    return add_member_general_purpose_flag(
        payload, member_name, 0x1, local=True, central=True
    )


def test_reference_round_trip_is_canonical_sided_and_owned(tmp_path: Path):
    full_crops, spent_crops = calibration_crops()
    refs = build_meter_references(Side.P1, full_crops, spent_crops)
    save_references(tmp_path, "profiles/default.json", "refs/p1.npz", refs)
    loaded = load_references(
        tmp_path,
        "profiles/default.json",
        "refs/p1.npz",
        expected_side=Side.P1,
    )
    assert loaded.full_gray.shape == METER_OUTPUT_SHAPE
    assert loaded.full_gray.dtype == np.uint8
    assert loaded.full_gray.flags.owndata and not loaded.full_gray.flags.writeable
    assert loaded.spent_gray.flags.owndata and not loaded.spent_gray.flags.writeable
    assert not np.shares_memory(loaded.full_gray, loaded.spent_gray)


def test_hsv_references_round_trip_with_exact_model_members(tmp_path: Path):
    original = MeterReferences(
        Side.P1,
        np.zeros(METER_OUTPUT_SHAPE, dtype=np.uint8),
        np.full(METER_OUTPUT_SHAPE, 255, dtype=np.uint8),
        color_model=colour_model(Side.P1),
    )
    assert original.schema_version == 2
    save_references(tmp_path, "profiles/default.json", "refs/p1.npz", original)
    loaded = load_references(
        tmp_path,
        "profiles/default.json",
        "refs/p1.npz",
        expected_side=Side.P1,
    )

    assert loaded.color_model is not None
    assert loaded.color_model.side is Side.P1
    assert np.array_equal(loaded.color_model.active_mask, original.color_model.active_mask)
    assert not np.shares_memory(loaded.color_model.active_mask, original.color_model.active_mask)
    assert (
        loaded.color_model.full_hue,
        loaded.color_model.hue_tolerance,
        loaded.color_model.saturation_floor,
        loaded.color_model.value_floor,
        loaded.color_model.full_fill,
        loaded.color_model.spent_fill,
    ) == (
        original.color_model.full_hue,
        original.color_model.hue_tolerance,
        original.color_model.saturation_floor,
        original.color_model.value_floor,
        original.color_model.full_fill,
        original.color_model.spent_fill,
    )


def test_schema_one_archive_stays_legacy_without_colour_model(tmp_path: Path):
    save_references(
        tmp_path,
        "profiles/default.json",
        "refs/p1.npz",
        reference_value(Side.P1),
    )
    loaded = load_references(
        tmp_path,
        "profiles/default.json",
        "refs/p1.npz",
        expected_side=Side.P1,
    )

    assert loaded.schema_version == REFERENCE_SCHEMA_VERSION
    assert loaded.color_model is None


def test_unknown_colour_classifier_fails_closed(tmp_path: Path):
    original = MeterReferences(
        Side.P1,
        np.zeros(METER_OUTPUT_SHAPE, dtype=np.uint8),
        np.full(METER_OUTPUT_SHAPE, 255, dtype=np.uint8),
        color_model=colour_model(Side.P1),
    )
    save_references(tmp_path, "profiles/default.json", "refs/p1.npz", original)
    target = tmp_path / "profiles" / "refs" / "p1.npz"
    with zipfile.ZipFile(target, "r") as source:
        members = {
            info.filename: source.read(info.filename) for info in source.infolist()
        }
    members["classifier.npy"] = npy_bytes(np.array("unknown", dtype="<U19"))
    replacement = io.BytesIO()
    with zipfile.ZipFile(replacement, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload, compress_type=zipfile.ZIP_DEFLATED)
    target.write_bytes(replacement.getvalue())

    with pytest.raises(ValueError, match="classifier"):
        load_references(
            tmp_path,
            "profiles/default.json",
            "refs/p1.npz",
            expected_side=Side.P1,
        )


def test_valid_saved_npz_round_trip_stays_below_every_envelope_bound(tmp_path: Path):
    references = reference_value()
    save_references(
        tmp_path, "profiles/default", "refs/reference.NpZ", references
    )
    target = tmp_path / "profiles" / "refs" / "reference.NpZ"
    assert target.exists()
    assert not (tmp_path / "refs" / "reference.NpZ").exists()
    assert target.stat().st_size <= MAX_REFERENCE_ARCHIVE_BYTES

    with zipfile.ZipFile(target) as archive:
        infos = archive.infolist()
        assert {info.filename for info in infos} == set(EXPECTED_MEMBER_NAMES)
        assert len({info.filename for info in infos}) == len(EXPECTED_MEMBER_NAMES)
        assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in infos)
        assert all(info.file_size <= MAX_NPY_MEMBER_BYTES for info in infos)
        assert sum(info.file_size for info in infos) <= MAX_NPZ_UNCOMPRESSED_BYTES

    loaded = load_references(
        tmp_path,
        "profiles/default",
        "refs/reference.NpZ",
        expected_side=Side.P1,
    )
    assert loaded.side is Side.P1
    assert np.array_equal(loaded.full_gray, references.full_gray)
    assert np.array_equal(loaded.spent_gray, references.spent_gray)
    assert not np.shares_memory(loaded.full_gray, references.full_gray)
    assert not np.shares_memory(loaded.spent_gray, references.spent_gray)
    assert list(target.parent.glob("*.tmp")) == []


def test_profile_schema_and_reference_side_fail_closed(
    tmp_path: Path, archive_factory: ArchiveFactory
):
    profile_path = tmp_path / "profiles" / "default.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_profile(tmp_path, "profiles/default.json")

    payload = archive_factory.archive()
    archive_factory.write_reference(tmp_path, "p1.npz", payload)
    with pytest.raises(ValueError, match="side"):
        load_references(
            tmp_path,
            "profiles/default.json",
            "refs/p1.npz",
            expected_side=Side.P2,
        )


def test_load_and_save_references_resolve_below_profile_parent_and_reject_escapes(
    tmp_path: Path,
):
    references = reference_value()
    with pytest.raises(ValueError):
        save_references(
            tmp_path, "../default", "refs/p1.npz", references
        )
    with pytest.raises(ValueError):
        save_references(
            tmp_path, "profiles/default", "../../escape.npz", references
        )
    with pytest.raises(ValueError):
        save_references(
            tmp_path, "profiles/default", "refs/p1.npz.exe", references
        )
    assert not (tmp_path / "escape.npz").exists()


def test_reference_load_cannot_retreat_above_user_root_via_profile_directory_alias(
    tmp_path: Path, archive_factory: ArchiveFactory
):
    user_data_root = tmp_path / "data"
    user_data_root.mkdir()
    outside_archive = tmp_path / "escape.npz"
    original = archive_factory.archive()
    outside_archive.write_bytes(original)

    with pytest.raises(ValueError):
        load_references(
            user_data_root,
            ". ",
            "escape.npz",
            expected_side=Side.P1,
        )

    assert outside_archive.read_bytes() == original
    assert list(user_data_root.iterdir()) == []


def test_load_references_requires_typed_expected_side_before_io(tmp_path: Path):
    with pytest.raises(TypeError):
        load_references(
            tmp_path,
            "profiles/default",
            "refs/missing.npz",
            expected_side="P1",
        )


def test_outer_archive_over_one_mib_is_rejected_before_zip_parsing(
    tmp_path: Path,
    archive_factory: ArchiveFactory,
    monkeypatch: pytest.MonkeyPatch,
):
    oversized = b"\xff" + b" " * MAX_REFERENCE_ARCHIVE_BYTES
    archive_factory.write_reference(tmp_path, "oversized.npz", oversized)
    original_open = Path.open
    read_sizes: list[int] = []

    class TrackingReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self.stream.__exit__(exc_type, exc_value, traceback)

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return self.stream.read(size)

    def traced_open(path_value: Path, *args, **kwargs):
        return TrackingReader(original_open(path_value, *args, **kwargs))

    def forbidden_zip_parse(*args, **kwargs):
        raise AssertionError("ZIP parser must not see oversized outer archive")

    monkeypatch.setattr(
        "sf6_music_trigger.meter_references.Path.open", traced_open
    )
    monkeypatch.setattr(
        "sf6_music_trigger.meter_references.zipfile.ZipFile", forbidden_zip_parse
    )
    with pytest.raises(ValueError, match="1 MiB"):
        load_references(
            tmp_path,
            "profiles/default",
            "refs/oversized.npz",
            expected_side=Side.P1,
        )
    assert read_sizes == [MAX_REFERENCE_ARCHIVE_BYTES + 1]


@pytest.mark.parametrize("variant", ["missing", "extra", "duplicate", "directory"])
def test_npz_rejects_missing_extra_duplicate_and_directory_entries(
    tmp_path: Path, archive_factory: ArchiveFactory, variant: str
):
    if variant == "missing":
        payload = archive_factory.archive(missing={"full_gray.npy"})
    elif variant == "extra":
        payload = archive_factory.archive(extra=("extra.npy", b"extra"))
    elif variant == "duplicate":
        payload = archive_factory.archive(duplicate="full_gray.npy")
    else:
        members = archive_factory.valid_member_bytes()
        members["full_gray.npy/"] = members.pop("full_gray.npy")
        payload = archive_factory.archive(raw_members=members)

    archive_factory.write_reference(tmp_path, f"{variant}.npz", payload)
    with pytest.raises(ValueError):
        load_references(
            tmp_path,
            "profiles/default",
            f"refs/{variant}.npz",
            expected_side=Side.P1,
        )


def test_npz_rejects_encrypted_flag_before_member_read(
    tmp_path: Path, archive_factory: ArchiveFactory
):
    payload = mark_member_encrypted(
        archive_factory.archive(), "schema_version.npy"
    )
    archive_factory.write_reference(tmp_path, "encrypted.npz", payload)
    with pytest.raises(ValueError):
        load_references(
            tmp_path,
            "profiles/default",
            "refs/encrypted.npz",
            expected_side=Side.P1,
        )


@pytest.mark.parametrize(
    ("header_location", "flag"),
    [
        ("local", 0x0001),
        ("local", 0x0040),
        ("local", 0x2000),
        ("central", 0x0001),
        ("central", 0x0040),
        ("central", 0x2000),
        ("both", 0x0001),
        ("both", 0x0040),
        ("both", 0x2000),
    ],
    ids=[
        "local-traditional",
        "local-strong",
        "local-central-directory",
        "central-traditional",
        "central-strong",
        "central-central-directory",
        "both-traditional",
        "both-strong",
        "both-central-directory",
    ],
)
def test_npz_rejects_every_encryption_flag_in_either_zip_header_before_numpy_load(
    tmp_path: Path,
    archive_factory: ArchiveFactory,
    monkeypatch: pytest.MonkeyPatch,
    header_location: str,
    flag: int,
):
    payload = add_member_general_purpose_flag(
        archive_factory.archive(),
        "schema_version.npy",
        flag,
        local=header_location in {"local", "both"},
        central=header_location in {"central", "both"},
    )
    archive_factory.write_reference(
        tmp_path, f"encrypted-{header_location}-{flag:x}.npz", payload
    )
    monkeypatch.setattr(
        meter_references_module.np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("np.load ran before ZIP flag preflight"),
    )

    with pytest.raises(ValueError):
        load_references(
            tmp_path,
            "profiles/default",
            f"refs/encrypted-{header_location}-{flag:x}.npz",
            expected_side=Side.P1,
        )


@pytest.mark.parametrize("header_location", ["local", "central"])
def test_npz_rejects_local_central_general_purpose_flag_mismatch_before_numpy_load(
    tmp_path: Path,
    archive_factory: ArchiveFactory,
    monkeypatch: pytest.MonkeyPatch,
    header_location: str,
):
    payload = add_member_general_purpose_flag(
        archive_factory.archive(),
        "schema_version.npy",
        0x0004,
        local=header_location == "local",
        central=header_location == "central",
    )
    archive_factory.write_reference(
        tmp_path, f"flag-mismatch-{header_location}.npz", payload
    )
    monkeypatch.setattr(
        meter_references_module.np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("np.load ran before ZIP flag preflight"),
    )

    with pytest.raises(ValueError):
        load_references(
            tmp_path,
            "profiles/default",
            f"refs/flag-mismatch-{header_location}.npz",
            expected_side=Side.P1,
        )


def test_npz_rejects_non_deflated_member(
    tmp_path: Path, archive_factory: ArchiveFactory
):
    payload = archive_factory.archive(
        compression_overrides={"schema_version.npy": zipfile.ZIP_STORED}
    )
    archive_factory.write_reference(tmp_path, "stored.npz", payload)
    with pytest.raises(ValueError):
        load_references(
            tmp_path,
            "profiles/default",
            "refs/stored.npz",
            expected_side=Side.P1,
        )


def test_npz_rejects_declared_member_over_16_kib(
    tmp_path: Path, archive_factory: ArchiveFactory
):
    payload = archive_factory.archive(
        replacements={"full_gray.npy": b"x" * (MAX_NPY_MEMBER_BYTES + 1)}
    )
    archive_factory.write_reference(tmp_path, "member-too-large.npz", payload)
    with pytest.raises(ValueError):
        load_references(
            tmp_path,
            "profiles/default",
            "refs/member-too-large.npz",
            expected_side=Side.P1,
        )


def test_npz_rejects_declared_total_over_64_kib(
    tmp_path: Path, archive_factory: ArchiveFactory
):
    member_sizes = [10_923, 10_923, 10_923, 10_923, 10_923, 10_922]
    raw_members = {
        name: b"x" * size
        for name, size in zip(EXPECTED_MEMBER_NAMES, member_sizes)
    }
    assert all(len(value) <= MAX_NPY_MEMBER_BYTES for value in raw_members.values())
    assert sum(map(len, raw_members.values())) == MAX_NPZ_UNCOMPRESSED_BYTES + 1
    payload = archive_factory.archive(raw_members=raw_members)
    archive_factory.write_reference(tmp_path, "total-too-large.npz", payload)
    with pytest.raises(ValueError):
        load_references(
            tmp_path,
            "profiles/default",
            "refs/total-too-large.npz",
            expected_side=Side.P1,
        )


@pytest.mark.parametrize("variant", ["truncated-member", "bad-crc"])
def test_npz_rejects_truncated_and_bad_crc_members(
    tmp_path: Path,
    archive_factory: ArchiveFactory,
    variant: str,
):
    valid = archive_factory.archive()
    payload = (
        truncate_last_compressed_member(valid, "spent_gray.npy")
        if variant == "truncated-member"
        else corrupt_compressed_member(valid, "full_gray.npy")
    )
    archive_factory.write_reference(tmp_path, f"{variant}.npz", payload)
    with pytest.raises(ValueError):
        load_references(
            tmp_path,
            "profiles/default",
            f"refs/{variant}.npz",
            expected_side=Side.P1,
        )


@pytest.mark.parametrize("variant", ["truncated-payload", "appended-payload"])
def test_npy_rejects_truncated_or_appended_bytes_after_valid_header(
    tmp_path: Path,
    archive_factory: ArchiveFactory,
    variant: str,
):
    original = archive_factory.valid_member_bytes()["full_gray.npy"]
    malformed = original[:-1] if variant == "truncated-payload" else original + b"\x00"
    payload = archive_factory.archive(replacements={"full_gray.npy": malformed})
    archive_factory.write_reference(tmp_path, f"{variant}.npz", payload)
    with pytest.raises(ValueError):
        load_references(
            tmp_path,
            "profiles/default",
            f"refs/{variant}.npz",
            expected_side=Side.P1,
        )


def test_npy_rejects_object_array_before_numpy_load(
    tmp_path: Path,
    archive_factory: ArchiveFactory,
    monkeypatch: pytest.MonkeyPatch,
):
    object_member = npy_bytes(
        np.array({"danger": "pickle"}, dtype=object), allow_pickle=True
    )
    payload = archive_factory.archive(
        replacements={"schema_version.npy": object_member}
    )
    archive_factory.write_reference(tmp_path, "object.npz", payload)
    monkeypatch.setattr(
        meter_references_module.np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("np.load ran before NPY preflight"),
    )
    with pytest.raises(ValueError):
        load_references(
            tmp_path,
            "profiles/default",
            "refs/object.npz",
            expected_side=Side.P1,
        )


@pytest.mark.parametrize(
    ("name", "member_payload"),
    [
        ("malformed", b"not an npy member"),
        ("oversized-shape", header_only_npy(shape=(1_000_000_000,))),
    ],
)
def test_npy_rejects_malformed_header_and_oversized_declared_shape_without_allocation(
    tmp_path: Path,
    archive_factory: ArchiveFactory,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    member_payload: bytes,
):
    payload = archive_factory.archive(
        replacements={"full_gray.npy": member_payload}
    )
    archive_factory.write_reference(tmp_path, f"{name}.npz", payload)
    monkeypatch.setattr(
        meter_references_module.np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("np.load ran before NPY preflight"),
    )
    with pytest.raises(ValueError):
        load_references(
            tmp_path,
            "profiles/default",
            f"refs/{name}.npz",
            expected_side=Side.P1,
        )


def test_npz_duplicate_member_and_oversized_header_fail_preflight(
    tmp_path: Path, archive_factory: ArchiveFactory
):
    duplicate = archive_factory.duplicate_member("full_gray.npy")
    oversized = archive_factory.npy_header(
        "full_gray.npy", shape=(1_000_000_000,)
    )
    for name, payload in (("duplicate.npz", duplicate), ("oversized.npz", oversized)):
        archive_factory.write_reference(tmp_path, name, payload)
        with pytest.raises(ValueError):
            load_references(
                tmp_path,
                "profiles/default",
                f"refs/{name}",
                expected_side=Side.P1,
            )


def test_npz_accepts_valid_version_2_npy_headers(
    tmp_path: Path, archive_factory: ArchiveFactory
):
    members = {
        f"{name}.npy": npy_bytes(array, version=(2, 0))
        for name, array in reference_arrays().items()
    }
    payload = archive_factory.archive(raw_members=members)

    loaded = write_and_load_archive(
        tmp_path, archive_factory, "version-2.npz", payload
    )

    assert loaded.side is Side.P1
    assert np.array_equal(loaded.full_gray, np.zeros(METER_OUTPUT_SHAPE, np.uint8))
    assert np.array_equal(
        loaded.spent_gray, np.full(METER_OUTPUT_SHAPE, 255, np.uint8)
    )


@pytest.mark.parametrize("variant", ["version-3-header", "fortran-order"])
def test_npy_rejects_wrong_header_version_and_fortran_order(
    tmp_path: Path,
    archive_factory: ArchiveFactory,
    variant: str,
):
    replacement = (
        npy_bytes(
            np.zeros(METER_OUTPUT_SHAPE, dtype=np.uint8), version=(3, 0)
        )
        if variant == "version-3-header"
        else npy_bytes(
            np.asfortranarray(np.zeros(METER_OUTPUT_SHAPE, dtype=np.uint8))
        )
    )
    payload = archive_factory.archive(
        replacements={"full_gray.npy": replacement}
    )
    archive_factory.write_reference(tmp_path, f"{variant}.npz", payload)
    with pytest.raises(ValueError):
        load_references(
            tmp_path,
            "profiles/default",
            f"refs/{variant}.npz",
            expected_side=Side.P1,
        )


@pytest.mark.parametrize(
    ("case_name", "member_name", "array"),
    [
        ("schema-shape", "schema_version.npy", np.array([1], dtype=np.uint16)),
        ("schema-dtype", "schema_version.npy", np.array(1, dtype=np.uint32)),
        ("schema-value", "schema_version.npy", np.array(2, dtype=np.uint16)),
        (
            "preprocessing-shape",
            "preprocessing_version.npy",
            np.array([1], dtype=np.uint16),
        ),
        (
            "preprocessing-dtype",
            "preprocessing_version.npy",
            np.array(1, dtype=np.int16),
        ),
        (
            "preprocessing-value",
            "preprocessing_version.npy",
            np.array(2, dtype=np.uint16),
        ),
        ("side-shape", "side.npy", np.array(["P1"], dtype="<U2")),
        ("side-dtype", "side.npy", np.array("P1", dtype="<U3")),
        ("side-value", "side.npy", np.array("P3", dtype="<U2")),
        (
            "shape-hw-shape",
            "shape_hw.npy",
            np.array([[32, 64]], dtype=np.uint16),
        ),
        ("shape-hw-dtype", "shape_hw.npy", np.array([32, 64], dtype=np.uint32)),
        ("shape-hw-value", "shape_hw.npy", np.array([64, 32], dtype=np.uint16)),
        (
            "full-shape",
            "full_gray.npy",
            np.zeros((METER_OUTPUT_SHAPE[0], METER_OUTPUT_SHAPE[1] - 1), dtype=np.uint8),
        ),
        (
            "full-dtype",
            "full_gray.npy",
            np.zeros(METER_OUTPUT_SHAPE, dtype=np.uint16),
        ),
        (
            "spent-shape",
            "spent_gray.npy",
            np.zeros((METER_OUTPUT_SHAPE[0] - 1, METER_OUTPUT_SHAPE[1]), dtype=np.uint8),
        ),
        (
            "spent-dtype",
            "spent_gray.npy",
            np.zeros(METER_OUTPUT_SHAPE, dtype=np.float32),
        ),
    ],
)
def test_npz_rejects_wrong_member_scalar_shape_dtype_value_and_output_contract(
    tmp_path: Path,
    archive_factory: ArchiveFactory,
    case_name: str,
    member_name: str,
    array: np.ndarray,
):
    payload = archive_factory.archive(
        replacements={member_name: npy_bytes(array)}
    )
    archive_factory.write_reference(tmp_path, f"{case_name}.npz", payload)
    with pytest.raises(ValueError):
        load_references(
            tmp_path,
            "profiles/default",
            f"refs/{case_name}.npz",
            expected_side=Side.P1,
        )


def test_npz_rejects_stored_side_mismatch(tmp_path: Path, archive_factory: ArchiveFactory):
    payload = archive_factory.archive(
        replacements={"side.npy": npy_bytes(np.array("P2", dtype="<U2"))}
    )
    archive_factory.write_reference(tmp_path, "p2.npz", payload)
    with pytest.raises(ValueError, match="side"):
        load_references(
            tmp_path,
            "profiles/default",
            "refs/p2.npz",
            expected_side=Side.P1,
        )


def test_atomic_npz_replace_failure_preserves_old_neighbor_and_cleans_owned_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "profiles" / "refs" / "p1.npz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old npz bytes")
    neighbor = target.parent / "neighbor.tmp"
    neighbor.write_bytes(b"keep neighbor")
    owned_sources: list[Path] = []

    def fail_replace(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        source_path = Path(source)
        owned_sources.append(source_path)
        assert source_path.parent == target.parent
        assert Path(destination) == target
        raise OSError("injected NPZ replace failure")

    monkeypatch.setattr("sf6_music_trigger.meter_references.os.replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        save_references(
            tmp_path, "profiles/default", "refs/p1.npz", reference_value()
        )

    assert target.read_bytes() == b"old npz bytes"
    assert neighbor.read_bytes() == b"keep neighbor"
    assert len(owned_sources) == 1 and not owned_sources[0].exists()
    assert sorted(path.name for path in target.parent.iterdir()) == [
        "neighbor.tmp",
        "p1.npz",
    ]


def test_atomic_npz_save_uses_unique_temp_names_across_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sources: list[Path] = []

    def fail_replace(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        sources.append(Path(source))
        raise OSError("injected")

    monkeypatch.setattr("sf6_music_trigger.meter_references.os.replace", fail_replace)
    for _ in range(2):
        with pytest.raises(OSError):
            save_references(
                tmp_path, "profiles/default", "refs/p1.npz", reference_value()
            )

    assert len(sources) == 2
    assert len(set(sources)) == 2
    assert all(source.suffix == ".tmp" and not source.exists() for source in sources)
    assert list((tmp_path / "profiles" / "refs").iterdir()) == []


def test_npz_save_revalidates_tampered_object_before_touching_target(tmp_path: Path):
    references = reference_value()
    object.__setattr__(references, "schema_version", 2)
    with pytest.raises(ValueError):
        save_references(
            tmp_path, "profiles/default", "refs/p1.npz", references
        )
    assert not (tmp_path / "profiles").exists()
