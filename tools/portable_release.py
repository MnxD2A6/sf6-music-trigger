"""Assemble privacy-safe files for a portable SF6 Music Trigger V2 release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence
from zipfile import ZIP_DEFLATED, ZipFile


WINRT_CAPTURE_MODULES = (
    "winrt.runtime",
    "winrt.windows.foundation",
    "winrt.windows.foundation.collections",
    "winrt.windows.graphics",
    "winrt.windows.graphics.capture",
    "winrt.windows.graphics.capture.interop",
    "winrt.windows.graphics.directx",
    "winrt.windows.graphics.directx.direct3d11",
    "winrt.windows.graphics.directx.direct3d11.interop",
)


# Imported lazily inside the app (importlib / function-level imports).
V2_HIDDEN_IMPORTS = (
    "pygame.sndarray",
    "sf6_music_trigger.game_ui",
    "sf6_music_trigger.ref_engine",
    "sf6_music_trigger.reframework_events",
)
RELEASE_VERSION = "2.0.0b2"


class ReleaseValidationError(ValueError):
    """Raised when release input cannot be copied safely."""


def pyinstaller_command(
    *,
    python: Path,
    repo_root: Path,
    build_root: Path,
) -> list[str]:
    command = [
        str(python),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        "SF6 Music Trigger",
        "--paths",
        str(repo_root / "src"),
        "--collect-submodules",
        "dxcam",
        "--copy-metadata",
        "dxcam",
        "--hidden-import",
        "pygame",
        "--add-data",
        f"{repo_root / 'src' / 'sf6_music_trigger' / 'game_scripts'};sf6_music_trigger/game_scripts",
        "--exclude-module",
        "pytest",
        "--distpath",
        str(build_root / "dist"),
        "--workpath",
        str(build_root / "work"),
        "--specpath",
        str(build_root / "spec"),
        str(repo_root / "tools" / "portable_entry.py"),
    ]
    entry = command.pop()
    for module_name in (*WINRT_CAPTURE_MODULES, *V2_HIDDEN_IMPORTS):
        command.extend(("--hidden-import", module_name))
    command.append(entry)
    return command


def group_test_readme() -> str:
    return f"""SF6 Music Trigger V2（{RELEASE_VERSION}）群测版

这是什么
- 在 Street Fighter 6（SF6）里放出 SA1/SA2/SA3/CA 时，自动播放你选的音乐；
  空大或被防时可以改放打空音效，或者淡出主音乐。
- V2 直接在游戏内读取超必杀事件（需要 REFramework），不再识别画面，
  所以换 UI、换分辨率、全屏都不受影响，全角色通用。
- 只用于离线：训练模式、CPU 对战、本地对战。检测到联机对战时会自动停用。
  打排位或联机前，建议把游戏目录里的 dinput8.dll 改名（例如 dinput8.dll.off），
  离线时再改回来。
- 不读写存档，不发送任何按键输入，也不上传任何数据。

安装（解压即用，不写注册表）
1. 完整解压本 ZIP 到任意文件夹，不要直接在压缩包里运行。
2. 把要用的音乐（wav / ogg / mp3）放进 music 文件夹；
   也可以放在任何位置，然后在面板里选择。
3. 关闭游戏，双击“SF6 Music Trigger.exe”，确认窗口标题包含“V2”。旧版窗口请先关闭。
4. “检测方式”选“REFramework 游戏内检测（推荐）”。
   “SF6 游戏目录”一般会自动找到；没找到就点“选择目录”。
5. 点“一键安装/更新”。它会把附带的 REFramework（dinput8.dll）和 V2 的两个游戏内脚本
   放进游戏目录；如果你已经装了自己的 REFramework，可以选择保留。
   以后更新 V2 时也点一次。
6. 为 P1 / P2 选择音乐、音量和大招范围，点“开始监听”，然后启动游戏。
7. 建议勾选“启动后自动开始监听并最小化”，以后打开就会自动在后台工作。

卸载
1. 关闭游戏，在面板里点“从游戏中移除”：
   - 删除 V2 放进游戏的两个脚本和日志文件夹；
   - 如果游戏里没有其他 REFramework Mod，还会询问是否连 REFramework 一起删除，
     选“是”后游戏目录恢复原样（游戏本身的文件从来不会被改动）。
2. 再删除本程序文件夹即可，没有其他残留。

在游戏里改设置
- 游戏里按 REFramework 的菜单键（默认是 Insert）打开菜单时，
  会同时出现“SF6 音乐设置”窗口，可以直接改：启用、音乐、音量、大招范围、
  空大/被防时的处理、打空音效、开始/停止监听、应用音频模板。
- 改动立即生效并自动保存，不需要切出游戏。
- 键盘没有 Insert 键：REFramework 菜单在游戏启动时会自动弹出，
  在 REFramework Config 里把 Menu Key 换成其他键（例如 F7）。
- 窗口显示“未连接 V2 播放器”时，说明 V2 没有在运行。

音量
- 100% 是音乐文件本身的音量。
- 勾选“解除音量上限”后最高 500%，放大的部分带防爆音处理。
- “重置音量”把 P1/P2 的音乐和打空音效都恢复到 100%。

播放规则
- 全部大招：SA1/SA2 播满 4 秒后淡出；SA3/CA 命中会一直播放；
  被防或空大时执行打空处理（播放打空音效，或者满 4 秒后淡出）。
- 仅 SA3/CA：SA1/SA2 不播放，其余同上。
- 空大在超必杀的攻击判定帧结束后立即判定（约 2 秒内）；被防在对手防住时立即判定。
- 连段断掉后接的超必杀，会按实际结果（被防/空大）判定，不会误当成命中。
- 双方都放超必杀时，后放的一方接管音乐。

已知限制
- 飞行道具类超必杀（波动拳一类）的空大要等收招后才判定。
- 已实测：杰米、不知火舞、桑吉尔夫，对手为茱莉。其他角色按游戏数据通用识别，欢迎反馈。
- 旧版画面识别仍然保留：“检测方式”选“画面识别（旧版 T03）”即可使用，
  这个模式需要“校准 HUD”。

反馈请附
- 面板里点“打开诊断文件夹”，会打开游戏目录下 reframework\\data\\SF6MusicTrigger，
  请把其中的 events.jsonl 一起发来。日志里只有角色编号和动作编号，
  不含截图、音乐或个人信息。
- 使用的角色、出了什么问题、大概的时间；方便的话附上录屏。
- 可以用 SHA256SUMS.txt 核对文件是否完整。
"""


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseValidationError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ReleaseValidationError(f"{label} must be a JSON object")
    return value


def _sanitized_side(raw: object) -> dict[str, object]:
    side = raw if isinstance(raw, dict) else {}
    scope = side.get("super_scope", "SA3_CA_ONLY")
    if scope not in {"ALL_SUPERS", "SA3_CA_ONLY"}:
        scope = "SA3_CA_ONLY"
    return {
        "enabled": bool(side.get("enabled", True)),
        "main_track": "",
        "volume": side.get("volume", 0.18),
        "whiff_track": "",
        "whiff_volume": side.get("whiff_volume", 0.18),
        "play_whiff_sound": False,
        "whiff_fade_ms": side.get("whiff_fade_ms", 1500),
        "super_scope": scope,
        "fade_non_three": side.get("fade_non_three", False) is True,
    }


def _profile_references(profile: dict[str, Any]) -> tuple[str, str]:
    references: list[str] = []
    for side_name in ("p1", "p2"):
        side = profile.get(side_name)
        if not isinstance(side, dict):
            raise ReleaseValidationError(f"profile {side_name} is missing")
        reference = side.get("references_path")
        if not isinstance(reference, str) or not reference:
            raise ReleaseValidationError(f"profile {side_name} reference is missing")
        references.append(reference)
    return references[0], references[1]


def _resolve_reference(profile_root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.drive:
        raise ReleaseValidationError("profile reference must be relative")
    try:
        resolved_root = profile_root.resolve(strict=True)
        resolved = (profile_root / candidate).resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ReleaseValidationError("profile reference is missing or escapes its root") from exc
    if not resolved.is_file():
        raise ReleaseValidationError("profile reference is missing or is not a file")
    return resolved


def prepare_payload(
    *,
    release_root: Path,
    source_config: Path,
    source_profile: Path,
    commit: str,
) -> None:
    config = _read_object(source_config, label="source config")
    profile = _read_object(source_profile, label="source profile")
    references = _profile_references(profile)
    resolved_references = tuple(
        _resolve_reference(source_profile.parent, reference) for reference in references
    )

    release_root.mkdir(parents=True, exist_ok=True)
    profile_root = release_root / "profiles"
    profile_root.mkdir(parents=True, exist_ok=True)
    destination_profile = profile_root / source_profile.name
    shutil.copy2(source_profile, destination_profile)
    for relative, source in zip(references, resolved_references, strict=True):
        destination = profile_root / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    sanitized = {
        "schema_version": config.get("schema_version", 1),
        "p1": _sanitized_side(config.get("p1")),
        "p2": _sanitized_side(config.get("p2")),
        "profile_path": f"profiles/{source_profile.name}",
        "capture_fps": config.get("capture_fps", 20),
        # V2 defaults: in-game detection; the game folder is found at startup.
        "detection_backend": "reframework",
        "game_dir": "",
        "volume_unlimited": False,
        "auto_start": False,
    }
    (release_root / "config.json").write_text(
        json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (release_root / "VERSION.txt").write_text(
        f"SF6 Music Trigger V2 {RELEASE_VERSION}\n"
        f"Source commit: {commit}\n"
        "Status: group-test beta; REFramework detection verified offline in SF6 training mode.\n",
        encoding="utf-8",
    )
    music = release_root / "music"
    music.mkdir(exist_ok=True)
    (music / "把音乐放在这里.txt").write_text(
        "把 wav / ogg / mp3 音乐放进这个文件夹，游戏内设置窗口的音乐列表里就能选到。\n",
        encoding="utf-8",
    )


REFRAMEWORK_BUNDLE_FILES = ("dinput8.dll", "LICENSE.txt", "revision.txt")


def copy_reframework_bundle(release_root: Path, reframework_dir: Path) -> Path:
    """Ship REFramework (MIT) next to the exe; the panel installs it on request."""

    missing = [name for name in REFRAMEWORK_BUNDLE_FILES if not (reframework_dir / name).is_file()]
    if missing:
        raise ReleaseValidationError(f"REFramework bundle is missing: {', '.join(missing)}")
    target = release_root / "REFramework"
    target.mkdir(parents=True, exist_ok=True)
    for name in REFRAMEWORK_BUNDLE_FILES:
        shutil.copy2(reframework_dir / name, target / name)
    (target / "README.txt").write_text(
        "这是 V2 附带的 REFramework（MIT 许可，见 LICENSE.txt）。\n"
        "不需要手动复制：在 V2 面板里点“一键安装/更新”即可装进游戏，"
        "点“从游戏中移除”可以再删掉。\n"
        "游戏大更新后如果 REFramework 失效，可以到官方 GitHub "
        "（praydog/REFramework-nightly）下载新版 dinput8.dll 替换这里的文件。\n",
        encoding="utf-8",
    )
    return target


def write_sha256_manifest(release_root: Path) -> Path:
    manifest = release_root / "SHA256SUMS.txt"
    lines: list[str] = []
    for path in sorted(
        (candidate for candidate in release_root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(release_root).as_posix(),
    ):
        if path == manifest:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(release_root).as_posix()
        lines.append(f"{digest}  {relative}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def validate_public_text(release_root: Path) -> None:
    text_suffixes = {".cmd", ".json", ".md", ".ps1", ".toml", ".txt"}
    drive_path = re.compile(r"(?i)\b[a-z]:[\\/]")
    for path in release_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ReleaseValidationError(f"public text is not readable: {path.name}") from exc
        if drive_path.search(text) or "\\\\Users\\" in text:
            raise ReleaseValidationError(
                f"public text contains an absolute Windows path: {path.name}"
            )


def create_release_zip(release_root: Path) -> Path:
    archive = release_root.parent / f"{release_root.name}.zip"
    with ZipFile(archive, mode="w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(
            (candidate for candidate in release_root.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(release_root).as_posix(),
        ):
            relative = path.relative_to(release_root.parent).as_posix()
            bundle.write(path, arcname=relative)
    return archive


def validate_no_forbidden_bytes(
    release_root: Path,
    markers: tuple[str, ...],
) -> None:
    needles = tuple(
        encoded
        for marker in markers
        if marker
        for encoded in (marker.encode("utf-8"), marker.encode("utf-16-le"))
    )
    if not needles:
        return
    overlap_size = max(len(needle) for needle in needles) - 1
    for path in release_root.rglob("*"):
        if not path.is_file():
            continue
        previous = b""
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                haystack = previous + chunk
                if any(needle in haystack for needle in needles):
                    raise ReleaseValidationError(
                        f"release contains a forbidden builder marker: {path.name}"
                    )
                previous = haystack[-overlap_size:] if overlap_size else b""


def install_tk_runtime(
    *,
    release_root: Path,
    python_root: Path,
    data_root: Path,
) -> None:
    internal = release_root / "_internal"
    required_files = ("_tkinter.pyd", "tcl86t.dll", "tk86t.dll")
    required_data = ("_tcl_data", "_tk_data")
    tkinter_source = python_root / "tkinter"
    if not tkinter_source.is_dir():
        raise ReleaseValidationError("portable tkinter package is missing")
    for name in required_files:
        if not (python_root / name).is_file():
            raise ReleaseValidationError(f"portable Tk runtime file is missing: {name}")
    for name in required_data:
        if not (data_root / name).is_dir():
            raise ReleaseValidationError(f"portable Tk data directory is missing: {name}")

    shutil.copytree(
        tkinter_source,
        internal / "tkinter",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in required_files:
        shutil.copy2(python_root / name, internal / name)
    for name in required_data:
        shutil.copytree(data_root / name, internal / name, dirs_exist_ok=True)


def build_portable_release(
    *,
    python: Path,
    repo_root: Path,
    build_root: Path,
    release_root: Path,
    source_config: Path,
    source_profile: Path,
    commit: str,
    reframework_dir: Path,
    forbidden_markers: tuple[str, ...] = (),
    tk_python_root: Path | None = None,
    tk_data_root: Path | None = None,
) -> Path:
    if build_root.exists():
        raise ReleaseValidationError("build output already exists; choose a new build root")
    archive = release_root.parent / f"{release_root.name}.zip"
    if release_root.exists() or archive.exists():
        raise ReleaseValidationError("release output already exists; refusing to overwrite it")

    build_root.mkdir(parents=True)
    subprocess.run(
        pyinstaller_command(
            python=python,
            repo_root=repo_root,
            build_root=build_root,
        ),
        cwd=repo_root,
        check=True,
    )
    frozen_root = build_root / "dist" / "SF6 Music Trigger"
    executable = frozen_root / "SF6 Music Trigger.exe"
    if not executable.is_file():
        raise ReleaseValidationError("PyInstaller did not produce the expected executable")

    shutil.copytree(frozen_root, release_root)
    if (tk_python_root is None) != (tk_data_root is None):
        raise ReleaseValidationError("both Tk runtime roots must be provided together")
    if tk_python_root is not None and tk_data_root is not None:
        install_tk_runtime(
            release_root=release_root,
            python_root=tk_python_root,
            data_root=tk_data_root,
        )
    prepare_payload(
        release_root=release_root,
        source_config=source_config,
        source_profile=source_profile,
        commit=commit,
    )
    copy_reframework_bundle(release_root, reframework_dir)
    for name in ("LICENSE", "PRIVACY.md", "THIRD_PARTY.md"):
        source = repo_root / name
        if not source.is_file():
            raise ReleaseValidationError(f"required release document is missing: {name}")
        shutil.copy2(source, release_root / name)
    (release_root / "README-群测说明.txt").write_text(
        group_test_readme(),
        encoding="utf-8",
    )

    validate_public_text(release_root)
    validate_no_forbidden_bytes(release_root, forbidden_markers)
    subprocess.run(
        [str(release_root / "SF6 Music Trigger.exe"), "--portable-self-check"],
        cwd=release_root,
        check=True,
        timeout=90,
    )
    write_sha256_manifest(release_root)
    return create_release_zip(release_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument(
        "--reframework-dir",
        type=Path,
        required=True,
        help="folder with REFramework dinput8.dll, LICENSE.txt and revision.txt",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--forbidden-marker", action="append", default=[])
    parser.add_argument("--tk-python-root", type=Path)
    parser.add_argument("--tk-data-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    archive = build_portable_release(
        python=args.python.resolve(),
        repo_root=args.repo_root.resolve(),
        build_root=args.build_root.resolve(),
        release_root=args.release_root.resolve(),
        source_config=args.source_config.resolve(),
        source_profile=args.source_profile.resolve(),
        commit=args.commit,
        reframework_dir=args.reframework_dir.resolve(),
        forbidden_markers=tuple(args.forbidden_marker),
        tk_python_root=args.tk_python_root.resolve() if args.tk_python_root else None,
        tk_data_root=args.tk_data_root.resolve() if args.tk_data_root else None,
    )
    print(f"Release folder: {args.release_root.resolve()}")
    print(f"Release archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
