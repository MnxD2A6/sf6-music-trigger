# SF6 Meter-Level Group-Test r6 Release Evidence

Date: 2026-08-23 (Asia/Singapore)

## Scope and source

- Branch: `background-independent-meter`
- Source commit: `9e4fefc0e3056022a726a98968ab04968600a00b`
- Local rollback tag: `backup/meter-level-detector-20260822`
- Rollback tag target: `9e4fefc0e3056022a726a98968ab04968600a00b`
- Python: 3.12.6
- PyInstaller: 6.22.2 with the already-installed local shared packages
- No network access, dependency download, push, PR, SF6 launch, or Computer Use

The old prototype `main.py`, runtime user configuration, selected audio,
recordings, and private profiles were not changed or packaged.

## Offline verification

The focused production-path suite was executed twice from fresh temp roots:

```text
1364 passed in 93.80s
1364 passed in 94.75s
```

The complete offline suite was then executed from another fresh temp root:

```text
1727 passed in 102.91s
```

The replay details and input hashes are recorded in
`docs/verification/2026-08-22-meter-level-replay.md`.

## Performance safety

SF6 and SF6 Music Trigger remained stopped. Before the final verification
runs, CPU was 8.2% then 6.5%, GPU utilization was 0%, dedicated GPU memory was
0 MB, and GPU temperature was 35-36°C. After packaging, CPU was 11.3%, GPU
utilization was 0%, dedicated GPU memory was 0 MB, and GPU temperature was
36°C. No 99% GPU condition was reproduced during the offline work.

## Portable artifact

- Folder: `SF6_Music_Trigger_GroupTest_0.1.0a1_20260822_r6`
- ZIP: `SF6_Music_Trigger_GroupTest_0.1.0a1_20260822_r6.zip`
- Folder files: 1,075
- Folder bytes: 178,420,612
- ZIP bytes: 69,374,184
- ZIP SHA-256:
  `BAC0FE6833EB75BBD9F44F8B01FA0CCBB55BE2F6C02F03667BA1F2B371F78388`
- EXE SHA-256:
  `95C58BE91CA116413FA29C83B5AA448230DC5A30400F406510B31234793C8F17`
- Manifest SHA-256:
  `C4EE90A37F94AA348CB7138BC8E0E8AC92A98B6A4F42CCA3CB55E65A897A1719`
- Sanitized config SHA-256:
  `B972A3B1EDC0F055655B57BF8391CEFC71BFD97204F32CD04D973C5BD0B79D42`
- Bundled profile SHA-256:
  `9D224830E7EB3980751E7FA315935FC49249E348BA05D84B81DC12F4F9D8D0B4`

The release config contains blank main/whiff audio paths, disables whiff audio
by default, selects `SA3_CA_ONLY`, and points only to the bundled profile.
There are zero packaged audio/video files.

## Independent extraction verification

The ZIP was expanded into a new directory. Both the original and extracted
EXE returned exit code 0 for `--portable-self-check`. All 1,074 manifest
entries matched their extracted files. The self-check confirms the sanitized
config/profile and bundled NumPy, OpenCV, Pygame, DXcam, and Tcl/Tk imports.

A binary/text scan found none of the builder's private user path, recording
names, recordings directory, or selected-audio directory in the release.
The repository worktree was clean after packaging.

## Acceptance status

SF6 实机验收尚未执行. This is an offline group-test candidate, not a claim of
real gameplay acceptance. A later SF6-only session must still verify natural
gain, SA1, SA2, SA3/CA, hit, block, whiff, round transition, both sides, and
gameplay performance. MK1 and other applications remain out of scope.
