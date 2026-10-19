# SF6 Meter-Level Offline Replay Evidence

Date: 2026-08-23 (Asia/Singapore)

## Scope

This is offline replay evidence only. It does not start or control SF6, does
not play audio, and is not SF6 real acceptance. MK1 and all other applications
were outside scope.

The replay uses the production `observe_gauge` and `GaugeTracker` path at the
configured 20 FPS. Output is scalar event data only; no frames, crops, glyph
templates, or audio paths are exported.

## Inputs

- Recording: `实际测试，误触，没触发，出发错误音频.mp4`
- Recording size: 890,666,208 bytes
- Recording SHA-256:
  `971E3DBB2DF827CED2C22F5BDB1EBE99E8ABB07D680CBDBDE65395C3F731F7D0`
- Profile ID: `sf6-screenshot-color-20260816`
- Profile SHA-256:
  `9D224830E7EB3980751E7FA315935FC49249E348BA05D84B81DC12F4F9D8D0B4`
- Python: 3.12.6
- Starting tracked commit: `e766fa610734eb5bd90c4cacbd6ac2eb87cf3f15`

## Reproduction

From the repository root, resolve the recording and calibrated profile from
their local roots, then run:

```powershell
& $python tools\replay_meter_levels.py `
  --video $recording `
  --profile-root $profileRoot `
  --profile profiles/screenshot-sf6-color-20260816.json `
  --csv $scalarCsv `
  --expect-count 4
```

Final command result:

```text
processed_frames=9373 sample_fps=20 events=4
event=137.733333,P1,3,0,3
event=234.400000,P1,3,0,3
event=387.800000,P1,3,0,3
event=399.533333,P2,3,0,3
```

The last confirmation follows the 399.067-second action landmark because the
numeric HUD is occluded until it restores. The scalar CSV SHA-256 is
`F0D2ECCE2632496E9E1C852793565B980717EEA323E0AFCBDF148F1CF35E2BA8`.

No event was emitted at the documented non-Super landmarks near 56.467,
154.800, 188.000, 297.067, 322.867, or 448.000 seconds.

The previously reported pre-action autoplay recording `不说了.mp4`
(SHA-256
`34D33D9701704F635034A69FA389A46A7E167D264019227E24B35A8ABD558AEB`)
was also replayed through the same production path. Before the cross-signal
gate it produced four false P2 `3->1` transitions while the trustworthy colour
fill remained between 0.81 and 0.99. After the gate it produced zero events
and zero audio action batches.

## Prior Positive Recordings

The same production path and profile were replayed without an expected-count
override:

| Recording | SHA-256 | Frames | Scalar events |
| --- | --- | ---: | --- |
| `杰米测试3.mp4` | `0D40A756732574B08F2449870128C04CCC03A6F3D3C3A42A6D933CB5E6599BF7` | 871 | `1.900 P1 3->0`, `14.300 P1 3->1`, `23.000 P1 3->2`, `24.333 P2 3->1` |
| `终于基本成功了，谢谢你S02.mp4` | `1F70392D3FB3EB76EA5076A49E54099DC81B0D594E2155DE1CF1B30BD183B0F0` | 889 | `3.733 P1 3->0`, `16.133 P1 3->0`, `29.867 P1 3->0`, `32.100 P2 3->2` |

Their scalar CSV hashes are respectively
`E3C4962DD52CFC2489F93258BF72B385ECEDA1954D483027DA462FDA495A20E2`
and
`F293E75B307E98347EB1BF1E46B44C798F1F58C261F99C1E54F01056B0A9D8B0`.

## Performance Safety

SF6 and SF6 Music Trigger were not running during replay. Before the final
full replay, system measurements were CPU 13.5%, GPU 0%, dedicated GPU memory
0 MB, and GPU temperature 38°C. Earlier monitored replay samples stayed at GPU
0%, 41-47°C, with CPU samples between 28.8% and 41.9%. The offline detector
therefore did not reproduce the user's prior 99% GPU condition.

Before the final focused and full-suite verification, measurements were CPU
8.2% then 6.5%, GPU utilization 0%, dedicated GPU memory 0 MB, and GPU
temperature 35-36°C. SF6 and the trigger remained stopped.

## Offline Test Suites

- Focused production-path suite, first run: 1364 passed in 93.80 seconds.
- Focused production-path suite, repeat run: 1364 passed in 94.75 seconds.
- Complete offline suite: 1727 passed in 102.91 seconds.

## Result

PASS for this recording's offline event-count and false-positive regression.
SF6 实机验收尚未执行. A later SF6-only acceptance must still cover natural
gain, SA1, SA2, SA3/CA, hit, block, whiff, round transition, and both sides
while monitoring gameplay performance.
