# R26 bounded CA health evidence (2026-09-10)

Source freeze:
- outcome.py SHA256: 980B4EF732A6116E1C86D733FCABD71A4B32CC4D4CEE21F711F4B37B3659B68F
- contact_hud.py SHA256: 10798851E7DBA82BF414F81D3D3EF09EA7AC4939C397D0E5C22827E39CD7E714

Only these two production files were edited. Existing uncommitted changes were
preserved. New tests: test_r26_ca_video.py and test_r26_ca_safety.py. No shared
replay assertion, engine, audio, profile, release configuration, or package edit.

## Independently labelled local sources

All sources are local under the user's Videos/屏幕录制/SF6 directory; no upload,
download, audio playback, live game, installation, or OCR generation was used.

| Source | Dimensions | Titlebar removed | SHA256 |
| --- | --- | --- | --- |
| R25.mp4 | 1902x1104, 30fps | 26 | 04D233FA87664B1781628B351C607545E0C38938AEB8D07515E6EF7870A43986 |
| R23.mp4 | 1916x1092, 30fps | 26 | 2FBE367DA03A1FAE3EE3C3CAA8EB59C9A2EBAF967AFBD6700CDC9765F1BA63D7 |
| 杰米测试3.mp4 | 2560x1440, 30fps | 0 | 0D40A756732574B08F2449870128C04CCC03A6F3D3C3A42A6D933CB5E6599BF7 |

R25 430..440 and R23 116/424 CA hit labels come from the user's visual diagnosis.
The tests decode actual frames. Absent originals alone cause skip; existing but
unreadable/wrong-sized video fails. The local 148-test run had zero skips.

R25 native log: candidate 99742.2974442, confirmed 99742.9100065;
UNKNOWN hud_unavailable_at_deadline and fade at 99750.2974533. Observer golden
uses audio-aligned 430.067/430.680 seconds. Native diagnostic has no frame IDs;
429.5s pre-spend anchor is borrowed from the existing offline meter replay.
This is explicitly an observer golden, not full-engine confirmation.

## Historical Jamie label correction evidence

Contact sheet: r26-ca-jamie-evidence.jpg. Independently viewed original frames:
at 3.20 seconds training combo damage is 2090; at 3.30 it is 2513 and the
opponent is in the landed cinematic. At 5 and 8 seconds the cinematic continues.
This supports HIT after 3.30, independently of the health reader's output.

Reader trace: source frames 45/47 (1.500/1.567) target health ~26.5%, attacker
25%; frame 96 (3.200) same; frame 98 (3.267) target ~23.1%; frames 99/100
(3.300/3.333) target ~21.6%, attacker still 25%. Two stable lower observations
give HIT at historical replay clock 3.35 (source frame 100 due 20Hz rounding).
The new historical golden rejects any earlier terminal HIT and any later UNKNOWN.

Actual full historical engine replay: 871 frames, zero stale rejections,
confirmations at 1.95/14.55/23.05, three candidates, zero cancellations.
One outcome: HIT at 3.35, evidence [opponent_health_changed]. Audio batches:
only stop/play at 1.90; no fade at 9.90. Existing shared UNKNOWN assertion is
intentionally left untouched for parent review. Parent four-second hold changes
the unrelated synthetic UNKNOWN fade from 3.65 to 4.65, not its label/time.

## Narrow production change

Default legacy contact path remains intact. The opt-in experimental contact
classifier stays off. A separate health-only reader checks original geometry
and the R25 -4 reference-pixel vertical offset, retaining outline, endpoint,
palette and live-anchor validation. Conflicting extents abstain; no live pixels
never implies zero health. No timer/drive BLOCK classifier is enabled.

Supplemental HIT requires an explicit confirmed three-bar spend, a cached
pre-spend anchor, and two stable valid baseline observations. Baseline is at
most 1.5s before activation and within .75s of the anchor; consecutive baseline
samples are at most .25s apart. A newer changed reading cannot borrow an older
plateau. Target loss must be >=2.5% of total health and stable within 1.2% for
the configured contact sample count (default two); sample gap <=.25s.
Only post-suppression evidence counts, including the immediately preceding
cached sample when confirmation arrives late. Observed attacker damage,
target recovery above the baseline or changed dimensions invalidates supplemental attribution.
Attacker occlusion alone is allowed when the target health remains validated.
No evidence is admitted after the existing hard deadline. No timeouts extend.

Limits: default HUD geometry/palette only; a short visible window may be missed
at lower capture rates (10Hz R25 failed, 20/30Hz passed). Missing baseline,
occlusion or other insufficient evidence retains the existing UNKNOWN fallback.
This is health-damage evidence tied to a confirmed spend, not a general move
classifier; chip/armor/trade and other unrepresented match states remain limits.
No full-engine R25 success is claimed here: parent owns that continuous replay
and live/listening acceptance remains outside this task.

## TDD and validation

Initial actual-video goldens: 3 failed, all observed UNKNOWN instead of HIT.
After geometry/baseline path, R23 passed but R25 remained red; 20/30Hz R25 both
failed before adding the immediately preceding post-suppression sample.
Safety tests then exposed borrowing an older plateau and ignoring attacker
damage in the cached confirmation sample; both failed before their corrections.

Final targeted command:
`../.r21-runtime/Scripts/python.exe -B -m pytest tests/test_r26_ca_video.py tests/test_r26_ca_safety.py tests/test_contact_hud.py tests/test_contact_outcome.py tests/test_outcome.py -q -p no:cacheprovider --tb=short`

Result: 148 passed in 37.04s, zero skips. Includes six CA video combinations,
three R25 non-super controls, historical Jamie, real blocked/unchanged HUD,
missing/stale baseline, tier scope, continuity, geometry, deadline, flash and
occlusion controls, legacy outcomes and all existing contact-reader tests.

Additional historical/CLI integration verification:
- Original historical private video tests: three passed; Jamie alone fails the
  preserved shared UNKNOWN assertion, with the independently verified HIT above.
- First broad invocation: 11 passed, one Jamie assertion failure, 29 setup errors
  due access denied to the shared Temp/pytest-of-<user> directory.
- Retried non-private integration tests with a fresh independent temporary
  directory: 36 passed, four private cases deselected, one filesystem failure.
  The remaining case attempts to create repository screenshots/task10-test-*;
  the sandbox denied creation of screenshots, outside this task's write scope.
  No permissions or unrelated directories were changed to bypass this check.
- Source SHA256 hashes rechecked after integration and still match the freeze.
