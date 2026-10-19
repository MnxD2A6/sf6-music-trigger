# R27 Task1 outcome evidence report

Date: 2026-09-10. Status: production source frozen for parent integration.
Focused required regression: **163 passed, zero skipped, 79.43 seconds**.
One separately checked old synthetic brightness-HIT assertion intentionally fails;
its exact replacement expectation is documented below. No full-suite claim.

## Scope and preservation

Repository: the existing `SF6_Music_Trigger_Full_MeterColor` linked worktree.
Branch `background-independent-meter`, initial HEAD `6e464b9`; linked worktree
and approved uncommitted R24-R26 changes preserved. No reset, commit, installs,
subagents, live capture, game input, real audio, upload or package work.

Task1 changes only:

- `src/sf6_music_trigger/outcome.py`: reject default brightness-only HIT;
  correct the default feature extractor docstring.
- `src/sf6_music_trigger/contact_hud.py`: require a spatially coherent endpoint.
- `tests/test_r27_outcome_video.py`: 13 original-video cases.
- This report. No shared old tests changed. No engine/audio/gauge edits.

Read TDD SKILL.md, its complete writing-good-tests.md, systematic-debugging
SKILL.md and root-cause-tracing.md; completion checked with verification skill.
The initial attempted old skill path was absent; the listed superpowers/6.3.0
plugin paths were found and read before any test/source edits.

## Verified provenance and root causes

Private diagnosis: `SF6_R23_Review/R26-diagnosis.md` in the private workspace.
Original: `Videos/屏幕录制/SF6/R26.mp4` relative to the local user profile.
SHA256 rechecked in this task:
`a0319cb9828358187a0dd3b082d78400983f1fef17cdb7576ac85068f06eef95`.
Decoded source shape 1948x1138, 30 FPS; exact client crop x14/y38/w1920/h1080,
no resizing. Tests decode sequential raw video with OpenCV and real production
outcome/health readers. Missing original skips only if the file is absent;
decode, shape, or seek failures assert. All required originals were tested locally.

### Blue parry

The existing private probe, run offline with fake audio, reproduced the actual
event anchor: pre-spend frame **7118** (237.266667), startup 237.566667,
confirmation 237.666667. Its broad target ROI baseline luma is 0.1042244211.
The diagnosis records current luma 0.2179592699/0.2176980227 and unchanged
target health 0.2613636364. The default legacy `_contact` produced HIT at238.6
using this brightness change and mislabeled it `opponent_health_changed`.

Default `_contact` now abstains for luma HIT just as it already abstained for
luma BLOCK. Explicit custom extractors retain their existing HIT/BLOCK contract.
No invented blue-parry/BLOCK/WHIFF label; unresolved result remains UNKNOWN.

### Final landed CA

The exact missing evidence was **not a late deadline or missing early damage**.
Offline probe confirmed pre-spend frame **13170** (439.0), startup439.1,
confirmation439.166667. Stable baseline frame13169 (438.966667) reads
P1/P2 health **0.1590909091 / 0.1212121212**. An earlier target drop at438.7333
was already present in this baseline and must not be borrowed as CA damage.

At frame13197 (439.9), the health reader incorrectly reported attacker health
0.1401515152. Frames13198..13200 similarly reported 0.14015 or0.14394. This
crossed the existing attacker-change tolerance0.012 and permanently invalidated
CA attribution. At441.0 (frame13230), real target health is **0.0416666667**,
with attacker0.1515151515, but the baseline had already been discarded.

Visual inspection of decoded HUD strips13169/13197/13230 showed cinematic
foreground covering the attacker's outer endpoint. The existing endpoint check
accepted white pixels in different columns across nine rows, so a curved
foreground edge impersonated a straight rectified HUD border. At13197 the best
three-column band supports only7/11 rows; at13199 only8/11. Genuine baseline
and landed-health endpoints support11/11 within a narrow band.

The reader now requires at least9/11 rows to contain an outline within one
three-sample-column band. Existing palette, live anchor, fill continuity,
row-end coherence, wide-flash rejection, upper/lower outlines and conflicting
offset abstention remain. This rejects the false reading at its pixel source;
the observer's attacker-damage invalidation is unchanged. No new architecture,
early-damage bypass, timer change, tolerance increase or deadline extension.

## RED -> GREEN commands and results

All commands run from the repository with the existing sibling runtime.

```powershell
../.r21-runtime/Scripts/python.exe -B -m pytest tests/test_r27_outcome_video.py -q -p no:cacheprovider --tb=short
```

Initial RED: **8 failed, 5 passed, 45.74s**. Four false shortened attacker
readings failed; final CA failed UNKNOWN on hidden gauge and premature luma HIT
on visible gauge. Initial parry pre-frame237.5 was an inaccurate reconstruction
and did not reproduce its luma baseline. Read-only actual-event probe identified
7118; the test was corrected before any production edit.

Corrected RED, same command: **10 failed, 3 passed, 46.49s**. Added expected
parry failures at20/30Hz with visible gauge: actual HIT versus expected UNKNOWN.
No reader mocks, xfails or weakened outcome expectations.

Then applied the two minimal production corrections with apply_patch.

```powershell
../.r21-runtime/Scripts/python.exe -B -m pytest tests/test_r27_outcome_video.py tests/test_r26_ca_video.py tests/test_r26_ca_safety.py tests/test_contact_hud.py tests/test_contact_outcome.py tests/test_outcome.py -q -p no:cacheprovider --tb=short
```

GREEN: **163 passed in79.43s, exit0, zero skips/xfails**.
The13 new cases comprise parry/final-CA x20/30Hz xvisible/hidden gauge (8),
four independently reviewed foreground frames (4), and readable baseline/loss (1).
20Hz uses original timestamps and omits source frames whose index modulo3 is1.
Gauge visibility is a controlled observer input via existing `_meters`; actual
gauge recognition is not under test and remains the parent's integration gate.

Parry produces exactly one UNKNOWN within existing3..8second fallback bounds.
Final CA produces exactly one HIT in **[441.0,442.1)** with either visibility,
before the existing447.1 deadline; observation continues through452 with no
second terminal decision. This protects a real hit without crediting startup luma.
R23/R25 landed goldens, historical Jamie CA, target occlusion/attacker damage,
stale/missing anchors, missing/gapped/duplicate samples, health recovery, shapes,
block/unchanged-health negatives, scene flashes and custom extractor tests pass.

## Old expectation changes for parent

No changes are needed in the six required focused suites above.

Confirmed intentional failure in shared `tests/test_replay_integration.py`:
`test_replay_cli_plays_on_candidate_then_confirms_hit_without_restart`.
Its synthetic320x480 image merely changes `[8:34,264:452]` from gray to black
atframe27; it supplies no readable health bar. The old assertion expects HIT
at1.6 from this brightness-only stimulus. Under R27 **outcomes is []** at the
end of its approximately2second replay, before the fallback deadline. Preserve
candidate/confirmation/no-restart assertions; update this outcome expectation
and rename it as an unresolved-brightness control, or supply a genuine health
fixture if the parent wants an integrated HIT-positive test. Do not restore
brightness HIT to satisfy it.

Command initially encountered WinError5 enumerating the shared pytest temp
directory (setup error, not a behavioral failure). Repeated with a fresh unique
temp root, with no deletion/ACL changes:

```powershell
$r27Temp = Join-Path ([System.IO.Path]::GetTempPath()) ('r27-outcome-' + [guid]::NewGuid().ToString('N'))
../.r21-runtime/Scripts/python.exe -B -m pytest tests/test_replay_integration.py::test_replay_cli_plays_on_candidate_then_confirms_hit_without_restart -q -p no:cacheprovider --tb=short --basetemp=$r27Temp
```

Result: **1 failed in1.97s**, specifically `[] != [HIT at1.6]`, as intended.
Other historical private full-engine replay assertions were only read, not run
or changed here. Parent must assess them from actual evidence, not blanket-relabel
landed positives. The required original-video R23/R25/Jamie goldens already pass.

## Self-review, risks and integration handoff

- Mutation review: restoring default luma HIT breaks parry negatives and final
  CA timing; removing coherent endpoint validation breaks four raw frames and
  hidden-gauge final CA. Removing attribution gates remains covered by existing
  safety tests. No test expectation is computed from the production classifier.
- No observer state, labels, public API, custom extractor contract, existing
  deadlines or experimental-reader opt-in changed. Health geometry check applies
  to both sides/palettes; current focused corpus passes, alternate HUD layouts or
  compression can still abstain and lead to UNKNOWN.
- Default non-three-bar events no longer obtain HIT from scene brightness.
  The supplemental health path still requires explicit three-bar spend. This is
  the intended conservative consequence, not a new claim of general SA1/SA2
  contact recognition. No lowering of uncertainty or synthetic WHIFF.
- Worker source freezes here. Parent owns full suite, continuous R26/R25 engine
  replay, audio ownership/fade integration, build and live/listening acceptance.
  No actual audio behavior is certified by these observer tests.
- `git diff --check -- src/sf6_music_trigger/outcome.py` exited0 (only Git's
  existing LF/CRLF conversion warning). New files/source reviewed directly;
  contact_hud.py was already untracked at handoff and remains so.
- Three private decoded HUD PNGs were generated under the existing diagnosis
  directory for visual inspection only, named `r27-hud-13169.png`,
  `r27-hud-13197.png`, `r27-hud-13230.png`; none added to release assets.

Frozen file SHA256 (before report, after successful focused run):

| File | SHA256 |
| --- | --- |
| src/sf6_music_trigger/outcome.py | f23cb10bf647170d4ed3bad3e3ab20a31e98ea5c7f304e9aead729e99e28c18c |
| src/sf6_music_trigger/contact_hud.py | fe541bf6ef77b87ab814d8dabfde8d84dc4c7568ad086453273a304e41097080 |
| tests/test_r27_outcome_video.py | 0dca4a01b38f5e8438d25c34a268dafcfbf4ed25378323a50014ee45be3937a5 |
