# R27 TASK2 meter false spend — source frozen

2026-09-10. Ready for parent integration/full-suite verification.

## Delivered scope

- Production delta for this task: `src/sf6_music_trigger/meter_level_features.py`
  extends the existing R26 P2 saturation-conflict veto from `(0, 3)` to
  `(0, 2, 3)`, with updated explanation. One conditional behavior change.
- `gauge_tracker.py` inspected, not changed by this task. Its existing R23/R24
  color-flicker work remains intact. Existing uncommitted R24–R26 changes in
  both owned files and the shared tree are preserved; no reset or commit.
- New `tests/test_r27_meter.py`: 33 tests including four private-original
  runs, four portable HUD episode runs, eight raw glyph regressions, and
  17 R23/R25 exact positive event-frame controls.
- Two new lossless HUD fixtures, named `test_r27_p2_158.npz` and
  `test_r27_p2_249.npz`. No old tests or fixtures were edited.
- No engine/audio/outcome/config/UI changes by this task. No subagents,
  installs, live capture, game inputs, real audio, uploads, or package build.

## Independently reproduced root cause

Used the real `observe_gauge -> GaugeTracker` path with original R26 pixels.
The client crop is exactly x=14,y=38,w=1920,h=1080; no resizing or aspect-gate
bypass. Original rate is 30 fps. At 20 Hz, source indices with `index % 3 == 1`
are omitted; timestamps remain `index / 30`.

Before any production edit, the 158–168s window reproduced the parent's exact
false candidate at frame4859 / 161.9666667s and confirmation at frame4901 /
163.3666667s, both 2->0. At 30 Hz it also invented a candidate (161.9s).

This is a reader/baseline defect, not a retired episode being replayed:

1. P2 visibly remains zero. Initial tracker baseline is zero at158.1s.
2. At frame4841 / 161.3666667s the permissive saturation<=65 extraction reports
   **2 with confidence .9**. Frames4845,4850,4854 repeat it. The classified
   component at4841 has bounds x25..35,y12..31, no hole (only11x20 canonical
   pixels), whereas the saturation<=20 extraction recovers the intact zero
   at x24..42,y8..46 (19x39 pixels, one hole). The fallback/fragment shape is
   accepted as2 despite the full visible ring being0.
3. The existing tracker promotes the false positive baseline to2 at161.5s.
   When the glyph reads0 again at161.8666667s, normal candidate/confirmation
   logic interprets it as spending. Fractional fill is only about .176–.188;
   temporal repetition of a wrong glyph is not independent correctness evidence.
4. R26's achromatic-core conflict veto already solves analogous false0/3,
   but never checks a permissive result of2. Here strict and permissive reads
   disagree reproducibly (2 vs0), so extending that veto fixes the input
   instead of altering the episode timers.

After the change, conflicting2 becomes unknown/zero confidence. During the
long contaminated section the existing unknown policy can clear the baseline;
when a clean0 returns it is re-established. Neither candidate nor confirmation
is emitted at either tested rate. No timestamp, side-disable, delay, or
recording-specific production rule was added.

## Separate rank verification near253s

Raw HUD images, visually inspected at original pixel content (enlarged only
for inspection), establish the rank independently of either runtime log:

- Frames7584–7586 (252.8–252.8666667s): a clear **1**.
- Frame7587 (252.9s) and7588: a clear **0**. Frames7589/7590 also visibly0.
- Thus the observed spend is **1->0, one bar**, not SA2.

Earlier contamination at frames7478/7487/7491 (249.2667/249.5667/249.7s)
and7572 (252.4s) appends background to the1, causing permissive2 while the
achromatic extraction reads1. At7491 the permissive component spans
x26..44,y6..46; the strict1 is x27..36,y8..46. The false baseline2 persists
through subsequent1 reads, explaining current-R26's incorrect spent2.

The same one-conditional fix prevents that false gain. The 249–257s actual
reader/tracker replay now confirms **1->0 at frame7596 / 253.2s**, unchanged
confirmation time, with no candidate in this 20Hz window. Both20/30Hz rank
tests pass. The native R25 diagnostic was read independently and agrees:
`diagnostics/20260910T081341279035Z-68254877df4d54c8-0000.jsonl`, line11,
monotonic104218.4882608, P2 CONFIRMED, before1/after0/spent1.
The log's monotonic time is not substituted for video time.

Private visual/probe artifacts are in the user's existing `SF6_R23_Review`
workspace: `r27_meter_probe.py`, `r27-meter-glyphs.png`, and
`r27-meter-rank-fine.png`. They contain only local HUD diagnostic evidence;
the portable tests use the NPZ crops, not these absolute paths.

## RED before GREEN — commands and results

All commands run from the requested MeterColor worktree, branch
`background-independent-meter`, HEAD `6e464b93e5c2b25d362a66697cbd7d22364c1d78`.
Runtime is the requested `..\.r21-runtime\Scripts\python.exe`, with `-B` and
pytest cache disabled. No full suite ran in this task.

1. First RED, unchanged R26 source:

   `..\.r21-runtime\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_r27_meter.py --tb=short`

   Initial single original-video negative: **1 failed in3.96s**, actual false
   candidate161.9667 and confirmation163.3667. This was an assertion failure,
   not an import or missing-original error.

2. Fixture extraction:

   `..\.r21-runtime\Scripts\python.exe -B tests/test_r27_meter.py --extract`

   Normal sandbox first denied writing the out-of-workspace repository NPZ.
   The same narrow command with reviewed escalation succeeded (exit0), writing
   only the two new fixtures. The extractor refuses to overwrite existing files.

3. Expanded RED, still before the production edit, same pytest command:
   **16 failed,17 passed in32.15s**. Failures: four original-video cases,
   four lossless-HUD cases, eight raw glyph cases. All17 existing positive
   event-frame controls passed. False2 and incorrect spent2 are the observed
   failures; expected labels are visual0/1 and spend1, not reader output.

4. GREEN after the single production conditional change:

   `..\.r21-runtime\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_r27_meter.py tests/test_r26_p2_sequences.py tests/test_r26_p2_glyphs.py tests/test_r23_video_regressions.py --tb=short`

   **100 passed,4 xfailed in59.30s**, exit0, zero skips/XPASS. Includes all33
   R27 tests and all60 R26 glyph/sequence tests. All present local originals
   were decoded and tested, not silently skipped. The original four strict
   XFAILs remain `[2-2]`, `[3-2]`, `[4-2]`, `[10-None]`; none were weakened.

5. Focused base/control GREEN:

   `..\.r21-runtime\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_meter_level_features.py tests/test_meter_level_tracker.py tests/test_meter_color_features.py tests/test_gauge_features.py tests/test_gauge_tracker.py tests/test_candidate_color_flicker.py --tb=short`

   **573 passed in4.56s**, exit0. Across these two non-overlapping commands:
   **673 passed,4 original XFAIL**. Not a whole-repository test result.

6. Direct trace commands, executed before and after the edit:

   `..\.r21-runtime\Scripts\python.exe -B tests/test_r27_meter.py 158 168`

   `..\.r21-runtime\Scripts\python.exe -B tests/test_r27_meter.py 249 257`

   Post-fix first window has no event; second confirms1->0 at253.2s.

7. `git diff --check -- src/sf6_music_trigger/meter_level_features.py src/sf6_music_trigger/gauge_tracker.py`
   passed, exit0. Git printed existing LF-to-CRLF working-copy notices, not
   whitespace errors. Reviewed source diff against the pre-task contents;
   the larger diff versus HEAD includes approved earlier work, not just R27.

## Positive timing controls and tradeoffs

All17 reviewed positive sequences preserve the exact R26 baseline event frames
at20Hz (candidate frames as well as confirmations). R25 confirmation times:
42.5,89,251.9666667,309.5666667,347.4666667,430.2. R23 confirmation times:
36.3,90.3666667,167.4,219.6,384.8,117.1,245.6,289.1,291.7,370.4,424.8666667.
The R23 P2 real2->0 at90.3667 and real3->1 at289.1 remain valid; P1 SA3/CA
and other positive ranks survive. The existing R26 suite also covers these
rank/negative windows at30Hz. Exact golden-frame literals are in the new test.

Self-review: removing2 from the veto recreates the observed RED cases; a
blanket unknown/suppression result fails the real spend and golden controls.
The strict reading is only a veto, never a replacement digit. Unreadable strict
cores still do not veto a readable permissive result. P1 and the already-vetoed
P2 levels0/3 take the same paths as before. No test-only production hook or
mock reader/tracker was introduced.

Residual risks: two extractors can agree on a wrong glyph, so this is bounded
contamination handling, not universal OCR correctness. An intact real2 with
a readable conflicting strict core may now abstain, potentially delaying a
baseline or missing a spend in untested lighting. Longer unknown sequences
can clear existing tracker history (observed here at161.6s); selected goldens
show no added latency, but all-character coverage is not established. A P2
permissive2 now incurs the existing second extraction's CPU work; live frame
budget/performance was not measured. Four historical glyph defects remain.

Parent owns full-engine continuous R26 replay, full suite on the integrated
tree, packaging, and runtime/listening acceptance. These focused tests do not
claim outcome/audio correctness or whole-recording absence of other events.

## Fixture provenance and portability

Source R26.mp4:1948x1138,30fps,14191frames,557334768bytes. SHA256 verified:
`a0319cb9828358187a0dd3b082d78400983f1fef17cdb7576ac85068f06eef95`.
The original remains in the user's private `Videos/屏幕录制/SF6` directory.

- `test_r27_p2_158.npz`: indices4740..5039,300frames,19852143bytes,
  SHA256 `493cbc98328ebc575db8e191ac1fd1643b09e9185bb26854aa4dafd3ff018c75`.
- `test_r27_p2_249.npz`: indices7470..7709,240frames,14183845bytes,
  SHA256 `2f0be52d19ee38d406d1f2017aa746449797c9abbda3b5f77b5dca20bb27c150`.

Each stores raw BGR meter+glyph crops from native1920x1080 client frames,
integer indices, shape/bounds, ROI and R25 calibrated full/spent references.
Crop bounds are `[1488,955,1896,1053]` (408x98). No audio, account identity,
full-frame screenshots, or private absolute paths are embedded. Loading uses
`allow_pickle=False`; playback reconstructs just the HUD on a black canvas.

Portable fixture tests require neither private recording nor release folder.
The four original tests may skip when originals/calibration are absent on
another machine; present unreadable originals fail. This local run had no
skips. No fixture refresh, old shared test edit, or assertion relaxation was
used to achieve GREEN.
