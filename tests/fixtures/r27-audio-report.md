# R27 TASK3 — exclusive newest music

Completed 2026-09-10 in the existing `background-independent-meter` linked
worktree. Focused verification: **398 passed, 1 deselected, zero warnings**.
The new R27 module contributes 24 passing cases. No full suite, game capture,
game input, real audio, installation, upload, commit, reset, or subagent work.
Parent retains integration, full-suite, recording replay and acceptance work.

## Scope and root cause

Read the complete TDD SKILL.md, its writing-good-tests.md, systematic-debugging
SKILL.md and verification-before-completion SKILL.md. Read the private
`SF6_R23_Review/R26-diagnosis.md` in the studio workspace. Its cross-side finding
was reproduced directly: both fixed mixer channels remained playing because
the port stopped only the requested side. The engine independently retained
both owners, and installed ownership before knowing whether PLAY succeeded.
That admitted old WHIFF playback and stale candidate/fade actions.

Production edits are exclusively `src/sf6_music_trigger/audio.py` and
`src/sf6_music_trigger/engine.py`. New tests are exclusively
`tests/test_r27_exclusive_audio.py`, with every new test named `test_r27_*`.
Updated audio/engine-only old policy expectations in `tests/test_audio.py`,
`tests/test_engine.py`, `tests/test_r26_fade_hold.py`, `tests/test_block_fade.py`.
The last two files were already untracked at entry. Existing R24–R26 work was
preserved. Git diff against HEAD includes that earlier work and must not be
interpreted as the TASK3-only delta. Parent/other task modifications visible in
status were neither reverted nor edited.

## Frozen integration boundary

- **Opposite-side STOP is port-only.** The engine does not emit a separate
  opposite-side STOP batch. Existing same-side STOP/PLAY scheduler batches and
  the `AudioPort.execute(actions) -> None` success contract remain intact.
- `PygameAudioPort` preloads every PLAY sound and sets its volume before any
  batch STOP. Load, invalid-path and set-volume failure therefore preserve
  existing music, ownership, held fade and playback clock.
- Before PLAY, the port hard-stops the opposite side if it holds a sound
  reference, under the existing process lock. Fixed P1 channel 0 and P2 channel
  1 remain separate. No shared music channel or global mixer stop was added.
- An opposite stop failure aborts before new PLAY. A partial PLAY exception
  triggers best-effort stop of the target. If cleanup also fails, its sound
  reference is retained; a later opposite PLAY must successfully stop that
  possibly audible channel first. No automatic PLAY retry occurs.
- `AudioPlaybackError.invalidated_sides: frozenset[Side]` is the only new error
  state field. It defaults to empty for pre-effect failures and existing fake
  ports; the real port sets it to sides whose stop/play was attempted before
  a batch failure. Engine drops those owners, clocks and held fades. It carries
  no path or free-form backend data. Constructor parameters, issue values and
  operation values are unchanged.
- The engine commits global ownership only after successful PLAY, clearing
  previous owners, clocks and held fades. The existing owner dictionary now
  has at most one entry. Pending candidates remain available for detector
  lifecycle/history but are marked superseded for audio; terminal confirmation
  and cancellation still record/deduplicate normally. Started-candidate status
  is recorded only after successful execution.
- Outcomes require the exact current `(side, event_id)` owner, including when
  no owner exists. This prevents old WHIFF replay, stale fades and post-reset
  outcomes from affecting new playback. Candidate confirmation transfers the
  live candidate identity without restarting its four-second clock.
- **Same-frame order is deterministic P1 then P2.** If both PLAY operations
  succeed, P2 owns music. Due held fades flush after frame transitions, so a
  new opposite-side play wins even at the exact old fade deadline.
- Normal held fades remain four seconds from successful playback completion
  with the configured existing fade duration (default 1500 ms). Supersession
  hard-stops immediately after new sound preparation, without waiting for the
  old hold or fade. A failed replacement load does not move the old deadline.

`tools.replay._RecordingAudioPort` and the parent's `full_engine_review.py`
merely recording engine batches will **not** observe port-internal opposite
STOP. They can still verify lifecycle routing, dedup and fade scheduling, but
cannot prove mixer exclusivity. For exclusivity replay, parent should use the
real `PygameAudioPort` with the fake mixer boundary, as the new test module does,
or explicitly implement this frozen contract in a private stateful fake.
No edits were made to tools/replay, diagnostic_log, panel, or BUILD_ID.
No shared diagnostic record/evidence schema change is required. Parent's
`R27-<random>` default session_id and BUILD_ID change are independent.
Recorded action batches continue to mean requested actions, not a transcript
of internal device operations; the new invalidation field is internal state
repair, not serialized evidence.

## RED → GREEN and commands

All commands ran in the `SF6_Music_Trigger_Full_MeterColor` linked worktree.
The private workspace is abbreviated below as `$r27Workspace`.

Initial test-only change, before production edits:

```powershell
..\.r21-runtime\Scripts\python.exe -m pytest -q -p no:cacheprovider --tb=short tests/test_r27_exclusive_audio.py
```

This first run had 19 setup errors: the shared pytest temp directory was
inaccessible (`WinError 5`). These are not counted as RED evidence. Used a new
explicit task temp directory; no ACL or existing directory changes:

```powershell
..\.r21-runtime\Scripts\python.exe -m pytest -q -p no:cacheprovider --tb=short --basetemp "$r27Workspace\.r27-audio-red-01" tests/test_r27_exclusive_audio.py
```

**RED: 17 failed, 2 passed in 1.10 s.** Expected failures included `[0, 1]`
playing instead of one side, same-side load failure leaving no music, stop
failure not raising, partial PLAY leaving both sides active, stale WHIFF and
candidate fade calls, and old post-reset outcomes issuing PLAY. The two
already-passing cases characterized opposite-side preparation failures.

After minimal implementation, same command with `--basetemp ...\.r27-audio-green-01`:
18 passed, 1 failed. The remaining test reused frame_id 0 in separate `_drive`
calls, causing the real freshness gate to reset correctly. Fixed the test to
use incrementing frame IDs; did not relax production freshness validation.

First focused old/new run used the final file selection below, without
`-X utf8`, and `--basetemp ...\.r27-audio-focused-01`:
381 passed, 12 failed, 1 deselected. These failures exposed obsolete independent
side/ownerless-outcome policies and changed preparation/cleanup ordering.
Updated those expectations; old outcome tests that intend active playback now
establish it through real engine confirmation before testing their result.
Three pre-existing subprocess GBK decoding warnings disappeared with `-X utf8`.

Repeat with `-X utf8`, `--basetemp ...\.r27-audio-focused-02`:
**393 passed, 1 deselected in 3.52 s**. Added five supplementary boundary cases
for both directions at the exact fade deadline, error-state invalidation and
failed-candidate confirmation timing. They exercise the implementation already
driven by the initial RED cases; no further production change was required.
Standalone `--basetemp ...\.r27-audio-extra-01` run: **24 passed in 0.92 s**.

Final command:

```powershell
..\.r21-runtime\Scripts\python.exe -X utf8 -m pytest -q -p no:cacheprovider --tb=short --basetemp "$r27Workspace\.r27-audio-final-01" tests/test_r27_exclusive_audio.py tests/test_audio.py tests/test_engine.py tests/test_r26_fade_hold.py tests/test_non_three_fade.py tests/test_block_fade.py tests/test_unknown_fade.py -k 'not sdl_dummy'
```

**GREEN: 398 passed, 1 deselected in 3.58 s; exit 0; no pytest warnings.**
The deselected test is the existing SDL dummy/default real pygame loader
roundtrip; all selected playback tests use injected fake device boundaries.

```powershell
git diff --check -- src/sf6_music_trigger/audio.py src/sf6_music_trigger/engine.py tests/test_audio.py tests/test_engine.py tests/test_r27_exclusive_audio.py tests/test_r26_fade_hold.py tests/test_block_fade.py
```

Passed with no whitespace errors. Git prints existing LF→CRLF normalization
notices. Inspected owned diffs and final git status; no commit/reset performed.

## Coverage and fixture provenance

The new module runs real `VisualTriggerEngine`, real `AudioScheduler` and real
`PygameAudioPort` together. It reuses the existing fake pygame/mixer fixture
from test_audio; fake audio never imports pygame or opens an audio device.
Engine lifecycle inputs use existing synthetic `_event`/`_candidate` helpers;
same-frame and timing tests exercise `engine.process` with synthetic frames,
controlled tracker transitions and an injected clock. Recognition and outcome
classification are intentionally outside this audio task.

24 parameterized cases cover both port interruption directions; load and
volume failure on same/opposite side; stop failure; partially successful PLAY
with successful/failed cleanup; both old WHIFF directions; stale confirmation
and cancellation after successful or failed candidate playback; P1→P2→P1
identity; P1/P2 same-frame processing; both opposite directions at exact held
fade deadline; successful-play timing; reset/runtime reset; failure invalidation;
failed candidate confirmation retry plus dedup. Existing selected suites retain
R26 load-time exclusion, delayed confirmation timing and default 1500 ms fade.

No private video/audio original is required by these lifecycle/device-boundary
tests; no missing-original skip was added. Private diagnosis is supporting
evidence, not a replacement for pixel/recording regression. Parent's video
tasks must run locally present originals and retain their own provenance.

## Self-review and remaining risks

- Verified real stop-before-opposite-play order, fixed channel identities,
  stale-event identity checks, two candidate lifecycle histories, failure
  timing and reset behavior. Checked realistic missing-side-stop, missing-owner
  guard and missing-preload mutations against the initial RED failures.
- A device STOP failure may leave old music physically audible. The port
  prevents a second PLAY; engine invalidates uncertain ownership instead of
  pretending a stop succeeded or allowing an old outcome to restart music.
  Subsequent successful replacement can recover through the retained reference.
- Playback preparation precedes interruption by design so a load failure does
  not kill old music; slow decoding can delay the switch. The new four-second
  clock begins after successful execute, retaining R26 timing semantics.
- Exclusivity is for P1/P2 gameplay music. Existing dedicated channel-2 manual
  preview behavior is unchanged; this task did not expand preview policy.
- Full application review, full suite, private recording replay, packaging and
  actual listening/game acceptance remain parent-owned. Passing synthetic
  mixer tests is not evidence of physical device or creative acceptance.

The verification skills influenced test-first sequencing, behavior assertions
at the real port boundary and explicit separation of focused test success from
the remaining integration/runtime acceptance gates.

## Review follow-up: preserve unresolved audible owner's result window

Status: repaired and source frozen. This follow-up changes only engine.py,
test_r27_exclusive_audio.py and this report. No audio.py, observer, other worker
files, shared schema or interface changes; no commit or full-suite run.
Parent's report path sanitization was verified before this append.

The review identified a gap in the original preparation-failure coverage:
those tests retained an already queued non-three fade, whereas an unresolved
three-bar event still depends on OutcomeObserver's pending window. The engine
called begin(B) before execute(B); real begin replaces the pending side entry.
Thus Sound/set_volume failure preserved audible owner A but replaced its
observer with B. B's eventual UNKNOWN was correctly rejected by the ownership
guard, leaving A without its fallback. The receiving-code-review workflow was
used to verify this behavior before changing production code.

Exact production delta relative to the preceding TASK3 freeze: in
VisualTriggerEngine._route_confirmation, relocate the existing scope/status,
event-context registration and observer.begin block from before audio execution
to immediately after it. Guard that block with the resulting owner being absent
or equal to ("event", event.event_id). If an older owner remains after preparation
failure, its pending observation, accumulated evidence, original deadline,
bounded context and status are preserved. Detection event_log and confirmation
dedup still record B. Successful B replaces A's result window normally. Existing
ownerless observation behavior is retained.

New test: test_r27_real_observer_retains_audible_owner_on_preparation_failure,
six parameterized cases (P1/P2 x Sound failure/set_volume failure/success).
It uses real OutcomeObserver with real engine, scheduler and PygameAudioPort,
injecting only the fake mixer boundary and synthetic gauge/frame inputs.
A starts at t=1, B confirms at t=2; process receives frames at 8.99, 9, 10, 12.
On preparation failure the test requires A's UNKNOWN plus exactly one 1500 ms
fade at t=9, and silence thereafter; on success it requires B's UNKNOWN/fade
at t=10. Both detections and B dedup are asserted independently.

Commands below use $testTemp to denote the task's writable temporary directory;
the actual personal absolute path is intentionally omitted. Run from repo root.

```powershell
..\.r21-runtime\Scripts\python.exe -X utf8 -m pytest -q -p no:cacheprovider --tb=short --basetemp "$testTemp/.r27-review-red-02" tests/test_r27_exclusive_audio.py -k real_observer
..\.r21-runtime\Scripts\python.exe -X utf8 -m pytest -q -p no:cacheprovider --tb=short --basetemp "$testTemp/.r27-review-green-01" tests/test_r27_exclusive_audio.py tests/test_engine.py tests/test_r26_fade_hold.py tests/test_non_three_fade.py tests/test_block_fade.py tests/test_unknown_fade.py
git diff --check -- src/sf6_music_trigger/engine.py
```

RED before production edit: **4 failed, 2 passed, 24 deselected in 1.00 s**.
All four failures observed B UNKNOWN at t=10 with no audio instead of A UNKNOWN
with FADE at t=9. The success controls passed. An earlier attempt at repo-local
basetemp produced six permission setup errors, not RED evidence; the test was
rerun in the writable task temp directory without changing ACLs.

GREEN after production edit: **204 passed in 1.67 s**, no pytest warnings.
All six new cases and all 24 earlier exclusive-audio cases are included.
Whitespace check passed; Git reports its existing LF-to-CRLF notice.
Engine SHA256 at freeze:
`3C83A2A26EFAFA8DF54FE8ED964B76D31465A406D6ED822E222B94BF0ED0CE3C`.

The first attempted test patch was rejected by automatic approval because of
quota exhaustion; it was verified absent after the user resumed, then retried
through the normal tool. No bypass was used. Current blocker: none.
Parent owns the post-freeze full-suite/package checks. These synthetic frames
prove lifecycle fallback, not real-video contact recognition or listening QA.
