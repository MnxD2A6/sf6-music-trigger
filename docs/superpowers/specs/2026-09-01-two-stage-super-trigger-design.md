# Two-Stage Super Trigger Recovery Design

## Goal

Restore responsive SF6 Super music without giving up the round-transition and
meter-glitch protections added after the early successful build. A plausible
Super starts that side's main track after a short, stable meter-drop candidate
is observed; a separate authority then confirms or rejects the candidate and
the existing result observer classifies hit, block, or whiff.

The design remains Windows-only, screen-only, local, SF6-only, and independent
for P1 and P2. It does not read game memory, inject input, inspect traffic,
upload frames, operate MK1, or modify the historical prototype `main.py`.

## Baseline and Recoverability

Implementation starts from branch `background-independent-meter` at commit
`3deff09d0e4a2863f11bfa17e758fc215254195d`. The complete offline suite at this
baseline contains 1,745 passing tests.

The following existing recovery points remain intact:

- `backup/stable-ca-audio-20260816` preserves the earlier responsive CA build;
- `backup/pre-meter-level-detector-20260822` preserves the pre-level detector;
- `backup/pre-round-transition-fix-20260825` preserves the state before the
  later round-transition stabilization.

The refactor is committed in small, test-backed units. No existing backup tag
or packaged candidate is overwritten.

## Root Cause

The current engine has only one gameplay event boundary:
`GaugeTracker` emits `ConfirmedSuperEvent`, and only then does
`AudioScheduler.on_confirmed_spend` start audio. The conservative confirmation
rules now require a clean downward level to survive additional scene guards,
including a 0.6-second clean-drop dwell. This protects against real-match
round transitions, but it also delays or loses the start boundary that the
user experienced as stable in the earlier recording.

Two previously approved requirements were therefore collapsed into one event:

1. start music quickly when a Super is plausible; and
2. mutate the authoritative meter baseline only when a Super is confirmed.

Threshold changes cannot satisfy both. Relaxing confirmation restores speed
but reintroduces known false events; retaining confirmation as the audio start
keeps the observed late and missed playback. The recovery separates the two
responsibilities.

## Approaches Considered

### Restore the old fractional-colour trigger

Rejected. Natural meter gain wraps the fractional fill from full to empty and
has already produced false starts in long real-match recordings. It also
cannot reliably distinguish SA1 and SA2.

### Run an independent fast detector beside the existing tracker

Rejected. Two components would maintain separate baselines and scene-reset
rules from the same meter readings. Their state could drift, making a
candidate impossible to explain or cancel deterministically.

### Emit candidate and authority transitions from one tracker

Selected. `GaugeTracker` remains the single owner of meter baseline and scene
trust. It emits a fast candidate transition without changing the trusted
baseline, then emits exactly one confirmation or cancellation for that
candidate. The engine starts audio on the candidate and uses the later
transition to continue, fade, or begin outcome resolution.

## Domain Model

`GaugeTracker.update(observation)` returns one immutable
`GaugeTrackerUpdate(transition=...)` per observation. It may contain at most
one transition for its side and has one of these forms:

- no transition;
- `SuperCandidateEvent`;
- `ConfirmedSuperEvent`; or
- `CancelledSuperCandidate`.

`SuperCandidateEvent` contains:

- `candidate_id`, `side`, `suspected_at`, and `profile_id`;
- `pre_spend_frame_id` and `candidate_frame_id`;
- the trusted `meter_before` baseline and the provisional `meter_after`; and
- privacy-safe evidence codes describing the stable downward readings.

`ConfirmedSuperEvent` retains the existing fields and adds an optional
`candidate_id`. Candidate-based confirmations require a matching nonblank ID;
legacy confirmed events use `None`. Its existing `event_id` remains the
identity used by outcome resolution. `CancelledSuperCandidate` contains the
candidate identity, side, decision time, and one of a closed set of reasons:

- `LEVEL_RECOVERED`;
- `LEVEL_CONFLICT`;
- `CONFIRMATION_TIMEOUT`;
- `CAPTURE_RESET`; or
- `TRACKER_RESET`.

IDs are unique per side and cannot be replayed. A confirmation or cancellation
must reference the currently active candidate for that side. P1 and P2 may
each own one candidate concurrently.

`GaugeTracker.reset(reason, decided_at=None)` also returns a
`GaugeTrackerUpdate`. If a candidate is active it contains one reset
cancellation, using `decided_at` when supplied or the last accepted sample time
otherwise. With no active candidate it contains no transition. This lets the
engine fade provisional audio before discarding tracker state.

## Candidate Detection

The tracker may produce a candidate only when all of these conditions hold:

1. meter-level extraction is supported for the calibrated profile;
2. the side has a trusted stable baseline held for the existing ten-sample
   arming dwell;
3. the observed level is lower than that baseline by one, two, or three;
4. the same lower level is visible in three consecutive fresh frames;
5. frame IDs and monotonic time remain ordered in one capture session; and
6. no reset, geometry change, profile change, long HUD loss, or conflicting
   level has occurred during the candidate streak.

At the configured 20 FPS sampling rate, three samples give a nominal
0.10–0.15-second candidate latency. A single glyph glitch, unknown frame,
ordinary meter gain, unchanged level, and fractional-colour wrap cannot create
a candidate.

Candidate emission does not lower the authoritative stable baseline. This is
the key isolation boundary: provisional audio may start, but a rejected
candidate cannot poison later meter comparisons.

## Confirmation and Cancellation

After candidate emission, the existing conservative cross-signal and scene
guards remain the confirmation authority. The candidate has a fixed
0.75-second confirmation deadline measured from `suspected_at`.

- If the lower level satisfies the current clean-drop authority before the
  deadline, the tracker emits one `ConfirmedSuperEvent`, updates the trusted
  baseline, and clears the candidate.
- If the level returns to the trusted baseline, the tracker cancels with
  `LEVEL_RECOVERED` immediately.
- If another incompatible level becomes stable, it cancels with
  `LEVEL_CONFLICT`.
- If no authority decision is possible by 0.75 seconds, it cancels with
  `CONFIRMATION_TIMEOUT`.
- Capture, profile, geometry, explicit stop, or tracker resets cancel with the
  matching reset reason before state is cleared.

Cancellation is expected to be rare but audible by design: because candidate
playback is immediate, a rejected candidate may produce at most approximately
0.75 seconds of music before fade begins. The engine must never leave such
audio playing after cancellation.

## Engine Data Flow

For every fresh frame, the engine performs this order:

1. validate capture session, geometry, profile, and freshness;
2. retain bounded pre-event outcome features;
3. derive P1 and P2 gauge observations;
4. update existing confirmed outcome resolutions;
5. update P1 and P2 trackers in deterministic side order;
6. route candidate, confirmation, or cancellation transitions;
7. write privacy-safe diagnostic records for state changes and audio actions.

On `SuperCandidateEvent`, the engine:

- suppresses duplicate candidate IDs;
- submits one atomic STOP/PLAY main-track batch for that side;
- marks the side diagnostic state as `疑似大招，正在确认`; and
- does not begin hit/block/whiff observation yet.

The engine retains the immutable candidate in
`_pending_candidates_by_side: dict[Side, SuperCandidateEvent]` until the
matching confirmation or cancellation. It also retains a bounded set of
terminal candidate IDs for duplicate suppression; reset clears the set only
after all pending audio has received a terminal action.

On matching `ConfirmedSuperEvent`, the engine:

- records the confirmed meter delta;
- for `SA3_CA_ONLY`, fades immediately when `levels_spent` is one or two;
- otherwise begins the existing outcome observer using the candidate's real
  pre-spend frame baseline; and
- never starts the main track a second time.

On `CancelledSuperCandidate`, the engine emits one fade action using that
side's configured `whiff_fade_ms`. Cancellation is not a whiff and therefore
never plays the whiff sound.

The audio policy boundary is explicit:

- `AudioScheduler.on_candidate(candidate, config)` returns the candidate
  STOP/PLAY batch;
- `AudioScheduler.on_confirmation(event, config, playback_started=...)`
  returns no action for an accepted candidate, returns FADE for a started
  SA1/SA2 filtered by `SA3_CA_ONLY`, starts an accepted legacy confirmation
  when `playback_started` is false, and does nothing for an unstarted filtered
  legacy event;
- `AudioScheduler.on_cancellation(cancellation, config)` returns one FADE for
  enabled provisional playback; and
- `AudioScheduler.on_outcome(decision, config)` retains its existing terminal
  result mapping.

An authority confirmation without a live matching candidate is a defensive
recovery case. The engine starts the main track once, applies scope filtering,
and begins outcome resolution so a lost candidate transition cannot silently
drop a genuine confirmed Super. This path is logged as
`CONFIRMED_WITHOUT_CANDIDATE`.

## Playback Scope and Outcome Rules

The user-visible rules remain:

- `ALL_SUPERS`: SA1, SA2, SA3, and CA candidates start immediately. A
  confirmed event proceeds to outcome resolution regardless of tier.
- `SA3_CA_ONLY`: every candidate still starts immediately. Confirmed SA1/SA2
  fades; confirmed SA3/CA continues to outcome resolution.
- confirmed hit or block keeps the main track playing;
- confirmed whiff with `play_whiff_sound=true` stops the main track and plays
  the whiff track;
- confirmed whiff with `play_whiff_sound=false` fades the main track;
- cancellation, unknown outcome, reset, or filtered tier fades the main
  track and never plays the whiff sound.

Whiff sound and fade remain mutually exclusive. No result may generate more
than one terminal audio batch for its side.

## Outcome Observation

`OutcomeObserver` continues to preserve pre-spend features, skip universal
Super flashes, wait for trustworthy HUD recovery, and classify hit, block,
whiff, filtered tier, unknown, or reset. Its observation window starts from
the candidate's `suspected_at` time, while `begin` is called only after
confirmation.

The interface becomes
`OutcomeObserver.begin(event, scope, *, activation_at)`. It rejects an
activation time later than `event.confirmed_at`. The suppression checkpoint
and hard deadline are derived from `activation_at`; the pre-spend feature is
still selected by `event.pre_spend_frame_id`. Legacy confirmation passes
`event.confirmed_at` as its activation time.

The initial contact checkpoint remains approximately 3.0 seconds after the
candidate. Expected cinematic obstruction may extend observation until HUD
recovery, with the existing 8.0-second hard deadline. A confirmed candidate
whose remaining HUD never becomes trustworthy resolves to `UNKNOWN` and
fades. Desktop or task-view pixels cannot confirm contact or extend the hard
deadline.

Tier filtering uses authoritative `levels_spent` from the confirmed meter
transition. The older normalized fractional-meter tier bucket remains only a
diagnostic fallback for confirmed legacy events lacking level metadata; it
cannot override authoritative level metadata.

## Diagnostic Event Log

Each application run creates one bounded JSON Lines log under a local
`diagnostics` directory beside the portable configuration. The filename uses
UTC date and a random session suffix; it contains no audio paths, screenshots,
window titles, usernames, or arbitrary exception text.

Records use a fixed schema with:

- schema version and monotonic timestamp;
- side and privacy-safe event/candidate ID;
- transition type and evidence/reason code;
- meter before, provisional after, and confirmed levels spent when available;
- selected scope;
- emitted audio action kinds only (`stop`, `play`, or `fade`); and
- final outcome label when available.

The writer appends off the capture hot path through a bounded queue. When the
queue is full or the directory is unwritable, it drops diagnostic records and
reports one privacy-safe panel warning; gameplay detection and audio continue.
Logs rotate at 2 MiB and retain the five newest session files. Rotation never
deletes configuration, audio, profiles, releases, or files outside the exact
`diagnostics` directory.

## Reset, Deduplication, and Failure Semantics

- Explicit stop, application close, capture loss, stale-frame rejection,
  profile change, window change, or geometry change first routes the
  cancellation returned by `GaugeTracker.reset`, then resets any confirmed
  outcome, and therefore terminates provisional audio exactly once before
  state is discarded.
- A repeated candidate, confirmation, cancellation, or outcome ID is ignored
  and logged as a duplicate; it cannot restart or stop audio twice.
- Candidate confirmation and cancellation are mutually exclusive terminal
  transitions.
- Audio backend errors are reported through the existing privacy-safe
  diagnostic callback and cannot mutate tracker authority state.
- Diagnostic-log errors cannot block frame capture, tracking, audio routing,
  or application shutdown.
- P1 and P2 transitions are processed in P1-then-P2 order but use independent
  tracker state, audio channels, pending candidates, and outcomes.

## Compatibility

- Existing schema-2 colour profiles and derived meter-level ROIs remain
  loadable without recalibration.
- Existing `config.json` files retain both playback scopes and whiff settings;
  no configuration migration is required.
- Profiles without reliable meter-level support remain fail-closed for the new
  candidate path. They may use the existing confirmed legacy event path but
  cannot claim immediate all-tier support.
- The GUI keeps the existing controls. Only privacy-safe candidate and
  cancellation status text plus a diagnostic-log warning are added.
- Existing portable builds remain untouched and recoverable.

## Test Strategy

Implementation is test-driven, with every production change preceded by a
failing behavioral test:

1. domain tests validate candidate/update/cancellation contracts, side
   identity, timing, meter metadata, and terminal-transition invariants;
2. tracker tests prove three-sample candidate latency, unchanged authoritative
   baseline before confirmation, clean confirmation, immediate recovery and
   conflict cancellation, 0.75-second timeout, duplicate suppression, reset
   cancellation, P1/P2 isolation, and no candidates from gain/wrap/unknown;
3. audio tests prove candidate STOP/PLAY, confirmation without replay,
   cancellation fade, tier-filter fade, and whiff mutual exclusion;
4. engine tests cover exact routing order, confirmed-without-candidate recovery,
   duplicate IDs, reset behavior, side isolation, and audio failure isolation;
5. outcome tests prove the candidate timestamp and pre-spend frame remain the
   baseline for 3-second contact resolution;
6. diagnostic-log tests use a temporary real directory and verify schema,
   privacy omissions, queue overflow handling, rotation boundaries, and
   shutdown without mocking filesystem behavior;
7. deterministic replay tests cover the known successful CA clips, long
   real-match false-trigger recordings, round-end transitions, ordinary meter
   gain, SA1/SA2/SA3/CA, hit/block/whiff, and the 22-minute/08 evidence when a
   local fixture can be derived without storing private video in the repo;
8. the complete offline suite, portable startup self-check, and release hygiene
   run before a new candidate is packaged; and
9. only after offline approval, SF6-only real acceptance covers candidate
   latency, round-end silence, natural gains, both scopes, hit, block, whiff,
   P1/P2, and performance. MK1 and other applications remain out of scope.

## Acceptance Criteria

- A valid Super candidate starts the configured side's music once within
  0.25 seconds nominally at 20 FPS.
- Candidate emission never changes the authoritative meter baseline.
- A rejected candidate begins fading no later than 0.75 seconds after its
  timestamp and never plays the whiff sound.
- A confirmed candidate never restarts music.
- Natural meter gain, unchanged level, single-frame glitches, and fractional
  fill wraps create no candidate.
- Known round-transition fixtures create no confirmed event; any provisional
  candidate is cancelled and faded within the fixed deadline.
- `ALL_SUPERS` keeps confirmed SA1/SA2/SA3/CA playback through hit or block.
- `SA3_CA_ONLY` fades confirmed SA1/SA2 and keeps confirmed SA3/CA.
- Whiff produces exactly one configured fade or one STOP/PLAY whiff batch.
- Diagnostic logs explain every audio start and terminal decision without
  recording private paths or pixels.
- The full offline suite and portable self-check pass before packaging.
- No build is described as real-match accepted until the SF6-only live matrix
  has been observed and recorded.

## Out of Scope

- character- or move-specific cinematic templates;
- OCR or machine-learning model downloads;
- memory reading, injection, macros, or network inspection;
- automatic operation of MK1 or unrelated applications;
- changing the user's selected audio or SF6 graphics configuration; and
- claiming complete all-character reliability from offline evidence alone.
