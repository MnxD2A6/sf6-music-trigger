# Immediate Super Playback and Deferred Resolution Design

## Goal

Keep the current responsive behavior—play the configured main track as soon as
an SF6 super activation is detected—then use a bounded post-activation window
to correct low-tier activations and whiffs without character- or move-specific
templates.

This design is screen-only, SF6-only, local, and side-independent. It does not
modify or use the old prototype `main.py`, inject into the game, read process
memory, operate MK1, or require network access.

## Stable Baseline and Backup

Implementation starts from local commit
`ffc6a51c5d7e78038ef5a5d2ae05ffcbb63bd454`.

The baseline is protected by:

- annotated Git tag `backup/stable-ca-audio-20260816`;
- snapshot directory
  `<workspace>\SF6_Music_Trigger_Snapshots\2026-08-16-ffc6a51-stable-ca-audio`.

The existing immediate meter trigger remains the activation boundary. The new
logic is layered after that boundary and must not weaken its desktop-overlay
or CA-animation protections.

## User-visible Rules

Each side has an independent super playback scope:

1. `ALL_SUPERS`: SA1, SA2, SA3, and CA all start the main track immediately.
2. `SA3_CA_ONLY`: every plausible super still starts the main track
   immediately. After the HUD becomes trustworthy again:
   - stable remaining meter of zero confirms SA3/CA and permits playback to
     continue;
   - stable remaining meter of one or two bars classifies SA1/SA2 and fades the
     provisional playback.

`SA3_CA_ONLY` is the default for newly loaded configurations that omit the new
field. SA3 and CA do not need to be distinguished from one another because
both consume all three bars and share the same playback policy.

Whiff behavior remains mutually exclusive per side:

- if `play_whiff_sound` is true, a confirmed whiff stops the main track and
  plays the configured whiff sound; it does not fade;
- if `play_whiff_sound` is false, a confirmed whiff fades the main track using
  `whiff_fade_ms`;
- a low-tier SA1/SA2 filtered by `SA3_CA_ONLY` is not a whiff and therefore
  never plays the whiff sound; it only fades the provisional main track;
- hit or block permits the main track to continue.

If capture is lost, the SF6 window changes, or the result remains genuinely
unknown at the hard deadline, the safe behavior is to fade provisional
playback. Unknown evidence must never play the whiff sound.

## Two-stage State Model

The current confirmed meter-spend event starts a `PendingSuperResolution` for
that side and immediately submits the existing STOP/PLAY main-track batch.
P1 and P2 own separate pending objects and may overlap.

Each pending resolution retains only bounded derived features:

- event identity, side, and activation time;
- the pre-spend frame ID supplied by the tracker;
- pre-spend opponent-health and training-result UI features;
- the calibrated meter endpoint model needed to quantify remaining bars;
- the selected playback scope;
- consecutive evidence counters and deadlines.

It does not retain full-resolution frames or write captured pixels to disk.
The engine keeps a small in-memory ring of precomputed outcome features keyed
by frame ID so the event's `pre_spend_frame_id` can recover the real pre-super
baseline. Missing baseline evidence fails safely to `UNKNOWN` rather than
inventing hit, block, or whiff.

The state progression is:

```text
IDLE
  -> PROVISIONAL_PLAYING
  -> OBSERVING
  -> CONTINUING_CINEMATIC (HUD still obscured; optional extension)
  -> HIT | BLOCK | WHIFF | FILTERED_TIER | UNKNOWN
  -> IDLE
```

Starting a new event for a side replaces only that side's previous completed
state. Duplicate event IDs cannot restart playback or create a second result.

## Timing and Evidence

The initial result checkpoint is 3.0 seconds after activation.

- During the first 0.4 seconds, features may be collected but no result may be
  finalized. This suppresses universal super flashes and camera cuts.
- From 0.4 through 3.0 seconds, only stable consecutive evidence contributes
  to a result. Single-frame changes are ignored.
- Frames whose meter structure is hidden or unverified do not resolve tier or
  outcome, but they also do not clear the pending event.
- If the same valid SF6 capture session remains active but trustworthy HUD has
  not returned at 3.0 seconds, playback continues and observation extends
  until trustworthy HUD returns, with a hard deadline of 8.0 seconds after
  activation. HUD absence pauses the decision; it is not itself proof of hit.
- If normal HUD has returned by the checkpoint and there is no stable contact
  evidence, the result is WHIFF.
- At the hard deadline, unresolved state is UNKNOWN and provisional playback
  fades.

No standalone animation classifier is required. “The follow-up animation is
still running” is represented conservatively by a valid unchanged SF6 capture
session whose post-event HUD has not yet become trustworthy again. Once
trustworthy HUD returns, persistent opponent-health change has priority as HIT;
otherwise persistent training-result or guard UI change may label BLOCK. HIT
and BLOCK are both contact outcomes with identical audio behavior, so an
uncertain distinction between them must never turn stable contact into WHIFF.
No contact evidence with a trustworthy restored HUD labels WHIFF. A desktop or
task-view frame cannot prove continuation, because it is not trustworthy HUD
and cannot supply outcome or tier evidence; it only reaches the UNKNOWN fade
deadline unless capture validation rejects it earlier.

Tier resolution uses the calibrated coloured-meter fill normalized between
the stored spent and full endpoints. Five consecutive trustworthy samples
must agree on the nearest remaining-bar bucket: zero, one, two, or three.
Intermediate or unstable values remain unresolved. In `ALL_SUPERS`, tier is
diagnostic only. In `SA3_CA_ONLY`, zero continues playback and one/two causes
`FILTERED_TIER` fade. A value of three after an activation is inconsistent and
must remain UNKNOWN rather than be treated as a whiff.

## Decision Ordering

For every pending side, decisions are applied in this order:

1. validate that capture, window identity, geometry, and profile remain valid;
2. preserve pending state through expected CA/super visual obstruction;
3. resolve stable remaining meter when the HUD returns;
4. apply the configured tier policy;
5. resolve stable hit/block evidence;
6. at the initial checkpoint, resolve trustworthy no-contact as WHIFF or wait
   for a still-obscured HUD to return;
7. at the hard deadline, resolve anything else as UNKNOWN.

This ordering prevents a filtered SA1/SA2 from being misreported as a whiff and
prevents cinematic pixels from becoming the outcome baseline.

## Audio Mapping

The existing audio port remains the only playback boundary. A result produces
at most one atomic action tuple:

| Result | Actions |
| --- | --- |
| HIT / BLOCK | no action; main track continues |
| WHIFF + sound enabled | STOP side, then PLAY whiff track |
| WHIFF + sound disabled | FADE side using `whiff_fade_ms` |
| FILTERED_TIER | FADE side using `whiff_fade_ms`; never whiff sound |
| UNKNOWN / RESET | FADE side using `whiff_fade_ms` |

Audio failures remain privacy-safe diagnostics and cannot mutate recognition
state or cause another side's channel to stop.

## Configuration and Panel

Add a strict string enum field to each `SideAudioConfig`, serialized under each
side as `super_scope` with values `ALL_SUPERS` and `SA3_CA_ONLY`. Unknown values
fail closed during configuration loading. Missing values use
`SA3_CA_ONLY` without requiring manual migration of the current local file.

The panel adds one compact selector per side:

- `全部大招（SA1/SA2/SA3/CA）`
- `仅 SA3/CA`

Existing whiff sound/fade controls remain mutually exclusive and unchanged.
Diagnostics may show privacy-safe states such as `等待大招结果`, `后续动画`,
`命中`, `格挡`, `打空`, `低等级已淡出`, or `结果不确定`; they must not expose
audio paths or captured pixels.

## Failure and Reset Semantics

- Expected super animation obstruction pauses evidence and preserves pending
  resolution.
- Explicit stop, application close, profile change, capture-session restart,
  window identity change, or geometry change resolves pending provisional
  playback to UNKNOWN/RESET and fades it exactly once.
- Desktop/task-view frames cannot produce contact or tier evidence. They may
  leave an existing pending event untouched only until its fixed hard deadline
  and can never extend that deadline.
- A black, stale, duplicate, or backward-time frame cannot advance evidence.
- A new valid capture session starts with no pre-super baseline and therefore
  cannot inherit a pending result from the previous session.
- Simultaneous P1/P2 events and outcomes are deterministic in P1-then-P2 order
  while retaining independent state and audio channels.

## Test Strategy

Implementation is test-driven and divided into independently reviewable
clusters:

1. configuration and panel tests for both scopes, missing-field default,
   rejection of unknown values, round-trip persistence, and unchanged
   whiff-control mutual exclusion;
2. pure meter-level tests for normalized zero/one/two/three buckets,
   five-sample stability, ambiguous boundaries, hidden/unverified frames, and
   color endpoints for both sides;
3. pure outcome tests proving the baseline comes from the pre-spend frame,
   early flash suppression, cinematic extension, hit/block priority,
   trustworthy no-contact whiff, and UNKNOWN hard deadline;
4. engine tests proving immediate one-time main playback, side isolation,
   duplicate suppression, decision ordering, and every result-to-audio mapping;
5. offline replay of the existing successful CA videos, desktop false-trigger
   video, whiff recordings, and `杰米测试3.mp4` with a silent fake audio port;
6. the complete offline regression suite, with the known Windows directory
   junction host limitation reported separately rather than hidden;
7. only after offline approval, a minimal SF6-only real acceptance covering
   ALL_SUPERS hit, SA3/CA-only zero-meter continuation, SA1/SA2 filtering,
   whiff fade, and whiff sound. No MK1 or other application is operated.

## Acceptance Criteria

- Main playback begins immediately at the same activation boundary as the
  current stable recording.
- ALL_SUPERS preserves playback for hit/block SA1, SA2, SA3, and CA.
- SA3_CA_ONLY preserves playback only when stable restored meter is zero;
  stable one/two-bar remainder fades and never plays the whiff sound.
- A genuine whiff results in exactly one fade or one STOP/PLAY whiff batch,
  according to the mutually exclusive user setting.
- An obscured cinematic may extend observation without being mistaken for
  hit/block.
- Task view, desktop, capture loss, and unknown results cannot leave
  provisional music playing indefinitely.
- The stable baseline remains recoverable by the backup tag and snapshot.
