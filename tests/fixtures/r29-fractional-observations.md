# R29 fractional-bar regression provenance

Source: user-supplied R28.mp4, SHA256
FA3B5B6B393E26AB972F37904E546F18502B18ABCC52DD1D88FE3D7F1BB1E478.
Recording 2018x1152, 30fps; exact client crop x87 y69 w1920 h1080,
no rescale. Production R28 observe_gauge using shipped screenshot profile.

Stored JSON contains anonymous scalar observations only, no private paths,
screenshots, audio, player names or raw runtime IDs. Side P1 covers
1015..1040s (751 frames); P2 covers425..442s (511 frames). Frame ID is
round(recording_seconds*30). Geometry is verified in the source review.
Distances are reconstructed from confidences for tracker tests: the tracker
does not consume distances. This is a tracker regression, not a new reader
or native-capture acceptance test.

Manual labels, independent of the software's output:
- P1 Jamie SA1 at1027.4s,1->0, visible fractional bar preserved~77%.
  Exactly one SA1 in this window, no later three-bar spend. At1030..1034s
  digit glitches must not create a second event. Original code emits none.
- P2 Juri SA1 at437.7667s,1->0. Exactly one SA1. Original20Hz confirmation
  439.1s,30Hz439.0s. Target confirmation within~0.6s of first visible0.

Tests run20Hz (skip frame_id%3==1) and30Hz without synthetic baseline seeds.
Changing only the old zero-fill veto makes the P1 realSA1 visible but also
creates a false3->0 at1031.4s. The delivered fix must reject that counterexample.
Capture/read/playback timing is measured separately; this does not prove
the native12:44.4 severe-lag incident is repaired.

Companion r29-severe-latency-observations.json uses the same source/profile,
755..778s P2 (691frames). Reader source is unchanged inR29. Manually reviewed
SA1at764.4s2->1; rawfirst1at764.2667s. It must still produce one timely
offline event and no second late event. This is a characterization regression:
the native audio at772.208s remains an unexplained live/offline difference.
