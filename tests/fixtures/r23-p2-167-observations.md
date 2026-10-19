# R23 P2 167-second observation segment

Derived from the user's R23.mp4 (source SHA256 in r23-p2-glyphs.md), not native
capture. Existing profile, top 26px titlebar removed, alternating 1/2 video frame
steps to approximate 20fps. Contains scalar GaugeObservation data only, 166-170s.
No private file paths, captured names, music, images, or object pickle payload.

The recording's P2 visible SA1 activation begins about 167.033s; waveform matching
against the selected P2 music gives 168.079s for recorded track start. The latest
runtime log has a confirmation-only event corresponding to this start.

Fresh continuous segment replay initializes from already-visible 1 at 166s (no
synthetic baseline assignment). R23 emits only confirmation at 167.4. The narrow
P2 color-flicker patch emits a candidate at 167.1667 with the same final
confirmation at 167.4, saving 233ms of recognition wait in this segment.
Whole-recording replay has a separate earlier untrusted-gain issue and misses
this event, so these are NOT equivalent to an end-to-end capture acceptance.

At 167.0667/167.100/167.1667, the glyph remains 0 at confidence .9 while spent
color confidence is approximately .739/.739/.691. Third-sample marginal color
must not erase the preceding two strong readings; missing/conflicting glyphs,
lower confidence, and weaker color must still abstain. P1 is unchanged.
