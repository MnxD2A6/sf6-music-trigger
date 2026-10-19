# R22 P2 glyph regression fixture

The accompanying NPZ contains 14 small P2 meter-glyph crops, not full frames,
from the user's R22 recording. No desktop, music, account data or paths are
included. Times are seconds in that recording; expected -1 means unreadable or
absent. Labels were visually inspected independently of detector output.

The 1920 x 1080 SF6 client was cropped from the recording at (480, 242).
The glyph ROI is derived from meter ROI (0.775, 0.885, 0.945, 0.975).
R22 mistook an unchanged zero at 662.0/662.2/749.0 seconds for 3/2/1 and
could not reliably read violet-blue digits at 346/761 seconds. These are
recognition regressions, not labelled Super activations.

The test reconstructs only this band into a blank client-sized frame so the
production ROI derivation and reader run unchanged. The fixture stays in
source tests; portable release packaging does not include tests.

`r22-p1-two-with-background.npz` contains five analogous P1 glyph-only crops at
299, 303.5, 303.767, 304 and 306 seconds. All visibly show 2. A bright background
appendage during the post-KO scene joins its lower edge; the old wide-1 rule
incorrectly classifies it as 1 at 303.767. The regression permits 2 or abstention,
not a fabricated meter spend. Its meter ROI is (0.065, 0.885, 0.225, 0.975).
