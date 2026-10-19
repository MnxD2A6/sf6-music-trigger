# R23 local recording regression crops

Source: user-provided R23.mp4, SHA256
`2fbe367da03a1fae3ee3c3caa8eb59c9a2ebaf967afbd6700cdc9765f1ba63d7`.
1916x1092, 30 fps, 13343 frames. Analysis removes the top 26-pixel title bar.
The NPZ contains only 11 P2 gauge-glyph HUD crops and their recording offsets;
no music, full frames, account identity, or absolute source path is included.

Read using allow_pickle=False. Reconstruct the derived P2 digit ROI on a
1916x1066 canvas with the bounds in test_r23_video_regressions.py.
These compressed/scaled recording samples do not substitute for native capture.

Known R23 failures: indexes 2/3/4 (85.867/85.9/85.967 seconds) read 2 as 0;
index 10 (432.667 seconds) reads a contaminated 2 as 1 instead of abstaining.
They are strict expected failures, not fixed or accepted functionality.
A trial pinhole-size/aspect filter was withdrawn after full replay introduced
a false event at 257.6 seconds (KO; actual P2 meter remains 2).
