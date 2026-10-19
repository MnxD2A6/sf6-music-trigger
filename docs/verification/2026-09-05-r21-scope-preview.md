# R21 scope-aware preview

R20 baseline: 992b77acb7ead2f452d12e2f4445fb31669e1706 (test/document
additions after production commit 0f12811). Source bundle, source archive,
original release ZIP and local configuration were backed up before edits.
The backed-up R20 ZIP SHA256 remains
65A326BE0EE58A1553E073BD26550479C1B76B2F28CC0A032F94CBF7088EBE0B.

The reported P2 preview ended because SA3_CA_ONLY confirmed a one-bar spend
approximately 0.10 seconds after playback and requested a 1500ms fade.

Restricted mode now previews only a candidate whose baseline and target are
3 and 0. Lower/partial candidates remain observable but silent. If the existing
tracker later confirms three bars, confirmation starts audio once. This can
delay playback for partial candidates until confirmation. ALL_SUPERS retains
immediate previews. Cancellation of an unplayed candidate does not fade an
older track. Tracker, capture and outcome detection were not changed.

No new fixed three-second merge window was added: the reported recording did
not demonstrate a split SA3, and merging independent spends without evidence
could manufacture a false SA3. Existing tracker episode rules remain in force.

Verification: new regression tests failed 18 cases before the fix and all 32
passed afterwards. Full offline run: 1872 passed, one subprocess import failure
caused by the isolated runtime path. After adding the project source path to
that local runtime, that single test passed. Earlier import failures were also
resolved in that environment. Tcl initialization: 8.6.12. Historical private
replay expectations now omit two filtered lower-tier play/fade pairs; event
counts and confirmation timestamps remain unchanged.

Live R21 acceptance remains pending: both sides, both scopes, SA1/SA2 silence
in restricted mode, immediate 3-to-0 preview, eventual partial-to-SA3 playback,
hit/block/whiff, round transitions and overlapping side audio. User recordings
and local audio paths are excluded from the distribution.
