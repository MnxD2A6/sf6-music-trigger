# SF6 Two-Stage Trigger r20 Verification Evidence

Date: 2026-09-02 (Asia/Singapore)

## Scope and provenance

- Branch: `background-independent-meter`
- Initial Task 8 source/test commit: `51739baf0052d200a704ec3614a9956b584bb123`
- Authoritative-baseline source/test commit: `e1c37147701fcfc0a8861be86c3e8fb892871494`
- Review-fix and artifact source commit: `b4ddbc329bcecb9262c66d60ea362a40c46469b5`
- Final-branch repair and r20 source commit: `0f12811f9c6a237f6245f02e769e0781ef18025d`
- Python: bundled Python 3.12 with the repository SDD `sitecustomize.py`
- PyInstaller: 6.22.2 from the existing offline shared runtime
- No network access, dependency download, private-frame export, or live-game launch

The artifact was built only after the exact source commit existed. Runtime user
configuration, selected audio, `main.py`, and the SF6/CU/MK1 installations
were not changed.

## Final-branch repair TDD and offline suites

Five final-review regressions were written before implementation. RED showed
that one incompatible active-candidate level cancelled immediately, same-ID
P1/P2 confirmations shared global engine/outcome state, a permanently blocked
diagnostic write kept its subprocess alive past the deadline, lifecycle
histories grew without a bound, and a real diagnostic log/controller could not
complete two start/close cycles. The isolated B-to-E RED run was
`7 failed, 3 passed, 411 deselected`; the corrected tracker RED was
`3 failed, 237 deselected`.

The minimal repair now requires three fresh side-local incompatible samples
before `LEVEL_CONFLICT`, while authoritative-baseline recovery still cancels
immediately. Unknown, target recovery, terminal transitions, authority loss,
and reset clear the streak. Confirmed event identity is consistently
`(Side, event_id)` through execution, outcome begin/terminal correlation, and
duplicate suppression. Started, terminal, executed, seen, and event-context
histories use bounded FIFO/LRU storage; terminal outcome releases its context
without releasing duplicate history. The diagnostic writer is a daemon with a
bounded close and explicit tail-loss warning, and both it and the controller's
warning latch are reusable across real start/close cycles. Changed suites then
passed `421 passed in 2.40 s`.

Final warnings-as-errors verification used writable sibling TEMP/TMP, disabled
the pytest cache, and used no downloaded or installed dependencies:

| Gate | Result |
| --- | --- |
| Changed tracker/engine/outcome/log/controller suites | 421 passed in 2.40 s |
| Focused product regression | 976 passed, 3 skipped in 102.47 s |
| Complete offline suite | 1,835 passed, 3 skipped in 115.69 s |

The three skips are the existing optional user-labeled screenshot fixtures;
there were no warnings or failures.

## Post-artifact test-only coverage

A final rereview found one remaining coverage gap, not a production defect.
Test-only commit `da41cbf9a0a6e87c4b629b3333f661ba94175e2e`
now uses the same `GaugeTracker` instance to preload two incompatible candidate
samples and independently proves that confirmation, stable-authority loss,
non-reset cancellation, and explicit reset discard the streak. Each sequence
then continues or rebuilds on that same instance and verifies that one later
incompatible sample cannot behave like a stale third sample. The confirmation
and authority-loss cases isolate their exact cleanup branches from the normal
target/unknown pre-cleanup.

Fresh warnings-as-errors evidence after the test-only commit is:

- Focused cleanup cases: 4 passed, 239 deselected in 0.82 s.
- Complete gauge tracker suite: 243 passed in 1.13 s.
- Complete offline suite: 1,838 passed, 3 skipped in 114.62 s.

No production file changed, no private replay was required, and no release
artifact was mutated. r20 therefore was not rebuilt: its exact production
source and embedded `VERSION.txt` provenance remain
`0f12811f9c6a237f6245f02e769e0781ef18025d`; the later commit adds tests only.

## Private-video aggregate replay

Five existing local recordings were decoded at 20 FPS through both the private
detector gate and the production observer, tracker, engine, outcome, and audio
scheduler. No frames, crops, audio, or private paths were copied into the
repository or release.

| Replay label | Processed | Candidate | Confirmed | Cancel | Outcome | Stop | Play | Fade |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stable positive | 889 | 3 | 3 | 0 | 3 | 3 | 3 | 0 |
| Long false-trigger regression | 9,373 | 5 | 4 | 1 | 4 | 5 | 5 | 3 |
| Hit / whiff / block | 1,753 | 3 | 4 | 0 | 4 | 4 | 4 | 1 |
| 22-minute match | 26,715 | 16 | 28 | 3 | 5 | 18 | 18 | 18 |
| Recording 08 | 16,309 | 2 | 2 | 0 | 1 | 2 | 2 | 2 |

Detector confirmation counts were respectively 3, 4, 4, 28, and 2. The long
false-trigger 184-to-197-second window has zero candidates, confirmations, and
audio batches. The stable-positive candidate landmarks remain 3.8, 16.2, and
29.9 seconds. The approved 22-minute staged-episode baseline remains 28: the
surviving event at 153.6667 seconds is P2 3-to-0, while candidate and audio
totals remain 16 and 18.

## Invalid artifact quarantine

r16, r17, r18, and r19 are **INVALID — NOT FOR DISTRIBUTION**. r16 contains
failed packaging evidence; r17 through r19 were built from source superseded
by later semantic fixes. Their folders, ZIPs, and in-folder manifests were
moved without overwrite or deletion into separate generic sibling
subdirectories below `INVALID_NOT_FOR_DISTRIBUTION`. Each quarantine
subdirectory has an explicit marker of the same name. The original eight
public-looking folder/ZIP paths no longer exist.

The deterministic tree digest hashes sorted `relative-path`, tab, file SHA-256,
and newline rows. Counts, bytes, and every digest matched before and after the
moves:

| Artifact | Files | Bytes | Tree SHA-256 | Manifest SHA-256 | ZIP SHA-256 |
| --- | ---: | ---: | --- | --- | --- |
| r16 | 1,121 | 199,156,426 | `6CF1522A191CEE861B6D875ED874AA26AE7B73C6CF664052B4979E6D5CDE7022` | `600A1639F9B3A7BA590BB9C725F93D493C69D5A5962725DDA1C53F433450F091` | `84D6BC1DF27A7466395524F4D77AB6F1FA5B4267B4F987CC167A51E63A3DB4C1` |
| r17 | 1,110 | 199,321,848 | `F2CFC156BD0A7FDA4AC1C42B172BE399A403DA093AE94090E698E504D74230FC` | `56D363714A6D58906D52D9BFE26C64BE627671CA986E0B02ED447A235C02CC9A` | `CA16D42202704450D4B334AC641172FD64392B1D092DB9A9AD4D74FA855BBA37` |
| r18 | 1,110 | 199,321,914 | `4A2CC419C40D630FD9C58941600A33F1CD147CDAC9A4D8462F21F7BF71D459B2` | `5019CC7DEA3B3BF1ECB331BF80EBD3CB267F28097B029CBF361045F07057D416` | `F56AE121992444864F39DB21DCB7FAC6F6E9A0A8CE11EFC30756AD177F21D083` |
| r19 | 1,110 | 199,321,941 | `8263CC4FD763A2E3759CF1D96F0D9EB4A234FCA035B68EEFDFF45E0CAA3500A1` | `212A99D213D0C9A510BE6294179D2306842F98BBE041AF07FC68F0CD56B5005B` | `842AC8DB38607CA6677350C369895FE76E18582C521635ED832D152FFECCF188` |

## r20 artifact gates

The accepted offline candidate is the fresh non-overwriting generic sibling
artifact:

- Folder: `SF6_Music_Trigger_GroupTest_0.1.0a1_20260902_r20`
- ZIP: `SF6_Music_Trigger_GroupTest_0.1.0a1_20260902_r20.zip`
- Folder files: 1,110
- Manifest entries: 1,109
- Folder bytes: 199,323,217
- ZIP bytes: 74,688,624
- ZIP SHA-256: `65A326BE0EE58A1553E073BD26550479C1B76B2F28CC0A032F94CBF7088EBE0B`
- EXE SHA-256: `ED47ECC2DF7CF0D7BAA858795FAB824E94086DE382C826ED1392E01D988FA6D0`
- Manifest SHA-256: `562298C4E93DF05D1C9F20638B0C85694CF4593E737702A59AB8AB1EDC88E77C`
- Sanitized config SHA-256: `B972A3B1EDC0F055655B57BF8391CEFC71BFD97204F32CD04D973C5BD0B79D42`
- Bundled profile SHA-256: `9D224830E7EB3980751E7FA315935FC49249E348BA05D84B81DC12F4F9D8D0B4`

The builder ran outside the filesystem sandbox without a legacy Tcl/Tk patch
and returned exit code 0. An independent hidden `--portable-self-check`
returned exit code 0. Every manifest filename and digest matched independently.
The ZIP has exactly one r20 top-level folder, its file set matches the folder,
and every decompressed entry hash matches its source file.

Public-payload checks found blank main/whiff audio paths, zero user-provided
audio/video/screenshot assets, no diagnostic capture directory or logs, no
absolute private path, no build-root or repository marker, and exact
`VERSION.txt` provenance for source commit `0f12811`. The only raster assets
are required third-party pygame/Tk runtime images below `_internal`.

## Acceptance status

r20 is an accepted **offline group-test candidate**. r16, r17, r18, and r19
are invalid and must not be distributed. `SF6 real acceptance not yet performed`.
A later authorized SF6-only session must still cover natural gain, SA1, SA2,
SA3/CA, hit, block, whiff, round transitions, both sides, and gameplay
performance.
