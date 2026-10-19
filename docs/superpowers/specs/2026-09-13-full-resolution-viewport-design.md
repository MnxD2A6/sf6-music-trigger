# SF6 Full-Resolution Viewport Design

## Goal

Extend the stable R34 detector to common 16:9, 16:10, and 21:9 client sizes from 720p through 4K, in windowed and borderless-windowed modes, without changing meter, outcome, fade, or audio policy thresholds.

## Supported boundary

- A client whose aspect already matches the selected HUD profile continues through the exact R34 path.
- A wider or taller client is supported automatically when it contains a centered, clearly dark letterbox or pillarbox around a content rectangle matching the selected profile aspect.
- A non-letterboxed layout with a different aspect ratio requires the existing HUD calibration flow and a profile created for that aspect.
- Clients whose effective content height is below 720 pixels are not release-qualified.
- Exclusive fullscreen is not claimed; windowed and borderless-windowed use the captured client area.

## Architecture

Create `viewport.py` as an isolated pre-processing boundary. `CenteredViewportResolver` receives a complete `CapturedFrame`, derives the largest centered rectangle matching the HUD profile aspect, and returns a cropped frame only after the expected outer bars are confirmed on consecutive frames. Matching-aspect frames pass through unchanged.

`VisualTriggerEngine` performs freshness checking on the original frame, then asks the resolver for the effective frame before the existing profile gate, outcome observer, and P1/P2 meter observers run. The effective frame therefore keeps every R34 recognition threshold and normalized ROI unchanged.

## Fail-closed behavior

- Invalid frame dimensions, insufficient effective resolution, bright/nonuniform borders, or an unstable viewport produce no observations or audio actions.
- A client-geometry change resets viewport confirmation and the existing tracker/outcome state before processing resumes.
- Once a non-native viewport is confirmed for one stable geometry, it remains locked until that geometry changes; dark super animations cannot move the crop.
- Capture loss and stale-frame behavior remain unchanged.

## Diagnostics

The engine exposes a privacy-safe viewport status: `native`, `confirming`, `normalized`, `unsupported`, or `too_small`. The panel may display this status but must not log screenshots or pixel data.

## Verification

- Pure geometry tests cover 1280x720, 1920x1080, 2560x1440, 3840x2160, 1680x1050, 1920x1200, 2560x1080, 3440x1440, and 3840x1600.
- Image tests prove centered dark bars normalize to a 16:9 frame and bright borders fail closed.
- State tests prove confirmation, locking, and reset on geometry change.
- Engine tests prove observers receive the effective frame and receive nothing while viewport status is uncertain.
- Existing R34 meter, outcome, fade, audio, and template suites must remain green.
- Automated tests and portable self-check do not replace live SF6 acceptance at 16:10 and 21:9.
