# SF6 Full Visual Music Trigger Roadmap

> **For agentic workers:** Each phase has its own implementation plan and review gate. Do not begin a later phase before the prior phase has a clean test suite and accepted interface.

**Goal:** Deliver a Windows-only, online-compatible, open-source SF6 music trigger that detects real SA3/CA spend from pixels and locally classifies hit, block, whiff, reset, or unknown.

**Architecture:** A deterministic screen-only GaugeTracker is the canonical event source. Outcome rules and an optional local ONNX classifier consume only confirmed events. Audio, capture, UI, diagnostics, and model inference communicate through typed event interfaces.

**Tech Stack:** Python 3.12, NumPy, OpenCV, DXcam, pygame, Tkinter, pytest, ONNX Runtime in the model phase, PyInstaller in the release phase.

## Global Constraints

- Support Windows 10 22H2 and Windows 11 23H2 or newer.
- Support Steam PC Street Fighter 6 only.
- Support P1/P2, modern/classic controls, and online/local/training play without using control inputs as a trigger prerequisite.
- Never read game memory, inject code, modify game files, synthesize input, inspect network traffic, or require administrator/kernel components.
- All capture and inference are local. No screenshot, clip, player name, Fighter ID, or telemetry upload without an explicit user export action.
- Do not commit commercial music, user configuration, logs, diagnostic captures, game screenshots, or machine-specific absolute paths.
- A low-confidence or stale-capture result is `UNKNOWN`; it must not be guessed into `HIT` or `WHIFF`.
- SF6 and MK1 must never be launched at the same time during live acceptance.

---

## Phase 1: Open-Source Foundation and Gauge Trigger

**Deliverable:** A clean MIT repository with a persistent Windows panel that captures SF6 pixels, loads a user-calibrated HUD profile, independently detects P1/P2 stable full-meter-to-spent transitions, and immediately plays the configured main track. No keyboard/gamepad mapping is required.

**Plan:** `docs/superpowers/plans/2026-08-08-foundation-gauge-trigger.md`

**Release gate:** Synthetic and recorded-frame replay tests pass; live status is explicitly marked pending until Computer Use can access the host SF6 window.

## Phase 2: Outcome Observer, Conservative Rules, and Labeling

**Deliverable:** A post-spend 2–6 second multi-ROI observer, rule-based `HIT/BLOCK/WHIFF/RESET/UNKNOWN` decisions, configurable result audio, and a local review/label/export tool.

**Plan boundary:** This plan may start only after Phase 1 freezes `ConfirmedSuperEvent`, `CapturedFrame`, `HudProfile`, and `AudioDirector` interfaces.

**Release gate:** Human-labeled replay set includes P1/P2 hit, block, whiff, reset, and capture loss. The classifier never uses current algorithm labels as truth.

## Phase 3: Optional Local Model and Benchmark

**Deliverable:** A versioned ONNX model-package format, local CPU inference adapter, rule/model fusion, grouped benchmark tooling, model card, license/provenance, checksum validation, and confidence fallback.

**Plan boundary:** The first training run requires manually labeled event clips split by user/session/environment. Model inference can improve Phase 2 confidence but cannot create a super event or bypass GaugeTracker.

**Release gate:** Held-out environment metrics and confusion matrix are published; low-confidence predictions become `UNKNOWN`; the rules-only mode remains fully functional.

## Phase 4: Windows Packaging and Online Acceptance

**Deliverable:** Reproducible PyInstaller build, clean installation bundle, privacy/license documentation, update policy, performance report, and live SF6 validation evidence.

**Acceptance matrix:**

- P1 and P2.
- Modern and classic controls.
- Hit, block, and whiff.
- Ranked-equivalent, casual, custom-room, training, and local visual layouts.
- 1920×1080, 2560×1440, and 3840×2160 at 16:9.
- Consecutive supers, rollback-like repeated frames, window movement, minimization, stale capture, and Start/Stop/Exit.

**Release gate:** Computer Use accesses the host SF6 window, only SF6 is launched, the controlled matrix passes with logs/screenshots, SF6 is closed after testing, and no result is described as live acceptance without that evidence.
