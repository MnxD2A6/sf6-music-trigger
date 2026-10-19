# Privacy

All processing is local to the PC. Nothing is uploaded and no personal data is
collected.

## REFramework detection (V2, default)

The game scripts only read game state (character ids, action ids, attack frame
data) and never write game objects or send input. They write these files into
`<SF6 folder>\reframework\data\SF6MusicTrigger`:

- `events.jsonl`: super starts/results/ends and a heartbeat (character and
  action numbers, timestamps). Truncated whenever the game starts and capped
  at 2 MB.
- `app_state.json` / `ui_commands.jsonl`: the settings exchanged with the
  in-game settings window: track file *names*, volumes and options.
- `detector_settings.json` / `ui_settings.json`: the two in-game checkboxes.

The panel stores its configuration (`config.json`) next to the program in the
portable release. "从游戏中移除" deletes all V2 files from the game folder.

## Screen detection (legacy T03 backend)

Screen capture saving is opt-in; the default workflow does not save captures.
The listener writes bounded local diagnostic logs under
`%LOCALAPPDATA%\SF6 Music Trigger\V2\diagnostics`, including anonymous timing
and meter metadata under the `latency` subfolder. These logs contain no
screenshots, audio samples, music paths or window titles.

Share any of these files only when you choose to provide troubleshooting
evidence.
