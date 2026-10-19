"""In-game settings window (sf6_music_trigger_ui.lua) against a scripted imgui."""

from pathlib import Path
import time

import pytest

from fake_reframework import FakeSF6

UI_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src" / "sf6_music_trigger" / "game_scripts" / "sf6_music_trigger_ui.lua"
)
STATE = "SF6MusicTrigger/app_state.json"
COMMANDS = "SF6MusicTrigger/ui_commands.jsonl"

SCRIPTED_IMGUI = r"""
local ui = { drawn = {}, inputs = {}, clicks = {}, windows = 0, ids = 0, current = nil, drawing = true }
rf_fake.ui = ui
reframework.is_drawing_ui = function(self) return ui.drawing end
local function key(label) return (ui.current and (ui.current .. "/") or "") .. label end
local function rec(kind, label) ui.drawn[#ui.drawn + 1] = kind .. ":" .. key(tostring(label)) end
local function take(label, default)
    local k = key(label)
    local value = ui.inputs[k]
    if value ~= nil then ui.inputs[k] = nil; return true, value end
    return false, default
end
imgui = {
    set_next_window_size = function() end,
    begin_window = function(name) ui.windows = ui.windows + 1; rec("window", name); return true end,
    end_window = function() ui.windows = ui.windows - 1 end,
    push_id = function(id) ui.ids = ui.ids + 1; ui.current = id end,
    pop_id = function() ui.ids = ui.ids - 1; ui.current = nil end,
    text = function(t) rec("text", t) end,
    separator = function() end,
    same_line = function() end,
    button = function(label)
        rec("button", label)
        local k = key(label)
        if ui.clicks[k] then ui.clicks[k] = nil; return true end
        return false
    end,
    checkbox = function(label, v) rec("checkbox", label); return take(label, v) end,
    slider_int = function(label, v, lo, hi) rec("slider", label); ui.max = hi; return take(label, v) end,
    combo = function(label, idx, values) rec("combo", label); return take(label, idx) end,
    tree_node = function() return false end,
    tree_pop = function() end,
}
log = { error = function(msg) ui.error = msg end }
"""


def side_state(**overrides):
    base = {"enabled": True, "main_track": "a.mp3", "volume": 60, "scope": "ALL_SUPERS",
            "whiff_mode": "sound", "whiff_track": "w.wav", "whiff_volume": 100}
    base.update(overrides)
    return base


def app_state(**overrides):
    state = {"v": 1, "session": "x", "updated": time.time(), "listening": True, "status": "监听中",
             "volume_max": 100, "volume_unlimited": False,
             "sides": {"p1": side_state(), "p2": side_state(enabled=False, whiff_mode="fade")},
             "tracks": ["a.mp3", "b.mp3", "w.wav"], "templates": ["模板 1", "模板 2"]}
    state.update(overrides)
    return state


@pytest.fixture
def game():
    g = FakeSF6(UI_SCRIPT)
    g.lua.execute(SCRIPTED_IMGUI)
    return g


def ui(game):
    return game.fake.ui


def drawn(game):
    items = ui(game).drawn
    return [items[i] for i in range(1, len(items) + 1)]


def commands(game):
    return [{k: v for k, v in c.items() if k not in {"v", "type", "session", "seq", "t"}}
            for c in game.lines(COMMANDS)]


def test_without_the_player_the_window_explains_what_to_do(game):
    game.load()
    game.frames(5)
    assert any("未连接 V2 播放器" in item for item in drawn(game))
    assert not any(item.startswith("slider:") for item in drawn(game))
    assert ui(game).windows == 0 and ui(game).ids == 0


def test_window_only_draws_while_the_menu_is_open(game):
    game.set_saved_json(STATE, app_state())
    game.load()
    ui(game).drawing = False
    game.frames(5)
    assert drawn(game) == []


def test_slider_drag_is_throttled_and_last_value_is_sent(game):
    game.set_saved_json(STATE, app_state())
    game.load()
    game.frames(2)
    for value in (61, 62, 63, 64, 65, 66):
        ui(game).inputs["p1/音量"] = value
        game.frames(1)
    game.frames(30)
    sent = [c for c in commands(game) if c.get("key") == "volume"]
    assert 1 <= len(sent) < 6
    assert sent[-1] == {"cmd": "set", "side": "p1", "key": "volume", "value": 66}
    assert ui(game).max == 100


def test_buttons_checkboxes_and_choices_send_commands(game):
    game.set_saved_json(STATE, app_state())
    game.load()
    game.frames(2)
    ui(game).clicks["停止监听"] = True
    ui(game).clicks["重置音量"] = True
    ui(game).clicks["p1/试听##whiff"] = True
    ui(game).inputs["解除音量上限（最高 500%）"] = True
    ui(game).inputs["p2/启用 P2"] = True
    ui(game).inputs["p1/主音乐"] = 3                 # options: (未选择), a.mp3, b.mp3, w.wav
    ui(game).inputs["p1/大招范围"] = 2
    ui(game).inputs["音频模板"] = 2
    game.frames(1)
    ui(game).clicks["应用模板"] = True
    game.frames(2)
    sent = commands(game)
    assert {"cmd": "stop"} in sent
    assert {"cmd": "reset_volume"} in sent
    assert {"cmd": "preview", "side": "p1", "value": "whiff"} in sent
    assert {"cmd": "set_option", "key": "volume_unlimited", "value": True} in sent
    assert {"cmd": "set", "side": "p2", "key": "enabled", "value": True} in sent
    assert {"cmd": "set", "side": "p1", "key": "main_track", "value": "b.mp3"} in sent
    assert {"cmd": "set", "side": "p1", "key": "scope", "value": "SA3_CA_ONLY"} in sent
    assert {"cmd": "apply_template", "value": "模板 2"} in sent
    assert ui(game).max == 500                         # unlimited applied locally right away


def test_fade_mode_hides_whiff_track_controls(game):
    game.set_saved_json(STATE, app_state())
    game.load()
    game.frames(1)
    items = drawn(game)
    assert "combo:p1/打空音效" in items and "combo:p2/打空音效" not in items


def test_stale_player_state_counts_as_disconnected(game):
    game.set_saved_json(STATE, app_state(updated=time.time() - 60))
    game.load()
    game.frames(2)
    assert any("未连接 V2 播放器" in item for item in drawn(game))


def test_widget_errors_keep_imgui_balanced(game):
    game.set_saved_json(STATE, app_state())
    game.lua.execute("imgui.slider_int = function() error('boom') end")
    game.load()
    game.frames(3)
    assert ui(game).windows == 0 and ui(game).ids == 0
    assert "boom" in ui(game).error
