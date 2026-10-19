-- SF6 Music Trigger V2 - in-game settings window (REFramework)
--
-- Shown while the REFramework menu is open (menu key, Insert by default). Lets the player
-- change the music settings without leaving a fullscreen game. The external
-- V2 player still plays the audio; this script only exchanges two files with
-- it inside reframework/data/SF6MusicTrigger:
--   app_state.json     written by the player: settings, track list, status
--   ui_commands.jsonl  written here: one JSON command per line
--     {v=1, type="ui", session, seq, t, cmd, side?, key?, value?}
--     cmd: set | set_option | reset_volume | apply_template | start | stop | preview

local UI_VERSION = "v2-ui-0.1"
local DATA_DIR = "SF6MusicTrigger"
local STATE_PATH = DATA_DIR .. "/app_state.json"
local COMMANDS_PATH = DATA_DIR .. "/ui_commands.jsonl"
local SETTINGS_PATH = DATA_DIR .. "/ui_settings.json"
local SESSION = os.date("%Y%m%d-%H%M%S") .. "-" .. tostring(math.random(1000, 9999))

local STATE_REFRESH_S = 0.5   -- re-read app_state.json this often (os.clock)
local STALE_S = 4             -- player considered gone after this (wall seconds)
local SLIDER_SEND_S = 0.15    -- slider drags are sent at most this often
local OVERRIDE_S = 2.0        -- show the local value until the player confirms
local COND_FIRST_USE = 4      -- ImGuiCond_FirstUseEver

local SCOPES = {
    { "ALL_SUPERS", "全部大招（SA1/SA2 播 4 秒后淡出）" },
    { "SA3_CA_ONLY", "仅 SA3/CA" },
}
local WHIFF_MODES = {
    { "sound", "播放打空音效" },
    { "fade", "淡出主音乐" },
}
local SIDES = { { "p1", "P1" }, { "p2", "P2" } }

local ui = { show_window = true }
do
    local ok, saved = pcall(json.load_file, SETTINGS_PATH)
    if ok and type(saved) == "table" and type(saved.show_window) == "boolean" then
        ui.show_window = saved.show_window
    end
end

------------------------------------------------------------------------------
-- Player state and commands
------------------------------------------------------------------------------
local state = nil
local state_read_at = -STATE_REFRESH_S
local writer = { file = nil, seq = 0, failed = false }
local pending = {}     -- key -> command waiting for the slider throttle
local last_sent = {}   -- key -> os.clock() of the last send
local overrides = {}   -- key -> { value, until }

local function send(rec)
    if not writer.file and not writer.failed then
        local ok, f = pcall(io.open, COMMANDS_PATH, "w")
        if ok and f then writer.file = f else writer.failed = true end
    end
    if not writer.file then return end
    writer.seq = writer.seq + 1
    rec.v, rec.type, rec.session, rec.seq, rec.t = 1, "ui", SESSION, writer.seq, os.clock()
    writer.file:write(json.dump_string(rec, -1), "\n")
    writer.file:flush()
end

local function refresh_state(now)
    if now - state_read_at < STATE_REFRESH_S then return end
    state_read_at = now
    local ok, loaded = pcall(json.load_file, STATE_PATH)
    if ok and type(loaded) == "table" and loaded.v == 1 then state = loaded end
end

local function player_alive()
    return state ~= nil and type(state.updated) == "number" and os.time() - state.updated <= STALE_S
end

local function flush_pending(now)
    for key, rec in pairs(pending) do
        if now - (last_sent[key] or -1) >= SLIDER_SEND_S then
            send(rec)
            last_sent[key] = now
            pending[key] = nil
        end
    end
end

local function side_value(side, key)
    local o = overrides[side .. "." .. key]
    if o and os.clock() < o.until_t then return o.value end
    local s = state and state.sides and state.sides[side]
    return s and s[key]
end

local function set_side(side, key, value, throttled)
    local id = side .. "." .. key
    overrides[id] = { value = value, until_t = os.clock() + OVERRIDE_S }
    local rec = { cmd = "set", side = side, key = key, value = value }
    if throttled then pending[id] = rec else send(rec) end
end

------------------------------------------------------------------------------
-- Widgets
------------------------------------------------------------------------------
local function choice(label, current, options)
    local labels, index = {}, 1
    for i, opt in ipairs(options) do
        labels[i] = opt[2]
        if opt[1] == current then index = i end
    end
    local changed, picked = imgui.combo(label, index, labels)
    if changed and options[picked] then return true, options[picked][1] end
    return false, current
end

local function track_choice(label, current)
    local options = { { "", "（未选择）" } }
    for _, name in ipairs(state.tracks or {}) do options[#options + 1] = { name, name } end
    return choice(label, current or "", options)
end

local function draw_side_body(side, title)
    imgui.separator()
    local changed, value
    changed, value = imgui.checkbox("启用 " .. title, side_value(side, "enabled") == true)
    if changed then set_side(side, "enabled", value) end

    changed, value = track_choice("主音乐", side_value(side, "main_track"))
    if changed and value ~= "" then set_side(side, "main_track", value) end
    imgui.same_line()
    if imgui.button("试听##main") then send({ cmd = "preview", side = side, value = "main" }) end

    local max = state.volume_max or 100
    changed, value = imgui.slider_int("音量", math.floor(side_value(side, "volume") or 0), 0, max, "%d%%")
    if changed then set_side(side, "volume", value, true) end

    changed, value = choice("大招范围", side_value(side, "scope"), SCOPES)
    if changed then set_side(side, "scope", value) end

    changed, value = choice("空大/被防时", side_value(side, "whiff_mode"), WHIFF_MODES)
    if changed then set_side(side, "whiff_mode", value) end

    if side_value(side, "whiff_mode") == "sound" then
        changed, value = track_choice("打空音效", side_value(side, "whiff_track"))
        if changed and value ~= "" then set_side(side, "whiff_track", value) end
        imgui.same_line()
        if imgui.button("试听##whiff") then send({ cmd = "preview", side = side, value = "whiff" }) end
        changed, value = imgui.slider_int("音效音量", math.floor(side_value(side, "whiff_volume") or 0), 0, max, "%d%%")
        if changed then set_side(side, "whiff_volume", value, true) end
    end
end

-- ImGui needs every push/begin matched even when a widget call fails.
local function draw_side(side, title)
    imgui.push_id(side)
    local ok, err = pcall(draw_side_body, side, title)
    imgui.pop_id()
    if not ok then error(err, 0) end
end

local template_index = 1

local function draw_body()
    if not player_alive() then
        imgui.text("未连接 V2 播放器。")
        imgui.text("请先运行 run_v2.bat；在面板里勾选“启动后自动开始监听”后，")
        imgui.text("以后打开播放器就会自动在后台工作。")
        return
    end

    imgui.text(state.status or "")
    if state.listening then
        if imgui.button("停止监听") then send({ cmd = "stop" }) end
    else
        if imgui.button("开始监听") then send({ cmd = "start" }) end
    end

    local templates = state.templates or {}
    if #templates > 0 then
        if template_index > #templates then template_index = 1 end
        local changed, picked = imgui.combo("音频模板", template_index, templates)
        if changed then template_index = picked end
        imgui.same_line()
        if imgui.button("应用模板") then send({ cmd = "apply_template", value = templates[template_index] }) end
    end

    local changed, unlimited = imgui.checkbox("解除音量上限（最高 500%）", state.volume_unlimited == true)
    if changed then
        state.volume_unlimited = unlimited
        state.volume_max = unlimited and 500 or 100
        send({ cmd = "set_option", key = "volume_unlimited", value = unlimited })
    end
    imgui.same_line()
    if imgui.button("重置音量") then
        overrides = {}
        send({ cmd = "reset_volume" })
    end

    for _, side in ipairs(SIDES) do draw_side(side[1], side[2]) end
    imgui.separator()
    imgui.text("关闭 REFramework 菜单后，这个窗口会一起隐藏。")
end

local function draw_window()
    imgui.set_next_window_size({ 460, 560 }, COND_FIRST_USE)
    imgui.begin_window("SF6 音乐设置", nil, 0)
    local ok, err = pcall(draw_body)
    imgui.end_window()
    if not ok then error(err, 0) end
end

------------------------------------------------------------------------------
-- REFramework callbacks
------------------------------------------------------------------------------
re.on_frame(function()
    local now = os.clock()
    flush_pending(now)
    if not ui.show_window or not reframework:is_drawing_ui() then return end
    refresh_state(now)
    local ok, err = pcall(draw_window)
    if not ok then
        pcall(log.error, "[SF6 Music Trigger UI] " .. tostring(err))
    end
end)

re.on_draw_ui(function()
    if imgui.tree_node("SF6 音乐设置") then
        local changed
        changed, ui.show_window = imgui.checkbox("打开菜单时显示音乐设置窗口", ui.show_window)
        if changed then pcall(json.dump_file, SETTINGS_PATH, ui, 2) end
        imgui.text(UI_VERSION)
        imgui.tree_pop()
    end
end)

re.on_script_reset(function()
    if writer.file then pcall(writer.file.close, writer.file) end
    writer.file = nil
end)
