-- SF6 Music Trigger V2 - in-game super detector (REFramework, read-only)
--
-- Detects SA1/SA2/SA3/CA for both sides from game state and writes versioned
-- JSON-lines events for the external music player. It plays no audio, never
-- writes game objects and never sends input.
--
-- Events: reframework/data/SF6MusicTrigger/events.jsonl (truncated at load)
--   hello       {script, game}
--   heartbeat   {battle, online, chara=[p1,p2], supers=[n1,n2]}         1 Hz
--               online=true: an online match is running; detection is off
--   chara       {side, chara, supers}
--   windows     {side, chara, acts, timed=[[act, last_active_frame, projectile]...]}
--   super_start {side, chara, level, label, act, episode}
--   super_result{side, chara, level, label, act, episode, result, reason, at}
--               first decisive result, as early as the game state allows:
--               hit/block on the opponent's first reaction, whiff once the
--               attack frames of the current super action have passed.
--   super_end   {side, chara, level, label, act, episode, result, reason, duration}
--               result: hit | block | whiff | unknown (final evidence)
-- Every line also carries v (protocol), session, seq and t (os.clock seconds).

local SCRIPT_VERSION = "v2-detector-0.4"
local PROTOCOL = 1
local DATA_DIR = "SF6MusicTrigger"
local EVENTS_PATH = DATA_DIR .. "/events.jsonl"
local SETTINGS_PATH = DATA_DIR .. "/detector_settings.json"
local SESSION = os.date("%Y%m%d-%H%M%S") .. "-" .. tostring(math.random(1000, 9999))

local MAX_EVENT_BYTES = 2 * 1024 * 1024  -- rotate (truncate) beyond this
local HEARTBEAT_S = 1.0
local EPISODE_TIMEOUT_S = 15.0
local SUPER_TABLE_RETRY_S = 1.0
local ERROR_REPEAT_S = 5.0
-- A whiff is declared this many frames after the last attack frame of the
-- current super action, and only if no contact shows up during a short grace
-- (the opponent's reaction can land a frame or two after the attacker's).
local WHIFF_MARGIN_FRAMES = 3
local WHIFF_GRACE_S = 0.1
-- An opponent reaction only counts as contact with the super when it starts
-- around the super's attack frames (the reaction can lag a few frames, and the
-- whiff grace above must still be able to catch it).
local CONTACT_EARLY_FRAMES = 4
local CONTACT_LATE_FRAMES = 12

-- BCM.TRIGGER category_flags (MMDK tables.lua; verified on 8 characters).
local CAT_SUPER = 0x8000
local CAT_LV = { [1] = 0x80, [2] = 0x100, [3] = 0x200, [4] = 0x400 }
local LEVEL_LABEL = { [1] = "SA1", [2] = "SA2", [3] = "SA3", [4] = "CA" }

-- Every character's super actions and their follow-up actions live in this
-- action-id block (MMDK data for 24 characters; Zangief starts at 1195).
local SUPER_BLOCK_MIN, SUPER_BLOCK_MAX = 1190, 1300

local settings = { enabled = true, overlay = false }
do
    local ok, saved = pcall(json.load_file, SETTINGS_PATH)
    if ok and type(saved) == "table" then
        if type(saved.enabled) == "boolean" then settings.enabled = saved.enabled end
        if type(saved.overlay) == "boolean" then settings.overlay = saved.overlay end
    end
end

------------------------------------------------------------------------------
-- Event writer
------------------------------------------------------------------------------
local writer = { file = nil, bytes = 0, dirty = false, seq = 0, failed = false }
local recent = {}

local function writer_open()
    local ok, f = pcall(io.open, EVENTS_PATH, "w")
    if ok and f then
        writer.file, writer.bytes, writer.failed = f, 0, false
        return true
    end
    writer.failed = true
    return false
end

local emit

local function writer_rotate()
    if writer.file then pcall(writer.file.close, writer.file) end
    writer.file = nil
    if writer_open() then
        local game_ok, game = pcall(function() return reframework:get_game_name() end)
        emit({ type = "hello", script = SCRIPT_VERSION, game = game_ok and game or "?", rotated = true })
    end
end

emit = function(rec)
    writer.seq = writer.seq + 1
    rec.v = PROTOCOL
    rec.session = SESSION
    rec.seq = writer.seq
    rec.t = os.clock()
    local line = json.dump_string(rec, -1) .. "\n"
    if not writer.file and not writer.failed then writer_open() end
    if writer.file then
        writer.file:write(line)
        writer.bytes = writer.bytes + #line
        writer.dirty = true
    end
    if rec.type == "super_start" or rec.type == "super_result" or rec.type == "super_end" or rec.type == "chara" then
        table.insert(recent, 1, string.format("%8.2f %s P%s %s %s", rec.t, rec.type,
            tostring(rec.side), tostring(rec.label or rec.chara), tostring(rec.result or "")))
        while #recent > 10 do table.remove(recent) end
    end
    if writer.bytes > MAX_EVENT_BYTES and rec.type ~= "hello" then writer_rotate() end
end

local function writer_flush()
    if writer.file and writer.dirty then
        writer.file:flush()
        writer.dirty = false
    end
end

------------------------------------------------------------------------------
-- Game access
------------------------------------------------------------------------------
local td_gBattle = sdk.find_type_definition("gBattle")

local function gbattle(name)
    local f = td_gBattle and td_gBattle:get_field(name)
    return f and f:get_data(nil)
end

local function toint(v)
    if type(v) == "number" then return math.tointeger(v) or math.floor(v) end
    if type(v) == "userdata" then
        local ok, f = pcall(function() return v:ToFloat() end)  -- via.sfix
        if ok and type(f) == "number" then return math.floor(f) end
    end
    return nil
end

-- Same traversal MMDK uses for List<T>/arrays (functions.lua lua_get_array).
local function lua_get_array(src)
    if not src then return {} end
    src = src._items or src.mItems or src
    local out = {}
    if src.get_Count then
        for i = 1, src:call("get_Count") do out[i] = src:get_Item(i - 1) end
    elseif src.get_elements then
        out = src:get_elements()
    end
    return out
end

local function key_field(key, ...)
    for _, name in ipairs({ ... }) do
        local ok, v = pcall(function() return key[name] end)
        if ok and v ~= nil then return toint(v) end
    end
    return nil
end

-- First and last attack frame (strike/throw/projectile collision with attack
-- data) of one FAB action, and whether it spawns a projectile (ShotKey).
local function attack_window(action)
    local first, last, projectile = nil, nil, false
    for _, keys_list in pairs(lua_get_array(action.Keys)) do
        local items = keys_list and (keys_list._items or keys_list)
        local head = items and items[0]
        if head then
            local kind = head:get_type_definition():get_name()
            if kind == "ShotKey" then
                projectile = true
            elseif kind == "AttackCollisionKey" or kind == "GimmickCollisionKey" or kind == "OtherCollisionKey" then
                for _, key in pairs(lua_get_array(keys_list)) do
                    if key then
                        local data = key_field(key, "AttackDataListIndex") or -1
                        local ctype = key_field(key, "CollisionType", "Kind")
                        local start = key_field(key, "_StartFrame", "StartFrame")
                        local stop = key_field(key, "_EndFrame", "EndFrame")
                        if data > -1 and ctype ~= 3 and stop then
                            if last == nil or stop > last then last = stop end
                            if start and (first == nil or start < first) then first = start end
                        end
                    end
                end
            end
        end
    end
    return first, last, projectile
end

-- Attack windows for every action in the super block of one player.
local function read_attack_windows(pl)
    local fab = gbattle("Resource").Data[pl].FAB
    local styles = fab.StyleDict
    local windows, count = {}, 0
    for si = 0, styles:call("get_Count()") - 1 do
        local style = styles[si]
        local entries = style and style.ActionList and style.ActionList._entries
        if entries then
            for _, entry in pairs(entries) do
                local action = entry and entry.value
                local act = action and toint(action.ActionID)
                if act and act >= SUPER_BLOCK_MIN and act < SUPER_BLOCK_MAX and windows[act] == nil then
                    local ok, first, last, projectile = pcall(attack_window, action)
                    if ok then
                        windows[act] = { first = first, last = last, projectile = projectile }
                        count = count + 1
                    end
                end
            end
        end
    end
    return windows, count
end

local function action_frame(eng)
    local ok, f = pcall(function() return eng.mParam.frame:ToFloat() end)
    if ok and type(f) == "number" then return f end
    ok, f = pcall(function() return eng:get_ActionFrame():ToFloat() end)
    if ok and type(f) == "number" then return f end
    return nil
end

-- Classify an action by the common action-id blocks shared by every
-- character (motion names in MMDK dumps for 24 characters):
--   1-99 basic movement, 100-191 guard (GRD/DRD/DGD), 199-299 hit reactions
--   (DMG/WALL/KNOCK...), 320-323 thrown / guard crash, 330-356 knockdown,
--   tech, stun, KO, 400-479 intro/win/lose demos, 480-489 drive parry.
--   Everything else is character-specific (attacks, specials, supers).
-- "down" is the aftermath of an earlier hit (falling after a dropped combo),
-- never proof that the current super connected.
local function act_class(act)
    if act == nil then return nil end
    if act >= 1 and act <= 99 then return "basic" end
    if (act >= 100 and act <= 191) or (act >= 480 and act <= 489) then return "guard" end
    if (act >= 199 and act <= 299) or (act >= 320 and act <= 323) then return "damage" end
    if act >= 330 and act <= 356 then return "down" end
    if act >= 400 and act <= 479 then return "demo" end
    return "other"
end

local function read_super_table(gCommand, pl)
    local triggers = gCommand:get_mUserEngine()[pl]:call("GetTrigger()")
    local supers, count = {}, 0
    local n = triggers:get_size()
    for i = 0, n - 1 do
        local trig = triggers:get_element(i)
        if trig then
            local flags = toint(trig.category_flags) or 0
            if (flags & CAT_SUPER) ~= 0 then
                local level = 0
                for lv = 4, 1, -1 do
                    if (flags & CAT_LV[lv]) ~= 0 then level = lv; break end
                end
                local act = toint(trig.action_id)
                if act and level > 0 and (not supers[act] or level > supers[act]) then
                    if not supers[act] then count = count + 1 end
                    supers[act] = level
                end
            end
        end
    end
    return supers, count
end

------------------------------------------------------------------------------
-- Super episodes
------------------------------------------------------------------------------
local function new_side()
    return { chara = nil, act = nil, act_since = 0, frame = nil, prev_act = nil, prev_frame = nil,
             supers = {}, count = 0,
             retry_at = 0, windows = nil, windows_retry_at = 0, episode = nil }
end
local sides = { [1] = new_side(), [2] = new_side() }
local episode_counter = 0
local last_heartbeat = -HEARTBEAT_S
local in_battle = false
local online = false

-- gBattle.Setting.IsOnline (app.BattleSetting, seen in the 2026-10-19 schema
-- dump). V2 is for offline play only: nothing is detected or reported while
-- an online match runs.
local function online_match()
    local ok, value = pcall(function()
        local setting = gbattle("Setting")
        return setting and setting.IsOnline
    end)
    return ok and value == true
end
local last_error = { msg = nil, at = -ERROR_REPEAT_S, suppressed = 0 }

local function in_super_block(s, act)
    return act ~= nil and (s.supers[act] ~= nil or (act >= SUPER_BLOCK_MIN and act < SUPER_BLOCK_MAX))
end

local function episode_fields(i, s, ep)
    return { side = i, chara = s.chara, level = ep.level, label = LEVEL_LABEL[ep.level],
             act = ep.act, episode = ep.id }
end

local function start_episode(i, s, act, level, now)
    episode_counter = episode_counter + 1
    s.episode = { id = episode_counter, level = level, act = act, started = now,
                  hit = false, guard = false, other = false, decided = nil, whiff_at = nil }
    local rec = episode_fields(i, s, s.episode)
    rec.type = "super_start"
    emit(rec)
end

local function decide(i, s, now, result, reason)
    local ep = s.episode
    if not ep or ep.decided then return end
    ep.decided = result
    local rec = episode_fields(i, s, ep)
    rec.type, rec.result, rec.reason, rec.at = "super_result", result, reason, now - ep.started
    emit(rec)
end

local function end_episode(i, s, now, reason)
    local ep = s.episode
    if not ep then return end
    local result
    if reason == "interrupted" or reason == "reset" or reason == "preempted" then
        result = ep.hit and "hit" or "unknown"
    elseif ep.hit then result = "hit"
    elseif ep.guard then result = "block"
    elseif ep.other then result = "unknown"
    else result = "whiff" end
    if not ep.decided then decide(i, s, now, result, reason) end
    local rec = episode_fields(i, s, ep)
    rec.type, rec.result, rec.reason, rec.duration = "super_end", result, reason, now - ep.started
    emit(rec)
    s.episode = nil
end

local function observe_own(i, s, act, now)
    local level = s.supers[act]
    local ep = s.episode
    if level then
        if not ep then
            start_episode(i, s, act, level, now)
        elseif level > ep.level then
            end_episode(i, s, now, "preempted")
            start_episode(i, s, act, level, now)
        end
        return
    end
    if not ep then return end
    local own_class = act_class(act)
    if own_class == "damage" or own_class == "down" then
        end_episode(i, s, now, "interrupted")
    elseif not in_super_block(s, act) then
        end_episode(i, s, now, "recovered")
    end
end

-- Is the attacker around the attack frames of its current action? Without
-- frame data (projectiles, unknown actions) any moment counts.
local function in_contact_window(s)
    local w = s.windows and s.windows[s.act]
    if not w or w.projectile or not w.first or not w.last or not s.frame then return true end
    return s.frame >= w.first - CONTACT_EARLY_FRAMES and s.frame <= w.last + CONTACT_LATE_FRAMES
end

-- Only reactions that START during the super count. The opponent's state when
-- the super begins (still in hitstun from a dropped combo, falling, blocking a
-- previous string) says nothing about whether the super itself connects.
local function observe_opponent(i, s, opp, now)
    local ep = s.episode
    local opp_act = opp.act
    if not ep or opp_act == nil or now <= ep.started then return end
    local class = act_class(opp_act)
    if class == "other" and not opp.episode and opp_act >= SUPER_BLOCK_MIN and opp_act < SUPER_BLOCK_MAX
        and not opp.supers[opp_act] then
        -- Opponent locked into this super's cinematic (grab supers, hit
        -- cinematics): the victim plays an action from the attacker's block.
        ep.hit = true
        decide(i, s, now, "hit", "contact")
        return
    end
    local fresh = opp_act ~= opp.prev_act
        or (opp.frame ~= nil and opp.prev_frame ~= nil and opp.frame + 0.5 < opp.prev_frame)
    if class == "damage" and fresh and in_contact_window(s) then
        ep.hit = true
        decide(i, s, now, "hit", "contact")
    elseif class == "guard" and fresh and in_contact_window(s) then
        ep.guard = true
        decide(i, s, now, "block", "contact")
    elseif class == "other" then
        -- Opponent doing its own character-specific action: not proof of a whiff.
        ep.other = true
    end
end

-- Declare a whiff once the attack frames of the current super action are over
-- without any contact. Projectile supers wait for the end of the episode.
local function check_whiff(i, s, now)
    local ep = s.episode
    if not ep or ep.decided or ep.hit or ep.guard then return end
    if ep.whiff_at then
        if now >= ep.whiff_at then decide(i, s, now, "whiff", "attack_frames_over") end
        return
    end
    local window = s.windows and s.windows[s.act]
    if not window then return end
    if window.projectile then
        ep.projectile = true
        return
    end
    if not window.last then
        -- Moved on to an action without attack frames (recovery) after the
        -- attacking action(s), still without contact.
        if ep.attack_seen and not ep.projectile then ep.whiff_at = now + WHIFF_GRACE_S end
        return
    end
    ep.attack_seen = true
    local frame = s.frame or (now - s.act_since) * 60
    if frame > window.last + WHIFF_MARGIN_FRAMES then
        ep.whiff_at = now + WHIFF_GRACE_S
    end
end

local function reset_all(now, reason)
    for i = 1, 2 do
        if sides[i].episode then end_episode(i, sides[i], now, reason) end
        sides[i].act = nil
    end
end

local function sample(now)
    online = online_match()
    if online then
        if in_battle then reset_all(now, "reset") end
        in_battle = false
        return
    end
    local gPlayer, gRollback, gCommand = gbattle("Player"), gbattle("Rollback"), gbattle("Command")
    local types = gPlayer and gPlayer.mPlayerType
    local engine = gRollback and gRollback:GetLatestEngine()
    local act_engines = engine and engine.ActEngines
    if not types or not act_engines or not gCommand then
        if in_battle then reset_all(now, "reset") end
        in_battle = false
        return
    end
    in_battle = true

    local acts, frames = {}, {}
    for i = 1, 2 do
        local s, pl = sides[i], i - 1
        local chara = toint(types[pl].mValue)
        if chara ~= s.chara then
            if s.episode then end_episode(i, s, now, "reset") end
            local fresh = new_side()
            fresh.chara = chara
            sides[i] = fresh
            s = fresh
        end
        if s.count == 0 and now >= s.retry_at then
            s.retry_at = now + SUPER_TABLE_RETRY_S
            local ok, supers, count = pcall(read_super_table, gCommand, pl)
            if ok and count > 0 then
                s.supers, s.count = supers, count
                emit({ type = "chara", side = i, chara = chara, supers = count })
            end
        end
        if s.count > 0 and s.windows == nil and now >= s.windows_retry_at then
            s.windows_retry_at = now + SUPER_TABLE_RETRY_S
            local ok, windows, count = pcall(read_attack_windows, pl)
            if ok and count > 0 then
                s.windows = windows
                local timed = {}
                for act, w in pairs(windows) do
                    if w.last or w.projectile then
                        timed[#timed + 1] = { act, w.last or -1, w.projectile }
                    end
                end
                table.sort(timed, function(a, b) return a[1] < b[1] end)
                emit({ type = "windows", side = i, chara = chara, acts = count, timed = timed })
            end
        end
        local parent = act_engines[pl] and act_engines[pl]._Parent
        local eng = parent and parent._Engine
        acts[i] = eng and toint(eng:get_ActionID()) or nil
        frames[i] = eng and action_frame(eng) or nil
    end

    for i = 1, 2 do
        local s = sides[i]
        s.prev_act, s.prev_frame = s.act, s.frame
        s.frame = frames[i]
        if acts[i] ~= nil and acts[i] ~= s.act then
            s.act = acts[i]
            s.act_since = now
            observe_own(i, s, acts[i], now)
        end
    end
    for i = 1, 2 do
        local s = sides[i]
        observe_opponent(i, s, sides[3 - i], now)
        check_whiff(i, s, now)
        if s.episode and now - s.episode.started >= EPISODE_TIMEOUT_S then
            end_episode(i, s, now, "timeout")
        end
    end
end

------------------------------------------------------------------------------
-- REFramework callbacks
------------------------------------------------------------------------------
do
    local game_ok, game = pcall(function() return reframework:get_game_name() end)
    emit({ type = "hello", script = SCRIPT_VERSION, game = game_ok and game or "?" })
    writer_flush()
end

local function save_settings()
    pcall(json.dump_file, SETTINGS_PATH, settings, 2)
end

re.on_frame(function()
    local now = os.clock()
    if settings.enabled then
        local ok, err = pcall(sample, now)
        if not ok then
            -- Game data is not readable while loading; report each distinct
            -- error at most every few seconds instead of once per frame.
            local msg = tostring(err)
            if msg ~= last_error.msg or now - last_error.at >= ERROR_REPEAT_S then
                emit({ type = "error", where = "sample", msg = msg, suppressed = last_error.suppressed })
                last_error.msg, last_error.at, last_error.suppressed = msg, now, 0
            else
                last_error.suppressed = last_error.suppressed + 1
            end
        end
    end
    if now - last_heartbeat >= HEARTBEAT_S then
        last_heartbeat = now
        emit({ type = "heartbeat", enabled = settings.enabled, battle = in_battle, online = online,
               chara = { sides[1].chara or -1, sides[2].chara or -1 },
               supers = { sides[1].count, sides[2].count } })
    end
    writer_flush()

    if settings.overlay and imgui.begin_window("SF6 Music Trigger", nil, 0) then
        imgui.text(string.format("%s  clock %.2f  battle=%s", SCRIPT_VERSION, now, tostring(in_battle)))
        for i = 1, 2 do
            local s = sides[i]
            imgui.text(string.format("P%d chara=%s act=%s supers=%d %s", i, tostring(s.chara),
                tostring(s.act), s.count, s.episode and LEVEL_LABEL[s.episode.level] or ""))
        end
        imgui.separator()
        for _, line in ipairs(recent) do imgui.text(line) end
        imgui.end_window()
    end
end)

re.on_draw_ui(function()
    if imgui.tree_node("SF6 Music Trigger") then
        local changed_a, changed_b
        changed_a, settings.enabled = imgui.checkbox("Detect supers", settings.enabled)
        changed_b, settings.overlay = imgui.checkbox("Show overlay", settings.overlay)
        if changed_a or changed_b then save_settings() end
        imgui.text("Events: reframework/data/" .. EVENTS_PATH)
        imgui.text(writer.failed and "Event file: WRITE FAILED" or "Event file: OK")
        imgui.tree_pop()
    end
end)

re.on_script_reset(function()
    reset_all(os.clock(), "reset")
    emit({ type = "bye" })
    writer_flush()
    if writer.file then pcall(writer.file.close, writer.file) end
    writer.file = nil
end)
