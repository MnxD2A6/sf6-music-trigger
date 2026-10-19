-- SF6 Music Trigger V2 - REFramework probe (read-only)
--
-- Phase-A verification script. It never writes to game objects, never calls
-- game methods that change state, and never sends input. It only:
--   * reads each side's character id and current action id,
--   * reads each side's BCM trigger table to find super-art actions
--     (category flag IsSuper + Lv1..Lv4, works for every character),
--   * logs action changes, super starts/ends and opponent reactions,
--   * snapshots integer fields of both players around each super so the
--     super-gauge / vital fields can be identified offline.
--
-- Output: <SF6>\reframework\data\SF6MusicTrigger\probe-<session>.jsonl
--         <SF6>\reframework\data\SF6MusicTrigger\schema-<session>.json

local PROBE_VERSION = "probe-0.1"
local LOG_DIR = "SF6MusicTrigger"
local SESSION = os.date("%Y%m%d-%H%M%S")
local LOG_PATH = LOG_DIR .. "/probe-" .. SESSION .. ".jsonl"
local SCHEMA_PATH = LOG_DIR .. "/schema-" .. SESSION .. ".json"

-- BCM.TRIGGER category_flags bits (MMDK tables.lua, cross-checked against
-- MMDK trigger dumps for Ryu/Ken/Chun-Li/JP/Marisa/Terry/M.Bison/Akuma).
local CAT_SUPER = 0x8000
local CAT_LV = { [1] = 0x80, [2] = 0x100, [3] = 0x200, [4] = 0x400 }
local LEVEL_LABEL = { [1] = "SA1", [2] = "SA2", [3] = "SA3", [4] = "CA" }

local DIFF_DELAY_S = 0.75     -- field diff taken this long after a super start
local HEARTBEAT_S = 1.0
local MAX_DIFF_FIELDS = 160
local MAX_OVERLAY_EVENTS = 12

local INT_TYPES = {
    ["System.Byte"] = true, ["System.SByte"] = true,
    ["System.Int16"] = true, ["System.UInt16"] = true,
    ["System.Int32"] = true, ["System.UInt32"] = true,
    ["System.Int64"] = true, ["System.UInt64"] = true,
}
local INTERESTING_NAME = { "cpu", "com", "pad", "input", "operat", "human", "ai_", "_ai", "side", "team", "control" }

local settings = { overlay = true, snapshots = true }

------------------------------------------------------------------------------
-- Logging
------------------------------------------------------------------------------
local log_file = nil
local log_dirty = false
local seq = 0
local overlay_events = {}
local last_error_at = {}

local function open_log()
    if log_file then return true end
    local ok, f = pcall(io.open, LOG_PATH, "a")
    if ok and f then
        log_file = f
        return true
    end
    return false
end

local function emit(rec)
    seq = seq + 1
    rec.seq = seq
    rec.t = os.clock()
    if open_log() then
        log_file:write(json.dump_string(rec, -1), "\n")
        log_dirty = true
    end
end

local function overlay(text)
    table.insert(overlay_events, 1, string.format("%8.2f  %s", os.clock(), text))
    while #overlay_events > MAX_OVERLAY_EVENTS do table.remove(overlay_events) end
end

local function report_error(where, err)
    local now = os.clock()
    if last_error_at[where] and now - last_error_at[where] < 5.0 then return end
    last_error_at[where] = now
    emit({ type = "error", where = where, msg = tostring(err) })
end

local function flush()
    if log_file and log_dirty then
        log_file:flush()
        log_dirty = false
    end
end

------------------------------------------------------------------------------
-- Game access helpers
------------------------------------------------------------------------------
local td_gBattle = sdk.find_type_definition("gBattle")
local td_act_enum = sdk.find_type_definition("nBattle.ACT_ID")

local function gbattle(name)
    local f = td_gBattle and td_gBattle:get_field(name)
    return f and f:get_data(nil)
end

local function toint(v)
    if type(v) == "number" then return math.tointeger(v) or math.floor(v) end
    if type(v) == "boolean" then return v and 1 or 0 end
    return nil
end

local function iter_array(arr)
    local out = {}
    if arr == nil then return out end
    local ok = pcall(function()
        local n = arr:get_size()
        for i = 0, n - 1 do out[#out + 1] = { i, arr:get_element(i) } end
    end)
    if not ok then
        out = {}
        for i, v in pairs(arr) do out[#out + 1] = { i, v } end
    end
    return out
end

-- Common action names (GRD_*, DMG_*, BAS_* ...) from the nBattle.ACT_ID enum.
local act_names = {}
if td_act_enum then
    for _, f in ipairs(td_act_enum:get_fields()) do
        if f:is_static() then
            local ok, v = pcall(f.get_data, f, nil)
            if ok and v ~= nil then act_names[toint(v)] = f:get_name() end
        end
    end
end

-- Same common action-id blocks as sf6_music_trigger.lua; the enum name is
-- logged alongside so the blocks can be checked against real data.
local function act_class(act)
    local name = act_names[act]
    local class
    if act == nil then class = nil
    elseif act >= 1 and act <= 99 then class = "basic"
    elseif (act >= 100 and act <= 191) or (act >= 480 and act <= 489) then class = "guard"
    elseif (act >= 199 and act <= 299) or (act >= 320 and act <= 356) then class = "damage"
    elseif act >= 400 and act <= 479 then class = "demo"
    else class = "other" end
    return class, name
end

------------------------------------------------------------------------------
-- Per-side state
------------------------------------------------------------------------------
local sides = {
    [1] = { chara = nil, act = nil, supers = {}, super_count = 0, episode = nil, fields = nil, snap = nil, prev_snap = nil },
    [2] = { chara = nil, act = nil, supers = {}, super_count = 0, episode = nil, fields = nil, snap = nil, prev_snap = nil },
}
local next_super_retry = { [1] = 0, [2] = 0 }
local schema_written = false
local last_heartbeat = 0

local function build_super_table(pl, gCommand)
    local ue = gCommand:get_mUserEngine()[pl]
    local triggers = ue:call("GetTrigger()")
    local supers, rows, count = {}, {}, 0
    for _, pair in ipairs(iter_array(triggers)) do
        local trig = pair[2]
        if trig then
            local flags = toint(trig.category_flags) or 0
            if (flags & CAT_SUPER) ~= 0 then
                local level = 0
                for lv = 4, 1, -1 do
                    if (flags & CAT_LV[lv]) ~= 0 then level = lv; break end
                end
                local act = toint(trig.action_id)
                if act and level > 0 then
                    local prev = supers[act]
                    if not prev or level > prev.level then
                        if not prev then count = count + 1 end
                        supers[act] = {
                            level = level,
                            label = LEVEL_LABEL[level],
                            gauge_consume = toint(trig.gauge_consume) or 0,
                            focus_consume = toint(trig.focus_consume) or 0,
                            vital_ratio = toint(trig.cond_vital_ratio) or 0,
                        }
                    end
                    rows[#rows + 1] = {
                        trigger_index = pair[1], act = act, level = level,
                        flags = string.format("0x%X", flags),
                        gauge_consume = toint(trig.gauge_consume),
                        gauge_need = toint(trig.gauge_need),
                        focus_consume = toint(trig.focus_consume),
                        vital_ratio = toint(trig.cond_vital_ratio),
                        kind_level = toint(trig.kind_level),
                    }
                end
            end
        end
    end
    return supers, rows, count
end

local function collect_int_fields(obj)
    local out = {}
    local td = obj:get_type_definition()
    local cur = td
    while cur do
        local full = cur:get_full_name()
        if full == "System.Object" or full == "System.ValueType" then break end
        for _, f in ipairs(cur:get_fields()) do
            if not f:is_static() then
                local ft = f:get_type()
                local tn = ft and ft:get_full_name() or ""
                local is_enum = ft and ft:is_value_type() and ft:get_parent_type() ~= nil
                    and ft:get_parent_type():get_full_name() == "System.Enum"
                if INT_TYPES[tn] or tn == "System.Boolean" or is_enum then
                    local ok, v = pcall(f.get_data, f, obj)
                    if ok and toint(v) ~= nil then
                        out[#out + 1] = { name = f:get_name(), field = f, type = tn }
                    end
                end
            end
        end
        cur = cur:get_parent_type()
    end
    return out, td:get_full_name()
end

local function snapshot(fields, obj)
    local s = {}
    for i = 1, #fields do
        local e = fields[i]
        s[e.name] = toint(e.field:get_data(obj))
    end
    return s
end

local function diff_snap(a, b)
    local changes, n = {}, 0
    if not a or not b then return changes, 0 end
    for k, v in pairs(b) do
        if a[k] ~= v then
            n = n + 1
            if n <= MAX_DIFF_FIELDS then changes[k] = { a[k], v } end
        end
    end
    return changes, n
end

local function interesting_values(s)
    local out = {}
    if not s then return out end
    for k, v in pairs(s) do
        local lk = k:lower()
        for _, needle in ipairs(INTERESTING_NAME) do
            if lk:find(needle, 1, true) then out[k] = v; break end
        end
    end
    return out
end

local function write_schema(gPlayer)
    local schema = { probe = PROBE_VERSION, session = SESSION, gBattle = {}, cPlayer = {} }
    for _, f in ipairs(td_gBattle:get_fields()) do
        local entry = { name = f:get_name(), type = f:get_type() and f:get_type():get_full_name() or "?" }
        local ok, obj = pcall(f.get_data, f, nil)
        if ok and obj and type(obj) == "userdata" then
            local ok2, td = pcall(obj.get_type_definition, obj)
            if ok2 and td then
                entry.runtime_type = td:get_full_name()
                entry.fields = {}
                for _, sf in ipairs(td:get_fields()) do
                    entry.fields[#entry.fields + 1] = sf:get_name() .. " : " .. (sf:get_type() and sf:get_type():get_full_name() or "?")
                end
            end
        end
        schema.gBattle[#schema.gBattle + 1] = entry
    end
    local s1 = sides[1]
    if s1.fields then
        for _, e in ipairs(s1.fields) do schema.cPlayer[#schema.cPlayer + 1] = e.name .. " : " .. e.type end
        schema.cPlayer_type = s1.cplayer_type
    end
    json.dump_file(SCHEMA_PATH, schema, 1)
    emit({ type = "schema_written", path = SCHEMA_PATH })
end

------------------------------------------------------------------------------
-- Super episode prototype (the rules the V2 detector will refine)
------------------------------------------------------------------------------
local function start_episode(side_idx, s, act, info, opp)
    s.episode = {
        act = act, level = info.level, label = info.label, started = os.clock(),
        base_self = s.prev_snap, base_opp = opp.prev_snap, diffed = false,
        opp_class = nil,
    }
    emit({ type = "super_start", side = side_idx, chara = s.chara, act = act,
           level = info.level, label = info.label, gauge_consume = info.gauge_consume,
           focus_consume = info.focus_consume, vital_ratio = info.vital_ratio })
    overlay(string.format("P%d %s start (act %d)", side_idx, info.label, act))
end

local function end_episode(side_idx, s, opp, reason)
    local ep = s.episode
    if not ep then return end
    local self_diff, self_n = diff_snap(ep.base_self, s.snap)
    local opp_diff, opp_n = diff_snap(ep.base_opp, opp.snap)
    emit({ type = "super_end", side = side_idx, label = ep.label, reason = reason,
           duration = os.clock() - ep.started, opp_class = ep.opp_class or "none",
           self_changed = self_n, opp_changed = opp_n,
           self_diff = settings.snapshots and self_diff or nil,
           opp_diff = settings.snapshots and opp_diff or nil })
    overlay(string.format("P%d %s end: %s, opp=%s", side_idx, ep.label, reason, tostring(ep.opp_class)))
    s.episode = nil
end

local function on_act_change(side_idx, s, opp, act)
    local class, name = act_class(act)
    emit({ type = "act", side = side_idx, act = act, name = name, class = class,
           super = s.supers[act] and s.supers[act].label or nil })

    local info = s.supers[act]
    local ep = s.episode
    if info then
        if not ep then
            start_episode(side_idx, s, act, info, opp)
        elseif info.level > ep.level then
            end_episode(side_idx, s, opp, "preempted_by_higher_level")
            start_episode(side_idx, s, act, info, opp)
        else
            emit({ type = "super_continue", side = side_idx, act = act, label = info.label })
        end
        return
    end

    if ep then
        if class == "damage" then
            end_episode(side_idx, s, opp, "interrupted_by_damage")
        elseif class == "basic" then
            end_episode(side_idx, s, opp, "returned_to_basic")
        end
    end
end

local function on_opp_act_change(side_idx, s, act)
    local ep = s.episode
    if not ep then return end
    local class, name = act_class(act)
    if class == "guard" or class == "guard_like" or class == "damage" then
        if ep.opp_class ~= class then
            local first = ep.opp_class == nil
            ep.opp_class = class
            emit({ type = "opp_reaction", side = side_idx, label = ep.label, opp_act = act,
                   opp_name = name, class = class, first = first,
                   since_start = os.clock() - ep.started })
            overlay(string.format("P%d %s -> opp %s (%s)", side_idx, ep.label, class, tostring(name)))
        end
    end
end

------------------------------------------------------------------------------
-- Per-frame sampling
------------------------------------------------------------------------------
local function sample()
    local gPlayer = gbattle("Player")
    local gRollback = gbattle("Rollback")
    local gCommand = gbattle("Command")
    if not gPlayer or not gRollback or not gCommand then return end

    local types = gPlayer.mPlayerType
    local engine = gRollback:GetLatestEngine()
    local act_engines = engine and engine.ActEngines
    if not types or not act_engines then return end

    local now = os.clock()
    local acts = {}
    for i = 1, 2 do
        local s = sides[i]
        local pl = i - 1
        local chara = toint(types[pl].mValue)
        if chara ~= s.chara then
            if s.episode then end_episode(i, s, sides[3 - i], "character_changed") end
            s.chara, s.supers, s.super_count, s.fields, s.snap, s.prev_snap = chara, {}, 0, nil, nil, nil
            next_super_retry[i] = 0
            emit({ type = "chara", side = i, chara = chara })
            overlay(string.format("P%d chara %s", i, tostring(chara)))
        end

        if s.super_count == 0 and now >= next_super_retry[i] then
            next_super_retry[i] = now + 1.0
            local ok, supers, rows, count = pcall(build_super_table, pl, gCommand)
            if ok and count > 0 then
                s.supers, s.super_count = supers, count
                emit({ type = "super_table", side = i, chara = chara, count = count, rows = rows })
                overlay(string.format("P%d super table: %d actions", i, count))
            elseif not ok then
                report_error("super_table_p" .. i, supers)
            end
        end

        local parent = act_engines[pl] and act_engines[pl]._Parent
        local eng = parent and parent._Engine
        acts[i] = eng and toint(eng:get_ActionID()) or nil

        -- Snapshot failures must never block action detection.
        local cp = gPlayer.mcPlayer[pl]
        if cp and settings.snapshots and not s.snapshots_broken then
            local ok, err = pcall(function()
                if not s.fields then
                    s.fields, s.cplayer_type = collect_int_fields(cp)
                    emit({ type = "fields", side = i, cplayer_type = s.cplayer_type, count = #s.fields })
                end
                s.prev_snap = s.snap
                s.snap = snapshot(s.fields, cp)
            end)
            if not ok then
                s.snapshots_broken = true
                report_error("snapshot_p" .. i, err)
            end
        end
    end

    if not schema_written and sides[1].fields then
        schema_written = true
        local ok, err = pcall(write_schema, gPlayer)
        if not ok then report_error("schema", err) end
    end

    for i = 1, 2 do
        local s, opp = sides[i], sides[3 - i]
        local act = acts[i]
        if act ~= nil and act ~= s.act then
            s.act = act
            on_act_change(i, s, opp, act)
        end
    end
    -- Opponent reactions are evaluated after both sides' own changes.
    for i = 1, 2 do
        local s, opp = sides[i], sides[3 - i]
        if acts[3 - i] ~= nil and acts[3 - i] ~= opp.last_seen_by_opp then
            opp.last_seen_by_opp = acts[3 - i]
            on_opp_act_change(i, s, acts[3 - i])
        end
        local ep = s.episode
        if ep and not ep.diffed and now - ep.started >= DIFF_DELAY_S then
            ep.diffed = true
            local self_diff, self_n = diff_snap(ep.base_self, s.snap)
            local opp_diff, opp_n = diff_snap(ep.base_opp, opp.snap)
            emit({ type = "super_diff", side = i, label = ep.label, delay = DIFF_DELAY_S,
                   self_changed = self_n, opp_changed = opp_n,
                   self_diff = self_diff, opp_diff = opp_diff })
        end
    end

    if now - last_heartbeat >= HEARTBEAT_S then
        last_heartbeat = now
        emit({ type = "heartbeat", move_ctr = toint(gPlayer.move_ctr),
               chara = { sides[1].chara, sides[2].chara }, act = { acts[1], acts[2] },
               supers = { sides[1].super_count, sides[2].super_count },
               p1_interesting = interesting_values(sides[1].snap),
               p2_interesting = interesting_values(sides[2].snap) })
    end
end

------------------------------------------------------------------------------
-- REFramework callbacks
------------------------------------------------------------------------------
local game_ok, game_name = pcall(function() return reframework:get_game_name() end)
emit({ type = "session", probe = PROBE_VERSION, session = SESSION,
       game = game_ok and game_name or "?",
       act_enum_count = (function() local n = 0 for _ in pairs(act_names) do n = n + 1 end return n end)() })
flush()

re.on_frame(function()
    local ok, err = pcall(sample)
    if not ok then report_error("sample", err) end
    flush()

    if settings.overlay then
        if imgui.begin_window("SF6 Music Trigger Probe", nil, 0) then
            imgui.text(PROBE_VERSION .. "  log: reframework/data/" .. LOG_PATH)
            -- Visible in screen recordings so video time can be aligned with log "t".
            imgui.text(string.format("clock %.2f", os.clock()))
            for i = 1, 2 do
                local s = sides[i]
                local ep = s.episode
                imgui.text(string.format("P%d  chara=%s  act=%s  supers=%d  %s", i,
                    tostring(s.chara), tostring(s.act), s.super_count,
                    ep and (ep.label .. " active") or ""))
            end
            imgui.separator()
            for _, line in ipairs(overlay_events) do imgui.text(line) end
            imgui.end_window()
        end
    end
end)

re.on_draw_ui(function()
    if imgui.tree_node("SF6 Music Trigger Probe") then
        local changed
        changed, settings.overlay = imgui.checkbox("Show overlay", settings.overlay)
        changed, settings.snapshots = imgui.checkbox("Field snapshots (gauge discovery)", settings.snapshots)
        if imgui.button("Mark: test note") then
            emit({ type = "mark" })
            overlay("MARK")
        end
        imgui.text("Log: reframework/data/" .. LOG_PATH)
        imgui.tree_pop()
    end
end)

re.on_script_reset(function()
    if log_file then
        emit({ type = "session_end" })
        log_file:flush()
        log_file:close()
        log_file = nil
    end
end)
