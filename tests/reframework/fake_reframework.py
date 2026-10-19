"""Minimal fake REFramework + SF6 battle state for running Lua scripts offline.

Only the API surface used by the V2 Lua scripts is modelled. The fake battle
exposes the same object paths MMDK uses in the real game:

    gBattle.Player.mPlayerType[i].mValue      -> character id
    gBattle.Player.mcPlayer[i]                 -> cPlayer (integer fields)
    gBattle.Rollback:GetLatestEngine().ActEngines[i]._Parent._Engine:get_ActionID()
    gBattle.Command:get_mUserEngine()[i]:call("GetTrigger()") -> BCM triggers
"""

from __future__ import annotations

import json
from pathlib import Path

from lupa import lua54

CAT_SUPER = 0x8000
CAT_LV = {1: 0x80, 2: 0x100, 3: 0x200, 4: 0x400}

FAKE_LUA = r"""
local fake = { now = 0.0, files = {}, dumped = {}, on_frame = {}, on_draw_ui = {}, on_reset = {} }
_G.rf_fake = fake

os.clock = function() return fake.now end

io.open = function(path, mode)
    if mode and mode:sub(1, 1) == "w" then fake.files[path] = {} end
    fake.files[path] = fake.files[path] or {}
    local lines = fake.files[path]
    return {
        write = function(self, ...)
            for _, s in ipairs({...}) do lines[#lines + 1] = s end
            return self
        end,
        flush = function(self) end,
        close = function(self) end,
    }
end

local function field(name, type_name, getter, static)
    return {
        get_name = function(self) return name end,
        get_type = function(self)
            return {
                get_full_name = function() return type_name end,
                is_value_type = function() return true end,
                get_parent_type = function() return nil end,
            }
        end,
        is_static = function(self) return static or false end,
        get_data = function(self, obj) return getter(obj) end,
    }
end

local function typedef(full_name, fields)
    local td = { fields = fields }
    function td:get_full_name() return full_name end
    function td:get_fields() return self.fields end
    function td:get_field(n)
        for _, f in ipairs(self.fields) do if f:get_name() == n then return f end end
        return nil
    end
    function td:get_parent_type() return nil end
    return td
end

local function array(items, n)
    local arr = {}
    for i = 0, n - 1 do arr[i] = items[i] end
    function arr:get_size() return n end
    function arr:get_element(i) return items[i] end
    return arr
end

-- Battle state controlled from Python ---------------------------------------
local state = { chara = { [0] = 1, [1] = 10 }, act = { [0] = 1, [1] = 1 },
                triggers = { [0] = {}, [1] = {} }, ints = { [0] = {}, [1] = {} } }
fake.state = state

local cplayer_fields = {}
local cplayer_td = typedef("app.battle.cPlayer", cplayer_fields)
local function cplayer(pl)
    local obj = { pl = pl }
    function obj:get_type_definition() return cplayer_td end
    return obj
end
fake.define_int_field = function(name, type_name)
    cplayer_fields[#cplayer_fields + 1] = field(name, type_name, function(obj) return state.ints[obj.pl][name] or 0 end)
end

local players = { [0] = cplayer(0), [1] = cplayer(1) }
local gPlayer = {
    mcPlayer = players,
    move_ctr = 1,
}
gPlayer.mPlayerType = setmetatable({}, { __index = function(_, i)
    if i == 0 or i == 1 then return { mValue = state.chara[i] } end
end })

fake.state.in_battle = true

-- Action frame counter: restarts at 0 whenever the action changes.
state.frame = { [0] = 0, [1] = 0 }
state.last_act = { [0] = nil, [1] = nil }
local function sfix(value)
    return { ToFloat = function(self) return value end }
end

local function engine_for(pl)
    local e = { mParam = { frame = sfix(state.frame[pl]) } }
    function e:get_ActionID() return state.act[pl] end
    return { _Parent = { _Engine = e } }
end
local gRollback = {}
function gRollback:GetLatestEngine()
    if state.engine_error then error("Invoke threw an exception") end
    if not state.in_battle then return nil end
    return { ActEngines = { [0] = engine_for(0), [1] = engine_for(1) } }
end

local gCommand = {}
function gCommand:get_mUserEngine()
    local function user_engine(pl)
        local ue = {}
        function ue:call(name)
            assert(name == "GetTrigger()", name)
            local list = state.triggers[pl]
            local items, n = {}, 0
            for _, t in ipairs(list) do items[n] = t; n = n + 1 end
            return array(items, n)
        end
        return ue
    end
    return { [0] = user_engine(0), [1] = user_engine(1) }
end

-- FAB action data: state.fab[pl][act] = { attack = {{start, stop}, ...}, projectile = bool }
state.fab = { [0] = {}, [1] = {} }

local function managed_array(list)          -- 0-based indexing + get_elements()
    local arr = {}
    for i, v in ipairs(list) do arr[i - 1] = v end
    function arr:get_elements() return list end
    return arr
end

local function typed(kind, fields)
    fields.get_type_definition = function()
        return { get_name = function() return kind end }
    end
    return fields
end

local function fab_action(act, spec)
    local key_lists = {}
    if spec.attack and #spec.attack > 0 then
        local keys = {}
        -- A proximity key (no attack data) must be ignored.
        keys[#keys + 1] = typed("AttackCollisionKey", { AttackDataListIndex = -1, CollisionType = 3,
                                                        _StartFrame = 0, _EndFrame = 400 })
        for _, w in ipairs(spec.attack) do
            keys[#keys + 1] = typed("AttackCollisionKey", { AttackDataListIndex = 1, CollisionType = 0,
                                                            _StartFrame = w[1], _EndFrame = w[2] })
        end
        key_lists[#key_lists + 1] = { _items = managed_array(keys) }
    end
    if spec.projectile then
        key_lists[#key_lists + 1] = { _items = managed_array({ typed("ShotKey", { ActionId = 0 }) }) }
    end
    key_lists[#key_lists + 1] = { _items = managed_array({ typed("MotionKey", {}) }) }
    return { ActionID = act, Keys = { _items = managed_array(key_lists) } }
end

local function person(pl)
    local entries = {}
    for act, spec in pairs(state.fab[pl]) do entries[#entries + 1] = { value = fab_action(act, spec) } end
    entries[#entries + 1] = { value = nil }  -- unused dictionary slot
    local style = { ActionList = { _entries = entries } }
    local styles = { [0] = style }
    function styles:call(name) assert(name == "get_Count()", name); return 1 end
    return { FAB = { StyleDict = styles } }
end
local gResource = {}
gResource.Data = setmetatable({}, { __index = function(_, pl) return person(pl) end })

state.online = false
local gSetting = setmetatable({}, { __index = function(_, k)
    if k == "IsOnline" then return state.online end
end })

local gbattle_td = typedef("gBattle", {
    field("Setting", "gBattle.Setting", function() return gSetting end, true),
    field("Resource", "gBattle.Resource", function() return gResource end, true),
    field("Player", "gBattle.Player", function() return gPlayer end, true),
    field("Rollback", "gBattle.Rollback", function() return gRollback end, true),
    field("Command", "gBattle.Command", function() return gCommand end, true),
})

local act_enum_fields = {}
local act_enum_td = typedef("nBattle.ACT_ID", act_enum_fields)
fake.define_act_name = function(name, value)
    act_enum_fields[#act_enum_fields + 1] = field(name, "nBattle.ACT_ID", function() return value end, true)
end

sdk = {
    find_type_definition = function(name)
        if name == "gBattle" then return gbattle_td end
        if name == "nBattle.ACT_ID" then return act_enum_td end
        return nil
    end,
}

re = {
    on_frame = function(fn) fake.on_frame[#fake.on_frame + 1] = fn end,
    on_draw_ui = function(fn) fake.on_draw_ui[#fake.on_draw_ui + 1] = fn end,
    on_script_reset = function(fn) fake.on_reset[#fake.on_reset + 1] = fn end,
}

imgui = setmetatable({}, { __index = function(_, k)
    if k == "begin_window" or k == "tree_node" or k == "button" then return function() return false end end
    if k == "checkbox" then return function(_, v) return false, v end end
    return function() end
end })

reframework = {}
function reframework:get_game_name() return "sf6" end

json = {
    dump_string = function(v, indent) return fake.py_dump(v) end,
    dump_file = function(path, v, indent) fake.dumped[path] = fake.py_dump(v); return true end,
    load_file = function(path) return fake.saved_json and fake.saved_json[path] or nil end,
}

fake.run_frame = function()
    for pl = 0, 1 do
        if state.act[pl] ~= state.last_act[pl] then
            state.frame[pl] = 0
            state.last_act[pl] = state.act[pl]
        else
            state.frame[pl] = state.frame[pl] + 1
        end
    end
    for _, fn in ipairs(fake.on_frame) do fn() end
end
"""


def lua_to_py(value):
    if lua54.lua_type(value) == "table":
        items = list(value.items())
        keys = [k for k, _ in items]
        if keys and all(isinstance(k, int) for k in keys) and sorted(keys) == list(range(1, len(keys) + 1)):
            return [lua_to_py(value[i]) for i in range(1, len(keys) + 1)]
        return {str(k): lua_to_py(v) for k, v in items}
    return value


class FakeSF6:
    def __init__(self, script: Path):
        self.lua = lua54.LuaRuntime(unpack_returned_tuples=True)
        self.lua.execute(FAKE_LUA)
        self.fake = self.lua.globals().rf_fake
        self.fake.py_dump = lambda v: json.dumps(lua_to_py(v), ensure_ascii=False)
        self.script = script
        self.frame_dt = 1.0 / 60.0

    # -- setup -------------------------------------------------------------
    def define_act_names(self, names: dict[str, int]) -> None:
        for name, value in names.items():
            self.fake.define_act_name(name, value)

    def define_int_fields(self, names: list[str]) -> None:
        for name in names:
            self.fake.define_int_field(name, "System.Int32")

    def set_chara(self, side: int, chara: int) -> None:
        self.fake.state.chara[side - 1] = chara

    def set_triggers(self, side: int, triggers: list[dict]) -> None:
        table = self.lua.table()
        for i, trig in enumerate(triggers, start=1):
            table[i] = self.lua.table_from(trig)
        self.fake.state.triggers[side - 1] = table

    def set_act(self, side: int, act: int) -> None:
        self.fake.state.act[side - 1] = act

    def set_fab(self, side: int, actions: dict[int, dict]) -> None:
        table = self.lua.table()
        for act, spec in actions.items():
            attack = self.lua.table_from([self.lua.table_from(list(w)) for w in spec.get("attack", [])])
            table[act] = self.lua.table_from({"attack": attack, "projectile": bool(spec.get("projectile"))})
        self.fake.state.fab[side - 1] = table

    def set_online(self, value: bool) -> None:
        self.fake.state.online = value

    def set_engine_error(self, value: bool) -> None:
        self.fake.state.engine_error = value

    def set_in_battle(self, value: bool) -> None:
        self.fake.state.in_battle = value

    def set_saved_json(self, path: str, value: dict) -> None:
        saved = self.fake.saved_json or self.lua.table()
        saved[path] = self.lua.table_from(value, recursive=True)
        self.fake.saved_json = saved

    def lines(self, path: str) -> list[dict]:
        chunks = self.fake.files[path]
        if chunks is None:
            return []
        text = "".join(chunks[i] for i in range(1, len(chunks) + 1))
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def set_int(self, side: int, name: str, value: int) -> None:
        self.fake.state.ints[side - 1][name] = value

    def load(self) -> None:
        self.lua.execute(self.script.read_text(encoding="utf-8"))

    # -- running -----------------------------------------------------------
    def frames(self, count: int = 1) -> None:
        for _ in range(count):
            self.fake.now = self.fake.now + self.frame_dt
            self.fake.run_frame()

    def events(self) -> list[dict]:
        out = []
        for path, lines in self.fake.files.items():
            for i in range(1, len(lines) + 1):
                text = lines[i]
                if text != "\n":
                    out.append(json.loads(text))
        return out

    def dumped_files(self) -> dict[str, dict]:
        return {k: json.loads(v) for k, v in self.fake.dumped.items()}


def super_trigger(action_id: int, level: int, gauge: int, *, extra_flags: int = 0, vital_ratio: int = 0) -> dict:
    return {
        "action_id": action_id,
        "category_flags": CAT_SUPER | CAT_LV[level] | extra_flags,
        "gauge_consume": gauge,
        "gauge_need": gauge,
        "focus_consume": 0,
        "cond_vital_ratio": vital_ratio,
        "kind_level": level,
    }


def normal_trigger(action_id: int) -> dict:
    return {"action_id": action_id, "category_flags": 0x2000, "gauge_consume": 0,
            "gauge_need": 0, "focus_consume": 0, "cond_vital_ratio": 0, "kind_level": 0}
