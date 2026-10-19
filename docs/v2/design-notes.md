# V2 设计笔记：用 REFramework 替代画面识别

## 目标

- 做一个全角色通用的 SA1/SA2/SA3/CA 音乐触发器，不依赖 HUD 像素，所以换 UI 也不会失效。
- 只在离线模式使用：训练模式、CPU 对战、本地对战。
- 继续沿用 T03 的 GUI、音频播放（`audio.py`）、配置和音频模板。OpenCV 画面识别保留为备用后端。

## 已核实的数据来源（2026-10-19）

全部来自本地克隆的 MMDK（alphazolam/MMDK）和 REFramework 源码，下表注明了各自依据。

| 需要的信息 | 读取路径 | 依据 |
|---|---|---|
| 角色编号 | `gBattle.Player.mPlayerType[i].mValue` | MMDK.lua:197 |
| 当前动作编号 | `gBattle.Rollback:GetLatestEngine().ActEngines[i]._Parent._Engine:get_ActionID()` | MMDK.lua:1232–1234、2049 |
| 招式触发表 | `gBattle.Command:get_mUserEngine()[i]:call("GetTrigger()")` | MMDK.lua:235 |
| 超必杀分类 | 触发项 `category_flags`：`IsSuper=0x8000`，`Lv1=0x80`、`Lv2=0x100`、`Lv3=0x200`、`Lv4=0x400`（Lv4 即 CA，带 `cond_vital_ratio=25`） | MMDK tables.lua；已用 8 个角色的 triggers.json 交叉验证 |
| 超必杀消耗 | 触发项 `gauge_consume` = 10000 / 20000 / 30000 | MMDK triggers.json |
| 通用动作名 | `nBattle.ACT_ID` 枚举：`GRD_*` 为防御，`DMG_*` 为受击，`BAS_*` 为基础动作 | MMDK.lua:162；Ryu Names.json |
| 文件输出 | Lua 的 `io.open` 只能写 `reframework/data/` 目录 | REFramework src/mods/bindings/FS.cpp |

触发表是在运行时从游戏里读取的，所以 MMDK 数据里没有的新角色（2025 年以后的 DLC）也能识别。这一点要在实测中确认（测试清单 F 项）。

还需要实测才能确定的：

- 超必杀槽字段和血量字段的名字（探针会通过数值变化自动找出）。
- 判断"哪边是玩家、哪边是 CPU"的字段（看 schema dump）。
- 特殊招式的边界情况：Terry SA2 派生、Bison SA2 派生、豪鬼瞬狱杀、蓄力超杀。

## 参考 Mod 4141（Arjun SA3 CA Aankhein Khuli）

- 只通过 Nexus 的"Preview file contents"文件清单查看，**没有下载**。原因是 Nexus 自动安全检查把这个文件隔离了，扫描结果标为"Some suspicious files"。
- 文件结构：一个 161 字节的 Lua 引导脚本，加上 `reframework/plugins/ArjunSuperAudioNative.dll`（4.6 MB，MinGW 编译，内含 ImGui 和 json 库）、两个 WAV 文件和 PowerShell 安装脚本。
- 核心逻辑都在没有源码的原生 DLL 里，无法从中得知它的检测方式。V2 不依赖它，只把"游戏内用 REFramework 插件放音乐"当作一种对照方案。

## 架构

```
SF6 进程（REFramework）                         外部 Python 程序（T03 代码继续使用）
┌────────────────────────────┐   JSONL 事件     ┌───────────────────────────────────┐
│ sf6_music_trigger.lua      │ ───────────────▶ │ RefEventSource（逐行读取新事件）   │
│  · 读角色、动作、触发表     │ reframework/data │  → 现有触发、淡出、抢占规则         │
│  · 判断 SA/CA 开始和结束    │ /SF6MusicTrigger │  → audio.py（pygame）              │
│  · 判断命中、格挡、空挥     │ /events.jsonl    │  → panel.py GUI、配置、音频模板      │
│  · 不播放声音               │                  │  OpenCV 后端保留为备用              │
└────────────────────────────┘                  └───────────────────────────────────┘
```

- 游戏内只负责检测，游戏外只负责播放，两边通过一个带版本号的事件格式通信。以后游戏更新，只需要改 Lua 脚本；改 GUI 或音频也不用碰游戏。

## 事件协议 v1（已实现）

文件：`<SF6>/reframework/data/SF6MusicTrigger/events.jsonl`。游戏启动时截断重写，超过 2 MB 也会截断。每行都带 `v`、`session`、`seq`、`t` 四个字段。

| type | 字段 | 说明 |
|---|---|---|
| `hello` | `script`, `game` | 脚本加载 |
| `heartbeat` | `enabled`, `battle`, `chara=[p1,p2]`, `supers=[n1,n2]` | 每秒一次，用来判断连接状态 |
| `chara` | `side`, `chara`, `supers` | 读到某一方的超必杀表 |
| `windows` | `side`, `chara`, `acts`, `timed=[[act, 最后攻击帧, 是否飞行道具]…]` | 读到超必杀各动作的攻击判定帧（v0.2 起） |
| `super_start` | `side`, `chara`, `level`(1–4), `label`, `act`, `episode` | 超必杀开始 |
| `super_result` | 同上，另加 `result`, `reason`, `at`（距开始的秒数） | 每段超必杀只发一次，是最早能确定的结果（v0.2 起） |
| `super_end` | 同上，另加 `result`(hit/block/whiff/unknown), `reason`, `duration` | 超必杀结束及最终结果 |

检测规则（`src/sf6_music_trigger/game_scripts/sf6_music_trigger.lua`）：

- **开始**：动作进入触发表里标记为超必杀的动作。更高等级可以打断较低等级，同等级的派生招式不会重复触发。
- **持续**：动作仍在 1190–1299 编号段内。
- **结束**：离开这个编号段时结束（recovered）；自己被打中时结束（interrupted）；换角色或离开对战时结束（reset）；超过 15 秒也会结束（timeout）。
- **结果**：期间对手出现过受击、被投或倒地，就算 hit；出现过防御，就算 block；对手做了自己的动作，就算 unknown；以上都没有，就算 whiff。对手在超必杀开始时已经处于受击状态（连段收尾），也算 hit。

这些编号段来自 MMDK 里 24 个角色的数据，已用 2026-10-19 的实测（杰米、不知火舞、桑吉尔夫）验证。

**提前判定结果（v0.2）**：实测时，空大要等到收招才判定（3.3–4.9 秒），音效来得太晚。现在 `super_result` 会尽早发出：

- **被防或命中**：对手第一次出现防御或受击时，立即判定。对手被锁进攻击方的超必杀动画（动作编号在 1190–1299，但又不是对手自己的超必杀）也算命中，这样投技类超必杀也能识别。
- **空大**：游戏内 FAB 动作数据里记录了每个动作的攻击判定帧（`AttackCollisionKey._EndFrame`，攻击数据索引大于 -1，且不是接近判定）。当前动作帧号超过最后一个攻击帧 3 帧后，再观察 0.1 秒，仍没有接触就判定为空大。
  - 实测里杰米 SA3/CA 约在 1.9 秒判定（原来 3.3 秒），桑吉尔夫 CA 约在 1.9 秒判定（原来 4.9 秒）。
  - 招式转到没有攻击帧的收招动作时，也按空大处理。
- **飞行道具类超必杀（带 `ShotKey`）和读不到帧数据的情况**：不提前判定，仍在收招时判定。

**掉招修正（v0.3，2026-10-19 实测）**：连段打断后接的 CA，被防或打空时会被误判为命中，原因是放招时对手还处在前一下的受击状态。日志里看到的两种情况：

- 对手还在受击 206，1.55 秒后防住了 CA，进入 163 防御。
- 对手在 1.5 秒时倒地（330/340），CA 打空。

现在的规则：

- **只认放招之后才开始的反应**：对手动作切换，或者同一个动作的帧号重新开始，才算一次新反应。放招那一刻对手已经处于什么状态，不作为判断依据。
- **受击只认 199–299 和 320–323**（受击、被投、破防）；倒地、受身等 330–356 只是之前那一下造成的后续，不算命中。
- **有攻击帧数据时**，对手的反应必须出现在攻击帧的前 4 帧到后 12 帧之间才算数。
- **对手被锁进攻击方的超必杀动画**，仍然直接判为命中。

`tests/reframework/test_real_replay.py` 用当天实测的 7 个回合（`fixtures/real_dropped_combos.json`）按 60 帧/秒重放，结果都与实际一致。

Python 端只执行每段超必杀的第一个结果，`super_end` 不会再重复执行。打空音效立即播放；淡出仍然遵守从开始播放起保留 4 秒的规则。

实测还发现：加载画面期间游戏数据读不到，每帧都会报错（24 秒里报了 1384 条）。现在同一个错误每 5 秒最多记录一次。

## Python 端（已实现）

- `reframework_events.py`：`RefEventSource` 负责持续读取事件文件。启动监听时会跳过已有的历史事件，能处理截断、半行、重复的 `seq`、新会话和坏行，并从心跳得出连接状态：没有文件、等待中、已关闭、菜单、对战中。另外还有检测脚本的安装函数，以及通过 Steam 库列表自动查找游戏目录的函数。
- `ref_engine.py`：`RefTriggerEngine` 按 T03 的规则决定怎么放音乐，包括 4 秒保留、后放的一方接管、过期的结果不生效、打空时放音效或淡出。
- `controller.py` 基本不变，只多了一个可选的 `runtime_message_provider`。`panel.py` 增加了"检测方式"下拉框、游戏目录和"安装检测脚本"按钮。REFramework 模式不需要 HUD 配置和校准。
- 配置新增 `detection_backend`（默认 `reframework`）和 `game_dir` 两项，T03 的旧配置文件可以直接读取。

## 游戏内设置（已实现）

全屏时切出游戏不方便，所以设置也可以在游戏里改。`game_scripts/sf6_music_trigger_ui.lua` 只在 REFramework 菜单打开时，画出"SF6 音乐设置"窗口（使用 REFramework 自带的 Roboto CJK 字体，可以显示中文）。

Lua 只能写 `reframework/data`，所以两边通过这个目录里的两个文件通信：

| 文件 | 写入方 | 内容 |
|---|---|---|
| `app_state.json` | 播放器（`game_ui.GameUiBridge.publish`，原子替换，至少每秒一次） | `v`、`updated`（Unix 时间）、`listening`、`status`、`volume_max`、`volume_unlimited`、`sides.p1/p2`（启用、主音乐名、音量%、大招范围、空大处理、打空音效名、音效音量%）、`tracks`、`templates` |
| `ui_commands.jsonl` | 游戏内窗口（脚本加载时截断） | 每行一条：`{v:1, type:"ui", session, seq, cmd, side?, key?, value?}`；`cmd` 可以是 `set` / `set_option` / `reset_volume` / `apply_template` / `start` / `stop` / `preview` |

- **播放器是唯一保存配置的一方**：命令先应用到面板的视图模型（`game_ui.apply_ui_command`），再刷新面板控件，保存 `config.json`，然后通过 `RefTriggerEngine.update_configs` 交给正在运行的监听器。正在播放的音乐不受影响，下一次超必杀就会用新设置。
- 面板里改的设置也走同样的流程：每 100 毫秒对比一次，有变化就立即交给监听器，0.5 秒内没有新变化就保存。
- 音乐列表（`game_ui.collect_tracks`）来自已配置音乐所在的文件夹和配置目录下的 `music/`，名字重复时会附上所在文件夹名。
- 游戏内拖滑块时，每 0.15 秒最多发送一次，并保证发出最后的数值。改完后窗口先显示本地的值，2 秒内等播放器确认。
- `app_state.json` 超过 4 秒没更新，窗口就显示"未连接 V2 播放器"。
- ImGui 的 `begin_window` / `push_id` 一定会配对结束，控件出错时也不会破坏 REFramework 菜单。

## 音量超过 100%（已实现）

- 音量上限 `config.MAX_VOLUME = 5.0`（500%）。`volume_unlimited` 只是界面选项，决定滑块最大到 100 还是 500；关闭时会把超过 100% 的值压回 100%。
- pygame 的 `set_volume` 最大只有 1.0，所以超过 1.0 时，`PygameAudioPort` 会用 `pygame.sndarray` 把采样乘以增益：
  - 满幅的 70% 以下按原样放大。
  - 70% 以上用 tanh 曲线软限幅，最高到满幅的 98%，避免爆音。
  - 放大后的声音缓存 6 首，`prepare()` 会在开始监听和设置变化时提前算好。
- 新增 `AppConfig.auto_start`：面板打开后自动开始监听并最小化。

## 阶段计划

1. **A：探针验证（待实测）**：`reframework/autorun/sf6_music_trigger_probe.lua` 只记录日志。实测方法见 `phase-a-probe-test-guide.md`。
2. **B：正式检测脚本（已完成离线版本）**：要等实测确认编号段和结果判断。待办：用超必杀槽字段和血量字段增强判断；判断是否在联机中，联机时自动停用。
3. **C：Python 桥接（已完成）**：GUI 里的 P1/P2 设置沿用原有的两侧独立配置。
4. **D：实测验收**：在训练模式里覆盖多个角色、P1/P2、命中/格挡/空挥/打断。
5. **E：打包 V2 群测版**：把 `BUILD_ID` 从 T03 改成 V2，PyInstaller 打包时带上 `game_scripts/*.lua`，并更新 README 和 VERSION。

## 离线测试工具

- `tests/reframework/fake_reframework.py`：用 lupa（内嵌 Lua 5.4）运行真实的 Lua 脚本，配一个模拟的 SF6 战斗状态。
- `tests/reframework/test_end_to_end.py`：真实的 Lua 检测脚本写出事件文件，`RefEventSource` 读取后交给 `RefTriggerEngine`，再驱动模拟的音频端口。
- `tools/analyze_probe_log.py`：汇总实测日志，自动找出超必杀槽字段和血量字段。
- `run_v2.bat`：从源码启动 V2 面板，使用项目里的 `.venv`。
