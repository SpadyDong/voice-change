# 实时AI变声器软件开发计划

## 1. 摘要 (Summary)

基于 Python + PyQt6 技术栈，开发一款支持 AI 实时声音转换（RVC 风格）的桌面变声器软件。软件能够实时捕获麦克风输入，通过 GPU 加速的 RVC 模型进行音色转换，并低延迟输出到扬声器或虚拟音频设备。目标平台为 Windows/Linux 桌面端，充分利用用户 RTX 4070S 显卡的 CUDA 加速能力。

## 2. 当前状态分析 (Current State Analysis)

- **代码库状态**：当前仓库 `/workspace` 几乎为空，仅包含 `README.md` 和 `LICENSE`，无现有代码架构约束。
- **技术约束**：
  - 用户已拥有 RTX 4070S，支持 CUDA 计算，显存约 12GB，足以运行 RVC 实时推理。
  - 需要低延迟实时音频处理（目标延迟 < 200ms）。
  - 需支持麦克风输入和扬声器/虚拟音频设备输出。
- **参考生态**：
  - RVC-Project/Retrieval-based-Voice-Conversion-WebUI 提供了成熟的 RVC 实时推理模块（`tools/rvc_for_realtime.py`）和 GUI 参考（`gui_v1.py`）。
  - 可复用其模型格式（`.pth` + `.index`）和核心推理逻辑，但需重构为模块化架构。

## 3. 技术架构决策 (Architecture Decisions)

| 模块 | 技术选型 | 理由 |
|------|----------|------|
| **GUI 框架** | PyQt6 | 跨平台、UI 现代化、信号槽机制适合实时音频状态更新 |
| **音频 I/O** | `sounddevice` + `PortAudio` | 低延迟、支持 ASIO（Windows）、回调式流处理 |
| **AI 推理** | PyTorch (CUDA) + RVC 模型 | 用户有 NVIDIA GPU，PyTorch CUDA 加速成熟；RVC 社区生态完善 |
| **音频处理** | `librosa`, `numpy`, `scipy` | 特征提取、重采样、STFT/ISTFT 处理 |
| **配置管理** | `pydantic` + JSON 配置文件 | 类型安全、验证方便 |
| **测试框架** | `pytest` + `pytest-qt` | 支持单元测试和 GUI 测试 |
| **打包发布** | `PyInstaller` | 打包为独立可执行文件，方便分发 |

## 4. 项目目录结构 (Project Structure)

```
voice-change/
├── src/
│   ├── __init__.py
│   ├── main.py                      # 应用入口，初始化 QApplication 和主窗口
│   ├── core/                        # 核心音频与推理引擎（无 GUI 依赖）
│   │   ├── __init__.py
│   │   ├── audio_pipeline.py        # 实时音频流捕获与播放（sounddevice Stream）
│   │   ├── rvc_engine.py            # RVC 模型推理封装（模型加载、特征提取、推理）
│   │   ├── model_manager.py         # 模型文件扫描、加载、切换管理
│   │   └── audio_utils.py           # 音频格式转换、重采样、缓冲区管理
│   ├── gui/                         # UI 层（仅依赖 core 的接口）
│   │   ├── __init__.py
│   │   ├── main_window.py           # 主窗口：设备选择、模型选择、参数调节
│   │   ├── control_panel.py         # 控制面板：启动/停止、音调、索引强度滑块
│   │   ├── model_browser.py         # 模型浏览与选择界面
│   │   ├── device_selector.py       # 输入/输出音频设备选择
│   │   └── audio_visualizer.py      # 实时音频波形/频谱可视化（可选增强）
│   └── config/
│       ├── __init__.py
│       └── settings.py              # 配置模型（pydantic）、默认配置、配置持久化
├── models/                          # 运行时模型存放目录（gitignored）
│   ├── rvc/                         # RVC .pth + .index 文件
│   └── hubert/                      # Hubert 特征提取器预训练权重
├── tests/                           # 测试目录
│   ├── __init__.py
│   ├── test_audio_pipeline.py       # 音频流捕获/播放测试
│   ├── test_rvc_engine.py           # 推理引擎测试（mock 模型或轻量模型）
│   ├── test_model_manager.py        # 模型管理逻辑测试
│   └── test_gui.py                  # GUI 组件基本测试（pytest-qt）
├── docs/
│   ├── ui_mockup.md                 # UI 原型与交互说明
│   └── architecture.md              # 详细架构文档
├── assets/
│   └── icons/                       # 应用图标资源
├── requirements.txt                 # Python 依赖
├── config.json                      # 默认运行配置（设备、模型路径等）
├── .gitignore
└── README.md
```

## 5. 模块详细设计 (Module Design)

### 5.1 核心音频管道 (`core/audio_pipeline.py`)

**职责**：管理 sounddevice 的 InputStream/OutputStream 或双工 Stream，在音频回调中执行推理。

**关键设计**：
- 使用 `sounddevice.Stream` 双工模式（同时输入输出），`blocksize` 设置为 1024-2048 samples（约 23-46ms @ 44.1kHz），平衡延迟与稳定性。
- 回调函数中：
  1. 接收 `indata`（麦克风输入块）。
  2. 调用 `RvcEngine.infer()` 进行音色转换。
  3. 将结果写入 `outdata`。
- 由于 RVC 推理可能在 GPU 上有抖动，引入一个线程安全的环形缓冲区（`collections.deque` 或 `numpy.ringbuffer`）平滑输出。
- 支持启动/停止/暂停状态控制。

**关键接口**：
```python
class AudioPipeline:
    def __init__(self, rvc_engine: RvcEngine, sample_rate=48000, block_size=1024): ...
    def set_input_device(self, device_id: int): ...
    def set_output_device(self, device_id: int): ...
    def start(self): ...
    def stop(self): ...
    def is_running(self) -> bool: ...
```

### 5.2 RVC 推理引擎 (`core/rvc_engine.py`)

**职责**：封装 RVC 模型的加载、Hubert 特征提取、F0 提取、检索增强和声码器合成。

**关键设计**：
- 复用 RVC 官方实现的核心逻辑（`infer.lib.infer_pack.models` 中的 `SynthesizerTrnMs256NSFsid` 或 `SynthesizerTrnMs768NSFsid`）。
- 模型加载时自动检测版本（v1/v2）和采样率（32k/40k/48k）。
- 支持 `rmvpe`、`harvest`、`crepe` 等 F0 提取算法，默认 `rmvpe`（精度与速度平衡）。
- 推理参数暴露给 GUI 调节：
  - `pitch_shift`（音调升降，半音）
  - `index_rate`（检索特征融合率，0.0-1.0）
  - `protect`（清音保护，0.0-0.5）
  - `filter_radius`（中值滤波半径）
- 使用 `torch.no_grad()` 和 `torch.cuda.stream()` 优化推理性能。
- 首次推理时进行 warm-up，避免后续延迟抖动。

**关键接口**：
```python
class RvcEngine:
    def load_model(self, pth_path: str, index_path: str | None): ...
    def infer(self, audio_chunk: np.ndarray) -> np.ndarray: ...
    def set_params(self, pitch_shift: int, index_rate: float, protect: float, filter_radius: int): ...
    def get_latency_ms(self) -> float: ...
```

### 5.3 模型管理器 (`core/model_manager.py`)

**职责**：扫描模型目录，维护可用模型列表，管理模型加载/卸载。

**关键设计**：
- 扫描 `models/rvc/` 目录，识别 `.pth` 和对应的 `.index` 文件。
- 模型元数据缓存（名称、版本、采样率、创建时间）。
- 异步加载模型（在独立线程中执行，避免阻塞 GUI）。
- 提供模型热切换能力（不重启音频流）。

### 5.4 GUI 主窗口 (`gui/main_window.py`)

**职责**：应用主界面，整合所有子控件。

**布局设计**：
- **顶部工具栏**：应用标题、设置按钮、帮助按钮。
- **左侧控制面板** (`gui/control_panel.py`)：
  - 启动/停止变声大按钮（带状态指示颜色）。
  - 音调调节滑块（-12 ~ +12 半音，带标签）。
  - 索引强度滑块（0 ~ 100%）。
  - 清音保护滑块。
  - 延迟显示标签（实时更新）。
- **中部模型选择** (`gui/model_browser.py`)：
  - 下拉框/列表展示已扫描模型。
  - 刷新按钮重新扫描。
  - 显示当前加载模型信息。
- **右侧设备选择** (`gui/device_selector.py`)：
  - 输入设备下拉框（列出所有麦克风）。
  - 输出设备下拉框（扬声器/虚拟音频设备如 VB-Cable）。
  - 采样率显示（根据模型自动或手动）。
- **底部状态栏**：
  - 运行状态（就绪/运行中/错误）。
  - GPU 使用率/显存占用（可选，通过 `pynvml`）。

### 5.5 配置管理 (`config/settings.py`)

**职责**：持久化用户配置（设备选择、模型路径、参数值、窗口位置）。

**关键设计**：
- 使用 `pydantic.BaseModel` 定义配置 schema。
- 配置文件存储在用户目录（`~/.voice-change/config.json`）。
- 启动时加载，退出时保存。

## 6. 开发阶段与任务分解 (Implementation Phases)

### Phase 1：项目骨架与基础配置（预计 1-2 天）
- [ ] 创建 `requirements.txt`，包含所有依赖（`PyQt6`, `torch`, `torchaudio`, `sounddevice`, `librosa`, `numpy`, `scipy`, `pydantic`, `pytest`, `pytest-qt`）。
- [ ] 创建项目目录结构（`src/`, `tests/`, `docs/`, `assets/`）。
- [ ] 实现 `config/settings.py` 配置模块。
- [ ] 创建 `src/main.py` 入口文件，初始化 PyQt6 应用和空主窗口。
- [ ] 编写 `docs/ui_mockup.md` UI 原型文档。
- [ ] 验证：能运行 `python src/main.py` 打开一个空窗口。

### Phase 2：音频管道与设备管理（预计 2-3 天）
- [ ] 实现 `core/audio_utils.py`（格式转换、重采样工具函数）。
- [ ] 实现 `core/audio_pipeline.py`：
  - 使用 `sounddevice` 查询可用音频设备。
  - 实现双工 Stream 回调框架。
  - 实现启动/停止/暂停控制。
  - 引入环形缓冲区平滑输出。
- [ ] 实现 `gui/device_selector.py`：下拉框绑定系统音频设备。
- [ ] 实现 `gui/audio_visualizer.py`（基础波形显示）。
- [ ] 验证：能在 GUI 中选择设备，启动后听到麦克风直通声音（无变声）。

### Phase 3：RVC 推理引擎集成（预计 3-5 天）
- [ ] 调研并引入 RVC 核心推理代码（从 RVC-WebUI 提取必要模块到 `src/core/rvc/`）。
  - `infer/lib/infer_pack/models.py`（声码器模型定义）。
  - `infer/lib/audio.py`（音频处理）。
  - `modules/vc/pipeline.py`（推理管道）。
  - Hubert 编码器加载。
  - RMVPE F0 提取器。
- [ ] 实现 `core/rvc_engine.py`，封装模型加载与推理：
  - 支持 `.pth` 和 `.index` 加载。
  - 实现 `infer()` 方法接收 numpy chunk，返回转换后 chunk。
  - GPU warm-up 和 CUDA stream 优化。
- [ ] 实现 `core/model_manager.py`：扫描、加载、切换模型。
- [ ] 实现 `gui/model_browser.py`：模型列表 UI。
- [ ] 实现 `gui/control_panel.py`：参数调节滑块 UI。
- [ ] 将 `audio_pipeline` 的回调与 `RvcEngine` 连接。
- [ ] 验证：加载预训练 RVC 模型后，能实时变声输出。

### Phase 4：UI 完善与用户体验（预计 2-3 天）
- [ ] 完善 `gui/main_window.py` 布局（左右分栏或标签页）。
- [ ] 实现状态栏与实时延迟/性能监控。
- [ ] 添加错误处理与弹窗提示（模型加载失败、设备不可用、CUDA OOM 等）。
- [ ] 添加键盘快捷键（如 Space 切换启动/停止）。
- [ ] UI 美化（QSS 样式表、图标资源）。
- [ ] 实现配置持久化（退出保存参数，启动恢复）。

### Phase 5：测试与验证（预计 2-3 天）
- [ ] 编写 `tests/test_audio_pipeline.py`：
  - 测试音频设备枚举。
  - 测试 Stream 启动/停止生命周期。
  - 使用 mock 回调验证数据流。
- [ ] 编写 `tests/test_rvc_engine.py`：
  - 使用轻量测试模型验证推理流程。
  - 测试参数设置接口。
- [ ] 编写 `tests/test_model_manager.py`：
  - 测试模型目录扫描。
  - 测试模型切换逻辑。
- [ ] 编写 `tests/test_gui.py`：
  - 使用 `pytest-qt` 测试控件信号与槽。
- [ ] 手动测试清单：
  - [ ] 延迟测试：使用外部录音对比输入输出，测量端到端延迟（目标 < 200ms）。
  - [ ] 音质测试：对比不同 `pitch_shift` 和 `index_rate` 的听感。
  - [ ] 稳定性测试：连续运行 30 分钟无崩溃、无内存泄漏。
  - [ ] GPU 显存测试：监控显存占用，确认 4070S 运行稳定。

### Phase 6：打包与发布（预计 1 天）
- [ ] 编写 `PyInstaller` spec 文件，打包为 Windows/Linux 可执行文件。
- [ ] 排除不必要的依赖（如训练相关代码），减小体积。
- [ ] 提供 `README.md` 使用说明（安装、模型下载、快速开始）。
- [ ] 创建 GitHub Release Tag（如 `v0.1.0`）。

## 7. 关键风险与应对 (Risks & Mitigations)

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| RVC 实时推理延迟过高 | 用户体验差，无法实时对话 | 调整 `blocksize`、使用更小的模型、开启 FP16 推理、启用 CUDA Graph |
| sounddevice + CUDA 回调冲突 | 音频爆音/卡顿 | 在回调中仅做数据搬运，推理放在独立线程/队列；或增大缓冲区 |
| 模型兼容性（v1/v2/不同采样率） | 加载失败或音质异常 | 模型管理器自动检测版本信息，UI 明确标注 |
| PyInstaller 打包体积过大 | 分发困难 | 仅包含推理依赖，排除训练库；使用 UPX 压缩 |
| PortAudio 在部分系统上 ASIO 支持差 | Windows 低延迟无法实现 | 提供 WASAPI/DS 备选方案；引导用户安装 ASIO4ALL |

## 8. 验证步骤 (Verification Steps)

1. **单元测试**：运行 `pytest tests/` 全部通过。
2. **集成测试**：启动应用 -> 选择输入输出设备 -> 加载模型 -> 点击启动 -> 对着麦克风说话，能从耳机听到转换后的声音。
3. **延迟测试**：使用物理环路（扬声器播放测试音 -> 麦克风接收 -> 变声输出），用 Audacity 录制并测量输入输出时间差。
4. **稳定性测试**：连续运行 30 分钟，观察内存和显存是否持续增长（`nvidia-smi` / 任务管理器）。
5. **模型切换测试**：运行中切换不同模型，确认无崩溃且新模型生效。

## 9. 假设与依赖 (Assumptions & Dependencies)

- 用户已安装 NVIDIA 显卡驱动和 CUDA Toolkit（或至少 PyTorch 能识别 CUDA）。
- 用户会自行准备或下载 RVC 预训练模型（`.pth` + `.index`）以及 Hubert 权重文件。
- 目标平台以 Windows 为主，Linux 为辅（PortAudio 跨平台）。
- 本项目不实现模型训练功能，仅实现推理与实时变声。
- 开发初期可参考/引入 RVC-WebUI 的开源代码（MIT/GPL 协议兼容）。
