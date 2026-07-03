# Voice Changer

基于 Python + PyQt6 + RVC 的实时 AI 变声器桌面应用。

## 功能特性

- 实时麦克风音频捕获与播放
- AI 音色转换（RVC 模型）与基础 Pitch Shift  fallback
- GPU 加速推理（支持 NVIDIA CUDA）
- 音调、索引强度、清音保护实时调节
- 实时音频波形可视化
- 音频设备热切换
- 暗色主题 UI

## 环境要求

- Python 3.10+
- Windows / Linux
- NVIDIA 显卡（推荐，用于 CUDA 加速）
- 麦克风与扬声器/耳机

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/SpadyDong/voice-change.git
cd voice-change
git checkout v0.1
```

### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装 PyTorch（带 CUDA）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

### 3. 准备模型

将 RVC 预训练模型文件放入 `models/rvc/` 目录：

- `.pth` 文件：RVC 生成器权重
- `.index` 文件（可选）：Faiss 特征索引
- `models/hubert/hubert_base.pt`：Hubert 特征提取器

> 模型可从 [RVC-WebUI 社区](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 或相关 HuggingFace 仓库下载。

### 4. 运行应用

```bash
python src/main.py
```

### 5. 使用说明

1. 在右侧选择输入（麦克风）和输出（扬声器/虚拟音频设备）设备
2. 从模型列表中选择要使用的 RVC 模型
3. 调节音调、索引强度等参数
4. 点击「启动变声」或按 `Space` 键开始实时变声

## 打包为可执行文件

```bash
pip install pyinstaller
pyinstaller voice-changer.spec
```

打包后的文件位于 `dist/VoiceChanger/`（Windows 为 `dist/VoiceChanger.exe`）。

## 项目结构

```
voice-change/
├── src/
│   ├── main.py              # 应用入口
│   ├── core/                # 音频管道与 RVC 推理
│   ├── gui/                 # PyQt6 界面组件
│   └── config/              # 配置管理
├── models/
│   ├── rvc/                 # RVC 模型存放
│   └── hubert/              # Hubert 权重
├── tests/                   # pytest 测试
├── assets/
│   └── style.qss            # UI 样式表
├── requirements.txt
└── voice-changer.spec       # PyInstaller 配置
```

## 测试

```bash
pytest tests/ -v
```

## 注意事项

- 首次启动若无 RVC 模型，软件将使用 librosa 的 Phase Vocoder 进行基础 Pitch Shift，效果较简单。
- 建议在 Windows 上搭配虚拟音频设备（如 VB-Cable）使用，以便将变声后的音频输出到其他应用（Discord、QQ、微信等）。
- 实时延迟与 `block_size` 和 GPU 性能相关，RTX 4070S 通常可做到 50-150ms。

## License

MIT
