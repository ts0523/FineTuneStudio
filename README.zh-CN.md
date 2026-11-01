# FineTune Studio（微调训练台）

**简体中文** · [English](README.md)

FineTune Studio 是一个 Windows 桌面小模型微调与多模态实验工作台。它把「下载模型 → 制作数据 → 训练 LoRA/全参 → 验证 → 对话 → 合并导出」做成可以点击完成的完整流程，针对无独显 CPU 环境优化。

![FineTune Studio 图标](studio/assets/app.png)

## 主要能力

- **从零到可用模型的 7 步教程**：目标、数据、训练、验收、导出都有明确入口。
- **纯 CPU 小模型训练**：默认适配 MiniMind-O 0.1B，LoRA 与全参 SFT 均可。
- **多模态理解**：语言、图像、语音输入；文本与本机语音输出。
- **多适配器同时激活**：图像适配器与语音适配器可多选，LoRA 和投影层一起工作。
- **实验层默认本地模型**：对话页适配器直接标为「图像适配器 / 音频适配器 / 语音适配器」，勾选即全模态输入输出。
- **对话/实验层录音**：支持录音直接理解、录音转文字、附加图片或语音附件。
- **自建问答数据集**：自定义数据集名称，逐条写问题和标准回答，可选图片或录音附件。
- **AI 帮训闭环**：AI 造数 → 训练 → 固定留出集验证 → AI 评分 → 下一轮补数据。
- **双语界面**：设置页可在中文与 English 之间切换，重启后保持。
- **合并与交付**：LoRA 合并成独立模型目录，保留多模态投影层，可加载验证、打包 ZIP。

## 快速开始

### 环境

- Windows 10/11
- Python 3.12
- 本地训练建议 16 GB 内存
- 不需要 CUDA 显卡

### 安装

```powershell
git clone https://github.com/ts0523/FineTuneStudio.git
cd FineTuneStudio
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

### 下载基座模型

仓库不包含模型权重。

```powershell
.\.venv\Scripts\python.exe scripts\download_model.py
```

### 启动

```powershell
.\.venv\Scripts\python.exe run_studio.py
```

或构建单文件 exe：

```powershell
powershell -ExecutionPolicy Bypass -File build_exe.ps1
.\FineTuneStudio.exe
```

## 双语界面

进入 **设置 → 外观 → 界面语言 / UI language**：

- `中文 / Chinese`
- `English`

语言写入 `studio_config.json` 的 `ui.lang`。也可以只对本次启动强制英文：

```powershell
$env:FTS_LANG = "en"
.\.venv\Scripts\python.exe run_studio.py
```

英文界面由 `studio/assets/i18n_en.json` 提供。用户输入、数据集名、模型回复和文件路径不会被翻译。

## 隐私与密钥

- `studio_config.json` 已被 Git 忽略，不会提交。
- 示例配置 `studio_config.example.json` 不含任何真实 Key。
- OpenRouter Key 从 `FTS_AI_KEY` / `OPENROUTER_API_KEY` 读取，或通过设置页写入 Windows DPAPI。
- Key 不写入配置文件或日志。
- 自定义模型代码默认不下载；信任代码必须明确确认并提供完整 40 位 commit SHA。
- 仓库不包含模型权重、训练数据、训练产物、日志、虚拟环境或打包 exe。

## 项目结构

```text
FineTuneStudio/
├─ run_studio.py             启动入口
├─ studio/
│  ├─ app.py                 主窗口与导航
│  ├─ core/                  配置、数据、模型、规划、AI
│  ├─ ui/                    主题、控件、i18n、14 个页面
│  ├─ workers/               训练、对话、下载、合并、体检
│  └─ assets/                图标与英文 UI 词典
├─ scripts/                  环境、下载、命令行工具、图标生成
├─ tests/                    单元、协议、进程与真实环境验收
├─ pyproject.toml            Ruff / mypy 配置
├─ requirements.txt          运行依赖
├─ setup_env.ps1             环境安装
└─ build_exe.ps1             PyInstaller 打包
```

## 验证

```powershell
# 单元与协议测试
.\.venv\Scripts\python.exe -m pytest

# 真实环境验收：OpenRouter、镜像下载、真实合并后验证失败
$env:FTS_LIVE_ACCEPTANCE = "1"
.\.venv\Scripts\python.exe -m pytest tests\test_live_acceptance.py

# 构建全部页面与主题
.\.venv\Scripts\python.exe run_studio.py --selftest

# 静态检查
.\.venv\Scripts\ruff.exe check studio tests scripts
.\.venv\Scripts\mypy.exe --config-file pyproject.toml
```

## 已知限制

- Windows 专用：回收站删除、进程树强杀、DPI 适配。
- 默认 64M 文本主干；图像/语音使用冻结的轻量编码器，效果受数据量限制。
- 不提供文字生成图片；输出模态为文本与本机语音。
- AI 取消会立即返回并关闭活动 socket，底层 daemon 读取仍受请求 timeout 限制。
- 自定义代码信任是「完整 commit + 显式信任 + 哈希校验」，不是签名/PKI。
- 英文词典覆盖全部内置静态界面文案；用户输入的中文目标、数据集名和模型回复保持原文。

## 许可证

MIT，详见 [LICENSE](LICENSE)。
