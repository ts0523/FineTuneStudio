# FineTune Studio

[简体中文](README.zh-CN.md) · **English**

FineTune Studio is a Windows desktop workbench for small-model fine-tuning, multimodal experiments, dataset authoring, and local inference. It runs a complete CPU-only loop: download a model, create or import data, train LoRA/full-parameter SFT, evaluate, chat, merge, and export.

![FineTune Studio icon](studio/assets/app.png)

## Why this project

Most fine-tuning demos stop at a Python script. FineTune Studio packages the whole workflow into a desktop application:

- **Guided from zero to a runnable model** — a seven-step tutorial, preflight checks, parameter estimates, and a one-click smoke test.
- **Small-model friendly** — designed for MiniMind-O 0.1B on CPUs without a discrete GPU.
- **Multimodal by design** — language, image, audio understanding, and local speech output.
- **Multiple adapters at once** — combine independent LoRA adapters and their vision/audio projectors in one chat session.
- **User-owned data** — build Q&A datasets directly in the app, with optional image or audio attachments.
- **Bilingual UI** — switch between Chinese and English in Settings; the choice persists.

## Features

### Data Studio

- Create named Q&A datasets from scratch.
- Write a question and reference answer, record the question with the microphone, and optionally attach an image or an audio clip.
- Append examples one by one to `data/<name>.jsonl` with atomic writes.
- Rename datasets, validate rows, inspect duplicates, split train/validation, and preview samples.
- Import JSONL/CSV/TSV/TXT and normalize records to chat `messages`.

### Training

- LoRA and full-parameter SFT through Transformers + PEFT.
- Live loss curve, throughput, ETA, validation loss, and step-level progress.
- Graceful stop at a step boundary with checkpoint save; resume from checkpoint.
- Hard-stop protection: a stuck worker is force-killed after 8 seconds.
- Smoke test with a two-step run.
- Every run is archived with hyperparameters, metrics, artifacts, and rerun support.

### Multimodal adapters

- Built-in capability toggles: language understanding, image understanding, audio understanding, speech output.
- Image path: MobileNetV3-Small encoder → projector → LLM embedding injection.
- Audio path: Whisper-tiny encoder → projector → LLM embedding injection.
- Speech output: local Windows TTS; no cloud speech service required.
- Multi-adapter selection: multiple LoRA adapters can be active together. Vision and audio projectors are combined from all selected adapters.
- A single message can contain both an image and an audio clip.
- Chat and Training pages support microphone recording and speech-to-text.

### AI AutoTrain

Give the app a goal. It searches public dataset metadata, asks a real AI model to generate SFT data, trains the model, evaluates on a fixed holdout set, scores the result, and uses the advice for the next round.

- Holdout is created once and excluded from every training round.
- Session IDs prevent late results from a previous run from mutating the current state.
- Metrics come from the actual training worker, not UI estimates.
- AI scoring records `judge_source` and the actual model name, so fallback local scoring cannot masquerade as a successful AI judgment.

### Chat, calls, merge, export

- Text/image/audio input and streaming text output.
- Voice and video calls through a cloud omni model when an OpenRouter key is configured.
- Merge LoRA into a standalone model directory and preserve `mm_projector.pt`.
- Load and verify the merged model immediately.
- Package ZIP, generate deployment notes, and open artifacts directly.

## Screenshots

Screenshots are not committed to keep the repository small. The app includes a `--selftest` mode that builds every page and reports JSON health data.

## Quick start

### Requirements

- Windows 10/11
- Python 3.12
- 16 GB RAM recommended for local training
- No CUDA GPU required

### 1. Clone and install

```powershell
git clone https://github.com/ts0523/FineTuneStudio.git
cd FineTuneStudio
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

`setup_env.ps1` creates `.venv`, installs CPU PyTorch and project dependencies, then runs the environment check.

### 2. Download the base model

The repository does not bundle model weights.

```powershell
.\.venv\Scripts\python.exe scripts\download_model.py
```

The default model is `jingyaogong/minimind-3o` from `hf-mirror.com`.

### 3. Start the app

```powershell
.\.venv\Scripts\python.exe run_studio.py
```

Or build a single-file Windows executable:

```powershell
powershell -ExecutionPolicy Bypass -File build_exe.ps1
.\FineTuneStudio.exe
```

## Bilingual UI

Open **Settings → Appearance → Interface language / UI language** and choose:

- `中文 / Chinese`
- `English`

The setting is stored as `ui.lang` in `studio_config.json`. You can also force a language for one launch:

```powershell
$env:FTS_LANG = "en"
.\.venv\Scripts\python.exe run_studio.py
```

The English UI is generated from a complete runtime catalog at `studio/assets/i18n_en.json`; user data, file names, model output, and paths are never translated.

## Configuration and secrets

- Never commit `studio_config.json`; it is ignored by Git.
- `studio_config.example.json` contains no key.
- OpenRouter keys are read from `FTS_AI_KEY` / `OPENROUTER_API_KEY`, or stored with Windows DPAPI from the Settings page.
- Keys are never written to config or logs.
- Downloaded model code is skipped by default. Trusting custom `*.py` requires an explicit action and a full 40-character commit SHA.
- Weight SHA-256 proves file integrity only; it does not make repository code trustworthy.

## Project layout

```text
FineTuneStudio/
├─ run_studio.py             application entry point
├─ studio/
│  ├─ app.py                 main window and navigation
│  ├─ core/                  paths, config, data, model, planning, AI
│  ├─ ui/                    themes, widgets, i18n, 14 pages
│  ├─ workers/               training, chat, download, merge, doctor
│  └─ assets/                icon and English UI catalog
├─ scripts/                  setup, download, CLI tools, icon generator
├─ tests/                    unit, protocol, process, live acceptance
├─ pyproject.toml            Ruff and mypy configuration
├─ requirements.txt          runtime dependencies
├─ setup_env.ps1             environment bootstrap
└─ build_exe.ps1             PyInstaller build
```

## Verification

```powershell
# Unit and protocol tests
.\.venv\Scripts\python.exe -m pytest

# Live tests: real OpenRouter generation/scoring, real mirror download,
# real merge followed by a forced verification failure
$env:FTS_LIVE_ACCEPTANCE = "1"
.\.venv\Scripts\python.exe -m pytest tests\test_live_acceptance.py

# Build every page and theme without entering the main loop
.\.venv\Scripts\python.exe run_studio.py --selftest

# Static analysis
.\.venv\Scripts\ruff.exe check studio tests scripts
.\.venv\Scripts\mypy.exe --config-file pyproject.toml
```

## Known limitations

- Windows-specific integrations: Recycle Bin deletion, process-tree termination, DPI handling.
- The default model is a 64M text backbone; image/audio branches use frozen lightweight encoders. Accuracy is limited by data scale.
- Image generation is not implemented. Output modalities are text and local speech.
- Audio cancel returns immediately and closes the active socket; the internal daemon read remains bounded by the request timeout.
- `trust_remote_code` is not enabled. Custom model code requires explicit trust, a full commit SHA, and hash verification.
- The English UI catalog covers all built-in static UI strings; user-provided Chinese goals, dataset names, and model replies remain in their original language.

## License

MIT. See [LICENSE](LICENSE).
