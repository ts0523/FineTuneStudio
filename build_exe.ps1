# FineTune Studio build script
# Usage: powershell -ExecutionPolicy Bypass -File build_exe.ps1
# Output: dist\FineTuneStudio.exe (onefile, no torch)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "找不到 .venv\Scripts\python.exe，请先运行 setup_env.ps1" }

Write-Host "[1/4] 检查 PyInstaller" -ForegroundColor Cyan
& $py -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "      安装固定版本 PyInstaller ..." -ForegroundColor Yellow
    & $py -m pip install -r requirements-build.txt --disable-pip-version-check
}

Write-Host "[2/4] 生成图标" -ForegroundColor Cyan
& $py "scripts\make_icon.py"
if (-not (Test-Path "studio\assets\app.ico")) {
    Write-Host "      图标生成失败，将使用默认图标" -ForegroundColor Yellow
    $iconArg = @()
} else {
    $iconArg = @("--icon", "$PSScriptRoot\studio\assets\app.ico")
}

Write-Host "[3/4] 打包（onefile / 无控制台）" -ForegroundColor Cyan
& $py -m PyInstaller --noconfirm --clean --onefile --windowed `
    --name FineTuneStudio `
    @iconArg `
    --add-data "$PSScriptRoot\studio\workers;workers" `
    --add-data "$PSScriptRoot\studio\assets;assets" `
    --hidden-import psutil `
    --hidden-import cv2 `
    --hidden-import PIL.ImageTk `
    --collect-all _sounddevice_data `
    --collect-submodules cv2 `
    --distpath dist --workpath build\work --specpath build `
    run_studio.py
if ($LASTEXITCODE -ne 0) { throw "打包失败" }

Write-Host "[4/4] 生成交付文件" -ForegroundColor Cyan
$src = Join-Path $PSScriptRoot "dist\FineTuneStudio.exe"
$dst = Join-Path $PSScriptRoot "FineTuneStudio.exe"
Copy-Item $src $dst -Force
$size = [math]::Round((Get-Item $dst).Length / 1MB, 1)
Write-Host "Done: $dst ($size MB)" -ForegroundColor Green
Write-Host "Self-test: .\FineTuneStudio.exe --selftest" -ForegroundColor DarkGray
