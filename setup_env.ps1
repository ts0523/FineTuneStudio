# 重建虚拟环境（纯环境搭建，不含训练）
# 用法: powershell -ExecutionPolicy Bypass -File setup_env.ps1
# 优先用 uv（快）；没有 uv 时退回 python -m venv + pip，两者都走清华镜像。
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Set-Location -Path $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$hasUv = $null -ne (Get-Command uv -ErrorAction SilentlyContinue)

function Find-BasePython {
    foreach ($cand in @("py", "python", "python3")) {
        $cmd = Get-Command $cand -ErrorAction SilentlyContinue
        if ($cmd) {
            try {
                $v = & $cmd.Source -c "import sys;print(sys.version_info[:2])" 2>$null
                if ($v -match "3, 1[0-3]") { return @{ Exe = $cmd.Source; Args = @() } }
            } catch { }
        }
    }
    return $null
}

if (Test-Path $py) {
    Write-Host "[skip] venv already exists: $py"
} elseif ($hasUv) {
    Write-Host "[1/4] creating venv with uv (Python 3.12) ..."
    $env:UV_LINK_MODE = "copy"
    uv venv ".venv" --python 3.12 --seed
} else {
    Write-Host "[1/4] uv not found, falling back to python -m venv ..."
    $base = Find-BasePython
    if (-not $base) { throw "找不到可用的 Python 3.12，请先安装 Python 3.12 后重试" }
    $venvArgs = @()
    $venvArgs += $base.Args
    $venvArgs += @("-m", "venv", ".venv")
    & $base.Exe $venvArgs
    if (-not (Test-Path $py)) { throw "venv 创建失败：$py 不存在" }
}

Write-Host "[2/4] installing dependencies from Tsinghua mirror ..."
$env:UV_LINK_MODE = "copy"
if ($hasUv) {
    uv pip install --python $py -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
} else {
    & $py -m pip install --upgrade pip --index-url https://pypi.tuna.tsinghua.edu.cn/simple
    & $py -m pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
}

Write-Host "[3/4] installing test dependencies (requirements-dev.txt) ..."
if (Test-Path "requirements-dev.txt") {
    if ($hasUv) {
        uv pip install --python $py -r requirements-dev.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
    } else {
        & $py -m pip install -r requirements-dev.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
    }
}

Write-Host "[4/4] self check ..."
if (-not (Test-Path "models\minimind-3o\config.json")) {
    Write-Host "[提示] 没找到 models\minimind-3o；可先运行 python scripts\download_model.py 或在应用「模型库」下载。"
}
& $py scripts\check_env.py
exit $LASTEXITCODE
