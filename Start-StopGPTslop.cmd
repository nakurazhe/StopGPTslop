@echo off
setlocal
cd /d "%~dp0"
set "HF_HOME=%~dp0.cache\huggingface"
set "TORCH_HOME=%~dp0.cache\torch"
set "TORCHINDUCTOR_CACHE_DIR=%~dp0.cache\torchinductor"
echo Starting StopGPTslop...
"%~dp0.venv\Scripts\python.exe" "%~dp0webui.py" --no-compile
if errorlevel 1 pause
