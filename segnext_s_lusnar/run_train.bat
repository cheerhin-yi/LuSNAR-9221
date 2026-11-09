@echo off
setlocal
cd /d "%~dp0"
if /I not "%CONDA_DEFAULT_ENV%"=="lusnar" (
  echo Please run: conda activate lusnar
  exit /b 1
)
python train_segnext_s.py train --data-root "D:\LuSNAR_Seg\data" --epochs 100 --batch-size 4 --size 512 --device cuda:1 %*
exit /b %errorlevel%
