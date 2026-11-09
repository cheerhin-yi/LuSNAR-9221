@echo off
setlocal
cd /d "%~dp0"
python -c "import os,sys; p=os.environ.get('CONDA_PREFIX',''); print('Python:',sys.executable); sys.exit(0 if p else 1)"
if errorlevel 1 (
  echo Activate your environment first: conda activate "F:\Conda\envs\lusnar"
  exit /b 1
)
python train_segformer_b3.py train --data-root "D:\LuSNAR_Seg\data" --device cuda:0 %*
exit /b %errorlevel%
