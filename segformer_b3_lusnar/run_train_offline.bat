@echo off
setlocal
cd /d "%~dp0"
python -c "import os,sys; print('Python:',sys.executable); sys.exit(0 if os.environ.get('CONDA_PREFIX') else 1)"
if errorlevel 1 (
  echo First activate: conda activate "F:\Conda\envs\lusnar"
  exit /b 1
)
python train_segformer_b3.py train --from-scratch --device cuda:0 %*
exit /b %errorlevel%
