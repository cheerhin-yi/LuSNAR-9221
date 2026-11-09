@echo off
setlocal
cd /d "%~dp0"
if not defined CONDA_PREFIX (
  echo First activate: conda activate "F:\Conda\envs\lusnar"
  exit /b 1
)
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
python train_segformer_b3.py train --check-only --device cuda:0 %*
exit /b %errorlevel%
