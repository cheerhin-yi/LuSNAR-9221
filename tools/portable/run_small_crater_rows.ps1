# Four paper rows. Scene5 I1a is the only gate. Does not score Moon_2/4/9.
$ErrorActionPreference = "Stop"
Set-Location D:\LuSNAR_Seg
$py = "F:\Conda\envs\lusnar\python.exe"
$best = "D:\LuSNAR_Seg\work_dirs\train_ROW_BLV_20k\best.pth"
$env:USE_LIBUV = "0"
$env:MASTER_ADDR = "127.0.0.1"
$env:MASTER_PORT = "29641"
$env:OMP_NUM_THREADS = "2"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue

function Resolve-Ckpt([string]$dir) {
    $best = Join-Path $dir "best.pth"
    $last = Join-Path $dir "last.pth"
    if (Test-Path $best) { return $best }
    if (Test-Path $last) { return $last }
    throw "no checkpoint in $dir"
}

function Score-Row($tag, $ckpt) {
    Write-Output "SCORE $tag"
    & $py -u tools\portable\lusnar_small_crater_e0.py --checkpoint $ckpt --tag $tag --project-root D:\LuSNAR_Seg
    if ($LASTEXITCODE -ne 0) { throw "score $tag failed" }
}

function Train-Row([string]$name, [string[]]$extra) {
    Write-Output "TRAIN $name"
    & $py -u -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 29641 tools\portable\lusnar_row_blv_ddp.py --recipe M_ROW_BLV --run-name $name --iters 19000 --batch 48 --workers 4 --amp bf16 --crop 512 --lr 6e-5 --val-every 500 --resume $best --reset-optimizer --list-dir D:\LuSNAR_Seg\data\lists_small_crater --project-root D:\LuSNAR_Seg --data-root D:\LuSNAR_Seg\data --train-subdir train @extra
    if ($LASTEXITCODE -ne 0) { throw "train $name failed" }
}

Score-Row "E0" $best
Train-Row "sc_E1_residual" @("--plain-residual")
Score-Row "E1" (Resolve-Ckpt "D:\LuSNAR_Seg\work_dirs\sc_E1_residual")
Train-Row "sc_E2_local" @("--local-margin")
Score-Row "E2" (Resolve-Ckpt "D:\LuSNAR_Seg\work_dirs\sc_E2_local")
Train-Row "sc_E3_both" -extra @("--plain-residual", "--local-margin")
Score-Row "E3" (Resolve-Ckpt "D:\LuSNAR_Seg\work_dirs\sc_E3_both")
Write-Output "SMALL_CRATER_ROWS_DONE"
