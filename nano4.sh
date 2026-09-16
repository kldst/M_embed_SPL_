#!/bin/bash

#SBATCH --job-name=vggt		# 工作名稱
#SBATCH --partition=normal2			# 使用的 partition (normal2 = hgpn[39-41,43-46]，MaxTime 2 天)
#SBATCH --time=2-00:00:00			# 執行時間上限 (天-小時:分鐘:秒)；normal2 的上限剛好是 2 天
#SBATCH --account=MST115419	####### 請記得換成您的計畫代碼 #######
#SBATCH --nodes=1				# (-N) Maximum number of nodes to be allocated
#SBATCH --gpus-per-node=4			# Gpus per node
#SBATCH --cpus-per-task=12			# (-c) Number of cores per MPI task
# 12 = 4 ranks x (2 dataloader workers + 1 main)。num_workers=0 時 GPU 有 57% 的
# 時間在等 dataloader (batch 11.94s / data 6.94s)，所以要開 worker；但節點是
# vm.overcommit_memory=2，fork 次數不能太多（詳見 config 的 num_workers 註解）。
#SBATCH --ntasks-per-node=1			# Maximum number of tasks on each nodes

# hgpn17 的第 4 張卡壞了：nvidia-smi (NVML) 看得到 4 張，但 CUDA runtime
# 只認得 3 張 (cudaGetDeviceCount()==3，local index 3 一碰就 "No CUDA GPUs are
# available"；GPU0 的 ECC 還回報 [GPU requires reset])。torch.cuda.device_count()
# 走 NVML 所以回 4，torchrun 就開 4 個 rank → rank3 crash：
#   RuntimeError: device >= 0 && device < num_gpus ... device=, num_gpus=
# 若之後其他節點也壞，把節點名加進這行。
# 註：hgpn17 不在 normal2 的節點清單 (hgpn[39-41,43-46]) 裡，所以這行對
# normal2 是 no-op；保留是為了之後切回 dev/normal 時仍然有保護。
#SBATCH --exclude=hgpn17

# Email 通知
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=crayon715@gmail.com

set -euo pipefail

ENV_NAME="vggt"
# 這份 repo 才有 training/config/mamma_harmony4d_mask_dpt.yaml
# (/work/crayon715/vggt 是上游 clone，config 目錄裡只有 default*.yaml)
PROJECT_ROOT="/work/crayon715/Multi_SMPL_Temporal"
CONFIG="mamma_harmony4d_mask_dpt"
NPROC=4

ml load miniconda3/24.11.1
# conda 的 activate script 會踩到 set -u，先關掉再開回來
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

cd "$PROJECT_ROOT/training"

# launch.py 的 sys.path.insert(PROJECT_ROOT) 發生在 `from trainer import Trainer`
# 之後才執行，而 trainer.py 頂層就 `from training.temporal import ...`，
# 所以 repo root 必須先進 PYTHONPATH，否則 ModuleNotFoundError: No module named 'training'。
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/training${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2026-09-03 實測 hgpn01：compute node 有外網（api.wandb.ai DNS + TCP 443 通，
# 真的 wandb.init(mode="online") 也成功上傳），~/.netrc 走 home 也讀得到，
# 所以直接 online，不用再事後 `wandb sync`。
# 網路真的斷了就 `WANDB_MODE=offline sbatch nano5.sh`，事後在 login node 用
# `WANDB_ENTITY=... wandb sync <log_dir>/wandb/offline-run-*` 補上傳。
# config 的 logging.wandb_writer.mode 是 ${oc.env:WANDB_MODE,online}，
# 所以這裡設環境變數才有效（wandb.init 的 kwarg 優先權高於 WANDB_MODE 本身）。
export WANDB_MODE="${WANDB_MODE:-online}"

# 帳號的預設 entity 是 team clchen-national-yang-ming-chiao-tung-university，
# 但那個 team 不給這把 key 建 run（wandb.init → CommError "the provided API key
# cannot access this resource"）。所有 config 都是 entity: null，而 entity=None
# 對 wandb 等於「未指定」，所以這個環境變數會生效。之後在 login node
# `wandb sync` 時也要一起設，否則同樣會退回壞掉的預設 entity。
export WANDB_ENTITY="${WANDB_ENTITY:-crayon715-national-yang-ming-university}"

# --- pre-flight: 缺套件時 2 秒內帶清楚訊息離開，不要拖出 torchrun 的 ChildFailedError ---
if ! python - <<'PYEOF'
import importlib.util, sys
missing = [m for m in ("hydra", "omegaconf", "iopath", "wcmatch", "yacs",
                       "fvcore", "tensorboard", "rich", "smplx", "cv2", "trimesh")
           if importlib.util.find_spec(m) is None]
if missing:
    print("MISSING PACKAGES:", ", ".join(missing), file=sys.stderr)
    print("在 login node 執行 build_conda_env.sh 剩下的安裝步驟後再重送 job", file=sys.stderr)
    sys.exit(1)
import trainer  # 順便驗 PYTHONPATH / repo import 鏈
print("dependency check: ok")
PYEOF
then
    echo "[nano5.sh] conda env '$ENV_NAME' 依賴或 PYTHONPATH 不正確，abort。" >&2
    exit 1
fi

# --- pre-flight: GPU 健康檢查 ---------------------------------------------
# NVML 看到的張數 != CUDA runtime 真的能用的張數時，torchrun 會開出對不到卡的
# rank，最後只吐一坨 ChildFailedError。這裡逐張 probe，壞了就早退並講清楚是
# 哪個節點的哪一張，方便加到上面的 --exclude。
GPU_PROBE=$(python - <<'PYEOF'
import os, subprocess, sys

visible = os.environ.get("CUDA_VISIBLE_DEVICES")
ids = visible.split(",") if visible else None
if ids is None:
    import torch
    ids = [str(i) for i in range(torch.cuda.device_count())]

bad = []
for i in ids:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=i)
    r = subprocess.run(
        [sys.executable, "-c", "import torch; torch.zeros(8, device='cuda:0')"],
        env=env, capture_output=True, text=True)
    if r.returncode != 0:
        bad.append(i)

print("%d %d %s" % (len(ids), len(ids) - len(bad), ",".join(bad) or "-"))
PYEOF
) || { echo "[nano5.sh] GPU probe 執行失敗" >&2; exit 1; }

read -r GPU_TOTAL GPU_OK GPU_BAD <<< "$GPU_PROBE"
echo "[nano5.sh] node=$(hostname) SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?} healthy=${GPU_OK}/${GPU_TOTAL} bad=${GPU_BAD}"

if [ "$GPU_OK" -lt "$NPROC" ]; then
    echo "[nano5.sh] 只有 ${GPU_OK} 張卡能用 (需要 ${NPROC})，壞卡 local index=${GPU_BAD} @ $(hostname)。" >&2
    echo "[nano5.sh] 把 $(hostname) 加進 sbatch 的 --exclude 後重送 job。" >&2
    exit 1
fi

# config 檔存在性檢查
if [ ! -f "config/${CONFIG}.yaml" ]; then
    echo "[nano5.sh] 找不到 config/${CONFIG}.yaml (cwd=$PWD)" >&2
    exit 1
fi

echo "[nano5.sh] env=$(which python) config=${CONFIG} nproc=${NPROC} $(date)"

python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC}" \
    launch.py \
    --config "${CONFIG}"
