

 %%writefile train_accelerate.sh
 #!/usr/bin/env bash
 set -euo pipefail
 # ===============================================================
 # Optimal Dual-Head RM Training on 2× T4 (Kaggle) with Accelerate 1.9.0
 # ===============================================================
 # Usage: ./train_accelerate.sh [CONFIG_PATH]
 # Default: configs/rm_train_config.yaml
 # ---------------------------------------------------------------

 CONFIG=${1:-"configs/rm_train_config.yaml"}
 RUN_DIR="runs"
 STAMP=$(date +%Y%m%d_%H%M%S)
 OUTLOG="${RUN_DIR}/train_${STAMP}.log"

 mkdir -p "${RUN_DIR}"

 echo "[accelerate] 🚀 Launching distributed training with 2x T4 GPUs"
 echo "[accelerate] Config: ${CONFIG}"
 echo "[accelerate] Logfile: ${OUTLOG}"
 echo "[accelerate] GPUs visible: $(nvidia-smi --query-gpu=name,index --format=csv,noheader,nounits)"

 # ---------------------------------------------------------------
 # Environment setup (robust for Kaggle multi-GPU)
 # ---------------------------------------------------------------
 export PYTHONPATH="${PWD}:$PYTHONPATH"
 echo "[accelerate] PYTHONPATH: ${PYTHONPATH}"

 export CUDA_VISIBLE_DEVICES=0,1
 export TOKENIZERS_PARALLELISM=false
 export NCCL_DEBUG=INFO
 export NCCL_IB_DISABLE=1
 export NCCL_P2P_DISABLE=1        # disable P2P on Kaggle T4 to avoid hangs
 export NCCL_SOCKET_IFNAME=lo
 export MASTER_PORT=$((12000 + RANDOM % 1000))   # avoid port conflicts
 export TORCH_DISTRIBUTED_DEBUG=INFO
 export PYTHONUNBUFFERED=1        # real-time log flushing
 export TF_CPP_MIN_LOG_LEVEL=3           # ✅ silences cuFFT/cuDNN/BLAS noise
 export PYTHONWARNINGS="ignore"          # ✅ suppresses all non-critical warnings (like Pydantic)
 export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  # ✅ stable protobuf backend
 unset LD_PRELOAD        # Avoid Kaggle-provided LD_PRELOAD interfering with NCCL
 export WANDB_DISABLED=true   # disable wandb auto-logging (if not using it)

 # ---------------------------------------------------------------
 # Launch with Accelerate (unbuffered + safe for Kaggle)
 # ---------------------------------------------------------------
 echo "[accelerate] ✅ Starting Accelerate Launch..."
 PYTHONUNBUFFERED=1 accelerate launch \
   --config_file configs/accelerate_config.yaml \
   --num_processes 2 \
   --multi_gpu \
   --main_process_port ${MASTER_PORT} \
   src/training/training_dual_rm.py "${CONFIG}" 2>&1 | tee "${OUTLOG}"

 echo "[accelerate] ✅ Training finished at $(date -u)"
 echo "[accelerate] Final GPU stats:"
 nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits

 ! chmod +x train_accelerate.sh
 ! ls -l train_accelerate.sh
 ! cat train_accelerate.sh
