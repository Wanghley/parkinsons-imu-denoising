#!/bin/bash
#SBATCH --job-name=ece685_full_run
#SBATCH --output=logs/run_%j.out
#SBATCH --error=logs/run_%j.err
#SBATCH -p gpu-common
#SBATCH --account=agashelab
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

usage() {
	cat <<'EOF'
Usage:
	./run_cluster.sh [wrapper-options] [-- main.py-options]

Wrapper options:
	--exp VALUE          Experiment: all|arch|latent|noise|noise_types|hyperparam
	--epochs VALUE       Number of epochs
	--lr VALUE           Learning rate
	--batch VALUE        Batch size
	--results DIR        Base results directory
	--tag VALUE          Optional custom tag (auto tag appended if missing)
	--venv PATH          Virtualenv path (default: /hpc/home/ws186/project-685/.venv)
	--dry-run            Print command and exit
	-h, --help           Show this help

Examples:
	./run_cluster.sh --exp arch --epochs 80 --batch 128
	./run_cluster.sh --exp hyperparam --epochs 20 --tag hp_tune
	./run_cluster.sh -- --exp noise --lr 5e-4 --batch 64
EOF
}

# 1. Prevent 'ctypes' or local library conflicts on the cluster
export PYTHONNOUSERSITE=1

# 2. Create a logs directory if it doesn't exist
mkdir -p logs

# 3. Wrapper defaults
EXP="all"
EPOCHS=""
LR=""
BATCH=""
RESULTS_DIR=""
CUSTOM_TAG=""
VENV_PATH="/hpc/home/ws186/project-685/.venv"
DRY_RUN=0
EXTRA_ARGS=()

# 4. Parse wrapper options
while [[ $# -gt 0 ]]; do
	case "$1" in
		--exp)
			EXP="$2"
			shift 2
			;;
		--epochs)
			EPOCHS="$2"
			shift 2
			;;
		--lr)
			LR="$2"
			shift 2
			;;
		--batch)
			BATCH="$2"
			shift 2
			;;
		--results)
			RESULTS_DIR="$2"
			shift 2
			;;
		--tag)
			CUSTOM_TAG="$2"
			shift 2
			;;
		--venv)
			VENV_PATH="$2"
			shift 2
			;;
		--dry-run)
			DRY_RUN=1
			shift
			;;
		--)
			shift
			EXTRA_ARGS+=("$@")
			break
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "Unknown option: $1"
			usage
			exit 1
			;;
	esac
done

# 5. Activate the virtual environment
source "${VENV_PATH}/bin/activate"

# 6. Generate a dynamic tag based on hardware and start time
# This detects the GPU (e.g., GeForce_RTX_2080_Ti) and appends a short timestamp.
# Fallback keeps wrapper functional on non-NVIDIA environments.
if command -v nvidia-smi >/dev/null 2>&1; then
	GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr ' ' '_' | sed 's/[^a-zA-Z0-9_]//g' | head -n 1)
else
	GPU_NAME="cpu_or_non_nvidia"
fi
RUN_TIME=$(date +%H%M)
DYNAMIC_TAG="${GPU_NAME}_${RUN_TIME}"

if [[ -z "${CUSTOM_TAG}" ]]; then
	FINAL_TAG="${DYNAMIC_TAG}"
else
	FINAL_TAG="${CUSTOM_TAG}_${DYNAMIC_TAG}"
fi

# 7. Build the python command
CMD=(python main.py --exp "${EXP}" --tag "${FINAL_TAG}")

if [[ -n "${EPOCHS}" ]]; then
	CMD+=(--epochs "${EPOCHS}")
fi
if [[ -n "${LR}" ]]; then
	CMD+=(--lr "${LR}")
fi
if [[ -n "${BATCH}" ]]; then
	CMD+=(--batch "${BATCH}")
fi
if [[ -n "${RESULTS_DIR}" ]]; then
	CMD+=(--results "${RESULTS_DIR}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
	CMD+=("${EXTRA_ARGS[@]}")
fi

# 8. Echo and run
echo "Launching: ${CMD[*]}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
	exit 0
fi

"${CMD[@]}"
