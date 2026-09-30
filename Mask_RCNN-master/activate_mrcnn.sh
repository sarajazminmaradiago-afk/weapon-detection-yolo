# Entorno Mask R-CNN sobre TF 2.x + GPU Blackwell.  Uso:  source activate_mrcnn.sh
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_ROOT/.venv/bin/activate"
_SP="$_ROOT/.venv/lib/python3.10/site-packages"
# TF no localiza solo las libs CUDA de las ruedas pip:
export LD_LIBRARY_PATH="$(find "$_SP/nvidia" -name '*.so*' -printf '%h\n' 2>/dev/null | sort -u | tr '\n' ':')${LD_LIBRARY_PATH:-}"
# ptxas, necesario para el JIT desde PTX en sm_120:
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$_SP/nvidia/cuda_nvcc"
export PATH="$_SP/nvidia/cuda_nvcc/bin:$PATH"
# El repo usa la API de Keras 2, no Keras 3:
export TF_USE_LEGACY_KERAS=1
export TF_CPP_MIN_LOG_LEVEL=2
echo "Entorno Mask R-CNN listo (TF2 + Keras2 legacy + GPU sm_120)"
