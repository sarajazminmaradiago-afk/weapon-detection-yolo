# Entorno torchvision Mask R-CNN R50-FPN.  Uso:  source activate.sh
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_ROOT/.venv/bin/activate"
echo "Entorno torchvision listo (torch 2.11+cu128, sm_120 nativo)"
