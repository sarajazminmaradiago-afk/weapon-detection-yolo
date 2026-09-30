"""Entrenamiento del detector YOLO sobre el conjunto Sohas weapon detection.

Uso:
    python src/train.py                         # entrenamiento completo (por defecto)
    python src/train.py --model yolov8n.pt --epochs 1 --name smoke   # prueba rápida
    # ajuste fino con capturas locales (ver build_local_dataset.py):
    python src/train.py --model runs/yolov8s_sohas/weights/best.pt \
        --data datasets/local/finetune.yaml --imgsz 1280 --batch 8 --epochs 30 \
        --optimizer AdamW --lr0 0.0005 --name ft_local

Los pesos preentrenados en COCO se descargan automáticamente la primera vez.
Los resultados (pesos, curvas, matriz de confusión) quedan en runs/<name>/.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

# Rutas relativas a la raíz del proyecto (este fichero vive en src/).
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "src" / "sohas.yaml"


def parse_args():
    p = argparse.ArgumentParser(description="Entrenar YOLO en Sohas weapon detection")
    p.add_argument("--data", default=str(DATA),
                   help="YAML del dataset. Por defecto src/sohas.yaml")
    p.add_argument("--model", default="yolov8s.pt",
                   help="Pesos de partida (yolov8n/s/m...). Por defecto yolov8s.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16,
                   help="Tamaño de lote. -1 para autobatch según VRAM.")
    p.add_argument("--device", default="0", help="GPU id ('0') o 'cpu'")
    p.add_argument("--name", default="yolov8s_sohas",
                   help="Nombre de la corrida bajo runs/")
    p.add_argument("--patience", type=int, default=20,
                   help="Épocas sin mejora antes de parada temprana")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--optimizer", default="auto",
                   help="auto elige solo el optimizador y la tasa (e ignora --lr0)")
    p.add_argument("--lr0", type=float, default=None,
                   help="Tasa de aprendizaje inicial; para ajuste fino conviene una baja")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=str(ROOT / "runs"),
        patience=args.patience,
        seed=args.seed,
        optimizer=args.optimizer,
        **({"lr0": args.lr0} if args.lr0 is not None else {}),
        plots=True,          # genera curvas P/R y matriz de confusión
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
