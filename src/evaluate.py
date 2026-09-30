"""Evaluación del detector YOLO entrenado sobre la partición de prueba de Sohas.

Uso:
    python src/evaluate.py --weights runs/yolov8s_sohas/weights/best.pt

Reporta mAP@0.5, mAP@0.5:0.95, precisión y recall por clase, y guarda la
matriz de confusión (clave para analizar los falsos positivos sobre los
objetos confundibles: smartphone, monedero, billete, tarjeta).
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "src" / "sohas.yaml"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluar YOLO en Sohas weapon detection")
    p.add_argument("--weights", required=True, help="Ruta a best.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="yolov8s_sohas_eval")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(
        data=str(DATA),
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        project=str(ROOT / "runs"),
        plots=True,
        exist_ok=True,
    )

    names = model.names
    print("\n===== Métricas globales =====")
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"Precisión    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")

    print("\n===== Métricas por clase =====")
    print(f"{'clase':<12}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for i, c in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = (
            metrics.box.p[i], metrics.box.r[i],
            metrics.box.ap50[i], metrics.box.ap[i],
        )
        print(f"{names[c]:<12}{p:>8.3f}{r:>8.3f}{ap50:>8.3f}{ap:>10.3f}")

    print(f"\nMatriz de confusión y curvas guardadas en: {metrics.save_dir}")


if __name__ == "__main__":
    main()
