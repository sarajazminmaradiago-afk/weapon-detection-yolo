"""Compara modelos sobre el test local (capturas etiquetadas que ningún modelo vio).

Métricas:
  - Por caja (Ultralytics): mAP@0.5 por clase y matriz de confusión.
  - Por imagen, que es lo que importa para una alarma:
      alarma con arma   de las imágenes con pistola/cuchillo, en cuántas salta la alarma
      clase correcta    ... y en cuántas además acierta el tipo de arma
      alarma sin arma   de las imágenes sin armas, en cuántas salta por error
  - Con --sohas, mAP@0.5 en el test de Sohas para comprobar que no olvidó lo aprendido.

Uso:
    python src/eval_local.py --weights runs/yolov8s_sohas/weights/best.pt \\
                                       runs/ft_local/weights/best.pt --sohas
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
NAMES = yaml.safe_load(open(ROOT / "src" / "sohas.yaml"))["names"]
ARMAS = {"pistol", "knife"}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluar modelos en el test local")
    p.add_argument("--weights", nargs="+", required=True)
    p.add_argument("--data-dir", default=str(ROOT / "datasets" / "local"))
    p.add_argument("--conf", type=float, default=0.35, help="Umbral de alarma")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--sohas", action="store_true", help="Evaluar también en el test de Sohas")
    return p.parse_args()


def nombre_modelo(w):
    return Path(w).parent.parent.name if Path(w).parent.name == "weights" else Path(w).stem


def por_imagen(model, test_dir, args):
    r = dict(con_arma=0, alarma_con=0, clase_ok=0, sin_arma=0, alarma_sin=0)
    for img in sorted((test_dir / "images").glob("*.jpg")):
        lab = test_dir / "labels" / f"{img.stem}.txt"
        gt = {NAMES[int(l.split()[0])] for l in lab.read_text().splitlines() if l.strip()} & ARMAS
        pred = model.predict(str(img), conf=args.conf, imgsz=args.imgsz,
                             device=args.device, verbose=False)[0]
        armas = {model.names[int(c)] for c in pred.boxes.cls} & ARMAS
        if gt:
            r["con_arma"] += 1
            r["alarma_con"] += bool(armas)
            r["clase_ok"] += bool(gt & armas)
        else:
            r["sin_arma"] += 1
            r["alarma_sin"] += bool(armas)
    return r


def pct(a, b):
    return f"{100 * a / b:5.1f}% ({a}/{b})" if b else "    -"


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    filas, por_clase = [], {}
    for w in args.weights:
        nombre = nombre_modelo(w)
        model = YOLO(w)
        # Las gráficas de validación contienen imágenes de las capturas: van a una
        # carpeta gitignorada (runs/eval_local_*).
        m = model.val(data=str(data_dir / "test.yaml"), imgsz=args.imgsz, device=args.device,
                      project=str(ROOT / "runs"), name=f"eval_local_{nombre}",
                      exist_ok=True, plots=True, verbose=False)
        por_clase[nombre] = {model.names[c]: m.box.ap50[i] for i, c in enumerate(m.box.ap_class_index)}
        fila = dict(nombre=nombre, map50=m.box.map50, **por_imagen(model, data_dir / "test", args))
        if args.sohas:
            s = model.val(data=str(ROOT / "src" / "sohas.yaml"), imgsz=args.imgsz, device=args.device,
                          project=str(ROOT / "runs"), name=f"eval_sohas_{nombre}",
                          exist_ok=True, plots=False, verbose=False)
            fila["sohas"] = s.box.map50
        filas.append(fila)

    print(f"\nTest local ({args.data_dir}), umbral de alarma {args.conf}, imgsz {args.imgsz}\n")
    cab = f"{'modelo':<18}{'mAP50 local':>12}  {'alarma con arma':<18}{'clase correcta':<18}{'alarma sin arma':<18}"
    print(cab + ("  mAP50 Sohas" if args.sohas else ""))
    for f in filas:
        linea = (f"{f['nombre']:<18}{f['map50']:>12.3f}  {pct(f['alarma_con'], f['con_arma']):<18}"
                 f"{pct(f['clase_ok'], f['con_arma']):<18}{pct(f['alarma_sin'], f['sin_arma']):<18}")
        print(linea + (f"  {f['sohas']:11.3f}" if args.sohas else ""))

    clases = sorted({c for d in por_clase.values() for c in d})
    print(f"\nmAP50 por clase (test local):\n{'modelo':<18}" + "".join(f"{c:>12}" for c in clases))
    for nombre, d in por_clase.items():
        print(f"{nombre:<18}" + "".join(f"{d[c]:>12.3f}" if c in d else f"{'-':>12}" for c in clases))
    print("\nMatrices de confusión en runs/eval_local_<modelo>/")


if __name__ == "__main__":
    main()
