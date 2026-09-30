"""Construye el conjunto local (capturas etiquetadas) para el ajuste fino.

Cada sesión de captures/ se asigna a entrenamiento (--train) o a evaluación
(--test), nunca a ambos: evaluar con imágenes vistas en el entrenamiento
inflaría los resultados. Solo entran las imágenes ya etiquetadas con
label_captures.py.

Genera en datasets/local/ (gitignorado; las imágenes son enlaces a captures/):
    train/ val/ test/     images/ + labels/
    finetune.yaml         train: Sohas train + local train (sobremuestreado)
                          val:   Sohas test + local val
    test.yaml             solo el test local (para eval_local.py)

Las capturas locales se sobremuestrean (--copias) porque son pocas frente a las
~5000 imágenes de Sohas; mezclar ambas evita que el modelo olvide lo aprendido.

Uso:
    python src/build_local_dataset.py --train "captures/*_train_*" --test "captures/*_test_*"
"""
import argparse
import glob
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SOHAS = yaml.safe_load(open(ROOT / "src" / "sohas.yaml"))
NAMES = SOHAS["names"]


def parse_args():
    p = argparse.ArgumentParser(description="Construir el dataset local para el ajuste fino")
    p.add_argument("--train", nargs="+", required=True, help="Sesiones (o patrones) de entrenamiento")
    p.add_argument("--test", nargs="+", required=True, help="Sesiones (o patrones) de evaluación")
    p.add_argument("--copias", type=int, default=5,
                   help="Veces que se repite cada imagen local de entrenamiento")
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="Fracción de las imágenes de entrenamiento reservada para validación")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "datasets" / "local"))
    return p.parse_args()


def sesiones(patrones):
    return sorted({Path(s).resolve() for pat in patrones for s in glob.glob(pat) if Path(s).is_dir()})


def etiquetadas(dirs):
    pares = []
    for ses in dirs:
        for img in sorted((ses / "raw").glob("*.jpg")):
            lab = ses / "labels" / f"{img.stem}.txt"
            if lab.exists():
                pares.append((img, lab))
    return pares


def volcar(pares, destino, copias=1):
    (destino / "images").mkdir(parents=True)
    (destino / "labels").mkdir()
    for img, lab in pares:
        base = f"{img.parent.parent.name}__{img.stem}"
        for k in range(copias):
            nombre = base if copias == 1 else f"{base}__c{k}"
            (destino / "images" / f"{nombre}.jpg").symlink_to(img)
            shutil.copy(lab, destino / "labels" / f"{nombre}.txt")


def contar(pares):
    clases, vacias = Counter(), 0
    for _, lab in pares:
        lineas = [l for l in lab.read_text().splitlines() if l.strip()]
        vacias += not lineas
        clases.update(NAMES[int(l.split()[0])] for l in lineas)
    return clases, vacias


def main():
    args = parse_args()
    ses_train, ses_test = sesiones(args.train), sesiones(args.test)
    comunes = set(ses_train) & set(ses_test)
    if comunes:
        sys.exit(f"Estas sesiones están en train y en test a la vez: {[s.name for s in comunes]}")
    train, test = etiquetadas(ses_train), etiquetadas(ses_test)
    if not train or not test:
        sys.exit(f"Faltan imágenes etiquetadas (train={len(train)}, test={len(test)}). "
                 "Etiqueta primero con src/label_captures.py.")

    random.Random(args.seed).shuffle(train)
    n_val = max(1, round(len(train) * args.val_frac))
    val, train = train[:n_val], train[n_val:]

    out = Path(args.out)
    if out.exists():
        if not (out / "finetune.yaml").exists():
            sys.exit(f"{out} existe y no parece generado por este script; no lo borro.")
        shutil.rmtree(out)
    volcar(train, out / "train", args.copias)
    volcar(val, out / "val")
    volcar(test, out / "test")

    sohas_root = Path(SOHAS["path"])
    (out / "finetune.yaml").write_text(yaml.safe_dump({
        "train": [str(sohas_root / SOHAS["train"]), str(out / "train" / "images")],
        "val": [str(sohas_root / SOHAS["val"]), str(out / "val" / "images")],
        "names": NAMES,
    }, allow_unicode=True, sort_keys=False))
    # Ultralytics exige la clave train aunque solo se evalúe.
    (out / "test.yaml").write_text(yaml.safe_dump({
        "train": str(out / "test" / "images"),
        "val": str(out / "test" / "images"),
        "names": NAMES,
    }, allow_unicode=True, sort_keys=False))

    print("Sesiones de entrenamiento:", ", ".join(s.name for s in ses_train))
    print("Sesiones de evaluación   :", ", ".join(s.name for s in ses_test))
    for nombre, pares, copias in (("train", train, args.copias), ("val", val, 1), ("test", test, 1)):
        clases, vacias = contar(pares)
        print(f"{nombre:<5}: {len(pares):4d} imágenes (x{copias}), sin objetos: {vacias:3d}, "
              f"cajas: {dict(clases)}")
    print(f"\nListo: {out / 'finetune.yaml'}")


if __name__ == "__main__":
    main()
