"""Etiquetador de capturas en formato YOLO, con pre-etiquetas del modelo.

Recorre las imágenes raw/ de una o varias sesiones de captures/, propone las
cajas que detecta el modelo actual y deja corregirlas. Guarda una etiqueta YOLO
por imagen en captures/<sesión>/labels/. Un fichero vacío significa "imagen
revisada, sin objetos", que también es información útil (ejemplo negativo).

Qué etiquetar:
    cuchillos -> knife          celular -> smartphone
    billetera -> monedero       tarjeta -> tarjeta      billete -> billete
    control de la tele, mando, etc. -> SIN caja (el modelo debe ignorarlos)

Controles:
    ratón      arrastrar = nueva caja con la clase actual; clic = seleccionar caja
    1-6        elegir clase (si hay una caja seleccionada, le cambia la clase)
    x / Supr   borrar la caja seleccionada (o la última)
    c          borrar todas las cajas
    espacio    guardar y pasar a la siguiente
    b          volver a la anterior (sin guardar la actual)
    q          guardar y salir          Esc   salir sin guardar la actual

Uso:
    python src/label_captures.py captures/20260930_*_train_*
    python src/label_captures.py captures/<sesión> --redo   # revisar también las ya hechas
    python src/label_captures.py captures/*_cuchillos --ignorar billete   # no proponer billetes
"""
import argparse
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
NAMES = yaml.safe_load(open(ROOT / "src" / "sohas.yaml"))["names"]
COLORES = {0: (0, 0, 255), 1: (255, 160, 0), 2: (0, 200, 255),
           3: (200, 0, 200), 4: (0, 180, 0), 5: (255, 255, 0)}
VENTANA = "Etiquetador (espacio = guardar y siguiente)"
FUENTE = cv2.FONT_HERSHEY_SIMPLEX
TECLA_SUPR = 0xFFFF


def ruta_label(img):
    return img.parent.parent / "labels" / f"{img.stem}.txt"


def leer_label(path, w, h):
    """YOLO normalizado -> lista de [clase, x1, y1, x2, y2] en píxeles."""
    cajas = []
    for linea in path.read_text().splitlines():
        partes = linea.split()
        if len(partes) != 5:
            continue
        c, xc, yc, bw, bh = int(partes[0]), *map(float, partes[1:])
        cajas.append([c, (xc - bw / 2) * w, (yc - bh / 2) * h,
                      (xc + bw / 2) * w, (yc + bh / 2) * h])
    return cajas


def escribir_label(path, cajas, w, h):
    path.parent.mkdir(exist_ok=True)
    lineas = []
    for c, x1, y1, x2, y2 in cajas:
        x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
        y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lineas.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                      f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    path.write_text("".join(f"{l}\n" for l in lineas))


def prelabels(model, frame, args):
    r = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                      device=args.device, verbose=False)[0]
    return [[int(c), *xyxy] for c, xyxy in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())
            if NAMES[int(c)] not in args.ignorar]


def area(caja):
    return (caja[3] - caja[1]) * (caja[4] - caja[2])


estado = {"cajas": [], "sel": None, "ini": None, "act": None, "clase": 2, "escala": 1.0}


def raton(evento, x, y, flags, _):
    s = estado["escala"]
    X, Y = x / s, y / s
    if evento == cv2.EVENT_LBUTTONDOWN:
        estado["ini"] = estado["act"] = (X, Y)
    elif evento == cv2.EVENT_MOUSEMOVE and estado["ini"]:
        estado["act"] = (X, Y)
    elif evento == cv2.EVENT_LBUTTONUP and estado["ini"]:
        (x0, y0), estado["ini"] = estado["ini"], None
        if abs(X - x0) > 4 / s and abs(Y - y0) > 4 / s:
            estado["cajas"].append([estado["clase"], min(x0, X), min(y0, Y), max(x0, X), max(y0, Y)])
            estado["sel"] = len(estado["cajas"]) - 1
        else:
            dentro = [i for i, (_, x1, y1, x2, y2) in enumerate(estado["cajas"])
                      if x1 <= X <= x2 and y1 <= Y <= y2]
            estado["sel"] = min(dentro, key=lambda i: area(estado["cajas"][i])) if dentro else None


def dibujar(frame, info):
    s = estado["escala"]
    vis = cv2.resize(frame, None, fx=s, fy=s) if s != 1 else frame.copy()
    for i, (c, x1, y1, x2, y2) in enumerate(estado["cajas"]):
        color = COLORES[c]
        p1, p2 = (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s))
        seleccionada = i == estado["sel"]
        cv2.rectangle(vis, p1, p2, (255, 255, 255) if seleccionada else color, 3 if seleccionada else 2)
        cv2.putText(vis, NAMES[c], (p1[0], max(60, p1[1] - 6)), FUENTE, 0.6, color, 2)
    if estado["ini"] and estado["act"]:
        (x0, y0), (x1, y1) = estado["ini"], estado["act"]
        cv2.rectangle(vis, (int(x0 * s), int(y0 * s)), (int(x1 * s), int(y1 * s)),
                      COLORES[estado["clase"]], 1)

    leyenda = "  ".join(f"{k + 1}:{v}" for k, v in NAMES.items())
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(vis, info, (8, 20), FUENTE, 0.55, (255, 255, 255), 1)
    cv2.putText(vis, f"clase: {NAMES[estado['clase']]}   |   {leyenda}   |   "
                "x=borrar c=vaciar espacio=siguiente b=anterior q=salir",
                (8, 42), FUENTE, 0.5, COLORES[estado["clase"]], 1)
    return vis


def parse_args():
    p = argparse.ArgumentParser(description="Etiquetar capturas (formato YOLO)")
    p.add_argument("sesiones", nargs="+", help="Carpetas de sesión en captures/")
    p.add_argument("--redo", action="store_true", help="Revisar también las ya etiquetadas")
    p.add_argument("--weights", default=str(ROOT / "runs" / "yolov8s_sohas" / "weights" / "best.pt"),
                   help="Modelo para las pre-etiquetas")
    p.add_argument("--no-prelabel", action="store_true", help="Empezar sin cajas propuestas")
    p.add_argument("--conf", type=float, default=0.25, help="Umbral de las pre-etiquetas")
    p.add_argument("--ignorar", nargs="+", default=[], choices=list(NAMES.values()),
                   metavar="CLASE", help="Clases que no se proponen (p. ej. billete)")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--max-width", type=int, default=1600, help="Ancho máximo de la ventana")
    return p.parse_args()


def main():
    args = parse_args()
    imagenes = sorted(img for s in args.sesiones for img in (Path(s) / "raw").glob("*.jpg"))
    if not imagenes:
        sys.exit("No hay imágenes en <sesión>/raw/")
    pendientes = [i for i, img in enumerate(imagenes) if args.redo or not ruta_label(img).exists()]
    if not pendientes:
        sys.exit("Todas las imágenes ya están etiquetadas (usa --redo para revisarlas).")

    model = None
    if not args.no_prelabel:
        from ultralytics import YOLO
        model = YOLO(args.weights)

    cv2.namedWindow(VENTANA, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(VENTANA, raton)
    i, salir = pendientes[0], False
    while 0 <= i < len(imagenes) and not salir:
        img = imagenes[i]
        frame = cv2.imread(str(img))
        h, w = frame.shape[:2]
        lab = ruta_label(img)
        cajas = leer_label(lab, w, h) if lab.exists() else (prelabels(model, frame, args) if model else [])
        estado.update(cajas=cajas, sel=None, ini=None, act=None, escala=min(1.0, args.max_width / w))
        hechas = sum(ruta_label(p).exists() for p in imagenes)
        info = f"{img.parent.parent.name}/{img.name}   [{i + 1}/{len(imagenes)}]   etiquetadas: {hechas}"

        while True:
            cv2.imshow(VENTANA, dibujar(frame, info))
            k = cv2.waitKeyEx(20)
            if cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1:
                salir = True
                break
            if k == -1:
                continue
            if ord("1") <= k <= ord(str(len(NAMES))):
                estado["clase"] = k - ord("1")
                if estado["sel"] is not None:
                    estado["cajas"][estado["sel"]][0] = estado["clase"]
            elif k in (ord("x"), 8, TECLA_SUPR):
                if estado["cajas"]:
                    estado["cajas"].pop(estado["sel"] if estado["sel"] is not None else -1)
                estado["sel"] = None
            elif k == ord("c"):
                estado["cajas"], estado["sel"] = [], None
            elif k in (32, 13):
                escribir_label(lab, estado["cajas"], w, h)
                i += 1
                break
            elif k == ord("b"):
                i = max(0, i - 1)
                break
            elif k == ord("q"):
                escribir_label(lab, estado["cajas"], w, h)
                salir = True
                break
            elif k == 27:
                salir = True
                break

    cv2.destroyAllWindows()
    hechas = sum(ruta_label(p).exists() for p in imagenes)
    print(f"Etiquetadas {hechas}/{len(imagenes)} imágenes.")


if __name__ == "__main__":
    main()
