"""Captura de muestras del stream para documentar pruebas en condiciones reales.

Cada captura guarda el cuadro limpio (raw/, útil para un futuro ajuste fino), el
cuadro anotado con las detecciones (annotated/, evidencia para la exposición) y
una fila en detections.csv. Al capturar indicas qué había REALMENTE en escena
(ground truth), lo que permite medir aciertos, fallos y falsas alarmas.

Teclas (ventana en vivo):
    k  capturar: en escena hay un CUCHILLO
    p  capturar: en escena hay una PISTOLA
    n  capturar: en escena NO hay arma (p. ej. el control de la tele)
    s  capturar sin etiqueta
    a  activar/desactivar captura automática al detectar un arma
    q  salir

Se registran todas las detecciones desde --log-conf (0.10) aunque en pantalla
solo se dibujen las que superan --conf: así se puede ver después si un cuchillo
"casi" se detecta y qué umbral convendría.

Uso (credenciales de la cámara en el .env del proyecto, ver camera.py):
    python src/capture_samples.py --tag cuchillo_cocina --nota "luz lámpara, 2 m"
    python src/capture_samples.py --source webcam --tag webcam_cuchillo
    python src/capture_samples.py --tag control_tv --gt none --auto
    python src/capture_samples.py --summary          # resumen de todas las sesiones

Las capturas quedan en captures/ (gitignorado: contienen imágenes de personas).
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera import FUENTES, ROOT, Camara, fuente_por_defecto
from stream_infer import DEFAULT_WEIGHTS, WEAPON_CLASSES

CAPTURES = ROOT / "captures"
GT_KEYS = {ord("k"): "knife", ord("p"): "pistol", ord("n"): "none", ord("s"): ""}
CSV_FIELDS = ["timestamp", "raw", "annotated", "reason", "ground_truth",
              "max_pistol", "max_knife", "detections"]
UMBRALES = (0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70)


def parse_args():
    p = argparse.ArgumentParser(description="Capturar muestras del stream RTSP")
    p.add_argument("--source", choices=FUENTES, default=fuente_por_defecto(),
                   help="Fuente de vídeo (por defecto CAMERA_SOURCE del .env, o tapo)")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--tag", default="", help="Nombre de la sesión (condición de prueba)")
    p.add_argument("--nota", default="", help="Condiciones: luz, distancia, ángulo...")
    p.add_argument("--gt", choices=["knife", "pistol", "none"], default=None,
                   help="Ground truth por defecto para capturas auto/interval")
    p.add_argument("--conf", type=float, default=0.35, help="Umbral de alerta/dibujo")
    p.add_argument("--log-conf", type=float, default=0.10,
                   help="Umbral mínimo para registrar detecciones en el CSV")
    p.add_argument("--auto", action="store_true", help="Empezar con captura automática")
    p.add_argument("--cooldown", type=float, default=1.5,
                   help="Segundos mínimos entre capturas automáticas")
    p.add_argument("--interval", type=float, default=0,
                   help="Capturar cada N segundos (0 = desactivado)")
    p.add_argument("--max-captures", type=int, default=0, help="Parar tras N capturas")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Resolución de inferencia. La cámara es gran angular y los "
                        "objetos quedan pequeños: 1280 detecta mejor que 640 y sigue "
                        "siendo tiempo real (~150 FPS en la RTX 5070).")
    p.add_argument("--device", default="0")
    p.add_argument("--display-scale", type=float, default=0.5)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--summary", action="store_true",
                   help="No captura: resume todas las sesiones guardadas")
    return p.parse_args()


def nueva_sesion(args, fuente):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag.strip().replace(" ", "_")
    sesion = CAPTURES / (f"{stamp}_{tag}" if tag else stamp)
    (sesion / "raw").mkdir(parents=True)
    (sesion / "annotated").mkdir()
    (sesion / "session.txt").write_text(
        f"inicio: {datetime.now().isoformat(timespec='seconds')}\n"
        f"tag: {tag}\nnota: {args.nota}\ngt_por_defecto: {args.gt}\n"
        f"pesos: {args.weights}\nconf: {args.conf}\nlog_conf: {args.log_conf}\n"
        f"imgsz: {args.imgsz}\nfuente: {fuente}\n"
    )
    return sesion


def guardar(r, r_vis, sesion, reason, gt, names, writer):
    nombre = datetime.now().strftime("%H%M%S_%f")[:-3]
    raw = sesion / "raw" / f"{nombre}.jpg"
    ann = sesion / "annotated" / f"{nombre}.jpg"
    cv2.imwrite(str(raw), r.orig_img)
    cv2.imwrite(str(ann), r_vis.plot())

    dets = [(names[int(c)], float(s)) for c, s in zip(r.boxes.cls, r.boxes.conf)]
    maximo = lambda clase: max((s for n, s in dets if n == clase), default=0.0)
    writer.writerow({
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "raw": raw.name, "annotated": ann.name, "reason": reason,
        "ground_truth": gt or "",
        "max_pistol": f"{maximo('pistol'):.3f}", "max_knife": f"{maximo('knife'):.3f}",
        "detections": ";".join(f"{n}:{s:.2f}" for n, s in dets),
    })
    return dets


def dibujar_estado(frame, auto, n, scale):
    vis = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1 else frame.copy()
    texto = (f"AUTO {'ON' if auto else 'OFF'} | capturas: {n} | "
             "k=cuchillo p=pistola n=sin arma s=guardar a=auto q=salir")
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(vis, texto, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return vis


def capturar(args):
    from ultralytics import YOLO

    if not Path(args.weights).exists():
        sys.exit(f"No existe el modelo {args.weights}.")
    model = YOLO(args.weights)
    names = model.names
    cam = Camara(args.source)
    print("Fuente:", cam.descripcion)

    sesion = nueva_sesion(args, cam.descripcion)
    print("Sesión:", sesion.relative_to(ROOT))
    f = open(sesion / "detections.csv", "w", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writeheader()

    auto, n = args.auto, 0
    ultimo_auto = ultimo_intervalo = time.time()
    ventana = "Captura de muestras (q para salir)"
    try:
        for img in cam:
            r = model.predict(img, conf=args.log_conf, imgsz=args.imgsz,
                              device=args.device, verbose=False)[0]
            # En pantalla y en la imagen anotada, solo lo que supera el umbral de alerta.
            r_vis = r[r.boxes.conf >= args.conf] if len(r.boxes) else r
            armas = {names[int(c)] for c in r_vis.boxes.cls} & WEAPON_CLASSES
            ahora = time.time()
            reason, gt = None, args.gt

            if auto and armas and ahora - ultimo_auto > args.cooldown:
                reason, ultimo_auto = "auto", ahora
            if args.interval and ahora - ultimo_intervalo >= args.interval:
                reason, ultimo_intervalo = reason or "interval", ahora

            if not args.no_display:
                cv2.imshow(ventana, dibujar_estado(r_vis.plot(), auto, n, args.display_scale))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("a"):
                    auto = not auto
                    print("Captura automática:", "ON" if auto else "OFF")
                if key in GT_KEYS:
                    reason, gt = "manual", GT_KEYS[key]

            if reason:
                dets = guardar(r, r_vis, sesion, reason, gt, names, writer)
                f.flush()
                n += 1
                resumen = ", ".join(f"{c}:{s:.2f}" for c, s in dets) or "sin detecciones"
                print(f"[{n}] {reason:<8} gt={gt or '-':<6} -> {resumen}")
                if args.max_captures and n >= args.max_captures:
                    break
    except KeyboardInterrupt:
        print("\nDetenido por el usuario.")
    finally:
        cam.cerrar()
        f.close()
        cv2.destroyAllWindows()
    print(f"\n{n} capturas guardadas en {sesion.relative_to(ROOT)}")
    print("Resumen de todas las sesiones: python src/capture_samples.py --summary")


def pct(a, b):
    return f"{100 * a / b:5.1f}% ({a}/{b})" if b else "   -"


def resumir(conf):
    filas = []
    for csv_path in sorted(CAPTURES.glob("*/detections.csv")):
        with open(csv_path) as f:
            for fila in csv.DictReader(f):
                fila["sesion"] = csv_path.parent.name
                filas.append(fila)
    etiq = [r for r in filas if r["ground_truth"]]
    if not etiq:
        sys.exit("No hay capturas con ground truth todavía (usa las teclas k/p/n).")

    def grupo(rows, gt):
        return [r for r in rows if r["ground_truth"] == gt]

    def alerta(r, t):
        return max(float(r["max_pistol"]), float(r["max_knife"])) >= t

    kn, pi, no = grupo(etiq, "knife"), grupo(etiq, "pistol"), grupo(etiq, "none")
    print(f"Capturas: {len(filas)} en total, {len(etiq)} etiquetadas "
          f"(cuchillo={len(kn)}, pistola={len(pi)}, sin arma={len(no)})\n")

    # Compensación exhaustividad / falsas alarmas en condiciones reales.
    print(f"{'umbral':<8}{'recall cuchillo':<22}{'recall pistola':<22}{'falsas alarmas':<22}")
    for t in UMBRALES:
        rk = sum(float(r["max_knife"]) >= t for r in kn)
        rp = sum(float(r["max_pistol"]) >= t for r in pi)
        fp = sum(alerta(r, t) for r in no)
        marca = "  <- actual" if abs(t - conf) < 1e-9 else ""
        print(f"{t:<8.2f}{pct(rk, len(kn)):<22}{pct(rp, len(pi)):<22}{pct(fp, len(no)):<22}{marca}")

    # Comparación entre sesiones (condiciones del entorno) al umbral actual.
    print(f"\nPor sesión (umbral {conf:.2f}):")
    print(f"{'sesión':<40}{'recall cuchillo':<22}{'falsas alarmas':<22}")
    for s in sorted({r["sesion"] for r in etiq}):
        rows = [r for r in etiq if r["sesion"] == s]
        k, nn = grupo(rows, "knife"), grupo(rows, "none")
        rk = sum(float(r["max_knife"]) >= conf for r in k)
        fp = sum(alerta(r, conf) for r in nn)
        print(f"{s:<40}{pct(rk, len(k)):<22}{pct(fp, len(nn)):<22}")


def main():
    args = parse_args()
    if args.summary:
        resumir(args.conf)
    else:
        capturar(args)


if __name__ == "__main__":
    main()
