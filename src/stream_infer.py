"""Despliegue del detector YOLO en tiempo real sobre la cámara Tapo o una webcam USB.

Detecta armas cuadro a cuadro y emite una ALERTA cuando aparece una pistola o un
cuchillo por encima del umbral de confianza.

Las fuentes de vídeo (y las credenciales, en el .env del proyecto) están en camera.py.

Uso:
    python src/stream_infer.py                       # ventana en vivo
    python src/stream_infer.py --source webcam       # webcam USB en lugar de la Tapo
    python src/stream_infer.py --conf 0.5 --save     # umbral alto + graba vídeo
    python src/stream_infer.py --no-display           # sin ventana (solo alertas)
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from camera import FUENTES, ROOT, Camara, fuente_por_defecto

DEFAULT_WEIGHTS = ROOT / "runs" / "yolov8s_sohas" / "weights" / "best.pt"
WEAPON_CLASSES = {"pistol", "knife"}   # clases que disparan alerta


def parse_args():
    p = argparse.ArgumentParser(description="Detección de armas en stream RTSP")
    p.add_argument("--source", choices=FUENTES, default=fuente_por_defecto(),
                   help="Fuente de vídeo (por defecto CAMERA_SOURCE del .env, o tapo)")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--conf", type=float, default=0.35,
                   help="Umbral de confianza. Súbelo para menos falsas alarmas.")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Resolución de inferencia. La cámara es gran angular y los "
                        "objetos quedan pequeños: 1280 detecta mejor que 640 y sigue "
                        "siendo tiempo real (~150 FPS en la RTX 5070).")
    p.add_argument("--device", default="0")
    p.add_argument("--vid-stride", type=int, default=1,
                   help="Procesar 1 de cada N cuadros (sube si va justo de cómputo)")
    p.add_argument("--no-display", action="store_true", help="Sin ventana de vídeo")
    p.add_argument("--save", action="store_true",
                   help="Guardar el vídeo anotado en runs/stream/")
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO
    import cv2

    if not Path(args.weights).exists():
        sys.exit(f"No existe el modelo {args.weights}. Entrena primero (src/train.py).")

    model = YOLO(args.weights)
    names = model.names
    cam = Camara(args.source)
    print("Fuente:", cam.descripcion)

    writer = None
    ultimo_aviso = 0.0
    ventana = f"Detección de armas - {args.source} (q para salir)"
    try:
        for i, img in enumerate(cam):
            if i % args.vid_stride:
                continue
            r = model.predict(img, conf=args.conf, imgsz=args.imgsz,
                              device=args.device, verbose=False)[0]
            frame = r.plot()  # BGR con las cajas dibujadas

            detectadas = {names[int(c)] for c in r.boxes.cls} if r.boxes is not None else set()
            armas = detectadas & WEAPON_CLASSES
            if armas and time.time() - ultimo_aviso > 1.0:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] ⚠️  ALERTA: {', '.join(sorted(armas))} detectada(s)")
                ultimo_aviso = time.time()

            if args.save:
                if writer is None:
                    out_dir = ROOT / "runs" / "stream"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_dir / "stream.mp4"), fourcc,
                                             cam.fps / args.vid_stride, (w, h))
                writer.write(frame)

            if not args.no_display:
                try:
                    cv2.imshow(ventana, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    print("Sin entorno gráfico; usa --no-display o --save.")
                    break
    except KeyboardInterrupt:
        print("\nDetenido por el usuario.")
    finally:
        cam.cerrar()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
