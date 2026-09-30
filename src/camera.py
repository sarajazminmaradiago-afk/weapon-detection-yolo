"""Fuentes de vídeo del proyecto: cámara Tapo (RTSP) o webcam USB.

Las credenciales y ajustes viven en el fichero .env de la raíz del proyecto
(gitignorado; ver .env.example). Las variables de entorno ya definidas tienen
prioridad sobre el .env.

Tapo:
    TAPO_IP, TAPO_USER, TAPO_PASSWORD   obligatorias (cuenta de cámara de la app Tapo)
    TAPO_PORT                           por defecto 554
    TAPO_STREAM                         stream1 (HD) o stream2 (720p), por defecto stream1
    WEAPON_RTSP_URL                     alternativa: URL RTSP completa (ignora las anteriores)
Webcam:
    WEBCAM_INDEX                        índice de /dev/videoN, por defecto 0
    WEBCAM_WIDTH, WEBCAM_HEIGHT         por defecto 1920x1080
General:
    CAMERA_SOURCE                       fuente por defecto: tapo o webcam (por defecto tapo)

Se abre la cámara con OpenCV en lugar de dejárselo a Ultralytics porque este
abre las webcams a su resolución por defecto (640x360) y con la cámara gran
angular los objetos quedan demasiado pequeños.
"""
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

import cv2

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
FUENTES = ("tapo", "webcam")

# Los avisos de OpenCV/FFmpeg pueden incluir la URL con la contraseña.
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)


def cargar_env(path=ENV_FILE):
    """Carga KEY=VALUE del .env sin sobrescribir variables ya definidas."""
    if not path.exists():
        return
    for linea in path.read_text().splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, valor = linea.split("=", 1)
        os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


def fuente_por_defecto():
    cargar_env()
    return os.environ.get("CAMERA_SOURCE", "tapo")


def rtsp_url(stream=None):
    """URL RTSP de la cámara. Contiene la contraseña: no imprimirla (usar redactar)."""
    cargar_env()
    url = os.environ.get("WEAPON_RTSP_URL")
    if url:
        return url

    faltan = [v for v in ("TAPO_IP", "TAPO_USER", "TAPO_PASSWORD") if not os.environ.get(v)]
    if faltan:
        sys.exit(f"Faltan credenciales de la cámara: {', '.join(faltan)}.\n"
                 f"Copia .env.example a .env y rellénalo.")
    user = quote(os.environ["TAPO_USER"], safe="")
    password = quote(os.environ["TAPO_PASSWORD"], safe="")
    port = os.environ.get("TAPO_PORT", "554")
    stream = stream or os.environ.get("TAPO_STREAM", "stream1")
    return f"rtsp://{user}:{password}@{os.environ['TAPO_IP']}:{port}/{stream}"


def redactar(url):
    """Oculta la contraseña al imprimir la URL (rtsp://user:****@host...)."""
    if "@" in url and "://" in url:
        proto, resto = url.split("://", 1)
        cred, host = resto.rsplit("@", 1)
        user = cred.split(":", 1)[0]
        return f"{proto}://{user}:****@{host}"
    return url


class Camara:
    """Itera sobre los cuadros (BGR) de la fuente elegida.

        with Camara("webcam") as cam:
            for frame in cam:
                ...
    """

    REINTENTOS_RTSP = 5      # reconexiones seguidas antes de rendirse
    FALLOS_WEBCAM = 30       # lecturas fallidas seguidas antes de rendirse

    def __init__(self, fuente):
        if fuente not in FUENTES:
            sys.exit(f"Fuente desconocida: {fuente}. Opciones: {', '.join(FUENTES)}")
        cargar_env()
        self.fuente = fuente
        self.cap = self._abrir()
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 15.0
        destino = (redactar(rtsp_url()) if fuente == "tapo"
                   else f"/dev/video{os.environ.get('WEBCAM_INDEX', '0')}")
        self.descripcion = f"{fuente} {destino} ({w}x{h} @ {self.fps:.0f} FPS)"

    def _abrir(self):
        if self.fuente == "tapo":
            # TCP evita los cuadros corruptos que produce UDP al perder paquetes.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            cap = cv2.VideoCapture(rtsp_url(), cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(int(os.environ.get("WEBCAM_INDEX", "0")), cv2.CAP_V4L2)
            # MJPG es necesario para que la webcam entregue 1080p a 30 FPS por USB.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(os.environ.get("WEBCAM_WIDTH", "1920")))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(os.environ.get("WEBCAM_HEIGHT", "1080")))
        if not cap.isOpened():
            sys.exit(f"No se pudo abrir la cámara '{self.fuente}'. "
                     "¿Está encendida/conectada y son correctos los datos del .env?")
        return cap

    def __iter__(self):
        fallos = reintentos = 0
        while True:
            ok, frame = self.cap.read()
            if ok:
                fallos = reintentos = 0
                yield frame
                continue
            if self.fuente == "tapo":
                reintentos += 1
                if reintentos > self.REINTENTOS_RTSP:
                    print("Se perdió el stream de la Tapo y no se pudo reconectar.")
                    return
                print(f"Stream interrumpido, reconectando ({reintentos}/{self.REINTENTOS_RTSP})...")
                self.cap.release()
                time.sleep(1)
                self.cap = self._abrir()
            else:
                fallos += 1
                if fallos > self.FALLOS_WEBCAM:
                    print("La webcam dejó de entregar imágenes.")
                    return

    def cerrar(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cerrar()
