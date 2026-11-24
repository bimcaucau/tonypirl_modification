"""Live 3-pane segmentation viewer for TonyPi head camera.

Panes (left → right):
  1. Camera          raw BGR
  2. Segmentation    YOLO instance overlay, or HSV tint
  3. Binary          red = obstacle, green = road/free
                     (full-res mask; HUD also shows the 64×64 policy occupancy)

Does NOT walk the robot. TonyPi backend opens the camera with dry_run=True.

Examples:
  # Laptop fake scene (no camera, no robot)
  python -m sim2real_implementation.view_segmentation --backend fake --mode color

  # Laptop webcam
  python -m sim2real_implementation.view_segmentation --backend webcam --mode yolo

  # On the Pi, stream to a laptop browser (camera is exclusive — stop tonypi.service first)
  sudo systemctl stop tonypi
  python3 -m sim2real_implementation.view_segmentation --backend tonypi --mode yolo --http-port 8080
  # instance-seg fallback: --yolo-weights yolo26n-seg.pt
  # then open http://<ROBOT_IP>:8080

Do not pass Functions/model/best.pt or yolov5n.pt — detect-only, no masks.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim2real_implementation.constants import (
    DEFAULT_YOLO_WEIGHTS,
    OCCUPANCY_SIZE,
    YOLO11_SEG_FALLBACK,
    YOLO26_SEG_FALLBACK,
)
from sim2real_implementation.occupancy_real import OccupancyEncoder, COLOR_OBSTACLE, COLOR_ROAD

PANE_W, PANE_H = 640, 480


def parse_args():
    p = argparse.ArgumentParser(description="TonyPi live 3-pane segmentation viewer.")
    p.add_argument("--backend", choices=["tonypi", "webcam", "fake"], default="fake")
    p.add_argument("--mode", choices=["yolo", "color"], default="color")
    p.add_argument("--camera-index", type=int, default=0, help="OpenCV index for --backend webcam")
    p.add_argument(
        "--yolo-weights",
        type=str,
        default=DEFAULT_YOLO_WEIGHTS,
        help=(
            f"Semantic default {DEFAULT_YOLO_WEIGHTS}; "
            f"instance {YOLO26_SEG_FALLBACK} or {YOLO11_SEG_FALLBACK}"
        ),
    )
    p.add_argument("--yolo-imgsz", type=int, default=320)
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--keep-all", action="store_true", help="YOLO: overlay every class (floor still free in occupancy)")
    p.add_argument("--hsv-lo", type=int, nargs=3, default=[0, 80, 80])
    p.add_argument("--hsv-hi", type=int, nargs=3, default=[12, 255, 255])
    p.add_argument("--display", action="store_true", default=None, help="Force cv2.imshow even without DISPLAY check")
    p.add_argument("--no-display", action="store_true", help="Never open a GUI window")
    p.add_argument("--http-port", type=int, default=0, help="If >0, serve MJPEG 3-pane at http://0.0.0.0:PORT/")
    p.add_argument("--save-dir", type=str, default=None, help="Write latest_camera/seg/mask/triple.jpg each frame")
    p.add_argument("--record", type=str, default=None, help="Optional .mp4 path for the 3-pane mosaic")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = forever)")
    p.add_argument("--source", type=str, default=None, help="Optional image or video file instead of live camera")
    return p.parse_args()


class FrameSource:
    def start(self) -> None: ...
    def read_bgr(self) -> np.ndarray | None: ...
    def stop(self) -> None: ...


class TonyPiSource:
    def __init__(self):
        self.hw = None

    def start(self) -> None:
        from sim2real_implementation.hardware import TonyPiHardware
        # dry_run=True: open camera / sensors, never play walk clips.
        # allow_missing_tof: vis is camera-only; ToF is still opened when the
        # Functions/vl53l0x_python.so import succeeds.
        self.hw = TonyPiHardware(dry_run=True, occupancy_mode="color", allow_missing_tof=True)
        self.hw.start()
        self.hw.center_head()

    def read_bgr(self) -> np.ndarray | None:
        return self.hw.read_bgr() if self.hw is not None else None

    def stop(self) -> None:
        if self.hw is not None:
            # skip stand clip: stop() would play stand unless dry_run
            if self.hw.camera is not None:
                try:
                    self.hw.camera.camera_close()
                except Exception:
                    pass
            if self.hw.tof is not None:
                try:
                    self.hw.tof.stop_ranging()
                    self.hw.tof.close()
                except Exception:
                    pass
            self.hw = None


class WebcamSource:
    def __init__(self, index: int = 0, path: str | None = None):
        self.index = index
        self.path = path
        self.cap = None

    def start(self) -> None:
        src = self.path if self.path else self.index
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source {src!r}")

    def read_bgr(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            if self.path:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                if not ok:
                    return None
            else:
                return None
        return frame

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class FakeSource:
    """Moving saturated-red box on a gray floor so color mode is obviously live."""

    def __init__(self):
        self.t = 0.0

    def start(self) -> None:
        pass

    def read_bgr(self) -> np.ndarray | None:
        img = np.full((480, 640, 3), 40, dtype=np.uint8)
        img[320:, :] = (50, 90, 50)  # green-ish floor strip
        x = int(180 + 140 * np.sin(self.t))
        img[140:300, x:x + 160] = (0, 0, 255)
        cv2.putText(img, "FAKE SCENE", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        self.t += 0.08
        return img

    def stop(self) -> None:
        pass


class JpegHub:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = b""
        self._event = threading.Event()

    def set_bgr(self, bgr: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
        self._event.set()

    def get(self) -> bytes:
        with self._lock:
            return self._jpeg


def start_http(port: int, hub: JpegHub) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = (
                    "<!doctype html><meta charset=utf-8><title>TonyPi seg</title>"
                    "<body style='margin:0;background:#111;color:#ddd;font-family:sans-serif'>"
                    "<p style='margin:8px 12px'>TonyPi segmentation — camera | overlay | obstacle/road"
                    " — q in the OpenCV window, Ctrl+C here to stop.</p>"
                    "<img src='/stream' style='width:100%;image-rendering:auto'/>"
                    "</body>"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        jpeg = hub.get()
                        if jpeg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                        time.sleep(0.04)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
            self.send_error(404)

    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"[http] 3-pane MJPEG at http://0.0.0.0:{port}/")
    return httpd


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _banner(img: np.ndarray, text: str, bgr=(255, 255, 255)) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 1, cv2.LINE_AA)


def compose_triple(views: dict, hud: str) -> np.ndarray:
    cam = _fit(views["camera"], PANE_W, PANE_H)
    seg = _fit(views["overlay"], PANE_W, PANE_H)
    binary = _fit(views["binary"], PANE_W, PANE_H)
    _banner(cam, "1  camera")
    _banner(seg, "2  segmentation", (0, 200, 255))
    _banner(binary, "3  binary  red=obstacle  green=road", COLOR_OBSTACLE)

    occ = views["occupancy"][0]
    occ_vis = cv2.resize(
        (occ * 255).astype(np.uint8),
        (128, 128),
        interpolation=cv2.INTER_NEAREST,
    )
    occ_bgr = cv2.cvtColor(occ_vis, cv2.COLOR_GRAY2BGR)
    occ_bgr[occ_vis > 0] = COLOR_OBSTACLE
    occ_bgr[occ_vis == 0] = COLOR_ROAD
    binary[PANE_H - 136:PANE_H - 8, 8:136] = occ_bgr
    cv2.putText(binary, "64x64 policy", (8, PANE_H - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1)

    mosaic = np.hstack([cam, seg, binary])
    cv2.rectangle(mosaic, (0, mosaic.shape[0] - 32), (mosaic.shape[1], mosaic.shape[0]), (15, 15, 15), -1)
    cv2.putText(mosaic, hud, (10, mosaic.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return mosaic


class LiveTriple:
    """Background 3-pane stream. One thread owns encode_views so the walk loop can block on clips."""

    def __init__(self, encoder, port: int, save_dir: str | None = None):
        self.encoder = encoder
        self.save_dir = save_dir
        self._lock = threading.Lock()
        self._views = None
        self._epoch = 0
        self._stop = threading.Event()
        self._thread = None
        self.hub = JpegHub()
        self.httpd = start_http(port, self.hub) if port else None

    def start(self, read_bgr, hud_fn) -> None:
        def loop():
            while not self._stop.is_set():
                with self._lock:
                    epoch = self._epoch
                frame = read_bgr()
                if frame is None:
                    time.sleep(0.05)
                    continue
                try:
                    views = self.encoder.encode_views(frame)
                    hud = hud_fn() or ""
                    triple = compose_triple(views, hud)
                except Exception:
                    time.sleep(0.05)
                    continue
                self.hub.set_bgr(triple)
                if self.save_dir:
                    save_latest(self.save_dir, views, triple)
                with self._lock:
                    # A ground-look drops the epoch so a frame grabbed while the
                    # head was pitched down cannot become the next policy obs.
                    if epoch == self._epoch:
                        self._views = views

        self._thread = threading.Thread(target=loop, name="live-triple", daemon=True)
        self._thread.start()

    def latest(self) -> dict | None:
        with self._lock:
            return self._views

    def invalidate(self) -> None:
        """Drop the buffered view. The next wait_views blocks until a new grab."""
        with self._lock:
            self._epoch += 1
            self._views = None

    def wait_views(self, timeout: float = 5.0) -> dict | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            views = self.latest()
            if views is not None:
                return views
            time.sleep(0.05)
        return None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd = None


def save_latest(save_dir: str, views: dict, triple: np.ndarray) -> None:
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, "latest_camera.jpg"), views["camera"])
    cv2.imwrite(os.path.join(save_dir, "latest_seg.jpg"), views["overlay"])
    cv2.imwrite(os.path.join(save_dir, "latest_mask.jpg"), views["binary"])
    cv2.imwrite(os.path.join(save_dir, "latest_triple.jpg"), triple)
    occ = (views["occupancy"][0] * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, "latest_occ64.png"), occ)


def open_source(args) -> FrameSource:
    if args.source:
        path = args.source
        ext = os.path.splitext(path)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
            img = cv2.imread(path)
            if img is None:
                raise RuntimeError(f"Could not read image {path}")

            class Still:
                def start(self): pass
                def read_bgr(self): return img.copy()
                def stop(self): pass

            return Still()
        return WebcamSource(path=path)
    if args.backend == "tonypi":
        return TonyPiSource()
    if args.backend == "webcam":
        return WebcamSource(index=args.camera_index)
    return FakeSource()


def main():
    args = parse_args()
    if args.mode == "yolo":
        try:
            from tonypi_rl.perception import refuse_detect_checkpoint
            refuse_detect_checkpoint(args.yolo_weights)
        except RuntimeError as e:
            print(f"[refuse] {e}", file=sys.stderr)
            sys.exit(2)

    want_display = False if args.no_display else (
        True if args.display else bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    )

    encoder = OccupancyEncoder(
        mode=args.mode,
        yolo_weights=args.yolo_weights,
        yolo_conf=args.yolo_conf,
        yolo_imgsz=args.yolo_imgsz,
        hsv_lo=args.hsv_lo,
        hsv_hi=args.hsv_hi,
        keep_all=args.keep_all,
    )

    src = open_source(args)
    src.start()

    hub = JpegHub()
    httpd = start_http(args.http_port, hub) if args.http_port else None
    writer = None
    win = "TonyPi segmentation  (q quit)"
    if want_display:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("mode=%s backend=%s display=%s http=%s" % (
        args.mode, args.backend, want_display, args.http_port or "off",
    ))
    print("red = obstacle, green = road/free. Ctrl+C or q to stop.")

    n = 0
    t_prev = time.time()
    fps = 0.0
    try:
        while True:
            frame = src.read_bgr()
            t0 = time.time()
            views = encoder.encode_views(frame)
            lat_ms = (time.time() - t0) * 1000.0
            n += 1
            now = time.time()
            dt = now - t_prev
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if n > 1 else 1.0 / max(dt, 1e-3)
            t_prev = now

            labels = views.get("labels") or []
            names = ",".join(n for n, _, _ in labels[:4]) or "-"
            occ_mean = float(views["occupancy"].mean())
            h, w = views["camera"].shape[:2]
            if args.mode == "yolo":
                det_tag = os.path.splitext(os.path.basename(args.yolo_weights))[0]
            else:
                det_tag = "hsv"
            hud = (
                f"{args.mode} {w}x{h}  fps={fps:4.1f}  enc={lat_ms:5.1f}ms  "
                f"occ_mean={occ_mean:.3f}  det={det_tag}:{names}  "
                f"occ={OCCUPANCY_SIZE}x{OCCUPANCY_SIZE}"
            )
            triple = compose_triple(views, hud)
            hub.set_bgr(triple)

            if args.save_dir:
                save_latest(args.save_dir, views, triple)

            if args.record:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.record, fourcc, 15.0, (triple.shape[1], triple.shape[0]))
                writer.write(triple)

            if want_display:
                cv2.imshow(win, triple)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if args.max_frames and n >= args.max_frames:
                break
            if frame is None:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        if writer is not None:
            writer.release()
        if want_display:
            cv2.destroyAllWindows()
        if httpd is not None:
            httpd.shutdown()
        src.stop()
        print(f"frames={n} last_fps={fps:.1f}")


if __name__ == "__main__":
    main()
