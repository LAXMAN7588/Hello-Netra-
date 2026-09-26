#!/usr/bin/env python3
"""
Raspberry Pi Camera Frame Sender
Sends camera frames over a local Wi-Fi hotspot TCP connection to the Android Object Detection App.

Architecture:
  Pi Camera -> JPEG Encode -> Binary Frame Packet -> Local TCP Socket -> Android Phone Hotspot
"""

import time
import socket
import struct
import sys
import io

# ==============================================================================
# CONFIGURATION SECTION (Modify these values to match your local setup)
# ==============================================================================

# Android phone's hotspot IP address (default is commonly 192.168.43.1)
ANDROID_IP = "192.168.43.1"
ANDROID_PORT = 5000

# Camera capture resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# JPEG compression quality (1-100, higher = better quality, lower = lower latency/bandwidth)
JPEG_QUALITY = 75

# Target frame rate to transmit
TARGET_FPS = 10

# Reconnect delay in seconds if connection fails or disconnects
RECONNECT_DELAY_SEC = 2.0

# Protocol Constants
MAGIC = b"PIFR"  # 4-byte Magic signature
PROTOCOL_VERSION = 1
FORMAT_JPEG = 1

# Header format: 4s (Magic) + B (Version) + B (Format) + I (FrameId) + q (TimestampMs) + I (PayloadLen) = 22 bytes
HEADER_STRUCT = struct.Struct("!4sBB I q I")


def create_frame_packet(frame_id: int, timestamp_ms: int, jpeg_bytes: bytes) -> bytes:
    """Packs the 22-byte binary protocol header and appends the JPEG payload."""
    header = HEADER_STRUCT.pack(
        MAGIC,
        PROTOCOL_VERSION,
        FORMAT_JPEG,
        frame_id & 0xFFFFFFFF,
        timestamp_ms,
        len(jpeg_bytes)
    )
    return header + jpeg_bytes


class CameraCapture:
    """Unified camera capture supporting Picamera2, OpenCV, or synthetic test frames."""

    def __init__(self, width: int, height: int, quality: int):
        self.width = width
        self.height = height
        self.quality = quality
        self.backend = None
        self.picam2 = None
        self.cap = None

        # 1. Try Picamera2 (Raspberry Pi Camera Module on modern Pi OS)
        try:
            from picamera2 import Picamera2
            print("[Camera] Initializing Picamera2...")
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
            self.backend = "picam2"
            print("[Camera] Picamera2 initialized successfully.")
            return
        except Exception as e:
            print(f"[Camera] Picamera2 not available ({e}). Trying OpenCV...")

        # 2. Try OpenCV VideoCapture (USB Camera or legacy V4L2)
        try:
            import cv2
            print("[Camera] Initializing OpenCV VideoCapture(0)...")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.cap.isOpened():
                self.backend = "opencv"
                print("[Camera] OpenCV camera initialized successfully.")
                return
        except Exception as e:
            print(f"[Camera] OpenCV capture not available ({e}).")

        # 3. Fallback to Synthetic Test Pattern
        print("[Camera] Using synthetic frame generator (for testing without physical camera).")
        self.backend = "synthetic"

    def capture_jpeg(self) -> bytes:
        """Captures one frame and returns JPEG compressed bytes."""
        if self.backend == "picam2":
            import cv2
            frame_rgb = self.picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            _, encoded_jpeg = cv2.imencode(".jpg", frame_bgr, encode_param)
            return encoded_jpeg.tobytes()

        elif self.backend == "opencv":
            import cv2
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to capture frame from OpenCV camera")
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            _, encoded_jpeg = cv2.imencode(".jpg", frame, encode_param)
            return encoded_jpeg.tobytes()

        else:
            # Generate synthetic test frame using PIL
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new("RGB", (self.width, self.height), color=(30, 30, 30))
            draw = ImageDraw.Draw(img)
            draw.rectangle([50, 50, self.width - 50, self.height - 50], outline="lime", width=3)
            draw.text((70, 70), "Raspberry Pi Test Feed", fill="white")
            draw.text((70, 100), f"Time: {time.strftime('%H:%M:%S')}", fill="yellow")

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.quality)
            return buf.getvalue()

    def release(self):
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass


def run_sender():
    print("=" * 65)
    print(" Raspberry Pi Camera Frame Sender -> Android Hotspot Receiver")
    print(f" Target Server : {ANDROID_IP}:{ANDROID_PORT}")
    print(f" Resolution    : {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f" Target Rate   : {TARGET_FPS} FPS")
    print(f" JPEG Quality  : {JPEG_QUALITY}")
    print("=" * 65)

    camera = CameraCapture(CAMERA_WIDTH, CAMERA_HEIGHT, JPEG_QUALITY)
    frame_interval = 1.0 / TARGET_FPS
    frame_id = 0

    while True:
        sock = None
        try:
            print(f"\n[Network] Connecting to Android Hotspot at {ANDROID_IP}:{ANDROID_PORT}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(10.0)
            sock.connect((ANDROID_IP, ANDROID_PORT))
            sock.settimeout(None)  # Blocking for sends
            print(f"[Network] Connected to Android app successfully!")

            while True:
                start_time = time.time()
                frame_id += 1
                timestamp_ms = int(time.time() * 1000)

                # Capture and encode JPEG
                jpeg_bytes = camera.capture_jpeg()

                # Build binary frame packet
                packet = create_frame_packet(frame_id, timestamp_ms, jpeg_bytes)

                # Transmit over TCP
                sock.sendall(packet)

                # Maintain target FPS
                elapsed = time.time() - start_time
                sleep_time = max(0.0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if frame_id % 50 == 0:
                    actual_fps = 1.0 / max(0.001, (time.time() - start_time))
                    print(f"[Stream] Sent Frame #{frame_id} | Size: {len(jpeg_bytes)/1024:.1f} KB | {actual_fps:.1f} FPS")

        except KeyboardInterrupt:
            print("\n[Sender] Stopped by user.")
            break
        except Exception as e:
            print(f"[Network] Connection error: {e}")
            print(f"[Network] Retrying in {RECONNECT_DELAY_SEC} seconds...")
            time.sleep(RECONNECT_DELAY_SEC)
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    camera.release()
    print("[Sender] Shutdown complete.")


if __name__ == "__main__":
    run_sender()
