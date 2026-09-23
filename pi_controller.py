#!/usr/bin/env python3
"""
Raspberry Pi System Controller for Hello-Netra
Coordinates ESP32-C3 BLE button commands, VL53L5CX ToF safety sensor streaming,
and Android Hotspot TCP control/streaming.

External Hardware Architecture:
  VL53L5CX (I2C) + Push Buttons -> ESP32-C3 -> BLE -> Raspberry Pi (pi_controller.py) -> Local Wi-Fi (Phone Hotspot) -> Android App

Features:
  - Connects to ESP32-C3 over BLE
  - Listens for function button selections (OBJECT, OCR, SOS, STOP)
  - Listens for 16-zone VL53L5CX distance data @ 5 Hz
  - Packs and forwards binary PITF (45-byte) or text TOF: packets over TCP socket to Android
  - Automatically manages camera streaming process (camera_sender.py)
  - Interactive CLI test mode with synthetic 4x4 ToF distance simulation
"""

import sys
import os
import time
import socket
import struct
import subprocess
import threading
import argparse
import random

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ANDROID_IP = "192.168.43.1"
ANDROID_PORT = 5000

# ESP32-C3 BLE Configuration
ESP32_DEVICE_NAME = "HelloNetra-Stick"
ESP32_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
ESP32_BUTTON_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
ESP32_TOF_CHAR_UUID = "6E400004-B5A3-F393-E0A9-E50E24DCCA9E"

# PITF Binary Protocol Constants
TOF_MAGIC = b"PITF"  # 4-byte Magic signature
TOF_VERSION = 1
# Format: 4s (Magic) + B (Version) + q (TimestampMs) + 16H (16 x uint16) = 45 bytes
TOF_STRUCT = struct.Struct("!4sB q 16H")


def pack_tof_pitf(distances: list, timestamp_ms: int = None) -> bytes:
    """Packs 16 zone distances (in mm) into the 45-byte binary PITF protocol."""
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    
    clamped_distances = []
    for i in range(16):
        d = distances[i] if i < len(distances) else 0
        if 20 <= d <= 4000:
            clamped_distances.append(int(d))
        else:
            clamped_distances.append(0)

    return TOF_STRUCT.pack(
        TOF_MAGIC,
        TOF_VERSION,
        timestamp_ms,
        *clamped_distances
    )


class PiController:
    def __init__(self, android_ip=ANDROID_IP, android_port=ANDROID_PORT, sim_tof=False):
        self.android_ip = android_ip
        self.android_port = android_port
        self.sim_tof = sim_tof
        self.current_mode = "IDLE"
        self.camera_process = None
        self.sock = None
        self.sock_lock = threading.Lock()
        self.is_running = True
        self.tof_thread = None

    def connect_android(self):
        """Connects or reconnects to the Android phone TCP server over local hotspot."""
        with self.sock_lock:
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None

            try:
                print(f"[Network] Connecting to Android Hotspot at {self.android_ip}:{self.android_port}...")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(5.0)
                sock.connect((self.android_ip, self.android_port))
                sock.settimeout(None)
                self.sock = sock
                print(f"[Network] Connected to Android app successfully!")
                return True
            except Exception as e:
                print(f"[Network] Could not connect to Android ({e}). Will retry on next packet.")
                return False

    def send_tcp_data(self, data: bytes):
        """Thread-safe dispatch of binary or text data over the TCP connection."""
        with self.sock_lock:
            for attempt in range(2):
                if not self.sock:
                    # Attempt connection outside lock to avoid deadlocks
                    pass
                if self.sock:
                    try:
                        self.sock.sendall(data)
                        return True
                    except Exception as e:
                        print(f"[Network] TCP send failed ({e}), reconnecting...")
                        try:
                            self.sock.close()
                        except Exception:
                            pass
                        self.sock = None
                        # Reconnect
                        try:
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                            sock.settimeout(5.0)
                            sock.connect((self.android_ip, self.android_port))
                            sock.settimeout(None)
                            self.sock = sock
                            self.sock.sendall(data)
                            return True
                        except Exception:
                            pass
            return False

    def send_mode_command(self, command: str):
        """Sends a control command string to Android over TCP."""
        payload = (command.strip() + "\n").encode("utf-8")
        success = self.send_tcp_data(payload)
        if success:
            print(f"[Control] Sent to Android: {command}")
        else:
            print(f"[Control] Failed to dispatch command: {command}")
        return success

    def send_tof_frame(self, distances: list, timestamp_ms: int = None):
        """Transmits a 16-zone ToF distance frame to Android using binary PITF protocol."""
        packet = pack_tof_pitf(distances, timestamp_ms)
        return self.send_tcp_data(packet)

    def set_mode(self, mode: str):
        """Switches system mode and manages camera streaming process."""
        mode = mode.upper().strip()
        print(f"\n=======================================================")
        print(f" >>> ACTION TRIGGERED: {mode} <<<")
        print(f"=======================================================")

        if mode in ["OBJECT", "MODE_OBJECT"]:
            self.current_mode = "OBJECT_RECOGNITION"
            self.send_mode_command("MODE_OBJECT")
            self.start_camera()

        elif mode in ["OCR", "MODE_OCR"]:
            self.current_mode = "OCR"
            self.send_mode_command("MODE_OCR")
            self.start_camera()

        elif mode in ["SOS", "MODE_SOS"]:
            self.current_mode = "SOS"
            self.send_mode_command("MODE_SOS")
            self.stop_camera()

        elif mode in ["STOP", "MODE_STOP", "IDLE", "MODE_IDLE"]:
            self.current_mode = "IDLE"
            self.send_mode_command("MODE_STOP")
            self.stop_camera()

    def start_camera(self):
        """Starts camera_sender.py background process if not already streaming."""
        if self.camera_process and self.camera_process.poll() is None:
            print("[Camera] Camera streaming is already running.")
            return

        script_dir = os.path.dirname(os.path.abspath(__file__))
        camera_script = os.path.join(script_dir, "camera_sender.py")

        if not os.path.exists(camera_script):
            print(f"[Camera] Error: {camera_script} not found!")
            return

        print(f"[Camera] Starting camera streaming: {camera_script}")
        try:
            self.camera_process = subprocess.Popen(
                [sys.executable, camera_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            threading.Thread(target=self._log_camera_output, daemon=True).start()
        except Exception as e:
            print(f"[Camera] Error starting camera process: {e}")

    def _log_camera_output(self):
        proc = self.camera_process
        if not proc:
            return
        for line in proc.stdout:
            print(f"  [CameraStream] {line.strip()}")

    def stop_camera(self):
        """Terminates active camera streaming process."""
        if self.camera_process and self.camera_process.poll() is None:
            print("[Camera] Stopping camera streaming...")
            self.camera_process.terminate()
            try:
                self.camera_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.camera_process.kill()
            self.camera_process = None
            print("[Camera] Camera stream stopped.")

    def start_synthetic_tof_stream(self):
        """Spawns background thread generating realistic 4x4 ToF distances @ 5 Hz for testing."""
        def worker():
            print("[ToF Sim] Started synthetic 5 Hz VL53L5CX telemetry generator.")
            distances = [1200] * 16
            while self.is_running:
                # Add slight random walk to simulate movement
                for i in range(16):
                    change = random.randint(-50, 50)
                    distances[i] = max(300, min(2500, distances[i] + change))
                # Occasionally introduce an obstacle in center corridor (zones 5, 6, 9, 10)
                if random.random() < 0.3:
                    distances[5] = random.randint(400, 750)
                    distances[6] = random.randint(400, 750)

                self.send_tof_frame(distances)
                time.sleep(0.2)  # 5 Hz

        self.tof_thread = threading.Thread(target=worker, daemon=True)
        self.tof_thread.start()

    def run_ble_listener(self):
        """Connects to ESP32-C3 BLE peripheral and listens for buttons & VL53L5CX ToF data."""
        try:
            import asyncio
            from bleak import BleakClient, BleakScanner

            async def ble_worker():
                print(f"[BLE] Scanning for ESP32-C3 device: '{ESP32_DEVICE_NAME}'...")
                device = await BleakScanner.find_device_by_name(ESP32_DEVICE_NAME, timeout=10.0)
                if not device:
                    print(f"[BLE] ESP32-C3 device '{ESP32_DEVICE_NAME}' not found.")
                    print("[BLE] Starting in interactive CLI test mode...")
                    self.run_interactive_cli()
                    return

                print(f"[BLE] Connecting to ESP32-C3 at {device.address}...")
                async with BleakClient(device) as client:
                    print(f"[BLE] Connected to ESP32-C3!")

                    # Button notification handler
                    def button_handler(sender, data):
                        cmd = data.decode("utf-8", errors="ignore").strip()
                        print(f"[BLE] Button event: {cmd}")
                        self.set_mode(cmd)

                    # ToF data notification handler
                    def tof_handler(sender, data):
                        if len(data) == 45 and data[:4] == TOF_MAGIC:
                            # Direct PITF binary packet
                            self.send_tcp_data(data)
                        elif len(data) == 32:
                            # 16 x uint16 distances from ESP32
                            distances = struct.unpack("!16H", data)
                            self.send_tof_frame(list(distances))
                        elif data.startswith(b"TOF:"):
                            # Text packet
                            self.send_tcp_data(data + b"\n")

                    try:
                        await client.start_notify(ESP32_BUTTON_CHAR_UUID, button_handler)
                    except Exception as e:
                        print(f"[BLE] Could not subscribe to button char: {e}")

                    try:
                        await client.start_notify(ESP32_TOF_CHAR_UUID, tof_handler)
                    except Exception as e:
                        print(f"[BLE] Could not subscribe to ToF char: {e}")

                    print(f"[BLE] Subscriptions active. Listening for events...")
                    while self.is_running and client.is_connected:
                        await asyncio.sleep(0.5)

            asyncio.run(ble_worker())

        except ImportError:
            print("[BLE] Note: 'bleak' Python library not installed. Running in interactive test mode.")
            self.run_interactive_cli()
        except Exception as e:
            print(f"[BLE] BLE error: {e}. Falling back to interactive test mode.")
            self.run_interactive_cli()

    def run_interactive_cli(self):
        """Interactive console interface for manual development testing."""
        if self.sim_tof:
            self.start_synthetic_tof_stream()

        print("\n" + "=" * 65)
        print(" HELLO-NETRA CONTROLLER — INTERACTIVE TEST MODE")
        print("=" * 65)
        print(" Controls:")
        print("   [1] or 'obj'  -> Mode 1: OBJECT RECOGNITION (starts camera stream)")
        print("   [2] or 'ocr'  -> Mode 2: OCR (starts camera stream)")
        print("   [3] or 'sos'  -> Mode 3: EMERGENCY SOS (triggers GPS + SMS)")
        print("   [4] or 'stop' -> Standby / IDLE (stops camera)")
        print("   [t] or 'tof'  -> Send sample 4x4 ToF distance frame")
        print("   [q]           -> Quit")
        print("=" * 65 + "\n")

        # Initial connection
        self.connect_android()

        while self.is_running:
            try:
                cmd = input("Select [1=Object, 2=OCR, 3=SOS, 4=Stop, t=SendToF, q=Quit]: ").strip().lower()
                if cmd in ["1", "obj", "object"]:
                    self.set_mode("OBJECT")
                elif cmd in ["2", "ocr"]:
                    self.set_mode("OCR")
                elif cmd in ["3", "sos"]:
                    self.set_mode("SOS")
                elif cmd in ["4", "stop", "idle"]:
                    self.set_mode("STOP")
                elif cmd in ["t", "tof"]:
                    sample_dists = [820, 760, 690, 710, 950, 840, 720, 680, 1200, 1100, 900, 850, 0, 0, 0, 0]
                    self.send_tof_frame(sample_dists)
                    print(f"[ToF] Sent sample 16-zone frame to Android: {sample_dists[:4]}...")
                elif cmd in ["q", "quit", "exit"]:
                    print("Exiting...")
                    self.set_mode("STOP")
                    break
                else:
                    print("Invalid choice. Enter 1, 2, 3, 4, t, or q.")
            except (KeyboardInterrupt, EOFError):
                break

        self.cleanup()

    def cleanup(self):
        self.is_running = False
        self.stop_camera()
        with self.sock_lock:
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Controller for Hello-Netra with VL53L5CX ToF Support")
    parser.add_argument("--ip", default=ANDROID_IP, help="Android hotspot IP (default: 192.168.43.1)")
    parser.add_argument("--port", type=int, default=ANDROID_PORT, help="Android TCP port (default: 5000)")
    parser.add_argument("--cli", action="store_true", help="Force interactive CLI mode for testing")
    parser.add_argument("--sim-tof", action="store_true", help="Automatically stream synthetic 4x4 ToF distances @ 5 Hz in CLI mode")
    args = parser.parse_args()

    controller = PiController(android_ip=args.ip, android_port=args.port, sim_tof=args.sim_tof or args.cli)

    if args.cli:
        controller.run_interactive_cli()
    else:
        controller.run_ble_listener()


if __name__ == "__main__":
    main()
