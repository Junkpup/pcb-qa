"""
AOI System: scrcpy Raw Camera Feed Launcher with Grid Overlay
============================================================

Launches scrcpy with camera feed (--video-source=camera) in headless mode,
streams raw video to stdout, decodes with ffmpeg, and overlays a digital grid.

Features:
- Subprocess-based scrcpy control with raw stdout streaming
- ffmpeg decoding to BGR frames for OpenCV
- Real-time frame capture from camera stream
- 100px green digital alignment grid overlay
- Frame rate monitoring and performance metrics
"""

import subprocess
import time
import sys
import cv2
import numpy as np
from typing import Optional
import signal
import multiprocessing as mp
import tkinter as tk


class DigitalAlignmentGrid:
    """Renders 100px grid overlay for PCB coordinate registration."""
    
    GRID_SIZE = 100  # pixels
    GRID_COLOR = (0, 255, 0)  # BGR: Green
    GRID_THICKNESS = 1
    GRID_ALPHA = 0.3  # Transparency blending factor
    
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_w = frame_width
        self.frame_h = frame_height
    
    def render(self, frame: np.ndarray) -> np.ndarray:
        """
        Overlay 100px digital alignment grid on frame.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Frame with grid overlay (BGR)
        """
        overlay = frame.copy()
        
        # Draw vertical lines
        for x in range(0, self.frame_w, self.GRID_SIZE):
            cv2.line(overlay, (x, 0), (x, self.frame_h), 
                    self.GRID_COLOR, self.GRID_THICKNESS)
        
        # Draw horizontal lines
        for y in range(0, self.frame_h, self.GRID_SIZE):
            cv2.line(overlay, (0, y), (self.frame_w, y), 
                    self.GRID_COLOR, self.GRID_THICKNESS)
        
        # Blend overlay with semi-transparency
        return cv2.addWeighted(overlay, self.GRID_ALPHA, frame, 
                              1 - self.GRID_ALPHA, 0)


class SCRCPYSubprocessLauncher:
    """Manages scrcpy subprocess with raw video streaming."""
    
    CAMERA_ID = 0
    CAMERA_SIZE = (1920, 1080)
    VIDEO_BITRATE = "16M"
    MAX_FPS = 30

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.decoder: Optional[subprocess.Popen] = None
        self.grid: Optional[DigitalAlignmentGrid] = None
        self.frame_count = 0
        self.running = True
        self.frame_width = self.CAMERA_SIZE[0]
        self.frame_height = self.CAMERA_SIZE[1]

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n⚠ Received signal {signum}. Shutting down...")
        self.running = False

    def _build_scrcpy_cmd(self) -> list:
        """Build scrcpy command for raw stdout video streaming."""
        width, height = self.CAMERA_SIZE
        return [
            "scrcpy",
            "--no-display",
            "--raw-video",
            "--video-source=camera",
            f"--camera-id={self.CAMERA_ID}",
            f"--camera-size={width}x{height}",
            f"--video-bit-rate={self.VIDEO_BITRATE}",
            f"--max-fps={self.MAX_FPS}",
        ]

    def _build_ffmpeg_cmd(self) -> list:
        """Build ffmpeg command to decode H.264 to raw BGR frames."""
        return [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-an",
            "pipe:1",
        ]
    
    def launch_scrcpy(self):
        """Launch scrcpy subprocess and attach ffmpeg decoder."""
        try:
            cmd = self._build_scrcpy_cmd()
            print("🚀 Launching scrcpy subprocess...")
            print(f"   Command: {' '.join(cmd)}")
            print(
                f"   Camera: {self.frame_width}x{self.frame_height} "
                f"@ {self.MAX_FPS} FPS"
            )

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=None,
            )
            if self.process.stdout is None:
                raise RuntimeError("scrcpy stdout pipe not available")

            self.decoder = subprocess.Popen(
                self._build_ffmpeg_cmd(),
                stdin=self.process.stdout,
                stdout=subprocess.PIPE,
                stderr=None,
            )
            self.process.stdout.close()

            if self.decoder.stdout is None:
                raise RuntimeError("ffmpeg stdout pipe not available")

            self.grid = DigitalAlignmentGrid(self.frame_width, self.frame_height)
            print("✓ scrcpy raw stream started")
            print("✓ ffmpeg decoder attached")
            print("✓ Digital alignment grid initialized")

        except FileNotFoundError:
            print("✗ Error: scrcpy/ffmpeg not found. Install via: brew install scrcpy ffmpeg")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error launching scrcpy: {str(e)}")
            sys.exit(1)
    
    def capture_and_overlay(self):
        """Read decoded frames from ffmpeg and apply grid overlay."""
        print(f"\n📸 Starting frame capture loop...")
        print(f"Press 'q' to quit | ESC to pause\n")

        frame_size = self.frame_width * self.frame_height * 3
        start_time = time.time()

        try:
            while self.running and self.decoder and self.decoder.stdout:
                raw = self.decoder.stdout.read(frame_size)
                if not raw or len(raw) < frame_size:
                    print("\n⚠ Raw stream ended")
                    break

                frame = np.frombuffer(raw, dtype=np.uint8)
                frame = frame.reshape((self.frame_height, self.frame_width, 3))

                # Apply grid overlay
                frame = self.grid.render(frame)

                # Add frame counter to display
                cv2.putText(
                    frame,
                    f"Frame: {self.frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Display
                cv2.imshow("AOI - Digital Alignment Grid Overlay", frame)
                self.frame_count += 1

                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print(f"\n✓ User quit signal received")
                    self.running = False
                elif key == 27:  # ESC
                    print(f"\n⏸ Paused. Press any key to resume, 'q' to quit...")
                    cv2.waitKey(0)

        except KeyboardInterrupt:
            print(f"\n⚠ Interrupted by user")
        except Exception as e:
            print(f"\n✗ Capture error: {str(e)}")
        finally:
            elapsed_time = time.time() - start_time
            avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            print(f"\n📊 Capture Statistics:")
            print(f"   Total Frames: {self.frame_count}")
            print(f"   Elapsed Time: {elapsed_time:.2f}s")
            print(f"   Average FPS: {avg_fps:.2f}")
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources and terminate scrcpy."""
        print("\n🧹 Cleaning up resources...")
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        # Terminate decoder and scrcpy processes
        for label, proc in (("ffmpeg", self.decoder), ("scrcpy", self.process)):
            if not proc:
                continue
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"✓ {label} process terminated")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"✓ {label} process killed (timeout)")
            except Exception as e:
                print(f"⚠ Error terminating {label}: {str(e)}")
    
    def run(self):
        """Main execution method."""
        try:
            self.launch_scrcpy()
            self.capture_and_overlay()
        except Exception as e:
            print(f"\n✗ Fatal error: {str(e)}")
            self.cleanup()
            sys.exit(1)


class SCRCPYControlUI:
    """Minimal UI to start and stop the scrcpy capture loop."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AOI scrcpy Control")
        self.root.resizable(False, False)
        self.worker: Optional[mp.Process] = None
        self._build_ui()

    def _build_ui(self):
        frame = tk.Frame(self.root, padx=16, pady=16)
        frame.pack()

        self.status_var = tk.StringVar(value="Status: idle")
        status = tk.Label(frame, textvariable=self.status_var)
        status.pack(pady=(0, 10))

        self.start_btn = tk.Button(frame, text="Start", width=12, command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.stop_btn = tk.Button(frame, text="Stop", width=12, command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _set_status(self, text: str):
        self.status_var.set(f"Status: {text}")

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        self.worker = mp.Process(target=run_launcher, daemon=True)
        self.worker.start()

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._set_status("running")

    def stop(self):
        if not self.worker:
            return
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join(timeout=2)
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._set_status("stopped")

    def _on_close(self):
        self.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def run_launcher():
    launcher = SCRCPYSubprocessLauncher()
    launcher.run()


if __name__ == "__main__":
    ui = SCRCPYControlUI()
    ui.run()
