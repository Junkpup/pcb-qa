"""
AOI System: Digital Alignment Grid Overlay
==========================================

Streams scrcpy raw video to stdout and overlays a 100px grid for PCB localization.
This avoids window capture and letterboxing issues from screen-grabbing.

Workflow:
1. Launch scrcpy in headless mode with raw video output
2. Decode H.264 stream with ffmpeg
3. Render 100px digital alignment grid
4. Display for real-time QA inspection
"""

import subprocess
import cv2
import numpy as np
from typing import Optional


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


class SCRCPYCaptureEngine:
    """Captures and displays scrcpy raw video with QA overlay."""

    DISPLAY_NAME = "Inspection Engine"
    CAMERA_ID = 0
    CAMERA_SIZE = (1920, 1080)
    VIDEO_BITRATE = "16M"
    MAX_FPS = 30

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.decoder: Optional[subprocess.Popen] = None
        self.frame_width = self.CAMERA_SIZE[0]
        self.frame_height = self.CAMERA_SIZE[1]
        self.grid = DigitalAlignmentGrid(self.frame_width, self.frame_height)

    def _build_scrcpy_cmd(self) -> list:
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

    def _start_pipeline(self):
        """Start scrcpy and attach ffmpeg decoder."""
        try:
            self.process = subprocess.Popen(
                self._build_scrcpy_cmd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            if self.process.stdout is None:
                raise RuntimeError("scrcpy stdout pipe not available")

            self.decoder = subprocess.Popen(
                self._build_ffmpeg_cmd(),
                stdin=self.process.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self.process.stdout.close()

            if self.decoder.stdout is None:
                raise RuntimeError("ffmpeg stdout pipe not available")

            print("✓ scrcpy raw stream started")
            print("✓ ffmpeg decoder attached")
        except FileNotFoundError:
            print("✗ Error: scrcpy/ffmpeg not found. Install via: brew install scrcpy ffmpeg")
            raise

    def run(self):
        """Main capture loop with grid overlay."""
        print(f"\nStarting inspection engine...")
        print(f"Press 'q' to quit\n")

        frame_size = self.frame_width * self.frame_height * 3
        frame_count = 0

        try:
            self._start_pipeline()
            while self.decoder and self.decoder.stdout:
                raw = self.decoder.stdout.read(frame_size)
                if not raw or len(raw) < frame_size:
                    print("\n⚠ Raw stream ended")
                    break

                frame = np.frombuffer(raw, dtype=np.uint8)
                frame = frame.reshape((self.frame_height, self.frame_width, 3))

                # Apply grid overlay
                frame = self.grid.render(frame)

                # Display
                cv2.imshow(self.DISPLAY_NAME, frame)
                frame_count += 1

                # Check for quit command
                if cv2.waitKey(1) == ord('q'):
                    print(f"\n✓ Inspection session ended ({frame_count} frames captured)")
                    break

        except KeyboardInterrupt:
            print("\n✗ Interrupted by user")
        except Exception as e:
            print(f"\n✗ Capture error: {str(e)}")
        finally:
            cv2.destroyAllWindows()
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


if __name__ == "__main__":
    engine = SCRCPYCaptureEngine()
    engine.run()