import time
from picamera2 import Picamera2, Preview
import cv2
import numpy as np


class PiCameraCapture:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.picam2 = Picamera2()
        self.config = self.picam2.create_preview_configuration(
            main={"size": resolution, "format": "XBGR8888"}
        )
        self.picam2.configure(self.config)

    def start_preview(self):
        self.picam2.start_preview(Preview.QTGL)
        self.picam2.start()
        time.sleep(2)  # Allow camera to warm up

    def stop_preview(self):
        self.picam2.stop_preview()
        self.picam2.stop()

    def capture_image(self):
        return self.picam2.capture_array()

    def save_image(self, filename="capture.jpg"):
        img = self.capture_image()
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Image saved as {filename}")

    def display_video(self, duration=30):
        print(
            f"Displaying video feed for {duration} seconds. Press Ctrl+C to stop early."
        )
        try:
            self.start_preview()
            time.sleep(duration)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_preview()
            print("Video display stopped")


def test_image_capture(camera):
    camera.start_preview()
    try:
        camera.save_image("test_capture.jpg")
    finally:
        camera.stop_preview()


def test_video_display(camera):
    camera.display_video(duration=30)  # Display for 30 seconds


if __name__ == "__main__":
    try:
        camera = PiCameraCapture()

        # Test image capture
        test_image_capture(camera)

        # Test video display
        test_video_display(camera)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Script execution completed")
