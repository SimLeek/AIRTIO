import numpy as np
import cv2
import logging
import subprocess
import asyncio
import threading
import time
from PyV4L2Cam.camera import Camera as pyv4lcamera
from PyV4L2Cam.convert_to_numpy import convert_mjpeg, convert_rgb24
import sounddevice as sd
from displayarray import display
from input_devices import setup_input_devices, process_keyboard_array, process_mouse_array, process_gamepad_array
from capture import get_camera_devices, get_audio_devices, start_ffmpeg_screen_capture

logging.basicConfig(level=logging.INFO)

# Global dictionary to manage display windows
display_windows = {}


def example_screen_capture():
    """Test screen capture in isolation using FFmpeg and PyV4L2Cam."""
    try:
        ffmpeg_process = start_ffmpeg_screen_capture()
        time.sleep(3)
        cam = pyv4lcamera("/dev/video10", int(1920), int(1080), None, "RGB3")
        start_time = time.time()

        while True:
            frame_bytes = cam.get_frame()
            if cam.dest_pixel_format == "MJPG":
                frame = convert_mjpeg(frame_bytes)
            elif cam.dest_pixel_format == "RGB3":
                frame = convert_rgb24(frame_bytes, cam.width, cam.height)
            else:
                logging.warning(f"Unsupported pixel format: {cam.dest_pixel_format}")
                continue
            if frame is not None:
                #frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
                print(f'frame received. fps={1.0 / (time.time() - start_time)}')
                start_time = time.time()
                continue
                if "screen" not in display_windows:
                    display_windows["screen"] = display([frame], window_names=["screen"], fps_limit=30)
                else:
                    display_windows["screen"].update(frame, "screen")
            time.sleep(1.0 / 30)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except (ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        logging.error(f"Screen capture test failed: {e}")
    except KeyboardInterrupt:
        logging.info("Stopping screen capture")
    finally:
        if 'cam' in locals():
            cam.close()
        if 'ffmpeg_process' in locals():
            ffmpeg_process.terminate()
        for disp in display_windows.values():
            disp.__exit__(None, None, None)
        display_windows.clear()


def example_camera_capture():
    """Test camera capture in isolation using PyV4L2Cam."""
    try:
        devices = get_camera_devices()
        if not devices:
            logging.error("No camera devices found")
            return
        cam = pyv4lcamera(devices[0], 640, 480)
        start_time = time.time()

        while True:
            frame_bytes = cam.get_frame()
            if cam.pixel_format == "MJPEG":
                frame = convert_mjpeg(frame_bytes)
            else:
                logging.warning(f"Unsupported pixel format: {cam.pixel_format}")
                continue
            if frame is not None:
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
                if "camera_0" not in display_windows:
                    display_windows["camera_0"] = display([frame], window_names=["camera_0"], fps_limit=30)
                else:
                    display_windows["camera_0"].update(frame, "camera_0")
                print(f'frame received. fps={1.0 / (time.time() - start_time)}')
                start_time = time.time()
            time.sleep(1.0 / 30)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except (ValueError, RuntimeError) as e:
        logging.error(f"Camera capture test failed: {e}")
    except KeyboardInterrupt:
        logging.info("Stopping camera capture")
    finally:
        if 'cam' in locals():
            cam.close()
        for disp in display_windows.values():
            disp.__exit__(None, None, None)
        display_windows.clear()


def example_mic_capture():
    """Test microphone capture in isolation using sounddevice."""
    try:
        devices, _ = get_audio_devices()
        if not devices:
            logging.error("No microphone devices found")
            return
        device_name = devices[0]

        def callback(indata, frames, time, status):
            if status:
                logging.warning(f"Audio capture status: {status}")
            arr = indata[:, 0]
            fft_result = np.fft.fft(arr)
            magnitude = np.abs(fft_result[:len(arr) // 2])
            if magnitude.max() > 0:
                magnitude = (magnitude / magnitude.max()) * 255
            else:
                magnitude = np.zeros_like(magnitude)
            magnitude = magnitude.astype(np.uint8)
            img = np.repeat(magnitude[np.newaxis, :], 100, axis=0)
            if f"mic_{device_name}_fft" not in display_windows:
                display_windows[f"mic_{device_name}_fft"] = display([img], window_names=[f"mic_{device_name}_fft"],
                                                                    fps_limit=60)
            else:
                display_windows[f"mic_{device_name}_fft"].update(img, f"mic_{device_name}_fft")

        stream = sd.InputStream(device=device_name, samplerate=44100, channels=1, callback=callback)
        stream.start()

        while True:
            time.sleep(0.1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except (ValueError, RuntimeError) as e:
        logging.error(f"Microphone capture test failed: {e}")
    except KeyboardInterrupt:
        logging.info("Stopping microphone capture")
    finally:
        if 'stream' in locals():
            stream.stop()
            stream.close()
        for disp in display_windows.values():
            disp.__exit__(None, None, None)
        display_windows.clear()


def example_sys_capture():
    """Test system audio capture in isolation using sounddevice."""
    try:
        _, monitors = get_audio_devices()
        if not monitors:
            logging.error("No monitor devices found")
            return
        device_name = monitors[0]

        def callback(indata, frames, time, status):
            if status:
                logging.warning(f"Audio capture status: {status}")
            arr = indata[:, 0]
            fft_result = np.fft.fft(arr)
            magnitude = np.abs(fft_result[:len(arr) // 2])
            if magnitude.max() > 0:
                magnitude = (magnitude / magnitude.max()) * 255
            else:
                magnitude = np.zeros_like(magnitude)
            magnitude = magnitude.astype(np.uint8)
            img = np.repeat(magnitude[np.newaxis, :], 100, axis=0)
            if f"sys_{device_name}_fft" not in display_windows:
                display_windows[f"sys_{device_name}_fft"] = display([img], window_names=[f"sys_{device_name}_fft"],
                                                                    fps_limit=60)
            else:
                display_windows[f"sys_{device_name}_fft"].update(img, f"sys_{device_name}_fft")

        stream = sd.InputStream(device=device_name, samplerate=44100, channels=1, callback=callback)
        stream.start()

        while True:
            time.sleep(0.1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except (ValueError, RuntimeError) as e:
        logging.error(f"System audio capture test failed: {e}")
    except KeyboardInterrupt:
        logging.info("Stopping system audio capture")
    finally:
        if 'stream' in locals():
            stream.stop()
            stream.close()
        for disp in display_windows.values():
            disp.__exit__(None, None, None)
        display_windows.clear()


async def example_keyboard_input():
    """Test keyboard input processing in isolation."""
    logging.basicConfig(level=logging.INFO)

    try:
        keyboard, _, _ = setup_input_devices()

        # Sample array: trigger key A (index 0) and Enter (index 31)
        keyboard_array = np.zeros(36, dtype=np.float32)
        keyboard_array[0] = 1.0  # Key A
        keyboard_array[31] = 1.0  # Enter

        while True:
            process_keyboard_array(keyboard_array, keyboard)
            await asyncio.sleep(0.5)  # Slow loop for testing
    except (ValueError, RuntimeError) as e:
        logging.error(f"Keyboard input test failed: {e}")
    except KeyboardInterrupt:
        logging.info("Stopping keyboard input test")


async def example_mouse_input():
    """Test mouse input processing in isolation."""
    logging.basicConfig(level=logging.INFO)

    try:
        _, mouse, _ = setup_input_devices()

        # Sample array: trigger left click (index 0) and move X (index 2)
        mouse_array = np.zeros(5, dtype=np.float32)
        mouse_array[0] = 1.0  # Left click
        mouse_array[2] = 0.5  # Move X (scaled to 64 pixels)

        while True:
            process_mouse_array(mouse_array, mouse)
            await asyncio.sleep(0.5)  # Slow loop for testing
    except (ValueError, RuntimeError) as e:
        logging.error(f"Mouse input test failed: {e}")
    except KeyboardInterrupt:
        logging.info("Stopping mouse input test")


async def example_gamepad_input():
    """Test gamepad input processing in isolation."""
    logging.basicConfig(level=logging.INFO)

    try:
        _, _, gamepad = setup_input_devices()

        # Sample array: trigger A button (index 0) and left stick X (index 13)
        gamepad_array = np.zeros(17, dtype=np.float32)
        gamepad_array[0] = 1.0  # A button
        gamepad_array[13] = -0.5  # Left stick X (scaled to -16384)

        while True:
            process_gamepad_array(gamepad_array, gamepad)
            await asyncio.sleep(0.5)  # Slow loop for testing
    except (ValueError, RuntimeError) as e:
        logging.error(f"Gamepad input test failed: {e}")
    except KeyboardInterrupt:
        logging.info("Stopping gamepad input test")


def main():
    """Run a specific example (modify to select one)."""
    # Example: Run screen capture test
    example_screen_capture()
    # Uncomment to test others:
    # example_camera_capture()
    # example_mic_capture()
    # example_sys_capture()
    # asyncio.run(example_keyboard_input())
    # asyncio.run(example_mouse_input())
    # asyncio.run(example_gamepad_input())


if __name__ == "__main__":
    main()