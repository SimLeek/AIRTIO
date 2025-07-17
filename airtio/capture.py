import asyncio
import subprocess
import os
import logging
import getpass
import numpy as np
import cv2
import sounddevice as sd
from PyV4L2Cam.camera import Camera as pyv4lcamera
from PyV4L2Cam.convert_to_numpy import convert_mjpeg
from displayarray import display
from input_devices import setup_input_devices, process_keyboard_array, process_mouse_array, process_gamepad_array
import time
import threading
import random

logging.basicConfig(level=logging.INFO)

# Global dictionary to manage display windows
display_windows = {}





def start_ffmpeg_screen_capture():
    """Start FFmpeg process for screen capture to /dev/video10."""
    try:
        cmd = [
            "ffmpeg", "-f", "x11grab", "-video_size", "1920x1080", "-framerate", "60",
            "-i", ":0.0", "-vf", "format=rgb24", "-f", "v4l2", "/dev/video10"
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logging.info("Started FFmpeg screen capture to /dev/video10")
        return process
    except Exception as e:
        logging.error(f"Failed to start FFmpeg screen capture: {e}")
        raise


def get_camera_devices():
    """List available camera devices using ls /dev/video*."""
    try:
        output = subprocess.check_output("ls /dev/video*", text=True, shell=True).strip()
        devices = output.split("\n")
        logging.info(f"Found camera devices: {devices}")
        return devices
    except subprocess.CalledProcessError:
        logging.warning("No camera devices found")
        return []


def get_audio_devices():
    """List available audio devices using pactl list sources short."""
    try:
        output = subprocess.check_output(["pactl", "list", "sources", "short"], text=True).strip()
        devices = []
        monitors = []
        for line in output.split("\n"):
            fields = line.split()
            if len(fields) < 2:
                continue
            device_name = fields[1]
            if ".monitor" in device_name:
                monitors.append(device_name)
            else:
                devices.append(device_name)
        logging.info(f"Found audio input devices: {devices}")
        logging.info(f"Found audio monitor devices: {monitors}")
        return devices, monitors
    except subprocess.CalledProcessError:
        logging.warning("No audio devices found")
        return [], []


def setup_screen_capture():
    """Set up PyV4L2Cam for screen capture from /dev/video10."""
    try:
        cam = pyv4lcamera("/dev/video10", 1920, 1080)
        logging.info("Initialized PyV4L2Cam for screen capture on /dev/video10")
        return cam
    except Exception as e:
        logging.error(f"Failed to initialize screen capture: {e}")
        raise


def setup_camera_capture(device_index=0):
    """Set up PyV4L2Cam for camera capture."""
    devices = get_camera_devices()
    if not devices or device_index >= len(devices):
        logging.error(f"No camera device at index {device_index}")
        raise ValueError(f"No camera device at index {device_index}")
    camera_device = devices[device_index]
    try:
        cam = pyv4lcamera(camera_device, 640, 480)
        logging.info(f"Initialized PyV4L2Cam for camera {camera_device}")
        return cam
    except Exception as e:
        logging.error(f"Failed to initialize camera capture: {e}")
        raise


def setup_audio_capture(device_index=0, is_monitor=False):
    """Set up sounddevice for audio capture (mic or system)."""
    devices, monitors = get_audio_devices()
    device_list = monitors if is_monitor else devices
    if not device_list or device_index >= len(device_list):
        logging.error(f"No {'monitor' if is_monitor else 'microphone'} device at index {device_index}")
        raise ValueError(f"No {'monitor' if is_monitor else 'microphone'} device at index {device_index}")
    device_name = device_list[device_index]
    return device_name, 44100, 1  # Sample rate 44100 Hz, mono


def capture_video(cam, window_name, fps_limit=30):
    """Capture and display video frames from PyV4L2Cam."""
    global display_windows
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
            if window_name not in display_windows:
                display_windows[window_name] = display([frame], window_names=[window_name], fps_limit=fps_limit)
            else:
                display_windows[window_name].update(frame, window_name)
            print(f'frame received. fps={1.0 / (time.time() - start_time)}')
            start_time = time.time()
        time.sleep(1.0 / fps_limit)


def capture_audio(device_name, sample_rate, channels, window_name):
    """Capture audio with sounddevice and display FFT."""

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
        if window_name not in display_windows:
            display_windows[window_name] = display([img], window_names=[window_name], fps_limit=60)
        else:
            display_windows[window_name].update(img, window_name)

    stream = sd.InputStream(device=device_name, samplerate=sample_rate, channels=channels, callback=callback)
    return stream


async def generate_ai_input(keyboard, mouse, gamepad):
    """Generate random AI input arrays periodically."""
    while True:
        keyboard_array = np.random.uniform(-1, 1, 36).astype(np.float32)
        mouse_array = np.random.uniform(-1, 1, 5).astype(np.float32)
        gamepad_array = np.random.uniform(-1, 1, 17).astype(np.float32)
        process_keyboard_array(keyboard_array, keyboard)
        process_mouse_array(mouse_array, mouse)
        process_gamepad_array(gamepad_array, gamepad)
        await asyncio.sleep(random.uniform(0.001, 0.030))


def install_system_packages():
    """Install required system packages and ensure uinput permissions."""
    packages = [
        "ffmpeg", "v4l2loopback-dkms", "python-pyV4L2Cam", "python-sounddevice",
        "python-opencv", "python-numpy", "python-displayarray"
    ]
    missing_packages = []
    for pkg in packages:
        try:
            subprocess.check_output(["pacman", "-Qs", pkg], stderr=subprocess.STDOUT)
            logging.info(f"Package {pkg} is already installed")
        except subprocess.CalledProcessError:
            missing_packages.append(pkg)

    if missing_packages:
        sudo_password = getpass.getpass("Enter sudo password: ")
        logging.info(f"Installing missing packages: {', '.join(missing_packages)}")
        cmd = ["sudo", "-S", "pacman", "-S", "--noconfirm"] + missing_packages
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
        process.communicate(f"{sudo_password}\n")
        if process.returncode != 0:
            logging.error("Failed to install system packages")
            raise subprocess.CalledProcessError(process.returncode, cmd)

    try:
        subprocess.run(["sudo", "-S", "chmod", "666", "/dev/uinput"], check=True,
                       input=f"{sudo_password}\n", text=True)
        logging.info("Set permissions for /dev/uinput")
    except subprocess.CalledProcessError:
        logging.warning("Failed to set /dev/uinput permissions; may require manual configuration")


def main(camera_index=0, mic_index=0, monitor_index=0):
    """Set up and run capture pipelines and AI input processing."""
    try:
        install_system_packages()
        setup_v4l2loopback()
        ffmpeg_process = start_ffmpeg_screen_capture()
        keyboard, mouse, gamepad = setup_input_devices()

        screen_cam = setup_screen_capture()
        camera_cam = setup_camera_capture(camera_index)
        mic_device, mic_sr, mic_ch = setup_audio_capture(mic_index, is_monitor=False)
        sys_device, sys_sr, sys_ch = setup_audio_capture(monitor_index, is_monitor=True)

        screen_thread = threading.Thread(target=capture_video, args=(screen_cam, "screen", 30))
        camera_thread = threading.Thread(target=capture_video, args=(camera_cam, f"camera_{camera_index}", 30))

        mic_stream = capture_audio(mic_device, mic_sr, mic_ch, f"mic_{mic_device}_fft")
        sys_stream = capture_audio(sys_device, sys_sr, sys_ch, f"sys_{sys_device}_fft")

        screen_thread.start()
        camera_thread.start()
        mic_stream.start()
        sys_stream.start()

        loop = asyncio.get_event_loop()
        loop.run_until_complete(generate_ai_input(keyboard, mouse, gamepad))

    except (ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        logging.error(f"Setup failed: {e}")
    except KeyboardInterrupt:
        logging.info("Stopping capture processes")
        if 'screen_cam' in locals():
            screen_cam.close()
        if 'camera_cam' in locals():
            camera_cam.close()
        if 'mic_stream' in locals():
            mic_stream.stop()
            mic_stream.close()
        if 'sys_stream' in locals():
            sys_stream.stop()
            sys_stream.close()
        if 'ffmpeg_process' in locals():
            ffmpeg_process.terminate()
        for disp in display_windows.values():
            disp.__exit__(None, None, None)
        display_windows.clear()


if __name__ == "__main__":
    main()