import subprocess
import os
import logging
import getpass

logging.basicConfig(level=logging.INFO)

def detect_package_manager():
    """Detect the system's package manager."""
    os_info = {
        "/etc/debian_version": "apt-get",
        "/etc/redhat-release": "yum",
        "/etc/arch-release": "pacman",
        "/etc/gentoo-release": "emerge",
        "/etc/SuSE-release": "zypp",
        "/etc/alpine-release": "apk"
    }
    for path, manager in os_info.items():
        if os.path.isfile(path):
            logging.info(f"Detected package manager: {manager}")
            return manager
    logging.error("No supported package manager detected")
    raise RuntimeError("Unsupported package manager")


def get_system_packages(manager):
    """Return system packages for the given package manager."""
    package_map = {
        "apt-get": [
            "gstreamer1.0-pipewire",
            "gstreamer1.0-plugins-good",
            "gstreamer1.0-plugins-bad",
            "python3-gi",
            "python3-opencv",
            "libudev-dev"
        ],
        "yum": [
            "gstreamer1-pipewire",
            "gstreamer1-plugins-good",
            "gstreamer1-plugins-bad",
            "python3-gobject",
            "opencv-python",
            "libudev-devel"
        ],
        "pacman": [
            "pipewire",
            "gstreamer",
            "gst-plugins-good",
            "gst-plugins-bad",
            "python-gobject",
            "opencv",
            "libudev",
            "gst-plugin-pipewire"
        ],
        "emerge": [
            "media-libs/gstreamer",
            "media-libs/gst-plugins-good",
            "media-libs/gst-plugins-bad",
            "dev-python/pygobject",
            "media-libs/opencv",
            "sys-libs/libudev"
        ],
        "zypp": [
            "gstreamer",
            "gstreamer-plugins-good",
            "gstreamer-plugins-bad",
            "python3-gobject",
            "opencv",
            "libudev1"
        ],
        "apk": [
            "gstreamer",
            "gst-plugins-good",
            "gst-plugins-bad",
            "py3-gobject3",
            "opencv",
            "libudev-zero"
        ]
    }
    return package_map.get(manager, [])


def check_package_installed(manager, pkg):
    """Check if a package is installed."""
    try:
        if manager == "apt-get":
            subprocess.check_output(["dpkg", "-s", pkg], stderr=subprocess.STDOUT)
        elif manager == "yum":
            subprocess.check_output(["rpm", "-q", pkg], stderr=subprocess.STDOUT)
        elif manager == "pacman":
            subprocess.check_output(["pacman", "-Qs", pkg], stderr=subprocess.STDOUT)
        elif manager == "emerge":
            subprocess.check_output(["qlist", "-I", pkg], stderr=subprocess.STDOUT)
        elif manager == "zypp":
            subprocess.check_output(["rpm", "-q", pkg], stderr=subprocess.STDOUT)
        elif manager == "apk":
            subprocess.check_output(["apk", "info", "-e", pkg], stderr=subprocess.STDOUT)
        logging.info(f"Package {pkg} is already installed")
        return True
    except subprocess.CalledProcessError:
        return False


def ensure_uinput_permissions():
    """Ensure permissions for /dev/uinput."""
    try:
        subprocess.run(["sudo", "-S", "chmod", "666", "/dev/uinput"], check=True,
                       input=f"{getpass.getpass('Enter sudo password: ')}\n", text=True)
        logging.info("Set permissions for /dev/uinput")
    except subprocess.CalledProcessError:
        logging.warning("Failed to set /dev/uinput permissions; may require manual configuration")


def install_system_packages():
    """Install required system packages and ensure uinput permissions."""
    manager = detect_package_manager()
    packages = get_system_packages(manager)
    missing_packages = [pkg for pkg in packages if not check_package_installed(manager, pkg)]

    if missing_packages:
        sudo_password = getpass.getpass("Enter sudo password: ")
        logging.info(f"Installing missing packages: {', '.join(missing_packages)}")

        if manager == "apt-get":
            cmd = ["sudo", "-S", "apt-get", "install", "-y"] + missing_packages
        elif manager == "yum":
            cmd = ["sudo", "-S", "yum", "install", "-y"] + missing_packages
        elif manager == "pacman":
            cmd = ["sudo", "-S", "pacman", "-S", "--noconfirm"] + missing_packages
        elif manager == "emerge":
            cmd = ["sudo", "-S", "emerge", "-av"] + missing_packages
        elif manager == "zypp":
            cmd = ["sudo", "-S", "zypper", "install", "-y"] + missing_packages
        elif manager == "apk":
            cmd = ["sudo", "-S", "apk", "add"] + missing_packages
        else:
            raise RuntimeError("Unsupported package manager")

        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
        process.communicate(f"{sudo_password}\n")
        if process.returncode != 0:
            logging.error("Failed to install system packages")
            raise subprocess.CalledProcessError(process.returncode, cmd)

    ensure_uinput_permissions()

def setup_v4l2loopback():
    """Set up v4l2loopback module for screen capture."""
    try:
        # Check if v4l2loopback is already loaded
        result = subprocess.run(["lsmod"], capture_output=True, text=True)
        if "v4l2loopback" not in result.stdout:
            sudo_password = getpass.getpass("Enter sudo password to load v4l2loopback: ")
            cmd = ["sudo", "-S", "modprobe", "v4l2loopback", "devices=1", "video_nr=10"]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True)
            stdout, stderr = process.communicate(f"{sudo_password}\n")
            if process.returncode != 0:
                logging.error(f"Failed to load v4l2loopback: {stderr}")
                raise subprocess.CalledProcessError(process.returncode, cmd)
            logging.info("v4l2loopback module loaded")
        else:
            logging.info("v4l2loopback module already loaded")

        # Ensure /dev/video10 permissions
        subprocess.run(["sudo", "-S", "chmod", "666", "/dev/video10"], check=True,
                       input=f"{sudo_password}\n", text=True)
        logging.info("Set permissions for /dev/video10")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to set up v4l2loopback: {e}")
        raise

if __name__ =='__main__':
    install_system_packages()
    setup_v4l2loopback()