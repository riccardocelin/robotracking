# setup.py
# run the script to create the exact same virtual environment as specified in requirements.txt

import os
import platform
import subprocess
import sys

def create_venv():
    print("-- Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"])

def activate_and_install():
    print("-- Installing dependencies...")

    # OS-specific activation and install
    if platform.system() == "Windows":
        # Activate the venv and install via a subprocess
        subprocess.run(r"venv\Scripts\python.exe -m pip install --upgrade pip", shell=True)
        subprocess.run(r"venv\Scripts\pip.exe install -r requirements.txt", shell=True)
    else:
        # Unix/Mac
        subprocess.run("source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt", shell=True, executable="/bin/bash")

def main():
    create_venv()
    activate_and_install()
    print("-- Setup complete. Virtual environment ready.")

if __name__ == "__main__":
    main()
