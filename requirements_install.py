import os
import platform
import shutil
import subprocess
import sys

def delete_venv():
    if os.path.exists("venv"):
        print("-- Removing existing virtual environment...")
        shutil.rmtree("venv")

def create_venv():
    print("-- Creating virtual environment with system site packages...")
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", "venv"], check=True)

def activate_and_install():
    print("-- Installing dependencies from requirements.txt...")

    if platform.system() == "Windows":
        subprocess.run(r"venv\Scripts\python.exe -m pip install --upgrade pip", shell=True, check=True)
        subprocess.run(r"venv\Scripts\pip.exe install -r requirements.txt", shell=True, check=True)
    else:
        # Unix/Linux (Raspberry Pi)
        command = "source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt"
        subprocess.run(command, shell=True, executable="/bin/bash", check=True)

def main():
    delete_venv()
    create_venv()
    activate_and_install()
    print("-- Setup complete. Virtual environment ready.")

if __name__ == "__main__":
    main()
