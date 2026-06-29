# use this script to freeze all packages in the virtual environment

import os
import platform
import subprocess
import sys

def get_pip_path():
    os_type = platform.system()
    if os_type == "Windows":
        pip_path = os.path.join(".venv", "Scripts", "pip.exe")
    else:
        pip_path = os.path.join(".venv", "bin", "pip")

    if not os.path.isfile(pip_path):
        print("Could not find pip in your virtual environment.")
        print("Make sure the venv is created in './venv/'")
        sys.exit(1)

    return pip_path

def freeze_requirements(pip_path):
    print("Freezing packages...")
    with open("requirements.txt", "w") as f:
        subprocess.run([pip_path, "freeze"], stdout=f, check=True)

def main():
    pip_path = get_pip_path()
    freeze_requirements(pip_path)

main()
