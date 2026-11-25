import shutil, subprocess, sys

def version(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception:
        return "NOT FOUND"

print("Python :", sys.version.split()[0])
print("In venv :", sys.prefix != sys.base_prefix)
print("Git :", version(["git", "--version"]))
print("Docker :", version(["docker", "--version"]))

for lib in ["sklearn", "pandas", "numpy", "joblib"]:
    try:
        __import__(lib)
        print(f"{lib:10}: OK")
    except ImportError:
        print(f"{lib:10}: MISSING")
