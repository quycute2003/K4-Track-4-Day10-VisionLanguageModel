"""Stream child-process output into notebook cells and stop children on interruption."""
import os
import subprocess
import sys


def run_module(module, *arguments, cwd=None):
    command = [sys.executable, "-u", "-m", module, *map(str, arguments)]
    print("Bắt đầu:", module, " ".join(map(str, arguments)), flush=True)
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               encoding="utf-8", errors="replace",
                               env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
