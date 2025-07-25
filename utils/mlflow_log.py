"""
This module contains functions to log details about the
execution environment, like GPU information, Python environment,
and Git repository status.
"""
import mlflow
import platform
import subprocess
import sys
import os
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetName, nvmlShutdown, nvmlDeviceGetMemoryInfo, nvmlDeviceG

def log_gpu_info():
    try:
        nvmlInit()
        count = nvmlDeviceGetCount()
        for i in range(count):
            handle = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()

            mem_info = nvmlDeviceGetMemoryInfo(handle)
            mlflow.set_tag(f"gpu.{i}.name", name)
            mlflow.set_tag(f"gpu.{i}.memory.total_MB", mem_info.total // (1024**2))

        mlflow.set_tag("gpu.count", count)
        nvmlShutdown()
    except ImportError:
        mlflow.set_tag("gpu.info", "pynvml not intalled")
    except Exception as e:
        mlflow.set_tag("gpu.error", str(e))

def log_python_env():
    """
    Logs basic Python version and platform info.
    Avoids logging installed packages if already handled by autolog.
    """
    mlflow.set_tag("python.version", sys.version.split()[0])
    mlflow.set_tag("platform", platform.platform())

    try:
        pip_freeze = subprocess.check_output(["pip", "freeze"]).decode()
        mlflow.log_text(pip_freeze, artifact_file="environment.txt")
    except Exception as e:
        mlflow.set_tag("pip_freeze_error", str(e))

def is_inside_git_repo():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return result.stdout.decode().strip() == "true"
    except subprocess.CalledProcessError:
        return False
        
def log_git_info():
    if not is_inside_git_repo():
        mlflow.set_tag("git.status", "Not a git repo")
        return

    try:
        commit = subprocess.getoutput("git rev-parse HEAD").strip()
        branch = subprocess.getoutput("git rev-parse --abbrev-ref HEAD").strip()
        mlflow.set_tag("git.commit", commit)
        mlflow.set_tag("git.branch", branch)

        diff = subprocess.getoutput("git diff")
        mlflow.log_text(diff, "git_diff.txt")

        message = subprocess.getoutput("git log -1 --pretty=%B").strip()
        mlflow.log_text(message, "git_commit_message.txt")
    except Exception as e:
        mlflow.set_tag("git_error", str(e))