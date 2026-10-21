"""Register processes we create, without requiring a system-wide process listing."""
import json
import os
import subprocess


def record(registry, pid):
    import psutil
    try:
        born = psutil.Process(pid).create_time()
        line = json.dumps({"pid": pid, "born": born}) + "\n"
        fd = os.open(registry, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)
    except (OSError, psutil.Error):
        pass


def install():
    registry = os.environ.get("INFRA_PROCESS_REGISTRY")
    if not registry or getattr(subprocess.Popen, "_infra_registered", False):
        return
    # multiprocessing/joblib can bypass subprocess.Popen when spawning Python.
    record(registry, os.getpid())
    original = subprocess.Popen.__init__

    def registered(self, *args, **kwargs):
        original(self, *args, **kwargs)
        record(registry, self.pid)

    subprocess.Popen.__init__ = registered
    subprocess.Popen._infra_registered = True
