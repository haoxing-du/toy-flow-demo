import os
import time
import fcntl
import subprocess

sub = subprocess.Popen(
    ["python", "training.py"],
    stdout=subprocess.PIPE,
    env={"PYTHONUNBUFFERED": "true"},
)

fl = fcntl.fcntl(sub.stdout, fcntl.F_GETFL)
fcntl.fcntl(sub.stdout, fcntl.F_SETFL, fl | os.O_NONBLOCK)

while True:
    print(sub.stdout.readline())
    time.sleep(0.5)