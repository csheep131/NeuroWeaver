#!/usr/bin/env python3
"""Watch GPU and automatically start training when free.

Continuously monitors GPU and runs the training queue.
"""

import subprocess
import sys
import time
from pathlib import Path


def main():
print(" GPU Watcher - Startet Training wenn GPU frei")
print("=" * 60)
print("Drücke Ctrl+C zum Beenden")
print()

script_dir = Path(__file__).parent

try:
while True:
# Check queue
result = subprocess.run(
[sys.executable, str(script_dir / "training_queue.py"), "list"],
capture_output=True,
text=True,
)

# Check if there are pending runs
if "pending" not in result.stdout and "waiting" not in result.stdout:
print("\n Keine pending Runs mehr - Beende Watcher")
break

# Try to run next
print("\n Prüfe Queue und GPU...")
run_result = subprocess.run(
[sys.executable, str(script_dir / "training_queue.py"), "run"]
)

if run_result.returncode != 0:
print(" Warte 60s vor nächstem Versuch...")
time.sleep(60)
else:
print(" Warte 10s vor nächstem Run...")
time.sleep(10)

except KeyboardInterrupt:
print("\n\n Watcher gestoppt")


if __name__ == "__main__":
main()
