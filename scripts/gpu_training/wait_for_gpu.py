#!/usr/bin/env python3
"""Warte auf freie GPU-Ressourcen.

Überwacht GPU-Speicher und startet Training wenn genug VRAM verfügbar.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def get_gpu_memory():
    """Get free GPU memory in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        free_memory = int(result.stdout.strip().split('\n')[0])
        return free_memory
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def get_gpu_utilization():
    """Get GPU utilization percentage."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        utilization = int(result.stdout.strip().split('\n')[0])
        return utilization
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def wait_for_gpu(
    required_memory_mb: int = 4096,
    max_utilization: int = 50,
    check_interval: int = 30,
    timeout: int = None,
):
    """Warte auf freie GPU.
    
    Args:
        required_memory_mb: Mindestens freier VRAM in MB
        max_utilization: Maximale GPU-Auslastung in %
        check_interval: Prüfintervall in Sekunden
        timeout: Maximale Wartezeit in Sekunden (None = unendlich)
    """
    print(f"⏳ Warte auf GPU: {required_memory_mb}MB frei, <{max_utilization}% Auslastung")
    print(f"   Prüfe alle {check_interval}s (Ctrl+C zum Abbrechen)")
    print()
    
    start_time = time.time()
    checks = 0
    
    try:
        while True:
            free_memory = get_gpu_memory()
            utilization = get_gpu_utilization()
            checks += 1
            
            if free_memory is None:
                print("❌ Keine GPU gefunden!")
                return False
            
            # Zeige Status
            status = f"Check #{checks}: {free_memory}MB frei, {utilization}% Auslastung"
            
            if free_memory >= required_memory_mb and utilization <= max_utilization:
                print(f"✅ {status} - GPU VERFÜGBAR!")
                return True
            else:
                print(f"⏳ {status} - Warte...")
            
            # Timeout prüfen
            if timeout and (time.time() - start_time) > timeout:
                print(f"❌ Timeout nach {timeout}s")
                return False
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Abbruch durch Benutzer")
        return False


def main():
    parser = argparse.ArgumentParser(description="Warte auf freie GPU")
    parser.add_argument("--memory", type=int, default=4096,
                       help="Erforderlicher freier VRAM in MB (default: 4096)")
    parser.add_argument("--utilization", type=int, default=50,
                       help="Maximale GPU-Auslastung in % (default: 50)")
    parser.add_argument("--interval", type=int, default=30,
                       help="Prüfintervall in Sekunden (default: 30)")
    parser.add_argument("--timeout", type=int, default=None,
                       help="Timeout in Sekunden (default: unendlich)")
    parser.add_argument("--command", type=str, default=None,
                       help="Befehl der nach Verfügbarkeit ausgeführt wird")
    
    args = parser.parse_args()
    
    # Warte auf GPU
    success = wait_for_gpu(
        required_memory_mb=args.memory,
        max_utilization=args.utilization,
        check_interval=args.interval,
        timeout=args.timeout,
    )
    
    if success and args.command:
        print(f"\n🚀 Führe Befehl aus: {args.command}")
        result = subprocess.run(args.command, shell=True)
        sys.exit(result.returncode)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
