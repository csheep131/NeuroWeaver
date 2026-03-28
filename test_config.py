#!/usr/bin/env python3
"""Test Config loading from ENV"""
import os
import sys
sys.path.insert(0, '/home/schaf/projects/NeuroWeave')

# Setze die ENV-Variablen wie in run.sh
os.environ['NUM_LAYERS'] = '11'
os.environ['D_MODEL'] = '384'
os.environ['NUM_HEADS'] = '6'
os.environ['USE_XSA'] = '1'
os.environ['XSA_LAYERS'] = '8,9,10'
os.environ['EMA_ENABLED'] = '1'
os.environ['EMA_DECAY'] = '0.997'
os.environ['ITERATIONS'] = '9000'
os.environ['DATA_PATH'] = '/home/schaf/projects/NeuroWeave/data/datasets/fineweb10B_sp1024'

from train_gpt import Config

cfg = Config.from_env()

print("=" * 60)
print("CONFIG TEST")
print("=" * 60)
print(f"num_layers: {cfg.num_layers} (expected: 11)")
print(f"d_model: {cfg.d_model}")
print(f"num_heads: {cfg.num_heads}")
print(f"use_xsa: {cfg.use_xsa} (expected: True)")
print(f"xsa_layers: {cfg.xsa_layers} (expected: [8,9,10])")
print(f"ema_decay: {cfg.ema_decay} (expected: 0.997)")
print(f"max_steps: {cfg.max_steps} (expected: 9000)")
print(f"data_path: {cfg.data_path}")
print("=" * 60)

if cfg.num_layers != 11:
    print("ERROR: NUM_LAYERS not loaded correctly!")
else:
    print("OK: Config loaded correctly")
