#!/usr/bin/env python3
"""
Convenience fine-tuning entry point. Forwards to train.py.
Usage:
    python train_finetune.py --model yolov9e --epochs 50
"""
from train import main

if __name__ == "__main__":
    main()
