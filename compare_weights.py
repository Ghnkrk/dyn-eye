import os
from pathlib import Path

def main():
    p1 = Path('models/versions/best_v1_initial.pt')
    p2 = Path('runs/train/finetune_20260613_150801/weights/best.pt')
    p3 = Path('runs/train/finetune_20260613_150801/weights/last.pt')
    
    for p in [p1, p2, p3]:
        if p.exists():
            stat = p.stat()
            print(f"{p}: size={stat.st_size}, mtime={stat.st_mtime}")

if __name__ == '__main__':
    main()
