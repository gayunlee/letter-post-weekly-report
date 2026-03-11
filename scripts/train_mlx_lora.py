"""
MLX LoRA 로컬 학습 — EXAONE 7.8B on Apple Silicon

사용법:
  python3 scripts/train_mlx_lora.py --variant strict
  python3 scripts/train_mlx_lora.py --variant moderate
"""

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VERTEX_DATA_DIR = BASE_DIR / "data/training_data/v3/vertex_ai"
MLX_DATA_DIR = BASE_DIR / "data/training_data/v3/mlx"
MODEL_DIR = BASE_DIR / "models/mlx"

HF_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"


def convert_to_mlx_format(input_path: Path, output_path: Path):
    """Vertex SFT JSONL → mlx-lm chat format JSONL"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            # mlx-lm expects {"messages": [{"role": ..., "content": ...}, ...]}
            mlx_item = {
                "messages": [
                    {"role": "system", "content": item["instruction"]},
                    {"role": "user", "content": item["input"]},
                    {"role": "assistant", "content": item["output"]},
                ]
            }
            fout.write(json.dumps(mlx_item, ensure_ascii=False) + "\n")
            count += 1
    print(f"  {output_path.name}: {count}건")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["strict", "moderate"], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()

    variant = args.variant
    mlx_dir = MLX_DATA_DIR / variant
    output_dir = MODEL_DIR / f"exaone_balanced_{variant}"

    # 1. 데이터 변환
    print(f"\n[1/3] 데이터 변환 (variant={variant})")
    train_src = VERTEX_DATA_DIR / f"sft_balanced_{variant}_train.jsonl"
    val_src = VERTEX_DATA_DIR / f"sft_balanced_{variant}_val.jsonl"
    train_dst = mlx_dir / "train.jsonl"
    val_dst = mlx_dir / "valid.jsonl"

    convert_to_mlx_format(train_src, train_dst)
    convert_to_mlx_format(val_src, val_dst)

    # 2. LoRA 학습
    print(f"\n[2/3] LoRA 학습 시작")
    print(f"  모델: {HF_MODEL}")
    print(f"  epochs: {args.epochs}, batch: {args.batch_size}, lr: {args.learning_rate}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  출력: {output_dir}")
    print()

    import subprocess
    import sys

    # config for LoRA rank/alpha
    config_path = mlx_dir / "lora_config.yaml"

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", HF_MODEL,
        "--data", str(mlx_dir),
        "--train",
        "--adapter-path", str(output_dir),
        "--iters", str(args.epochs * 400),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--num-layers", "16",
        "--val-batches", "20",
        "--steps-per-eval", "100",
        "--steps-per-report", "10",
        "--mask-prompt",
        "-c", str(config_path),
    ]

    print(f"실행: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(BASE_DIR))

    if result.returncode == 0:
        print(f"\n[3/3] ✅ 학습 완료! 어댑터: {output_dir}")
    else:
        print(f"\n[3/3] ❌ 학습 실패 (exit code: {result.returncode})")
        return

    print(f"\n벤치마크 실행:")
    print(f"  python3 scripts/benchmark_mlx.py --variant {variant}")


if __name__ == "__main__":
    main()
