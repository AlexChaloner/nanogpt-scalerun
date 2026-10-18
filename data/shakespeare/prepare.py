"""Prepare Tiny Shakespeare with the GPT-2 BPE tokenizer (optional tiktoken)."""

import argparse
from pathlib import Path
import urllib.request

from evogpt.datasets import prepare_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.input or args.output / 'input.txt'
    if args.input is None and not source.exists():
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with urllib.request.urlopen(url, timeout=60) as response:
            source.write_bytes(response.read())
    manifest = prepare_text(source, args.output, 'gpt2', args.val_fraction)
    print(f"Prepared GPT-2 token streams; dataset_id={manifest['dataset_id']}")


if __name__ == '__main__':
    main()
