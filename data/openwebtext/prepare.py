"""Import existing nanoGPT GPT-2 uint16 token files; no corpus download."""

import argparse
from pathlib import Path

from scalerun.datasets import import_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True, type=Path)
    parser.add_argument('--val', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    manifest = import_tokens(args.train, args.val, args.output, tokenizer='gpt2')
    print(f"Imported GPT-2 uint16 token files; dataset_id={manifest['dataset_id']}")


if __name__ == '__main__':
    main()
