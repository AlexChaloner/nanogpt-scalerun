"""Prepare the familiar Tiny Shakespeare character dataset as uint16 IDs."""

import argparse
from pathlib import Path
import urllib.request

from evogpt.datasets import prepare_text


URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Use a local UTF-8 corpus instead of downloading')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.input
    if source is None:
        source = args.output / 'input.txt'
        if not source.exists():
            print(f'Downloading Tiny Shakespeare from {URL}')
            with urllib.request.urlopen(URL, timeout=60) as response:
                source.write_bytes(response.read())
    manifest = prepare_text(source, args.output, 'char', args.val_fraction)
    print(f"Prepared {manifest['train']['tokens']:,} train / {manifest['val']['tokens']:,} validation tokens; "
          f"vocab_size={manifest['tokenizer']['vocab_size']}; dataset_id={manifest['dataset_id']}")


if __name__ == '__main__':
    main()
