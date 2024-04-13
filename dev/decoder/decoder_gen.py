from __future__ import annotations

import sys
from pathlib import Path
from tiktoken_ext.openai_public import ENCODING_CONSTRUCTORS


root = Path(__file__).parent.parent.parent


def _txt_to_c_readable(txt: str | bytes) -> str:
    bytes_repr = txt.encode('utf-8') if isinstance(txt, str) else txt
    c_array_elements = []
    for byte in bytes_repr:
        # printable ASCII except `\`(92)
        if 32 <= byte <= 126 and byte != ord('\\'):
            c_array_elements.append(f"{chr(byte)}")
        else:
            # use `\xHH` style with hex value for other characters
            c_array_elements.append(f"\\x{byte:02X}")
    return ''.join(c_array_elements)


def gen(name: str) -> None:
    """
    Generate the decoding tokens for the given encoding constructor defined in https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
    """
    if name not in ENCODING_CONSTRUCTORS:
        raise ValueError(f"Unknown encoding constructor name: {name}. Available names: {list(ENCODING_CONSTRUCTORS.keys())}")

    gen_file = f"decode_{name}.txt"
    print(f"Generating {gen_file} ...")

    encoder = ENCODING_CONSTRUCTORS[name]()
    mergeable_ranks = encoder['mergeable_ranks']
    special_tokens = encoder['special_tokens']
    all_tokens = {**mergeable_ranks, **special_tokens}

    decoding = {v: _txt_to_c_readable(k) for k, v in all_tokens.items()}
    all_tokens_str = '\n'.join([v for _, v in decoding.items()])

    with open(root / gen_file, 'w') as f:
        f.write(all_tokens_str)

    print("Done!")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else 'gpt2'
    gen(name)
