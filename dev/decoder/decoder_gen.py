from __future__ import annotations

import sys
from pathlib import Path
from tiktoken_ext.openai_public import ENCODING_CONSTRUCTORS


root = Path(__file__).parent.parent.parent
tpl = """char* decode(int key) {{
    static char s[][{max_len}] = {{
{c_char_arrays}
    }};
    int numKeys = sizeof(s) / sizeof(s[0]);
    if (key < 0 || key >= numKeys) {{
        return "";
    }}
    return s[key];
}}
"""

def _txt_to_c_char_array(txt: str | bytes) -> str:
    bytes_repr = txt.encode('utf-8') if isinstance(txt, str) else txt
    c_array_elements = []
    for byte in bytes_repr:
        # ASCII
        if 32 <= byte <= 126 and byte not in (ord("'"), ord('\\')):
            c_array_elements.append(f"'{chr(byte)}'")
        else:
            c_array_elements.append(f"0x{byte:02x}")
    c_array_initializer = '{ ' + ', '.join(c_array_elements) + ", '\\0' }"  # add null
    return c_array_initializer


def gen(name: str) -> None:
    """
    Generate the C code for the given encoding constructor defined in https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
    """
    if name not in ENCODING_CONSTRUCTORS:
        raise ValueError(f"Unknown encoding constructor name: {name}. Available names: {list(ENCODING_CONSTRUCTORS.keys())}")

    gen_file = f"decode_{name}.c"
    print(f"Generating {gen_file} ...")
    encoder = ENCODING_CONSTRUCTORS[name]()
    mergeable_ranks = encoder['mergeable_ranks']
    special_tokens = encoder['special_tokens']
    all_tokens = {**mergeable_ranks, **special_tokens}

    max_len = max(len(k) for k, _ in all_tokens.items()) + 1
    decoding = {v: _txt_to_c_char_array(k) for k, v in all_tokens.items()}
    c_char_arrays = '\n'.join([f'        {v},' for _, v in decoding.items()])

    with open(root / gen_file, 'w') as f:
        f.write(tpl.format(max_len=max_len, c_char_arrays=c_char_arrays))

    print("Done!")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else 'gpt2'
    gen(name)
