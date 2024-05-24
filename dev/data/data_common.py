"""
Common utilities for the datasets
"""

import requests
from tqdm import tqdm
import numpy as np


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def write_datafile(filename, toks):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

def write_evalfile(filename, datas):
    """
    Saves eval data as a .bin file, for reading in C.
    Used for multiple-choice style evals, e.g. HellaSwag and MMLU
    - First comes a header with 256 int32s
    - The examples follow, each example is a stream of uint16_t:
        - <START_EXAMPLE> delimiter of 2**16-1, i.e. 65,535
        - <EXAMPLE_BYTES>, bytes encoding this example, allowing efficient skip to next
        - <EXAMPLE_INDEX>, the index of the example in the dataset
        - <LABEL>, the index of the correct completion
        - <NUM_COMPLETIONS>, indicating the number of completions (usually 4)
        - <NUM><CONTEXT_TOKENS>, where <NUM> is the number of tokens in the context
        - <NUM><COMPLETION_TOKENS>, repeated NUM_COMPLETIONS times
    """
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240522 # magic
    header[1] = 1 # version
    header[2] = len(datas) # number of examples
    header[3] = 0 # reserved for longest_example_bytes, fill in later
    # now write the individual examples
    longest_example_bytes = 0 # in units of uint16s
    full_stream = [] # the stream of uint16s, we'll write a single time at the end
    assert len(datas) < 2**16, "too many examples?"
    for idx, data in enumerate(datas):
        stream = []
        # header of the example
        stream.append(2**16-1) # <START_EXAMPLE>
        stream.append(0) # <EXAMPLE_BYTES> (fill in later)
        stream.append(idx) # <EXAMPLE_INDEX>
        stream.append(data["label"]) # <LABEL>
        ending_tokens = data["ending_tokens"]
        assert len(ending_tokens) == 4, "expected 4 completions for now? can relax later"
        stream.append(len(ending_tokens)) # <NUM_COMPLETIONS>
        # the (shared) context tokens
        ctx_tokens = data["ctx_tokens"]
        assert all(0 <= t < 2**16-1 for t in ctx_tokens), "bad context token"
        stream.append(len(ctx_tokens))
        stream.extend(ctx_tokens)
        # the completion tokens
        for end_tokens in ending_tokens:
            assert all(0 <= t < 2**16-1 for t in end_tokens), "bad completion token"
            stream.append(len(end_tokens))
            stream.extend(end_tokens)
        # write to full stream
        nbytes = len(stream)*2 # 2 bytes per uint16
        assert nbytes < 2**16, "example too large?"
        stream[1] = nbytes # fill in the <EXAMPLE_BYTES> field
        longest_example_bytes = max(longest_example_bytes, nbytes)
        full_stream.extend(stream)
    # construct the numpy array
    stream_np = np.array(full_stream, dtype=np.uint16)
    # fill in the longest_example field
    assert 0 < longest_example_bytes < 2**16, f"bad longest_example"
    header[3] = longest_example_bytes
    # write to file (for HellaSwag val this is 10,042 examples, 3.6MB file)
    print(f"writing {len(datas):,} examples to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(stream_np.tobytes())
