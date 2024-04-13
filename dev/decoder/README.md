## Generate Decoder in C

`python decoder_gen.py gpt2` will generate `decode_gpt2.txt` in root directory, which can be loaded as a `char [52507][129]` array in C as the decoder.
* printable characters are readable as is except for `\`
* all other characters are represented as `\xHH` where `HH` is the hex value of the character
