## Generate Decoder in C

* `python decoder_gen.py gpt2` will generate `decode_gpt2.c` in root directory, the generated code is like below.

```c
char* decode(int key) {
    static char s[][{max_len}] = {
         { '!', '\0' },
         { '"', '\0' },
         { '#', '\0' },
         { '$', '\0' },
         { '%', '\0' },
         ...
    };
    int numKeys = sizeof(s) / sizeof(s[0]);
    if (key < 0 || key >= numKeys) {
        return "";
    }
    return s[key];
}
```
