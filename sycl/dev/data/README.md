# dev/data organization

The idea is that each dataset has a .py file here in the root of `dev/data`, and each dataset then creates a directory here, and writes and caches anything inside that directory. So for example:

- running `python tinystories.py` will create a directory `tinystories` with its .bin files inside it
- running `python tinyshakespeare.py` will create a directory `tinyshakespeare` with its .bin files inside it

And so on. This way we can nicely organize multiple datasets here, share common utilities between them, and then point the .py/.c code in the root of the project accordingly to these.
