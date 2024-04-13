#!/usr/bin/bash
if [ "x$1" = "x" ]; then
  file="python-packages.txt"
else
  file=$1
fi
if [ ! -f "$file" ]; then
  echo "install-python-packages.sh <FILE>"
  exit 1
fi
while IFS= read -r line; do
  TEST=${line:0:1}
  if [ ! "$TEST" = "#" ]; then
    package=$line
    pip install $package
  fi
done < "$file"
