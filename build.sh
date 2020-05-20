#!/bin/sh

# make install_deps
make build
mkdir -p bin
mv tracker bin/
