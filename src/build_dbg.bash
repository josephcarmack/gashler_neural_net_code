#!/bin/bash
set -u -e
echo "Building..."
make dbg
echo "Running..."
cd ../bin
./proj1dbg
