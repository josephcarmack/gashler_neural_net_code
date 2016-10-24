#!/bin/bash
set -u -e
echo "Building..."
make opt
echo "Running..."
cd ../bin
./proj1
