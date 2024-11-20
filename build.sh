#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build
make

# Copy library to lib directory
mkdir -p ../lib
cp backtester_core*.so ../lib/
