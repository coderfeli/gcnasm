#!/bin/sh
BUILD=build
ARCH=${ARCH:-gfx950}
CXXFLAGS="-std=c++17 -O3 --offload-arch=$ARCH"

rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD
/opt/rocm/bin/hipcc -x hip $CXXFLAGS ../main.hip.cc -save-temps -o test.exe
