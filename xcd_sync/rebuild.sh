#!/bin/sh
ARCH=${ARCH:-gfx950}
BUILD=build/$ARCH          # per-arch, so gfx950 and gfx1250 can coexist
CXXFLAGS="-std=c++17 -O3 --offload-arch=$ARCH"

rm -rf $BUILD ; mkdir -p $BUILD ; cd $BUILD
/opt/rocm/bin/hipcc -x hip $CXXFLAGS ../../main.hip.cc -save-temps -o test.exe
