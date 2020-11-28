#!/bin/sh
set -ex
wget https://codeload.github.com/stevengj/nlopt/tar.gz/v2.7.0
tar -xzvf v2.7.0
cd nlopt-2.7.0 && mkdir build && cd build && cmake .. && make && sudo make install && cd ../..
rm -f v2.7.0
rm -rf nlopt-2.7.0
