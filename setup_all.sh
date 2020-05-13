#!/bin/bash

# Download pretrained embeddings.
# wget http://www.clips.uantwerpen.be/dutchembeddings/combined-320.tar.gz
# tar -xf combined-320.tar.gz
# mv 320/combined-320.txt combined-320.txt
# rm -rf 320/
# rm combined-320.tar.gz

# Build custom kernels.
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# Linux (pip)
g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# Linux (build from source)
#g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2

# Mac
#g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -I -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -undefined dynamic_lookup
