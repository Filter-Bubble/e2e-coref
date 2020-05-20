#!/bin/bash

# Download pretrained embeddings.
# wget http://www.clips.uantwerpen.be/dutchembeddings/combined-320.tar.gz
# tar -xf combined-320.tar.gz
# mv 320/combined-320.txt data/combined-320.txt
# rm -rf 320/
# rm combined-320.tar.gz
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nl.300.vec.gz
gzip -d cc.nl.300.vec.gz
mv cc.nl.300.vec data/fasttext.300.vec


# Download BERT-NL model
wget http://textdata.nl/bert-nl/dutch_cased_punct_L-12_H-768_A-12.zip
unzip dutch_cased_punct_L-12_H-768_A-12.zip -x "dutch_cased_punct_L-12_H-768_A-12/*"
mv dutch_cased_punct_L-12_H-768_A-12-NEW data/bert-nl
rm dutch_cased_punct_L-12_H-768_A-12.zip

# Build custom kernels.
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# Linux (pip)
g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# Linux (build from source)
#g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2

# Mac
#g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -I -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -undefined dynamic_lookup
