#!/bin/bash

dlx() {
  wget $1/$2
  tar -xvzf $2
  rm $2
}

dlx http://conll.cemantix.org/download reference-coreference-scorers.v8.01.tar.gz
mkdir conll-2012
mv reference-coreference-scorers conll-2012/scorer

# TODO: download data
python minimize.py data/train.dutch.conll
python minimize.py data/dev.dutch.conll

python get_char_vocab.py

python filter_embeddings.py false data/combined-320.txt data/train.dutch.jsonlines data/dev.dutch.jsonlines

# If using bertje, need to install transformers
python cache_bert.py bertje data/train.dutch.jsonlines data/dev.dutch.jsonlines
