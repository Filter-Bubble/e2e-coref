from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import h5py
import json
import sys


def load_bertje():
    tokenizer = BertTokenizer.from_pretrained("bert-base-dutch-cased")
    model = BertModel.from_pretrained("bert-base-dutch-cased")
    model.eval()
    return tokenizer, model


def cache_dataset(data_path, out_file, tokenizer, model):
    with open(data_path) as in_file:
        for doc_num, line in enumerate(in_file.readlines()):
            example = json.loads(line)
            sentences = example["sentences"]
            max_sentence_length = max(len(s) for s in sentences)
            tokens = [[""] * max_sentence_length for _ in sentences]
            text_len = np.array([len(s) for s in sentences])
            for i, sentence in enumerate(sentences):
                for j, word in enumerate(sentence):
                    tokens[i][j] = word

            # Encode
            indexed_tokens = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor(indexed_tokens)
            with torch.no_grad():
                # torch tensor of shape
                # (nr_sentences, sequence_length, hidden_size=768
                bert_output, _ = model(tokens_tensor)

            file_key = example["doc_key"].replace("/", ":")
            if file_key in out_file.keys():
                del out_file[file_key]
            group = out_file.create_group(file_key)
            for i, (e, l) in enumerate(zip(bert_output, text_len)):
                e = np.array(e[:l, :])
                # Add extra dim because elmo has this
                e = np.expand_dims(e, axis=2)
                group[str(i)] = e
            if doc_num % 10 == 0:
                print("Cached {} documents in {}".format(doc_num + 1, data_path))


if __name__ == "__main__":
    tokenizer, model = load_bertje()
    with h5py.File("bertje_cache.hdf5", "a") as out_file:
        for json_filename in sys.argv[1:]:
            cache_dataset(json_filename, out_file, tokenizer, model)
