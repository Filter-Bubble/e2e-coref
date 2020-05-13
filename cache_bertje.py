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

            # Use BERT tokenizer
            sentences_tokenized = [[tokenizer.tokenize(word) for word in sentence] for sentence in sentences]
            sentences_tokenized_flat = [[tok for word in sentence for tok in word] for sentence in sentences_tokenized]
            indices_flat = [[i for i,word in enumerate(sentence) for tok in word] for sentence in sentences_tokenized]

            max_nrtokens = max(len(s) for s in sentences_tokenized_flat)
            indexed_tokens = np.zeros((len(sentences), max_nrtokens), dtype=int)
            for i, sent in enumerate(sentences_tokenized_flat):
                idx = tokenizer.convert_tokens_to_ids(sent)
                indexed_tokens[i,:len(idx)] = np.array(idx)

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor(indexed_tokens)
            with torch.no_grad():
                # torch tensor of shape
                # (nr_sentences, sequence_length, hidden_size=768
                bert_output, _ = model(tokens_tensor)

            # Add up tensors for subtokens coming from same word
            max_sentence_length = max(len(s) for s in sentences)
            bert_final = torch.tensor(np.zeros((bert_output.shape[0],
                                                max_sentence_length,
                                                bert_output.shape[2])))
            for sent_id in range(len(sentences)):
                for tok_id, word_id in enumerate(indices_flat[sent_id]):
                    bert_final[sent_id, word_id, :] += bert_output[i,tok_id,:]

            text_len = np.array([len(s) for s in sentences])

            file_key = example["doc_key"].replace("/", ":")
            if file_key in out_file.keys():
                del out_file[file_key]
            group = out_file.create_group(file_key)
            for i, (e, l) in enumerate(zip(bert_final, text_len)):
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
