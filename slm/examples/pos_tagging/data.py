#   Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data utilities for POS tagging (char level input, IOB2 labels)."""

import json

CHAR_DELIMITER = "\002"

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


def _to_char_level(words, tags):
    """Convert word level (word, tag) pairs to char level IOB2 labels."""
    chars, char_tags = [], []
    for word, tag in zip(words, tags):
        if not word:
            continue
        chars.extend(list(word))
        if tag.endswith(("-B", "-I", "-E", "-S")):
            char_tags.extend([tag] * len(word))
        else:
            char_tags.append(tag + "-B")
            char_tags.extend([tag + "-I"] * (len(word) - 1))
    return chars, char_tags


def parse_sentence(line):
    """Parse one line into (chars, char_tags).

    Supported formats:
    1. char/label format used by slm/examples/lexical_analysis:
       "我\\002爱\\002北\\002京\\tt n-B\\002v-B\\002ns-B\\002ns-I"
    2. word/tag pairs: "我/r 爱/v 北京/ns"
    3. two tab separated columns: "我 爱 北京\\tr v ns"
    """
    line = line.strip()
    if not line:
        return None

    if CHAR_DELIMITER in line:
        text, tags = line.split("\t")
        return text.split(CHAR_DELIMITER), tags.split(CHAR_DELIMITER)

    if "\t" in line:
        text, tag_str = line.split("\t")
        words, tags = text.split(), tag_str.split()
        assert len(words) == len(tags), "word and tag number mismatch: %s" % line
        return _to_char_level(words, tags)

    words, tags = [], []
    for item in line.split():
        idx = item.rfind("/")
        if idx == -1:
            raise ValueError("Expected `word/tag` but got: %s" % item)
        word, tag = item[:idx], item[idx + 1 :]
        words.append(word if word else "/")
        tags.append(tag)
    assert len(words) == len(tags), "word and tag number mismatch: %s" % line
    return _to_char_level(words, tags)


def load_dataset(data_path, max_samples=None):
    """Load a corpus file, every line is one sentence."""
    dataset = []
    with open(data_path, "r", encoding="utf-8") as fin:
        for line in fin:
            example = parse_sentence(line)
            if example is None:
                continue
            chars, tags = example
            if not chars:
                continue
            dataset.append([chars, tags])
            if max_samples and len(dataset) >= max_samples:
                break
    return dataset


def build_vocab(sequences, min_freq=1, special_tokens=(PAD_TOKEN, UNK_TOKEN)):
    """Build token -> id mapping, insert special tokens at the beginning."""
    counter = {}
    for sequence in sequences:
        for token in sequence:
            counter[token] = counter.get(token, 0) + 1
    vocab = {token: index for index, token in enumerate(special_tokens)}
    for token, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def convert_example(example, max_seq_len, word_vocab, label_vocab):
    """Convert one sentence to (input_ids, length, label_ids)."""
    chars, tags = example
    chars, tags = chars[:max_seq_len], tags[:max_seq_len]
    input_ids = [word_vocab.get(ch, word_vocab[UNK_TOKEN]) for ch in chars]
    label_ids = [label_vocab.get(tag, label_vocab.get("O", 0)) for tag in tags]
    return input_ids, len(input_ids), label_ids


def decode(chars, tags):
    """Merge char level IOB2 tags back to words and their POS tags."""
    words, pos_tags = [], []
    current, current_tag = "", ""
    for ch, tag in zip(chars, tags):
        if tag == "O" or tag.endswith("-B") or tag.endswith("-S"):
            if current:
                words.append(current)
                pos_tags.append(current_tag)
            current, current_tag = ch, tag.split("-")[0]
        elif current:
            current += ch
        else:  # a leading "-I" tag, treat it as a new word
            current, current_tag = ch, tag.split("-")[0]
    if current:
        words.append(current)
        pos_tags.append(current_tag)
    return list(zip(words, pos_tags))


def save_vocab(data_dir, word_vocab, label_vocab):
    with open("%s/vocab.json" % data_dir, "w", encoding="utf-8") as fout:
        json.dump({"word_vocab": word_vocab, "label_vocab": label_vocab}, fout, ensure_ascii=False)


def load_vocab(data_dir):
    with open("%s/vocab.json" % data_dir, "r", encoding="utf-8") as fin:
        vocab = json.load(fin)
    return vocab["word_vocab"], vocab["label_vocab"]
