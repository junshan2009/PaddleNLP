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

import argparse
import json
import os

import paddle
from data import UNK_TOKEN, decode, load_vocab
from model import BiGRUCRF

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Directory with model and vocab.")
parser.add_argument("--text", type=str, default="张伟在北京大学研究自然语言处理", help="Text to tag.")
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="The device to predict on.")
args = parser.parse_args()
# yapf: enable


def predict(model, word_vocab, label_vocab, text):
    id2label = {index: label for label, index in label_vocab.items()}
    chars = [ch for ch in text if not ch.isspace()]
    input_ids = paddle.to_tensor([[word_vocab.get(ch, word_vocab[UNK_TOKEN]) for ch in chars]], dtype="int64")
    lengths = paddle.to_tensor([len(chars)], dtype="int64")

    model.eval()
    preds = model(input_ids, lengths)[0].numpy().tolist()[: len(chars)]
    tags = [id2label[index] for index in preds]
    return decode(chars, tags)


def main(args):
    paddle.set_device(args.device)
    word_vocab, label_vocab = load_vocab(args.model_dir)
    with open(os.path.join(args.model_dir, "config.json"), "r") as fin:
        config = json.load(fin)

    model = BiGRUCRF(len(word_vocab), len(label_vocab), emb_dim=config["emb_dim"], hidden_size=config["hidden_size"])
    model.load_dict(paddle.load(os.path.join(args.model_dir, "best_model.pdparams")))

    result = predict(model, word_vocab, label_vocab, args.text)
    print(" ".join("%s/%s" % (word, tag) for word, tag in result))


if __name__ == "__main__":
    main(args)
