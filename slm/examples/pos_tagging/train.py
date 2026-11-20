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
import random

import paddle
from data import PAD_TOKEN, build_vocab, convert_example, load_dataset, save_vocab
from model import BiGRUCRF

from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.datasets import MapDataset
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.utils.log import logger

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--train_file", type=str, required=True, help="Training corpus, one sentence per line.")
parser.add_argument("--dev_file", type=str, default=None, help="Development corpus. 10% of train is used if not set.")
parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model and vocab.")
parser.add_argument("--init_checkpoint", type=str, default=None, help="Path to a pretrained model.")
parser.add_argument("--epochs", type=int, default=10, help="Corpus iteration num.")
parser.add_argument("--batch_size", type=int, default=64, help="The number of sequences contained in a mini-batch.")
parser.add_argument("--max_seq_len", type=int, default=128, help="Number of chars of the longest sequence.")
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="The device to train on.")
parser.add_argument("--lr", type=float, default=0.001, help="The basic learning rate.")
parser.add_argument("--crf_lr", type=float, default=0.2, help="The learning rate ratio of the crf layer.")
parser.add_argument("--emb_dim", type=int, default=128, help="The dimension of the char embedding.")
parser.add_argument("--hidden_size", type=int, default=128, help="The number of hidden nodes in the GRU layer.")
parser.add_argument("--min_freq", type=int, default=1, help="Min frequency of a char to be kept in vocab.")
parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
# yapf: enable


def create_dataloader(dataset, word_vocab, label_vocab, args, shuffle=False):
    def trans_func(example):
        return convert_example(example, args.max_seq_len, word_vocab, label_vocab)

    dataset = MapDataset([trans_func(example) for example in dataset])
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=word_vocab.get(PAD_TOKEN, 0), dtype="int64"),  # input_ids
        Stack(dtype="int64"),  # lengths
        Pad(axis=0, pad_val=label_vocab.get("O", 0), dtype="int64"),  # label_ids
    ): fn(samples)
    sampler = paddle.io.BatchSampler(dataset, batch_size=args.batch_size, shuffle=shuffle, drop_last=False)
    return paddle.io.DataLoader(dataset, batch_sampler=sampler, return_list=True, collate_fn=batchify_fn)


@paddle.no_grad()
def evaluate(model, metric, data_loader):
    model.eval()
    metric.reset()
    correct, total = 0, 0
    for input_ids, lengths, labels in data_loader:
        preds = model(input_ids, lengths)
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(lengths, preds, labels)
        metric.update(num_infer_chunks.numpy(), num_label_chunks.numpy(), num_correct_chunks.numpy())

        mask = paddle.arange(labels.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        correct += int(paddle.sum(paddle.cast((preds == labels) & mask, "int64")))
        total += int(paddle.sum(lengths))
    precision, recall, f1_score = metric.accumulate()
    accuracy = correct / max(total, 1)
    logger.info(
        "eval acc: %.4f, word precision: %.4f, recall: %.4f, f1: %.4f" % (accuracy, precision, recall, f1_score)
    )
    model.train()
    return f1_score


def train(args):
    paddle.set_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    train_set = load_dataset(args.train_file)
    if args.dev_file:
        dev_set = load_dataset(args.dev_file)
    else:
        random.seed(42)
        random.shuffle(train_set)
        split = max(1, int(len(train_set) * 0.9))
        train_set, dev_set = train_set[:split], train_set[split:]

    word_vocab = build_vocab([chars for chars, _ in train_set], min_freq=args.min_freq)
    label_vocab = build_vocab([tags for _, tags in train_set], special_tokens=("O",))
    save_vocab(args.save_dir, word_vocab, label_vocab)
    with open(os.path.join(args.save_dir, "config.json"), "w") as fout:
        json.dump({"emb_dim": args.emb_dim, "hidden_size": args.hidden_size}, fout)
    logger.info(
        "train sentences: %d, dev sentences: %d, vocab: %d, labels: %d"
        % (len(train_set), len(dev_set), len(word_vocab), len(label_vocab))
    )

    train_loader = create_dataloader(train_set, word_vocab, label_vocab, args, shuffle=True)
    dev_loader = create_dataloader(dev_set, word_vocab, label_vocab, args, shuffle=False)

    model = BiGRUCRF(
        len(word_vocab),
        len(label_vocab),
        emb_dim=args.emb_dim,
        hidden_size=args.hidden_size,
        crf_lr=args.crf_lr,
    )
    if args.init_checkpoint:
        model.load_dict(paddle.load(args.init_checkpoint))

    optimizer = paddle.optimizer.Adam(learning_rate=args.lr, parameters=model.parameters())
    chunk_evaluator = ChunkEvaluator(label_list=list(label_vocab.keys()), suffix=True)

    best_f1 = -1.0
    for epoch in range(args.epochs):
        model.train()
        for step, (input_ids, lengths, labels) in enumerate(train_loader):
            loss = paddle.mean(model(input_ids, lengths, labels))
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if (step + 1) % args.logging_steps == 0:
                logger.info("epoch %d, step %d, loss: %.6f" % (epoch + 1, step + 1, float(loss)))
        f1_score = evaluate(model, chunk_evaluator, dev_loader)
        if f1_score > best_f1:
            best_f1 = f1_score
            paddle.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pdparams"))
            logger.info("Save best model, f1: %.4f" % best_f1)


if __name__ == "__main__":
    train(parser.parse_args())
