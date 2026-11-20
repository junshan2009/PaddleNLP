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

import paddle
import paddle.nn as nn

from paddlenlp.layers.crf import LinearChainCrf, LinearChainCrfLoss, ViterbiDecoder


class BiGRUCRF(nn.Layer):
    """Two layers of bidirectional GRU followed by a linear-chain CRF.

    Args:
        vocab_size (int): The size of the char vocabulary.
        num_labels (int): The number of tags, including `B`/`I` suffixes.
        emb_dim (int): The dimension of the char embedding.
        hidden_size (int): The number of hidden nodes in the GRU layer.
        emb_lr (float, optional): The learning rate ratio of the embedding layer. Defaults to 2.0.
        crf_lr (float, optional): The learning rate ratio of the crf layer. Defaults to 0.2.
        with_start_stop_tag (bool, optional): Whether to add [START] and [STOP] tags. Defaults to True.
    """

    def __init__(
        self,
        vocab_size,
        num_labels,
        emb_dim=128,
        hidden_size=128,
        emb_lr=2.0,
        crf_lr=0.2,
        with_start_stop_tag=True,
    ):
        super(BiGRUCRF, self).__init__()
        self.num_labels = num_labels
        self.init_bound = 0.1

        self.word_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            weight_attr=paddle.ParamAttr(
                learning_rate=emb_lr,
                initializer=nn.initializer.Uniform(low=-self.init_bound, high=self.init_bound),
            ),
        )

        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=2,
            direction="bidirectional",
            weight_ih_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(low=-self.init_bound, high=self.init_bound),
                regularizer=paddle.regularizer.L2Decay(coeff=1e-4),
            ),
            weight_hh_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(low=-self.init_bound, high=self.init_bound),
                regularizer=paddle.regularizer.L2Decay(coeff=1e-4),
            ),
        )

        self.fc = nn.Linear(
            in_features=hidden_size * 2,
            out_features=num_labels + 2 if with_start_stop_tag else num_labels,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(low=-self.init_bound, high=self.init_bound),
                regularizer=paddle.regularizer.L2Decay(coeff=1e-4),
            ),
        )

        self.crf = LinearChainCrf(num_labels, crf_lr, with_start_stop_tag)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions, with_start_stop_tag)

    def forward(self, input_ids, lengths, labels=None):
        """Return the crf loss if labels is given, otherwise the best tag paths."""
        word_embed = self.word_embedding(input_ids)
        bigru_output, _ = self.gru(word_embed, sequence_length=lengths)
        emission = self.fc(bigru_output)
        if labels is not None:
            return self.crf_loss(emission, lengths, labels)
        _, prediction = self.viterbi_decoder(emission, lengths)
        return prediction
