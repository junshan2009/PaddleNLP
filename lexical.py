# coding:utf-8
"""
把 `Taskflow("lexical_analysis")` 一路透传到 PaddlePaddle 底层的单文件实现。

原始调用链（左边是 Taskflow 的封装层级，右边是本文件中对应的展开位置）
=====================================================================

    from paddlenlp import Taskflow
    lac = Taskflow("lexical_analysis")
    lac("LAC是个优秀的分词工具")

  ① paddlenlp/taskflow/taskflow.py:110   TASKS["lexical_analysis"]["models"]["lac"]
       -> task_class = LacTask, emb_dim = 128, hidden_size = 128,
          task_flag = "lexical_analysis-gru_crf", task_priority_path = "lac"
       -> default model = "lac"                              [本文件 LAC_MODEL_CONFIG]

  ② paddlenlp/taskflow/taskflow.py:834   Taskflow.__init__
       -> get_env_device() + paddle.set_device(...)          [Lac.__init__ / _setup_device]
       -> kwargs["device_id"] = 0, kwargs.update(config)
       -> task_instance = LacTask(task, model, priority_path, **kwargs)

  ③ paddlenlp/taskflow/task.py:44        Task.__init__
       -> _task_path = $PPNLP_HOME/taskflow/lac   (task_priority_path="lac")
       -> _num_threads = ceil(cpu_count() / 2), _predictor_type = "paddle-inference"
       -> download_check(task_flag)

  ④ paddlenlp/taskflow/lexical_analysis.py:106  LacTask.__init__
       -> _check_task_files()     下载/校验 model_state.pdparams, tag.dic, q2b.dic, word.dic
                                                             [check_and_download_resources]
       -> _construct_vocabs()     load_vocab + id2word / id2tag 反转
                                                             [Lac._construct_vocabs]
       -> _get_inference_model()                             [Lac._get_inference_model]
       -> _max_seq_len = 512

  ⑤ paddlenlp/taskflow/task.py:310       Task._get_inference_model
       非静态模型分支（is_static_model=False）：
       -> inference_model_path = _task_path/static/inference
       -> 若 .pdiparams 不存在或参数已更新：
            dygraph_mode_guard():                            [paddle.disable_static()]
              _construct_model()      BiGruCrf(...) + paddle.load(...) + set_dict + eval()
                                                             [build_bigru_crf]
              _construct_input_spec() InputSpec(token_ids[int64,None,None]),
                                      InputSpec(length[int64,None])
                                                             [input_spec]
              _convert_dygraph_to_static()
                                      paddle.jit.to_static(model, input_spec, full_graph=True)
                                      paddle.jit.save(model, inference_model_path)
       -> _static_model_file  = inference_model_path + (.json | .pdmodel)   # PIR 开关决定后缀
       -> _static_params_file = inference_model_path + .pdiparams

  ⑥ paddlenlp/taskflow/task.py:216       Task._prepare_static_mode
       -> config = paddle.inference.Config(model_file, params_file)
       -> CPU : config.disable_gpu(); config.enable_mkldnn()
          GPU : config.enable_use_gpu(100, device_id); config.delete_pass(...)
       -> config.switch_ir_optim(True); config.enable_new_executor()
          config.set_cpu_math_library_num_threads(n); config.switch_use_feed_fetch_ops(False)
          config.disable_glog_info(); config.enable_memory_optim()
       -> predictor      = paddle.inference.create_predictor(config)     # <== 到这里就是 PaddlePaddle C++ 推理引擎
       -> input_handles  = [predictor.get_input_handle(n)  for n in predictor.get_input_names()]
       -> output_handle  = [predictor.get_output_handle(n) for n in predictor.get_output_names()]

  ⑦ paddlenlp/taskflow/task.py:567       Task.__call__ = _preprocess -> _run_model -> _postprocess
       -> lexical_analysis.py:158 _preprocess()
            _check_input_text / _auto_splitter(512) / q2b 归一化 / word.dic 查表 -> (ids, len)
            paddle.io.DataLoader(collate_fn=Tuple(Pad(int64), Stack(int64)))
                                                             [Lac._preprocess，直接用 numpy 组 batch]
       -> lexical_analysis.py:209 _run_model()
            input_handles[0].copy_from_cpu(input_ids)
            input_handles[1].copy_from_cpu(seq_len)
            predictor.run()                                  # <== PaddlePaddle 底层执行
            output_handle[0].copy_to_cpu()  -> tag_ids
                                                             [Lac._run_model]
       -> lexical_analysis.py:228 _postprocess()
            id2tag -> IOB2 边界还原 segs/tags -> _auto_joiner 还原长句
                                                             [Lac._postprocess]

模型结构（slm/examples/lexical_analysis/README.md）：
    one-hot 字 id -> Embedding(128) -> 2 层 BiGRU(hidden=128) -> FC(num_tags+2) -> CRF/Viterbi 解码

用法
====
    python lexical.py "LAC是个优秀的分词工具"
    python lexical.py --backend dygraph "三亚是一个美丽的城市"   # 不走 paddle.inference，直接跑动态图
    python lexical.py --verbose ...                              # 打印每层的中间产物

未包含的分支（与推理主链路无关，故略去）
======================================
    - LacTask(user_dict=...) 的用户词典干预：taskflow/utils.py:684 Customization + TriedTree(AC 自动机)
    - fp16/int8/onnxruntime 后端：task.py:264 _prepare_onnx_mode、npu 混合精度转换
    - is_static_model=True（直接加载已导出的静态模型，跳过 jit.to_static）
"""

import hashlib
import math
import os
import re
import sys
import urllib.request
from multiprocessing import cpu_count

import numpy as np
import paddle
import paddle.nn as nn

try:
    from paddle.text import ViterbiDecoder
except Exception:  # pragma: no cover
    raise ImportError(
        "Taskflow requires paddle version >= 2.2.0, but current paddle version is {}".format(
            paddle.version.full_version
        )
    )


# --------------------------------------------------------------------------------------
# ① 任务配置：对应 taskflow.py:110 TASKS["lexical_analysis"]["models"]["lac"]
# --------------------------------------------------------------------------------------
LAC_MODEL_CONFIG = {
    "hidden_size": 128,
    "emb_dim": 128,
    "task_flag": "lexical_analysis-gru_crf",
    "task_priority_path": "lac",
}

# 对应 lexical_analysis.py:79 LacTask.resource_files_names / resource_files_urls["lac"]
RESOURCE_FILES = {
    "model_state": (
        "model_state.pdparams",
        "https://bj.bcebos.com/paddlenlp/taskflow/lexical_analysis/lac/model_state.pdparams",
        "3d4008c6c9d29424465829c9acf909bd",
    ),
    "tags": (
        "tag.dic",
        "https://bj.bcebos.com/paddlenlp/taskflow/lexical_analysis/lac/tag.dic",
        "b11b616926b9f7f0a40a8087f84a8a99",
    ),
    "q2b": (
        "q2b.dic",
        "https://bj.bcebos.com/paddlenlp/taskflow/lexical_analysis/lac/q2b.dic",
        "4ef2cd16f8002fe7cd7dd31cdff47e0d",
    ),
    "word": (
        "word.dic",
        "https://bj.bcebos.com/paddlenlp/taskflow/lexical_analysis/lac/word.dic",
        "f1dfc68139bb6dd58c9c4313c341e436",
    ),
}

# 对应 paddlenlp/utils/env.py:67 PPNLP_HOME（可用环境变量 PPNLP_HOME 覆盖）
PPNLP_HOME = os.environ.get("PPNLP_HOME", os.path.join(os.path.expanduser("~"), ".paddlenlp"))

# 对应 paddlenlp/utils/env.py:24-29,160-165：PIR 模式下静态模型后缀是 .json，否则 .pdmodel
try:
    from paddle.base.framework import use_pir_api

    _PIR_ENABLED = use_pir_api()
except ImportError:
    _PIR_ENABLED = False
MODEL_SUFFIX = ".json" if _PIR_ENABLED else ".pdmodel"
PARAMS_SUFFIX = ".pdiparams"


# --------------------------------------------------------------------------------------
# ③④ 资源下载：对应 Task._check_task_files + taskflow/utils.py:61 download_file
# --------------------------------------------------------------------------------------
def md5file(fname):
    """对应 paddle.dataset.common.md5file"""
    m = hashlib.md5()
    with open(fname, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            m.update(block)
    return m.hexdigest()


def download_file(save_dir, filename, url, md5=None):
    """对应 taskflow/utils.py:61 download_file（底层是 get_path_from_url）"""
    os.makedirs(save_dir, exist_ok=True)
    fullname = os.path.join(save_dir, filename)
    print("Downloading {} from {}".format(filename, url))
    urllib.request.urlretrieve(url, fullname)
    if md5 and md5file(fullname) != md5:
        raise RuntimeError("md5 check failed for {}: expect {}, got {}".format(fullname, md5, md5file(fullname)))
    return fullname


def check_and_download_resources(task_path, model="lac"):
    """对应 lexical_analysis.py:154 Task._check_task_files：逐个校验 md5，缺失或过期才下载"""
    for file_id, (file_name, url, md5) in RESOURCE_FILES.items():
        path = os.path.join(task_path, file_name)
        if not os.path.exists(path) or md5file(path) != md5:
            download_file(task_path, file_name, url, md5)


# --------------------------------------------------------------------------------------
# ④ 词表：对应 lexical_analysis.py:45 load_vocab / lexical_analysis.py:129 _construct_vocabs
# --------------------------------------------------------------------------------------
def load_vocab(dict_path):
    """
    Load vocab from file
    """
    vocab = {}
    reverse = None
    with open(dict_path, "r", encoding="utf8") as fin:
        for i, line in enumerate(fin):
            terms = line.strip("\n").split("\t")
            if len(terms) == 2:
                if reverse is None:
                    reverse = True if terms[0].isdigit() else False
                if reverse:
                    value, key = terms
                else:
                    key, value = terms
            elif len(terms) == 1:
                key, value = terms[0], str(i)
            else:
                raise ValueError("Error line: %s in file: %s" % (line, dict_path))
            vocab[key] = value
    return vocab


# --------------------------------------------------------------------------------------
# ⑤ 模型：对应 paddlenlp/taskflow/models/lexical_analysis_model.py:31 BiGruCrf
#          + paddlenlp/layers/crf.py:31 LinearChainCrf
#   说明：推理只用得到 CRF 的 transitions 参数（喂给 paddle.text.ViterbiDecoder），
#         LinearChainCrf.forward / gold_score / LinearChainCrfLoss 只在训练算 loss 时用到，
#         这里省略；参数名（crf.transitions）与官方 model_state.pdparams 完全一致，
#         paddle 的 set_dict 只按名字取参数，多余的 key 会被忽略。
# --------------------------------------------------------------------------------------
class LinearChainCrf(nn.Layer):
    """paddlenlp/layers/crf.py:31 的推理子集：只保留 transitions 参数"""

    def __init__(self, num_labels, crf_lr=0.1, with_start_stop_tag=True):
        super().__init__()
        if with_start_stop_tag:
            self.num_tags = num_labels + 2  # Additional [START] and [STOP]
            self.start_idx = int(self.num_tags - 1)
            self.stop_idx = int(self.num_tags - 2)
        else:
            self.num_tags = num_labels

        # 对应 crf.py:56 create_parameter(attr=ParamAttr(learning_rate=crf_lr), shape=[num_tags, num_tags])
        self.transitions = self.create_parameter(
            attr=paddle.ParamAttr(learning_rate=crf_lr), shape=[self.num_tags, self.num_tags], dtype="float32"
        )
        self.with_start_stop_tag = with_start_stop_tag


class BiGruCrf(nn.Layer):
    """
    Embedding -> 2 层 BiGRU -> FC -> CRF(Viterbi)
    对应 paddlenlp/taskflow/models/lexical_analysis_model.py:31
    """

    def __init__(
        self, word_emb_dim, hidden_size, vocab_size, num_labels, emb_lr=2.0, crf_lr=0.2, with_start_stop_tag=True
    ):
        super().__init__()
        self.word_emb_dim = word_emb_dim
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.init_bound = 0.1

        # lexical_analysis_model.py:54
        self.word_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.word_emb_dim,
            weight_attr=paddle.ParamAttr(
                learning_rate=emb_lr,
                initializer=nn.initializer.Uniform(low=-self.init_bound, high=self.init_bound),
            ),
        )

        # lexical_analysis_model.py:63  两层双向 GRU
        self.gru = nn.GRU(
            input_size=self.word_emb_dim,
            hidden_size=self.hidden_size,
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

        # lexical_analysis_model.py:78  emission = FC(bigru_out)，输出维度 num_labels + 2（含 START/STOP）
        self.fc = nn.Linear(
            in_features=self.hidden_size * 2,
            out_features=self.num_labels + 2 if with_start_stop_tag else self.num_labels,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(low=-self.init_bound, high=self.init_bound),
                regularizer=paddle.regularizer.L2Decay(coeff=1e-4),
            ),
        )

        # lexical_analysis_model.py:87-89
        self.crf = LinearChainCrf(self.num_labels, crf_lr, with_start_stop_tag)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions, with_start_stop_tag)

    def forward(self, inputs, lengths, labels=None):
        """
        lexical_analysis_model.py:91
            word_embed = Embedding(inputs)                 # [B, T] -> [B, T, 128]
            bigru_output, _ = GRU(word_embed, seq_len)     # [B, T, 128] -> [B, T, 256]
            emission = FC(bigru_output)                    # [B, T, 256] -> [B, T, num_tags]
            _, prediction = ViterbiDecoder(emission, lengths)   # CRF 维特比解码 -> [B, T] int64
        """
        word_embed = self.word_embedding(inputs)
        bigru_output, _ = self.gru(word_embed, sequence_length=lengths)
        emission = self.fc(bigru_output)
        if labels is not None:  # 训练分支，推理走不到
            raise NotImplementedError("LinearChainCrfLoss 只在训练时使用，本文件只做推理")
        _, prediction = self.viterbi_decoder(emission, lengths)
        return prediction


def input_spec():
    """对应 lexical_analysis.py:120 LacTask._construct_input_spec"""
    return [
        paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_ids"),
        paddle.static.InputSpec(shape=[None], dtype="int64", name="length"),
    ]


# --------------------------------------------------------------------------------------
# Taskflow 主体：把 ①~⑦ 全部串起来
# --------------------------------------------------------------------------------------
class Lac(object):
    """
    等价于 Taskflow("lexical_analysis")，但内部全部展开到 paddle 层。

    Args:
        task_path(str): 资源目录，默认 $PPNLP_HOME/taskflow/lac（对应 task_priority_path="lac"）
        device_id(int): 对应 Taskflow.__init__ 的 device_id
        backend(str): "paddle-inference"（默认，同 Taskflow）或 "dygraph"（直接跑动态图，便于看算子）
        batch_size(int) / num_threads(int) / split_sentence(bool) / max_seq_len(int)
    """

    def __init__(
        self,
        task_path=None,
        device_id=0,
        backend="paddle-inference",
        batch_size=1,
        num_threads=None,
        split_sentence=False,
        max_seq_len=512,
        emb_dim=None,
        hidden_size=None,
        verbose=False,
    ):
        self.verbose = verbose
        self.kwargs = dict(LAC_MODEL_CONFIG)
        self.kwargs["device_id"] = device_id
        self.emb_dim = emb_dim or self.kwargs["emb_dim"]
        self.hidden_size = hidden_size or self.kwargs["hidden_size"]

        # ② Taskflow.__init__ -> paddle.set_device
        self.device_id = device_id
        self.device = self._get_env_device()
        if self.device == "cpu" or device_id == -1:
            paddle.set_device("cpu")
            self.device = "cpu"
        else:
            paddle.set_device(self.device + ":" + str(device_id))

        # ③ Task.__init__ -> _task_path / _num_threads / _predictor_type
        self.task_path = task_path or os.path.join(PPNLP_HOME, "taskflow", self.kwargs["task_priority_path"])
        self.num_threads = num_threads if num_threads is not None else math.ceil(cpu_count() / 2)
        self.predictor_type = backend
        self.infer_precision = "fp32"
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.split_sentence = split_sentence

        # ④ LacTask.__init__
        check_and_download_resources(self.task_path)
        self._construct_vocabs()
        self._get_inference_model()

    # ----------------------------- 设备 / 词表 -----------------------------
    @staticmethod
    def _get_env_device():
        """对应 paddlenlp/utils/tools.py:119 get_env_device"""
        if paddle.is_compiled_with_cuda():
            return "gpu"
        elif "npu" in paddle.device.get_all_custom_device_type():
            return "npu"
        elif "mlu" in paddle.device.get_all_custom_device_type():
            return "mlu"
        elif "gcu" in paddle.device.get_all_custom_device_type():
            return "gcu"
        elif "intel_hpu" in paddle.device.get_all_custom_device_type():
            return "intel_hpu"
        elif paddle.is_compiled_with_rocm():
            return "rocm"
        elif paddle.is_compiled_with_xpu():
            return "xpu"
        return "cpu"

    def _construct_vocabs(self):
        """对应 lexical_analysis.py:129 LacTask._construct_vocabs"""
        self._word_vocab = load_vocab(os.path.join(self.task_path, "word.dic"))
        self._tag_vocab = load_vocab(os.path.join(self.task_path, "tag.dic"))
        # q2b.dic：全角 -> 半角归一化表
        self._q2b_vocab = load_vocab(os.path.join(self.task_path, "q2b.dic"))
        self._id2tag_dict = dict(zip(self._tag_vocab.values(), self._tag_vocab.keys()))
        self._log("word_vocab={} tag_vocab={} q2b_vocab={}".format(
            len(self._word_vocab), len(self._tag_vocab), len(self._q2b_vocab)))

    # ------------------- ⑤⑥ 动态图 -> 静态图 -> paddle.inference -------------------
    def _get_inference_model(self):
        """对应 task.py:310 Task._get_inference_model"""
        self.inference_model_path = os.path.join(self.task_path, "static", "inference")
        if not os.path.exists(self.inference_model_path + PARAMS_SUFFIX):
            # task.py:381 dygraph_mode_guard() -> paddle.disable_static()
            paddle.disable_static()
            self._model = self._construct_model()
            self._convert_dygraph_to_static()

        self._static_model_file = self.inference_model_path + MODEL_SUFFIX
        self._static_params_file = self.inference_model_path + PARAMS_SUFFIX
        self._log("static model: {} / {}".format(self._static_model_file, self._static_params_file))

        if self.predictor_type == "paddle-inference":
            self._config = paddle.inference.Config(self._static_model_file, self._static_params_file)
            self._prepare_static_mode()
        else:
            # backend="dygraph"：不导出静态图，直接用 paddle 动态图跑 BiGruCrf.forward
            paddle.disable_static()
            self._model = self._construct_model()

    def _construct_model(self):
        """对应 lexical_analysis.py:139 LacTask._construct_model"""
        model_instance = BiGruCrf(self.emb_dim, self.hidden_size, len(self._word_vocab), len(self._tag_vocab))
        # 加载官方预训练参数：paddle.load + Layer.set_dict（按参数名逐个 set_value）
        state_dict = paddle.load(os.path.join(self.task_path, "model_state.pdparams"))
        model_instance.set_dict(state_dict)
        model_instance.eval()
        self._log("loaded params: {}".format(sorted(k for k in state_dict if not k.startswith("crf_loss"))))
        return model_instance

    def _convert_dygraph_to_static(self):
        """对应 task.py:417 Task._convert_dygraph_to_static"""
        self._log("Converting to the inference model cost a little time.")
        static_model = paddle.jit.to_static(self._model, input_spec=input_spec(), full_graph=True)
        paddle.jit.save(static_model, self.inference_model_path)
        self._log("The inference model save in the path:{}".format(self.inference_model_path))

    def _prepare_static_mode(self):
        """对应 task.py:216 Task._prepare_static_mode：这里是真正进入 PaddlePaddle 推理引擎的地方"""
        if self.device == "cpu":
            self._config.disable_gpu()
            self._config.enable_mkldnn()
        else:
            self._config.enable_use_gpu(100, self.device_id)
            # TODO(linjieccc): enable after fixed
            self._config.delete_pass("embedding_eltwise_layernorm_fuse_pass")
            self._config.delete_pass("fused_multi_transformer_encoder_pass")
            self._config.delete_pass("fused_rotary_position_embedding_pass")

        self._config.switch_ir_optim(True)
        self._config.enable_new_executor()
        self._config.set_cpu_math_library_num_threads(self.num_threads)
        self._config.switch_use_feed_fetch_ops(False)
        self._config.disable_glog_info()
        self._config.enable_memory_optim()

        # <== 到此就是 paddle 的 C++ inference API（AnalysisPredictor）
        self.predictor = paddle.inference.create_predictor(self._config)
        self.input_names = [name for name in self.predictor.get_input_names()]
        self.input_handles = [self.predictor.get_input_handle(name) for name in self.predictor.get_input_names()]
        self.output_handle = [self.predictor.get_output_handle(name) for name in self.predictor.get_output_names()]
        self._log("predictor inputs={} outputs={}".format(self.input_names, self.predictor.get_output_names()))

    # --------------------------------- ⑦ 推理三段式 ---------------------------------
    def __call__(self, *args, **kwargs):
        """对应 task.py:567 Task.__call__"""
        inputs = self._preprocess(*args)
        outputs = self._run_model(inputs, **kwargs)
        results = self._postprocess(outputs)
        return results

    def _check_input_text(self, inputs):
        """对应 task.py:434 Task._check_input_text"""
        inputs = inputs[0]
        if isinstance(inputs, str):
            if len(inputs) == 0:
                raise ValueError("Invalid inputs, input text should not be empty text, please check your input.")
            inputs = [inputs]
        elif isinstance(inputs, list):
            if not (isinstance(inputs[0], str) and len(inputs[0].strip()) > 0):
                raise TypeError(
                    "Invalid inputs, input text should be list of str, and first element of list should not be empty text."
                )
        else:
            raise TypeError(
                "Invalid inputs, input text should be str or list of str, but type of {} found!".format(type(inputs))
            )
        return inputs

    def _cut_chinese_sent(self, para):
        """对应 taskflow/utils.py:129 cut_chinese_sent（split_sentence=True 时用）"""
        para = re.sub(r"([。！？\?])([^”’])", r"\1\n\2", para)
        para = re.sub(r"(\.{6})([^”’])", r"\1\n\2", para)
        para = re.sub(r"(\…{2})([^”’])", r"\1\n\2", para)
        para = re.sub(r"([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)
        para = para.rstrip()
        return para.split("\n")

    def _auto_splitter(self, input_texts, max_text_len, split_sentence=False):
        """对应 task.py:454 Task._auto_splitter：超长文本按 max_seq_len 切片，并记录还原映射"""
        input_mapping = {}
        short_input_texts = []
        cnt_org = 0
        cnt_short = 0
        for idx in range(len(input_texts)):
            sens = self._cut_chinese_sent(input_texts[idx]) if split_sentence else [input_texts[idx]]
            for sen in sens:
                lens = len(sen)
                if lens <= max_text_len:
                    short_input_texts.append(sen)
                    input_mapping.setdefault(cnt_org, []).append(cnt_short)
                    cnt_short += 1
                else:
                    temp_text_list = [sen[i : i + max_text_len] for i in range(0, lens, max_text_len)]
                    short_input_texts.extend(temp_text_list)
                    short_idx = cnt_short
                    cnt_short += math.ceil(lens / max_text_len)
                    input_mapping.setdefault(cnt_org, []).extend(
                        [short_idx + i for i in range(cnt_short - short_idx)]
                    )
            cnt_org += 1
        return short_input_texts, input_mapping

    def _preprocess(self, inputs, padding=True, add_special_tokens=True):
        """
        对应 lexical_analysis.py:158 LacTask._preprocess。
        原实现用 load_dataset + paddle.io.DataLoader(collate_fn=Tuple(Pad(int64), Stack(int64)))，
        这里直接用 numpy 组 batch，产物与 Pad/Stack 完全一致（右侧补 0，dtype=int64）。
        """
        inputs = self._check_input_text(inputs)
        oov_token_id = int(self._word_vocab["OOV"])  # lexical_analysis.py:169

        filter_inputs = [i for i in inputs if isinstance(i, str) and len(i.strip()) > 0]
        short_input_texts, self.input_mapping = self._auto_splitter(
            filter_inputs, self.max_seq_len, split_sentence=self.split_sentence
        )

        # 逐字查表：先 q2b 全半角归一化，再查 word.dic，未登录字用 OOV id
        samples = []
        for text in short_input_texts:
            ids = [int(self._word_vocab.get(self._q2b_vocab.get(ch, ch), oov_token_id)) for ch in text]
            samples.append((ids, len(ids)))
        self._log("token_ids(first sample)={}".format(samples[0][0] if samples else []))

        batches = []
        for start in range(0, len(samples), self.batch_size):
            chunk = samples[start : start + self.batch_size]
            max_len = max(l for _, l in chunk)
            # Tuple(Pad(axis=0, pad_val=0, dtype="int64"), Stack(dtype="int64"))
            input_ids = np.zeros((len(chunk), max_len), dtype="int64")
            for i, (ids, _) in enumerate(chunk):
                input_ids[i, : len(ids)] = ids
            seq_len = np.asarray([l for _, l in chunk], dtype="int64")
            batches.append((input_ids, seq_len))

        return {"text": short_input_texts, "batches": batches}

    def _run_model(self, inputs, **kwargs):
        """
        对应 lexical_analysis.py:209 LacTask._run_model。
        paddle-inference 分支：copy_from_cpu -> predictor.run() -> copy_to_cpu，
        这三行就是 Taskflow 与 PaddlePaddle 推理引擎的全部接口。
        """
        results = []
        lens = []
        for input_ids, seq_len in inputs["batches"]:
            if self.predictor_type == "paddle-inference":
                self.input_handles[0].copy_from_cpu(input_ids)  # token_ids
                self.input_handles[1].copy_from_cpu(seq_len)  # length
                self.predictor.run()  # <== PaddlePaddle 执行静态图
                tags_ids = self.output_handle[0].copy_to_cpu()
            else:
                # 动态图分支：直接调 BiGruCrf.forward（Embedding -> BiGRU -> FC -> ViterbiDecoder）
                with paddle.no_grad():
                    tags_ids = self._model(paddle.to_tensor(input_ids), paddle.to_tensor(seq_len)).numpy()
            results.extend(tags_ids.tolist())
            lens.extend(seq_len.tolist())
            self._log("tag_ids(first sample)={}".format(results[-len(seq_len)] if len(seq_len) else []))

        inputs["result"] = results
        inputs["lens"] = lens
        return inputs

    def _postprocess(self, inputs):
        """对应 lexical_analysis.py:228 LacTask._postprocess：tag id -> 词性标签 -> IOB2 边界还原分词"""
        lengths = inputs["lens"]
        preds = inputs["result"]
        sents = inputs["text"]
        final_results = []
        for sent_index in range(len(lengths)):
            single_result = {}
            tags = [self._id2tag_dict[str(index)] for index in preds[sent_index][: lengths[sent_index]]]
            sent = sents[sent_index]
            sent_out = []
            tags_out = []
            parital_word = ""
            for ind, tag in enumerate(tags):
                if parital_word == "":
                    parital_word = sent[ind]
                    tags_out.append(tag.split("-")[0])
                    continue
                if tag.endswith("-B") or (tag == "O" and tags[ind - 1] != "O"):
                    sent_out.append(parital_word)
                    tags_out.append(tag.split("-")[0])
                    parital_word = sent[ind]
                    continue
                parital_word += sent[ind]

            if len(sent_out) < len(tags_out):
                sent_out.append(parital_word)

            single_result["text"] = sent
            single_result["segs"] = sent_out
            single_result["tags"] = tags_out
            final_results.append(single_result)
        return self._auto_joiner(final_results, self.input_mapping, is_dict=True)

    def _auto_joiner(self, short_results, input_mapping, is_dict=False):
        """对应 task.py:515 Task._auto_joiner：把切片结果按 input_mapping 拼回用户原始输入"""
        concat_results = []
        elem_type = {} if is_dict else []
        for k, vs in input_mapping.items():
            single_results = elem_type
            for v in vs:
                if len(single_results) == 0:
                    single_results = short_results[v]
                elif isinstance(elem_type, list):
                    single_results.extend(short_results[v])
                elif isinstance(elem_type, dict):
                    for sk in single_results.keys():
                        if isinstance(single_results[sk], str):
                            single_results[sk] += short_results[v][sk]
                        else:
                            single_results[sk].extend(short_results[v][sk])
                else:
                    raise ValueError(
                        "Invalid element type, the type of results "
                        "for each element should be list of dict, "
                        "but {} received.".format(type(single_results))
                    )
            concat_results.append(single_results)
        return concat_results

    def _log(self, msg):
        if self.verbose:
            print("[lexical] {}".format(msg), file=sys.stderr)


def main(argv):
    import argparse

    parser = argparse.ArgumentParser(description="Taskflow('lexical_analysis') 的底层展开实现")
    parser.add_argument("inputs", nargs="*", default=["LAC是个优秀的分词工具"], help="待分析文本")
    parser.add_argument("--backend", default="paddle-inference", choices=["paddle-inference", "dygraph"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_threads", type=int, default=None)
    parser.add_argument("--split_sentence", action="store_true")
    parser.add_argument("--task_path", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", help="打印每一层的中间产物")
    args = parser.parse_args(argv)

    lac = Lac(
        task_path=args.task_path,
        device_id=args.device_id,
        backend=args.backend,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        split_sentence=args.split_sentence,
        verbose=args.verbose,
    )
    texts = args.inputs if args.inputs else ["LAC是个优秀的分词工具"]
    if len(texts) == 1:
        print(lac(texts[0]))
    else:
        print(lac(texts))


if __name__ == "__main__":
    main(sys.argv[1:])
