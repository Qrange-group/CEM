#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))

import pickle as pkl
import logging
import time
import jieba
import numpy as np
import pandas as pd
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or any {'0', '1', '3'}
from utility import *
from sklearn.metrics import f1_score, auc, roc_curve
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


class Network(object):
    """
    Parent class for all tensorflow networks.
    """

    def __init__(
        self,
        vocab=None,
        sent_max_len=64,
        dia_max_len=50,
        nb_classes=2,
        nb_words=10000,
        embedding_dim=200,
        dense_dim=128,
        rnn_dim=128,
        num_layers=1,
        lr_min=0.0,
        loss_lambda=1.0,
        keep_prob=0.5,
        lr=0.001,
        weight_decay=0.0,
        l2_reg_lambda=0.0,
        warmup_steps=0,
        optim="adam",
        gpu="0",
        memory=0,
        batch_size=128,
        is_only_ssa=False,
        is_only_cf=False,
        weigth_way='score',
        add_senti_loss=False,
        **kwargs
    ):
        # ablation study
        self.is_only_ssa = is_only_ssa
        self.is_only_cf = is_only_cf
        self.weigth_way = weigth_way
        self.add_senti_loss = add_senti_loss

        # logging
        self.logger = logging.getLogger("Tensorflow")

        # data config
        self.sent_max_len = sent_max_len
        self.dia_max_len = dia_max_len
        self.nb_classes = nb_classes
        self.nb_words = nb_words
        self.batch_size = batch_size

        # network config
        self.embedding_dim = embedding_dim
        self.dense_dim = dense_dim
        self.rnn_dim = rnn_dim
        self.keep_prob = keep_prob
        self.lamb = 0.0

        # initializer
        self.initializer = tf.initializers.glorot_normal()

        # optimizer config
        self.lr = lr
        self.lr_min = lr_min
        self.weight_decay = weight_decay
        self.l2_reg_lambda = l2_reg_lambda
        self.optim = optim
        self.warmup_steps = warmup_steps
        self.loss_lambda = loss_lambda

        self.model_name = "Network"
        self.data = "normal"

        # session info config
        self.gpu = gpu
        self.memory = memory
        self.vocab = vocab
        print(self.vocab.embeddings.shape)

        if self.memory > 0:
            num_threads = os.environ.get("OMP_NUM_THREADS")
            self.logger.info("Memory use is %s." % self.memory)
            gpu_options = tf.compat.v1.GPUOptions(
                per_process_gpu_memory_fraction=float(self.memory)
            )
            config = tf.compat.v1.ConfigProto(
                gpu_options=gpu_options, intra_op_parallelism_threads=num_threads
            )
            self.sess = tf.compat.v1.Session(config=config)
        else:
            config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(config=config)

    def set_nb_words(self, nb_words):
        self.nb_words = nb_words
        self.logger.info("set nb_words.")

    def set_data(self, data):
        self.data = data
        self.logger.info("set data.")

    def set_name(self, model_name):
        self.model_name = model_name
        self.logger.info("set model_name.")

    def set_from_model_config(self, model_config):
        self.embedding_dim = model_config["embedding_dim"]
        self.optim = model_config["optimizer"]
        self.lr = model_config["learning_rate"]
        self.weight_decay = model_config["weight_decay"]
        self.l2_reg_lambda = model_config["l2_reg_lambda"]
        self.logger.info("set from model_config.")

    def set_from_data_config(self, data_config):
        self.nb_classes = data_config["nb_classes"]
        self.logger.info("set from data_config.")

    def build_dir(self):
        # experimental results saving path
        self.save_dir = curdir + "/weights/" + self.data + "/" + self.model_name + "/"
        if not os.path.exists(curdir + "/weights/" + self.data):
            os.makedirs(curdir + "/weights/" + self.data)
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)
        os.makedirs(self.save_dir + "best")

        self.tensorboard_dir = (
            curdir + "/tensorboard_dir/" + self.data + "/" + self.model_name + "/"
        )

        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
        # remove history tensorboard events
        try:
            shutil.rmtree(self.tensorboard_dir + "train")
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        try:
            shutil.rmtree(self.tensorboard_dir + "eval")
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        self.summary_writer_train = tf.compat.v1.summary.FileWriter(
            self.tensorboard_dir + "train", self.sess.graph
        )
        self.summary_writer_eval = tf.compat.v1.summary.FileWriter(
            self.tensorboard_dir + "eval"
        )

    def build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed(self.vocab.embeddings)
        self._inference()
        self._compute_loss()
        self._create_train_op()
        self.saver = tf.compat.v1.train.Saver(max_to_keep=10000)

        # self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        # print_trainable_variables(output_detail=True, logger=self.logger)
        # param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        # self.logger.info('There are {} parameters in the models'.format(param_num))
        # embedding_param_num = sum([np.prod(self.sess.run(tf.shape(v)))
        #                            for v in self.all_params if 'word_embedding' in v.name or 'weights' in v.name])
        # self.logger.info('There are {} parameters in the models for word embedding'.format(embedding_param_num))
        # pure_param_num = sum([np.prod(self.sess.run(tf.shape(v)))
        #                       for v in self.all_params if 'word_embedding' not in v.name and 'weights' not in v.name])
        # self.logger.info('There are {} parameters in the models without word embedding'.format(pure_param_num))

        # initialize the models
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.input_x1 = tf.compat.v1.placeholder(
            tf.int32, [None, self.dia_max_len, self.sent_max_len], name="input_x1"
        )
        self.role_list = tf.compat.v1.placeholder(
            tf.int32, [None, self.dia_max_len], name="role_list"
        )
        self.sent_len = tf.compat.v1.placeholder(
            tf.int32, [None, self.dia_max_len], name="sent_len"
        )
        self.dia_len = tf.compat.v1.placeholder(tf.int32, [None], name="dia_len")

        self.main_y = tf.compat.v1.placeholder(
            tf.float32, [None, self.dia_max_len, self.nb_classes], name="handoff_y"
        )
        self.apha = tf.compat.v1.placeholder(
            tf.float32, [None, self.dia_max_len, self.nb_classes], name="handoff_y"
        )

        self.senti_y = tf.compat.v1.placeholder(
            tf.float32, [None, self.dia_max_len, 3], name="senti_y"
        )
        self.score_y = tf.compat.v1.placeholder(tf.float32, [None, 3], name="score_y")

        self.dropout_keep_prob = tf.compat.v1.placeholder(
            tf.float32, name="dropout_keep_prob"
        )
        self.logger.info("setup placeholders.")

    def _embed(self, embedding_matrix=np.array([None])):
        """
        The embedding layer
        """
        with tf.compat.v1.variable_scope("word_embedding"):
            if embedding_matrix.any() is None:
                print("Using random initialized emebeddings")
                self.word_embeddings = tf.compat.v1.get_variable(
                    "word_embeddings",
                    shape=(self.nb_words, self.embedding_dim),
                    initializer=self.initializer,
                    trainable=True,
                )
            else:
                print("Using pre trained embeddings")
                self.word_embeddings = tf.compat.v1.get_variable(
                    "word_embeddings",
                    shape=(self.nb_words, self.embedding_dim),
                    initializer=tf.constant_initializer(embedding_matrix),
                    trainable=True,
                )

    def _inference(self):
        """
        encode sentence information
        """
        # [B, D_len, S_len, E_dim]
        self.embedded = tf.nn.embedding_lookup(self.word_embeddings, self.input_x1)
        if self.dropout_keep_prob != 1:
            self.embedded = tf.nn.dropout(
                self.embedded, keep_prob=self.dropout_keep_prob
            )
        show_layer_info_with_memory("embedded", self.embedded, self.logger)

        self.embedded_reshaped = tf.reshape(
            self.embedded, shape=[-1, self.sent_max_len, self.embedding_dim]
        )

        show_layer_info_with_memory(
            "embedded_reshaped", self.embedded_reshaped, self.logger
        )
        self.sent_len_reshape = tf.reshape(self.sent_len, shape=[-1])

        with tf.name_scope("sent_encoding"):
            self.sent_encoder_output, self.sent_encoder_state = self._bidirectional_rnn(
                self.embedded_reshaped, self.sent_len_reshape
            )
            show_layer_info_with_memory(
                "sent_encode_output", self.sent_encoder_state, self.logger
            )
        if self.dropout_keep_prob != 1:
            # [B * D_len, 2 * H]
            self.sent_encoder_state = tf.nn.dropout(
                self.sent_encoder_state, keep_prob=self.dropout_keep_prob
            )

        with tf.name_scope("sequence_context_encoding"):
            # RNN
            cells = tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(self.rnn_dim, state_is_tuple=True),
                output_keep_prob=self.dropout_keep_prob,
            )
            # [B, D_len, 2 * H]  [B, 2 * H]
            self.dia_encoder_output, self.dia_encoder_state = tf.nn.dynamic_rnn(
                cell=cells,
                inputs=self.sent_encoder_state,
                sequence_length=self.dia_len,
                dtype=tf.float32,
            )
            self.dia_encoder_state = self.dia_encoder_state[1]

        with tf.name_scope("dialogue_mask"):

            self.dia_seq_mask = tf.sequence_mask(
                self.dia_len, maxlen=self.dia_max_len, dtype=tf.float32
            )

        with tf.name_scope("output"):
            # [B, D_len, nb_classes]
            self.main_logits = tf.keras.layers.Dense(
                units=self.nb_classes, activation="softmax"
            )(self.dia_encoder_output)
            self.senti_logits = tf.keras.layers.Dense(units=3, activation="softmax")(
                self.dia_encoder_output
            )
            self.score_logits = tf.keras.layers.Dense(units=3, activation="softmax")(
                self.dia_encoder_state
            )

            # [B, D_len]
            self.output = tf.argmax(self.main_logits, axis=-1)
            self.senti = tf.argmax(self.senti_logits, axis=-1)
            # [B]
            self.pre_score = tf.argmax(self.score_logits, axis=-1)

            self.proba = tf.nn.softmax(self.main_logits)

        self.logger.info("network inference.")

    def _bidirectional_rnn(self, inputs, length, rnn_type="lstm"):
        if rnn_type == "lstm":
            fw_rnn_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_dim)
            fw_rnn_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                fw_rnn_cell,
                input_keep_prob=self.dropout_keep_prob,
                output_keep_prob=self.dropout_keep_prob,
            )
            bw_rnn_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_dim)
            bw_rnn_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                bw_rnn_cell,
                input_keep_prob=self.dropout_keep_prob,
                output_keep_prob=self.dropout_keep_prob,
            )
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                fw_rnn_cell,
                bw_rnn_cell,
                inputs,
                sequence_length=length,
                dtype=tf.float32,
            )
            fw_state, bw_state = output_states
            fw_c, fw_h = fw_state
            bw_c, bw_h = bw_state
            fw_state, bw_state = fw_h, bw_h

            final_output = tf.concat(outputs, -1)
            final_state = tf.concat([fw_state, bw_state], -1)

        elif rnn_type == "gru":
            fw_rnn_cell = tf.nn.rnn_cell.GRUCell(self.rnn_dim)
            fw_rnn_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                fw_rnn_cell,
                input_keep_prob=self.dropout_keep_prob,
                output_keep_prob=self.dropout_keep_prob,
            )
            bw_rnn_cell = tf.nn.rnn_cell.GRUCell(self.rnn_dim)
            bw_rnn_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                bw_rnn_cell,
                input_keep_prob=self.dropout_keep_prob,
                output_keep_prob=self.dropout_keep_prob,
            )
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                fw_rnn_cell,
                bw_rnn_cell,
                inputs,
                sequence_length=length,
                dtype=tf.float32,
            )

            final_output = tf.concat(outputs, -1)
            final_state = tf.concat(output_states, -1)

        return final_output, final_state

    def multihead_attention(
        self,
        queries,
        keys,
        values,
        key_masks,
        num_heads=8,
        dropout_rate=0,
        causality=False,
        scope="multihead_attention",
    ):
        """Applies multihead attention. See 3.2.2
        queries: A 3d tensor with shape of [N, T_q, d_model].
        keys: A 3d tensor with shape of [N, T_k, d_model].
        values: A 3d tensor with shape of [N, T_k, d_model].
        key_masks: A 2d tensor with shape of [N, key_seqlen]
        num_heads: An int. Number of heads.
        dropout_rate: A floating point number.
        causality: Boolean. If true, units that reference the future are masked.
        scope: Optional scope for `variable_scope`.

        Returns
        A 3d tensor with shape of (N, T_q, C)
        """
        d_model = queries.get_shape().as_list()[-1]
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Linear projections
            Q = tf.layers.dense(queries, d_model, use_bias=True)  # (N, T_q, d_model)
            K = tf.layers.dense(keys, d_model, use_bias=True)  # (N, T_k, d_model)
            V = tf.layers.dense(values, d_model, use_bias=True)  # (N, T_k, d_model)

            # Split and concat
            Q_ = tf.concat(
                tf.split(Q, num_heads, axis=2), axis=0
            )  # (h*N, T_q, d_model/h)
            K_ = tf.concat(
                tf.split(K, num_heads, axis=2), axis=0
            )  # (h*N, T_k, d_model/h)
            V_ = tf.concat(
                tf.split(V, num_heads, axis=2), axis=0
            )  # (h*N, T_k, d_model/h)

            # Attention
            outputs = self.scaled_dot_product_attention(
                Q_, K_, V_, key_masks, causality, dropout_rate
            )

            # Restore shape
            outputs = tf.concat(
                tf.split(outputs, num_heads, axis=0), axis=2
            )  # (N, T_q, d_model)

            # Residual connection
            outputs += queries

            # Normalize
            outputs = self.ln(outputs)

        return outputs

    def scaled_dot_product_attention(
        self,
        Q,
        K,
        V,
        key_masks,
        causality=False,
        dropout_rate=0.0,
        training=True,
        scope="scaled_dot_product_attention",
    ):
        """See 3.2.1.
        Q: Packed queries. 3d tensor. [N, T_q, d_k].
        K: Packed keys. 3d tensor. [N, T_k, d_k].
        V: Packed values. 3d tensor. [N, T_k, d_v].
        key_masks: A 2d tensor with shape of [N, key_seqlen]
        causality: If True, applies masking for future blinding
        dropout_rate: A floating point number of [0, 1].
        training: boolean for controlling droput
        scope: Optional scope for `variable_scope`.
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            d_k = Q.get_shape().as_list()[-1]

            # dot product
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

            # scale
            outputs /= d_k ** 0.5

            # causality or future blinding masking
            if causality:
                outputs = self.mask(outputs, type="future")
            else:
                # key masking
                outputs = self.mask(outputs, key_masks=key_masks, type="key")

            # softmax
            outputs = tf.nn.softmax(outputs)

            # dropout
            outputs = tf.nn.dropout(outputs, rate=dropout_rate)

            # weighted sum (context vectors)
            outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

        return outputs

    def mask(self, inputs, key_masks=None, type=None):
        """Masks paddings on keys or queries to inputs
        inputs: 3d tensor. (h*N, T_q, T_k)
        key_masks: 3d tensor. (N, 1, T_k)
        type: string. "key" | "future"

        e.g.,
        >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
        >> key_masks = tf.constant([[0., 0., 1.],
                                    [0., 1., 1.]])
        >> mask(inputs, key_masks=key_masks, type="key")
        array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
            [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

           [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
            [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],

           [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
            [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

           [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
            [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
        """
        padding_num = -(2 ** 32) + 1
        if type in ("k", "key", "keys"):
            key_masks = tf.to_float(key_masks)
            key_masks = tf.tile(
                key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1]
            )  # (h*N, seqlen)
            key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
            outputs = inputs + key_masks * padding_num

        elif type in ("f", "future", "right"):
            diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(
                diag_vals
            ).to_dense()  # (T_q, T_k)
            future_masks = tf.tile(
                tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1]
            )  # (N, T_q, T_k)

            paddings = tf.ones_like(future_masks) * padding_num
            outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
        else:
            print("Check if you entered type correctly!")

        return outputs

    def ff(self, inputs, num_units, scope="positionwise_feedforward"):
        """position-wise feed forward net. See 3.3

        inputs: A 3d tensor with shape of [N, T, C].
        num_units: A list of two integers.
        scope: Optional scope for `variable_scope`.

        Returns:
          A 3d tensor with the same shape and dtype as inputs
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Inner layer
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

            # Outer layer
            outputs = tf.layers.dense(outputs, num_units[1])

            # Residual connection
            outputs += inputs

            # Normalize
            outputs = self.ln(outputs)

        return outputs

    def ln(self, inputs, epsilon=1e-8, scope="ln"):
        """Applies layer normalization. See https://arxiv.org/abs/1607.06450.
        inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
        epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        scope: Optional scope for `variable_scope`.

        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.compat.v1.get_variable(
                "beta", params_shape, initializer=tf.zeros_initializer()
            )
            gamma = tf.compat.v1.get_variable(
                "gamma", params_shape, initializer=tf.ones_initializer()
            )
            normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
            outputs = gamma * normalized + beta

        return outputs

    def positional_encoding(
        self, inputs, maxlen, masking=True, scope="positional_encoding"
    ):
        """
        Sinusoidal Positional_Encoding. See 3.5
        inputs: 3d tensor. (N, T, E)
        maxlen: scalar. Must be >= T
        masking: Boolean. If True, padding positions are set to zeros.
        scope: Optional scope for `variable_scope`.

        returns
        3d tensor that has the same shape as inputs.
        """

        E = inputs.get_shape().as_list()[-1]  # static
        N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # position indices
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

            # First part of the PE function: sin and cos argument
            position_enc = np.array(
                [
                    [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
                    for pos in range(maxlen)
                ]
            )

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

            # lookup
            outputs = tf.nn.embedding_lookup(position_enc, position_ind)

            # masks
            if masking:
                outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

            return tf.to_float(outputs)

    def _compute_loss(self):
        """
        The loss function
        """

        def nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                losses = -tf.reduce_sum(labels * tf.math.log(probs + epsilon), -1)
            return losses

        # masked main loss
        self.cross_entropy_main = self.apha * nll_loss(self.main_logits, self.main_y)
        # balanced combine
        self.mask_cross_entropy = self.dia_seq_mask * self.cross_entropy_main
        # reduction
        self.scale_cross_entropy = tf.reduce_sum(self.mask_cross_entropy, -1) / tf.cast(
            self.dia_len, tf.float32
        )
        self.main_loss = tf.reduce_mean(self.scale_cross_entropy)

        self.score_loss = tf.reduce_mean(
            -tf.reduce_sum(self.score_y * tf.math.log(self.score_logits + 1e-9), -1), -1
        )
        self.loss = self.main_loss + self.loss_lambda * self.score_loss

        self.loss_pre = tf.reduce_mean(
            tf.square(self.main_logits[:, 1] - self.main_y[:, 1])
        )

        self.logger.info("Calculate Loss.")

        self.all_params = tf.compat.v1.trainable_variables()
        if self.l2_reg_lambda > 0:
            with tf.compat.v1.variable_scope("l2_loss"):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.l2_reg_lambda * l2_loss
            self.logger.info("Add L2 Loss.")

    def _lr_with_warmup(
        self,
        warmup_step,
        lr_base,
        global_step,
        lr_step=20,
        lr_decay=0.96,
        staircase=False,
    ):
        """参数设置个人建议最终训练完学习率大致为learning_rate_base的1/50,learning_rate_decay在0.95到0.995之间"""
        if warmup_step != 0:
            linear_increase = lr_base * tf.cast(global_step / warmup_step, tf.float32)
            exp_decay = tf.compat.v1.train.exponential_decay(
                lr_base,
                global_step - warmup_step,
                lr_step,
                lr_decay,
                staircase=staircase,
            )
            final_lr = tf.cond(
                global_step <= warmup_step, lambda: linear_increase, lambda: exp_decay
            )
        else:
            final_lr = lr_base

        return final_lr

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        self.global_step = tf.compat.v1.train.get_or_create_global_step()

        self.true_lr = self._lr_with_warmup(
            self.warmup_steps, self.lr, self.global_step, lr_step=50
        )

        if self.optim == "adagrad":
            self.optimizer = tf.compat.v1.train.AdagradOptimizer(self.true_lr)
        elif self.optim == "adamw":
            AdamWOptimizer = tf.contrib.opt.extend_with_decoupled_weight_decay(
                tf.compat.v1.train.AdamOptimizer
            )
            self.optimizer = AdamWOptimizer(
                weight_decay=self.weight_decay, learning_rate=self.true_lr
            )
        elif self.optim == "adam":
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.true_lr)
        elif self.optim == "sgdmoment":
            self.optimizer = tf.compat.v1.train.MomentumOptimizer(
                self.true_lr, 0.9, use_nesterov=True
            )
        elif self.optim == "rmsprop":
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.true_lr)
        elif self.optim == "adadelta":
            self.optimizer = tf.compat.v1.train.AdadeltaOptimizer(self.true_lr)
        elif self.optim == "sgd":
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(self.true_lr)
        else:
            raise NotImplementedError("Unsupported optimizer: {}".format(self.optim))

        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        tf.compat.v1.summary.scalar("loss", self.loss)

        self.summaries = tf.compat.v1.summary.merge_all()

    def save(self, model_dir, model_prefix):
        """
        Saves the models into model_dir with model_prefix as the models indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info(
            "Model saved in {}, with prefix {}.".format(model_dir, model_prefix)
        )

    def restore(self, model_dir, model_prefix):
        """
        Restores the models into model_dir from model_prefix as the models indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info(
            "Model restored from {}, with prefix {}".format(model_dir, model_prefix)
        )

    @staticmethod
    def save_model(sess, signature, path):
        builder = tf.saved_model.builder.SavedModelBuilder(path)
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={"predict": signature},
            legacy_init_op=tf.saved_model.main_op.main_op(),
        )
        builder.save()

    def train_cmhch(
        self,
        data_generator,
        keep_prob,
        epochs,
        data,
        task="train",
        batch_size=20,
        nb_classes=2,
        shuffle=True,
        is_val=True,
        is_test=True,
        is_save=True,
        save_best=True,
        save_frequency=10,
        val_task="eval",
        test_task="test",
    ):

        print("Using cmhch trainer.")
        max_val = 0

        for epoch in range(epochs):
            self.logger.info("------------- Epoch {} -------------".format(epoch))
            print("Training Epoch: {}".format(epoch))
            counter, total_loss = 0.0, 0.0
            # For handoff and GTT
            total_handoff = []
            total_pre_handoff = []
            # For senti
            total_senti = []
            total_pre_senti = []
            # For satisfaction
            score_list = []
            pre_score_list = []

            for (
                batch_dialog_ids,
                batch_sent_len,
                batch_dia_len,
                batch_ids,
                batch_role_ids,
                batch_handoff,
                batch_senti,
                batch_score,
            ) in data_generator(
                task=task,
                batch_size=batch_size,
                nb_classes=nb_classes,
                shuffle=shuffle,
                epoch=epoch,
            ):
                feed_dict = {
                    self.input_x1: batch_dialog_ids,
                    self.role_list: batch_role_ids,
                    self.dia_len: batch_dia_len,
                    self.sent_len: batch_sent_len,
                    self.main_y: batch_handoff,
                    self.senti_y: batch_senti,
                    self.score_y: batch_score,
                    self.dropout_keep_prob: keep_prob,
                }
                try:
                    if (not self.is_only_ssa) and epoch < 20:
                        (
                            _,
                            step,
                            summaries,
                            loss,
                            sequence,
                            senti,
                            score,
                        ) = self.sess.run(
                            [
                                self.train_op,
                                self.global_step,
                                self.summaries,
                                self.loss_pre,
                                self.output,
                                self.senti,
                                self.pre_score,
                            ],
                            feed_dict,
                        )
                    else:
                        (
                            _,
                            step,
                            summaries,
                            loss,
                            sequence,
                            senti,
                            score,
                        ) = self.sess.run(
                            [
                                self.train_op,
                                self.global_step,
                                self.summaries,
                                self.loss,
                                self.output,
                                self.senti,
                                self.pre_score,
                            ],
                            feed_dict,
                        )

                    self.summary_writer_train.add_summary(summaries, step)
                    # for handoff metric
                    handoff_y = np.argmax(batch_handoff, -1)
                    # for senti metric
                    senti_y = np.argmax(batch_senti, -1)
                    # for satisfaction score
                    score_y = np.argmax(batch_score, -1)
                    for batch_id in range(len(batch_dia_len)):
                        total_handoff.append(
                            handoff_y[batch_id, : batch_dia_len[batch_id]]
                        )
                        total_pre_handoff.append(
                            sequence[batch_id, : batch_dia_len[batch_id]]
                        )

                        total_senti.append(senti_y[batch_id, : batch_dia_len[batch_id]])
                        total_pre_senti.append(
                            senti[batch_id, : batch_dia_len[batch_id]]
                        )
                    score_list.append(score_y)
                    pre_score_list.append(score)
                    total_loss += loss
                    counter += 1
                except ValueError as e:
                    self.logger.info("Wrong batch.{}".format(e))

            total_handoff_flat = np.concatenate(total_handoff)
            total_pre_handoff_flat = np.concatenate(total_pre_handoff)
            # handoff
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_handoff, total_pre_handoff, lamb=self.lamb
            )
            (
                acc_handoff,
                p_handoff,
                r_handoff,
                f1_handoff,
                macro_handoff,
            ) = get_a_p_r_f_sara(
                target=total_handoff_flat, predict=total_pre_handoff_flat
            )

            print(str(confusion_matrix(total_handoff_flat, total_pre_handoff_flat)))
            print(
                "MHCH %s: Loss:%.3f\tF1Score:%.3f\tMacro_F1Score:%.3f\tGT-I:%.3f\tGT-II:%.3f\tGT-III:%.3f"
                % (
                    task,
                    total_loss / float(counter),
                    f1_handoff,
                    macro_handoff,
                    gtt_1,
                    gtt_2,
                    gtt_3,
                )
            )
            self.logger.info(
                "MHCH %s: Loss:%.3f\tF1Score:%.3f\tMacro_F1Score:%.3f\tGT-I:%.3f\tGT-II:%.3f\tGT-III:%.3f"
                % (
                    task,
                    total_loss / float(counter),
                    f1_handoff,
                    macro_handoff,
                    gtt_1,
                    gtt_2,
                    gtt_3,
                )
            )

            # satisfaction score
            score_list = np.concatenate(score_list)
            pre_score_list = np.concatenate(pre_score_list)

            ssa_acc = accuracy_score(score_list, pre_score_list)
            ssa_macro_f1 = f1_score(score_list, pre_score_list, average="macro")
            ssa_f1_0, ssa_f1_1, ssa_f1_2 = f1_score(
                score_list, pre_score_list, labels=[0, 1, 2], average=None
            )

            print(confusion_matrix(score_list, pre_score_list))
            print(
                "SSA %s: Loss:%.3f\tWS F1:%.3f\tF1:%.3f\tUS F1:%.3f\tMacro F1:%.3f\tAcc.:%.3f"
                % (
                    task,
                    total_loss / float(counter),
                    ssa_f1_2,
                    ssa_f1_1,
                    ssa_f1_0,
                    ssa_macro_f1,
                    ssa_acc,
                )
            )
            self.logger.info(
                "SSA %s: Loss:%.3f\tWS F1:%.3f\tF1:%.3f\tUS F1:%.3f\tMacro F1:%.3f\tAcc.:%.3f"
                % (
                    task,
                    total_loss / float(counter),
                    ssa_f1_2,
                    ssa_f1_1,
                    ssa_f1_0,
                    ssa_macro_f1,
                    ssa_acc,
                )
            )

            if is_val:
                (
                    eval_loss,
                    eval_f1_handoff,
                    eval_macro_handoff,
                    eval_auc,
                    gs1,
                    gs2,
                    gs3,
                    eval_f1_2_ssa,
                    eval_f1_1_ssa,
                    eval_f1_0_ssa,
                    eval_macro_ssa,
                    eval_acc_ssa,
                ) = self.evaluate_batch_cmhch(
                    data_generator,
                    data,
                    task=val_task,
                    global_step=step,
                    batch_size=batch_size,
                    nb_classes=nb_classes,
                    shuffle=False,
                )
                total_macro = eval_macro_handoff + eval_macro_ssa
                total_metrices = eval_macro_handoff + gs3 + eval_macro_ssa
                metrics_dict = {
                    "loss": eval_loss,
                    "f1_handoff": eval_f1_handoff,
                    "macro_handoff": eval_macro_handoff,
                    "gs1": gs1,
                    "gs2": gs2,
                    "gs3": gs3,
                    "acc_ssa": eval_acc_ssa,
                    "macro_ssa": eval_macro_ssa,
                    "f1_0_ssa": eval_f1_0_ssa,
                    "f1_1_ssa": eval_f1_1_ssa,
                    "f1_2_ssa": eval_f1_2_ssa,
                    "total_macro": total_macro,
                    "total_metrices": total_metrices,
                }

                if metrics_dict["total_macro"] > max_val and save_best:
                    shutil.rmtree(self.save_dir + "best")
                    max_val = metrics_dict["total_macro"]
                    self.save(self.save_dir + "best", "cmhch")
                    print("Saved!")

            if is_save and (epoch + 1) % save_frequency == 0:
                if os.path.exists(self.save_dir + str(epoch)):
                    shutil.rmtree(self.save_dir + str(epoch))
                os.makedirs(self.save_dir + str(epoch))
                self.save(self.save_dir + str(epoch), "cmhch")

        if is_test:
            self.restore(self.save_dir + "best", "cmhch")
            self.evaluate_batch_cmhch(
                data_generator,
                data,
                task=test_task,
                global_step=0,
                batch_size=batch_size,
                nb_classes=nb_classes,
                shuffle=False,
            )

        self.summary_writer_train.close()
        self.summary_writer_eval.close()

    def evaluate_batch_cmhch(
        self,
        data_generator,
        data,
        global_step,
        task="eval",
        batch_size=32,
        nb_classes=2,
        shuffle=False,
    ):
        counter, total_loss = 0, 0.0

        # For handoff and GTT
        total_handoff = []
        total_pre_handoff = []
        total_pre_handoff_scores = []
        # For senti
        total_senti = []
        total_pre_senti = []
        # For satisfaction
        score_list = []
        pre_score_list = []

        for (
            batch_dialog_ids,
            batch_sent_len,
            batch_dia_len,
            batch_ids,
            batch_role_ids,
            batch_handoff,
            batch_senti,
            batch_score,
        ) in data_generator(
            task=task, batch_size=batch_size, nb_classes=nb_classes, shuffle=shuffle
        ):

            feed_dict = {
                self.input_x1: batch_dialog_ids,
                self.role_list: batch_role_ids,
                self.dia_len: batch_dia_len,
                self.sent_len: batch_sent_len,
                self.main_y: batch_handoff,
                self.senti_y: batch_senti,
                self.score_y: batch_score,
                self.dropout_keep_prob: 1.0,
            }
            try:
                loss, sequence, handoff_prob, senti, score = self.sess.run(
                    [self.loss, self.output, self.proba, self.senti, self.pre_score],
                    feed_dict,
                )
                # for handoff metric
                handoff_y = np.argmax(batch_handoff, -1)
                # for senti metric
                senti_y = np.argmax(batch_senti, -1)
                # for satisfaction score
                score_y = np.argmax(batch_score, -1)
                for batch_id in range(len(batch_dia_len)):
                    total_handoff.append(handoff_y[batch_id, : batch_dia_len[batch_id]])
                    total_pre_handoff.append(
                        sequence[batch_id, : batch_dia_len[batch_id]]
                    )
                    total_pre_handoff_scores.append(
                        handoff_prob[batch_id, : batch_dia_len[batch_id], 1]
                    )

                    for idx, flag in enumerate(
                        batch_role_ids[batch_id, : batch_dia_len[batch_id]]
                    ):
                        if int(flag) == 1:
                            total_senti.append(senti_y[batch_id, idx])
                            total_pre_senti.append(senti[batch_id, idx])

                score_list.append(score_y)
                pre_score_list.append(score)

                total_loss += loss
                counter += 1

            except ValueError as e:
                self.logger.info("Wrong batch.{}".format(e))

        total_handoff_flat = np.concatenate(total_handoff)
        total_pre_handoff_flat = np.concatenate(total_pre_handoff)
        total_pre_handoff_scores_flat = np.concatenate(total_pre_handoff_scores)
        # handoff
        gtt_1, gtt_2, gtt_3 = get_gtt_score(
            total_handoff, total_pre_handoff, lamb=self.lamb
        )
        acc_handoff, p_handoff, r_handoff, f1_handoff, macro_handoff = get_a_p_r_f_sara(
            target=total_handoff_flat, predict=total_pre_handoff_flat
        )
        # calc AUC score
        fpr, tpr, thresholds = roc_curve(
            total_handoff_flat, total_pre_handoff_scores_flat, pos_label=1
        )
        auc_score = auc(fpr, tpr)

        print(confusion_matrix(total_handoff_flat, total_pre_handoff_flat))
        print(
            "MHCH %s : Loss:%.3f\tF1Score:%.3f\tMacro_F1Score:%.3f\tGT-I:%.3f\tGT-II:%.3f\tGT-III:%.3f"
            % (
                task,
                total_loss / float(counter),
                f1_handoff,
                macro_handoff,
                gtt_1,
                gtt_2,
                gtt_3,
            )
        )
        self.logger.info(
            "MHCH %s : Loss:%.3f\tF1Score:%.3f\tMacro_F1Score:%.3f\tGT-I:%.3f\tGT-II:%.3f\tGT-III:%.3f"
            % (
                task,
                total_loss / float(counter),
                f1_handoff,
                macro_handoff,
                gtt_1,
                gtt_2,
                gtt_3,
            )
        )

        # sentiment
        total_senti_flat = total_senti
        total_pre_senti_flat = total_pre_senti
        senti_acc = accuracy_score(total_senti_flat, total_pre_senti_flat)
        senti_macro_f1 = f1_score(
            total_senti_flat, total_pre_senti_flat, average="macro"
        )
        senti_f1_0, senti_f1_1, senti_f1_2 = f1_score(
            total_senti_flat, total_pre_senti_flat, labels=[0, 1, 2], average=None
        )

        # satisfaction score
        score_list = np.concatenate(score_list)
        pre_score_list = np.concatenate(pre_score_list)
        ssa_acc = accuracy_score(score_list, pre_score_list)
        ssa_macro_f1 = f1_score(score_list, pre_score_list, average="macro")
        ssa_f1_0, ssa_f1_1, ssa_f1_2 = f1_score(
            score_list, pre_score_list, labels=[0, 1, 2], average=None
        )

        print(confusion_matrix(score_list, pre_score_list))
        print(
            "SSA %s : Loss:%.3f\tWS F1:%.3f\tF1:%.3f\tUS F1:%.3f\tMacro F1:%.3f\tAcc.:%.3f"
            % (
                task,
                total_loss / float(counter),
                ssa_f1_2,
                ssa_f1_1,
                ssa_f1_0,
                ssa_macro_f1,
                ssa_acc,
            )
        )
        self.logger.info(
            "SSA %s : Loss:%.3f\tWS F1:%.3f\tF1:%.3f\tUS F1:%.3f\tMacro F1:%.3f\tAcc.:%.3f"
            % (
                task,
                total_loss / float(counter),
                ssa_f1_2,
                ssa_f1_1,
                ssa_f1_0,
                ssa_macro_f1,
                ssa_acc,
            )
        )

        if task == "test":
            self.logger.info(
                "Handoff Test Metrics:\tF1Score\tMacro_F1Score\tAUC\tGT-I\tGT-II\tGT-III"
            )
            self.logger.info(
                "Metrics %s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f"
                % (task, f1_handoff, macro_handoff, auc_score, gtt_1, gtt_2, gtt_3)
            )
            self.logger.info(
                "\n"
                + classification_report(
                    total_handoff_flat, total_pre_handoff_flat, digits=4
                )
            )
            self.logger.info(
                "\n" + str(confusion_matrix(total_handoff_flat, total_pre_handoff_flat))
            )

            self.logger.info(
                "Sentiment Test Metrics:\tPO F1\tNE F1\tNG F1\tMacro F1\tAcc."
            )
            self.logger.info(
                "Metrics %s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f"
                % (task, senti_f1_2, senti_f1_1, senti_f1_0, senti_macro_f1, senti_acc)
            )
            self.logger.info(
                "\n"
                + classification_report(
                    total_senti_flat, total_pre_senti_flat, digits=4
                )
            )
            self.logger.info(
                "\n" + str(confusion_matrix(total_senti_flat, total_pre_senti_flat))
            )

            self.logger.info("SSA Test Metrics:\tWS F1\tF1\tUS F1\tMacro F1\tAcc.")
            self.logger.info(
                "Metrics %s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f"
                % (task, ssa_f1_2, ssa_f1_1, ssa_f1_0, ssa_macro_f1, ssa_acc)
            )
            self.logger.info(
                "\n" + classification_report(score_list, pre_score_list, digits=4)
            )
            self.logger.info("\n" + str(confusion_matrix(score_list, pre_score_list)))
            tmp_lambda = 0.99
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_handoff, total_pre_handoff, lamb=tmp_lambda
            )
            self.logger.info(
                "Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3)
            )
            tmp_lambda = 0.75
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_handoff, total_pre_handoff, lamb=tmp_lambda
            )
            self.logger.info(
                "Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3)
            )
            tmp_lambda = 0.5
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_handoff, total_pre_handoff, lamb=tmp_lambda
            )
            self.logger.info(
                "Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3)
            )
            tmp_lambda = 0.25
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_handoff, total_pre_handoff, lamb=tmp_lambda
            )
            self.logger.info(
                "Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3)
            )
            tmp_lambda = 0.0
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_handoff, total_pre_handoff, lamb=tmp_lambda
            )
            self.logger.info(
                "Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3)
            )
            tmp_lambda = -0.25
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_handoff, total_pre_handoff, lamb=tmp_lambda
            )
            self.logger.info(
                "Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3)
            )
            tmp_lambda = -0.5
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_handoff, total_pre_handoff, lamb=tmp_lambda
            )
            self.logger.info(
                "Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3)
            )
            tmp_lambda = -0.75
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_handoff, total_pre_handoff, lamb=tmp_lambda
            )
            self.logger.info(
                "Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3)
            )
            tmp_lambda = -0.99
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_handoff, total_pre_handoff, lamb=tmp_lambda
            )
            self.logger.info(
                "Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3)
            )

        else:
            val_summary = tf.compat.v1.Summary(
                value=[
                    tf.compat.v1.Summary.Value(
                        tag="loss", simple_value=total_loss / float(counter)
                    ),
                    tf.compat.v1.Summary.Value(
                        tag="metrics/score_acc", simple_value=ssa_acc
                    ),
                    tf.compat.v1.Summary.Value(
                        tag="metrics/GT-III", simple_value=gtt_3 * 100
                    ),
                ]
            )
            self.summary_writer_eval.add_summary(val_summary, global_step)

        return (
            total_loss / float(counter),
            f1_handoff,
            macro_handoff,
            auc_score,
            gtt_1,
            gtt_2,
            gtt_3,
            ssa_f1_2,
            ssa_f1_1,
            ssa_f1_0,
            ssa_macro_f1,
            ssa_acc,
        )

    def test_cmhch(
        self,
        data_generator,
        data,
        batch_size=20,
        nb_classes=2,
        test_task="test",
        model_path="",
    ):
        self.restore(model_path, "cmhch")
        self.evaluate_batch_cmhch(
            data_generator,
            data,
            task=test_task,
            global_step=0,
            batch_size=batch_size,
            nb_classes=nb_classes,
            shuffle=False,
        )

    def train_mhch(
        self,
        data_generator,
        keep_prob,
        epochs,
        data,
        task="train",
        batch_size=20,
        nb_classes=2,
        shuffle=True,
        is_val=True,
        is_test=True,
        save_best=True,
        val_task="eval",
        test_task="test",
    ):

        print("Using mhch classification trainer.")
        max_val = 0

        for epoch in range(epochs):
            self.logger.info(
                "Training the models for epoch {} with batch size {}".format(
                    epoch, batch_size
                )
            )
            print("Training Epoch: {}".format(epoch))
            counter, total_loss = 0.0, 0.0
            total_label = []
            total_pre_label = []
            total_pre_scores = []

            for (
                batch_dialog_ids,
                batch_sent_len,
                batch_dia_len,
                batch_ids,
                batch_role_ids,
                batch_handoff,
                batch_senti,
                batch_score,
            ) in data_generator(
                task=task,
                batch_size=batch_size,
                nb_classes=nb_classes,
                shuffle=shuffle,
                epoch=epoch,
            ):
                feed_dict = {
                    self.input_x1: batch_dialog_ids,
                    self.role_list: batch_role_ids,
                    self.dia_len: batch_dia_len,
                    self.sent_len: batch_sent_len,
                    self.main_y: batch_handoff,
                    self.dropout_keep_prob: keep_prob,
                }
                try:
                    _, step, loss, output, scores = self.sess.run(
                        [
                            self.train_op,
                            self.global_step,
                            self.loss,
                            self.output,
                            self.proba,
                        ],
                        feed_dict,
                    )
                    # for handoff metric
                    true_y = np.argmax(batch_handoff, -1)
                    for batch_id in range(len(batch_dia_len)):
                        total_label.append(true_y[batch_id, : batch_dia_len[batch_id]])
                        total_pre_label.append(
                            output[batch_id, : batch_dia_len[batch_id]]
                        )
                        total_pre_scores.append(
                            scores[batch_id, : batch_dia_len[batch_id], 1]
                        )

                    total_loss += loss
                    counter += 1

                except ValueError as e:
                    self.logger.info("Wrong batch.{}".format(e))

            total_label_flat = np.concatenate(total_label)
            total_pre_label_flat = np.concatenate(total_pre_label)

            acc, p, r, f1, macro = get_a_p_r_f_sara(
                target=total_label_flat, predict=total_pre_label_flat
            )
            print(confusion_matrix(total_label_flat, total_pre_label_flat))

            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_label, total_pre_label, lamb=self.lamb
            )

            print(
                "MHCH Metrics %s: Loss:%.3f\tF1Score:%.3f\tMacro_F1Score:%.3f\tGT-I:%.3f\tGT-II:%.3f\tGT-III:%.3f"
                % (task, total_loss / float(counter), f1, macro, gtt_1, gtt_2, gtt_3)
            )
            self.logger.info(
                "MHCH Metrics %s: Loss:%.3f\tF1Score:%.3f\tMacro_F1Score:%.3f\tGT-I:%.3f\tGT-II:%.3f\tGT-III:%.3f"
                % (task, total_loss / float(counter), f1, macro, gtt_1, gtt_2, gtt_3)
            )

            if is_val:
                (
                    eval_loss,
                    eval_acc,
                    eval_f1,
                    eval_macro,
                    eval_auc,
                    gs1,
                    gs2,
                    gs3,
                ) = self.evaluate_batch_mhch(
                    data_generator,
                    data,
                    task=val_task,
                    batch_size=batch_size,
                    nb_classes=nb_classes,
                    shuffle=False,
                )

                metrics_dict = {
                    "loss": eval_loss,
                    "acc": eval_acc,
                    "f1": eval_f1,
                    "macro": eval_macro,
                    "gs1": gs1,
                    "gs2": gs2,
                    "gs3": gs3,
                }
                if metrics_dict["macro"] > max_val and save_best:
                    max_val = metrics_dict["macro"]
                    self.save(self.save_dir, self.model_name + ".best")
                    print("Saved!")

        self.save(self.save_dir, self.model_name + ".last")
        if is_test:
            self.restore(self.save_dir, self.model_name + ".best")
            self.evaluate_batch_mhch(
                data_generator,
                data,
                task=test_task,
                batch_size=batch_size,
                nb_classes=nb_classes,
                shuffle=False,
            )
        self.summary_writer_train.close()
        self.summary_writer_eval.close()

    def evaluate_batch_mhch(
        self,
        data_generator,
        data,
        task="eval",
        batch_size=32,
        nb_classes=2,
        shuffle=False,
    ):
        counter, total_loss = 0, 0.0
        total_label = []
        total_pre_label = []
        total_pre_scores = []

        for (
            batch_dialog_ids,
            batch_sent_len,
            batch_dia_len,
            batch_ids,
            batch_role_ids,
            batch_handoff,
            batch_senti,
            batch_score,
        ) in data_generator(
            task=task, batch_size=batch_size, nb_classes=nb_classes, shuffle=shuffle
        ):

            feed_dict = {
                self.input_x1: batch_dialog_ids,
                self.role_list: batch_role_ids,
                self.dia_len: batch_dia_len,
                self.sent_len: batch_sent_len,
                self.main_y: batch_handoff,
                self.dropout_keep_prob: 1.0,
            }
            try:
                loss, output, probas = self.sess.run(
                    [self.loss, self.output, self.proba], feed_dict
                )
                # for handoff metric
                true_y = np.argmax(batch_handoff, -1)
                for batch_id in range(len(batch_dia_len)):
                    total_label.append(true_y[batch_id, : batch_dia_len[batch_id]])
                    total_pre_label.append(output[batch_id, : batch_dia_len[batch_id]])
                    total_pre_scores.append(
                        probas[batch_id, : batch_dia_len[batch_id], 1]
                    )

                total_loss += loss
                counter += 1

            except ValueError as e:
                self.logger.info("Wrong batch.{}".format(e))

        total_label_flat = np.concatenate(total_label)
        total_pre_label_flat = np.concatenate(total_pre_label)
        total_pre_scores_flat = np.concatenate(total_pre_scores)

        acc, p, r, f1, macro = get_a_p_r_f_sara(
            target=total_label_flat, predict=total_pre_label_flat
        )
        print(confusion_matrix(total_label_flat, total_pre_label_flat))
        gtt_1, gtt_2, gtt_3 = get_gtt_score(
            total_label, total_pre_label, lamb=self.lamb
        )

        # calc AUC score
        fpr, tpr, thresholds = roc_curve(
            total_label_flat, total_pre_scores_flat, pos_label=1
        )
        auc_score = auc(fpr, tpr)

        print(
            "MHCH %s: Loss:%.3f\tF1Score:%.3f\tMacro_F1Score:%.3f\tAUC:%.3f\tGT-I:%.3f\tGT-II:%.3f\tGT-III:%.3f"
            % (
                task,
                total_loss / float(counter),
                f1,
                macro,
                auc_score,
                gtt_1,
                gtt_2,
                gtt_3,
            )
        )
        self.logger.info(
            "MHCH %s: Loss:%.3f\tF1Score:%.3f\tMacro_F1Score:%.3f\tAUC:%.3f\tGT-I:%.3f\tGT-II:%.3f\tGT-III:%.3f"
            % (
                task,
                total_loss / float(counter),
                f1,
                macro,
                auc_score,
                gtt_1,
                gtt_2,
                gtt_3,
            )
        )

        if task == "test":
            self.logger.info(
                "Handoff Test Metrics:\tF1Score\tMacro_F1Score\tAUC\tGT-I\tGT-II\tGT-III"
            )
            self.logger.info(
                "Metrics %s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f"
                % (task, f1, macro, auc_score, gtt_1, gtt_2, gtt_3)
            )

            self.logger.info(
                "\n"
                + classification_report(
                    total_label_flat, total_pre_label_flat, digits=4
                )
            )
            self.logger.info(
                "\n" + str(confusion_matrix(total_label_flat, total_pre_label_flat))
            )
            tmp_lambda = 0.99
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_label, total_pre_label, lamb=tmp_lambda
            )
            self.logger.info(
                "Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3)
            )
            tmp_lambda = 0.75
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_label, total_pre_label, lamb=tmp_lambda
            )
            self.logger.info(
                "Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3)
            )
            tmp_lambda = 0.5
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_label, total_pre_label, lamb=tmp_lambda
            )
            self.logger.info(
                "Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3)
            )
            tmp_lambda = 0.25
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_label, total_pre_label, lamb=tmp_lambda
            )
            self.logger.info(
                "Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3)
            )
            tmp_lambda = 0.0
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_label, total_pre_label, lamb=tmp_lambda
            )
            self.logger.info(
                "Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3)
            )
            tmp_lambda = -0.25
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_label, total_pre_label, lamb=tmp_lambda
            )
            self.logger.info(
                "Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3)
            )
            tmp_lambda = -0.5
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_label, total_pre_label, lamb=tmp_lambda
            )
            self.logger.info(
                "Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3)
            )
            tmp_lambda = -0.75
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_label, total_pre_label, lamb=tmp_lambda
            )
            self.logger.info(
                "Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3)
            )
            tmp_lambda = -0.99
            gtt_1, gtt_2, gtt_3 = get_gtt_score(
                total_label, total_pre_label, lamb=tmp_lambda
            )
            self.logger.info(
                "Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3)
            )

        return (
            total_loss / float(counter),
            acc,
            f1,
            macro,
            auc_score,
            gtt_1,
            gtt_2,
            gtt_3,
        )

    def train_ssa(
        self,
        data_generator,
        keep_prob,
        epochs,
        data,
        task="train",
        batch_size=20,
        nb_classes=2,
        shuffle=True,
        is_val=True,
        is_test=True,
        save_best=True,
        val_task="eval",
        test_task="test",
    ):

        print("Using SSA classification trainer.")
        max_val = 0

        for epoch in range(epochs):
            self.logger.info(
                "Training the models for epoch {} with batch size {}".format(
                    epoch, batch_size
                )
            )
            print("Training Epoch: {}".format(epoch))
            counter, total_loss = 0.0, 0.0
            total_label = []
            total_pre_label = []

            for (
                batch_dialog_ids,
                batch_sent_len,
                batch_dia_len,
                batch_ids,
                batch_role_ids,
                batch_handoff,
                batch_senti,
                batch_score,
            ) in data_generator(
                task=task,
                batch_size=batch_size,
                nb_classes=nb_classes,
                shuffle=shuffle,
                epoch=epoch,
            ):

                feed_dict = {
                    self.input_x1: batch_dialog_ids,
                    self.role_list: batch_role_ids,
                    self.dia_len: batch_dia_len,
                    self.sent_len: batch_sent_len,
                    self.score_y: batch_score,
                    self.dropout_keep_prob: keep_prob,
                }
                try:
                    _, step, loss, output = self.sess.run(
                        [self.train_op, self.global_step, self.loss, self.output],
                        feed_dict,
                    )
                    true_y = np.argmax(batch_score, -1)

                    total_label.append(true_y)
                    total_pre_label.append(output)

                    total_loss += loss
                    counter += 1

                except ValueError as e:
                    self.logger.info("Wrong batch.{}".format(e))
            label_list = np.concatenate(total_label)
            pre_label_list = np.concatenate(total_pre_label)
            acc = accuracy_score(label_list, pre_label_list)
            macro_f1 = f1_score(label_list, pre_label_list, average="macro")
            f1_0, f1_1, f1_2 = f1_score(
                label_list, pre_label_list, labels=[0, 1, 2], average=None
            )

            print(confusion_matrix(label_list, pre_label_list))
            print(
                "SSA Training Metrics %s: Loss:%.3f\tWS F1:%.3f\tF1:%.3f\tUS F1:%.3f\tMacro F1:%.3f\tAcc.:%.3f"
                % (task, total_loss / float(counter), f1_2, f1_1, f1_0, macro_f1, acc)
            )
            self.logger.info(
                "SSA Training Metrics %s: Loss:%.3f\tWS F1:%.3f\tF1:%.3f\tUS F1:%.3f\tMacro F1:%.3f\tAcc.:%.3f"
                % (task, total_loss / float(counter), f1_2, f1_1, f1_0, macro_f1, acc)
            )

            if is_val:

                (
                    eval_loss,
                    eval_f1_2,
                    eval_f1_1,
                    eval_f1_0,
                    eval_macro,
                    eval_acc,
                ) = self.evaluate_batch_ssa(
                    data_generator,
                    data,
                    task=val_task,
                    batch_size=batch_size,
                    nb_classes=nb_classes,
                    shuffle=False,
                )

                metrics_dict = {
                    "loss": eval_loss,
                    "acc": eval_acc,
                    "macro": eval_macro,
                    "f1_0": eval_f1_0,
                    "f1_1": eval_f1_1,
                    "f1_2": eval_f1_2,
                }

                if metrics_dict["macro"] > max_val and save_best:
                    max_val = metrics_dict["macro"]
                    self.save(self.save_dir, self.model_name + ".best")
                    print("Saved!")

        self.save(self.save_dir, self.model_name + ".last")
        if is_test:
            test_t = time.time()
            self.restore(self.save_dir, self.model_name + ".best")
            self.evaluate_batch_ssa(
                data_generator,
                data,
                task=test_task,
                batch_size=batch_size,
                nb_classes=nb_classes,
                shuffle=False,
            )
        self.summary_writer_train.close()
        self.summary_writer_eval.close()

    def evaluate_batch_ssa(
        self,
        data_generator,
        data,
        task="eval",
        batch_size=32,
        nb_classes=2,
        shuffle=False,
    ):
        counter, total_loss = 0, 0.0
        total_label = []
        total_pre_label = []

        for (
            batch_dialog_ids,
            batch_sent_len,
            batch_dia_len,
            batch_ids,
            batch_role_ids,
            batch_handoff,
            batch_senti,
            batch_score,
        ) in data_generator(
            task=task, batch_size=batch_size, nb_classes=nb_classes, shuffle=shuffle
        ):

            feed_dict = {
                self.input_x1: batch_dialog_ids,
                self.role_list: batch_role_ids,
                self.dia_len: batch_dia_len,
                self.sent_len: batch_sent_len,
                self.score_y: batch_score,
                self.dropout_keep_prob: 1.0,
            }
            try:
                loss, output = self.sess.run([self.loss, self.output], feed_dict)
                true_y = np.argmax(batch_score, -1)
                total_label.append(true_y)
                total_pre_label.append(output)

                total_loss += loss
                counter += 1

            except ValueError as e:
                self.logger.info("Wrong batch.{}".format(e))
        label_list = np.concatenate(total_label)
        pre_label_list = np.concatenate(total_pre_label)
        acc = accuracy_score(label_list, pre_label_list)
        macro_f1 = f1_score(label_list, pre_label_list, average="macro")
        f1_0, f1_1, f1_2 = f1_score(
            label_list, pre_label_list, labels=[0, 1, 2], average=None
        )
        print(confusion_matrix(label_list, pre_label_list))

        print(
            "SSA Training Metrics %s: Loss:%.3f\tWS F1:%.3f\tF1:%.3f\tUS F1:%.3f\tMacro F1:%.3f\tAcc.:%.3f"
            % (task, total_loss / float(counter), f1_2, f1_1, f1_0, macro_f1, acc)
        )
        self.logger.info(
            "SSA Training Metrics %s: Loss:%.3f\tWS F1:%.3f\tF1:%.3f\tUS F1:%.3f\tMacro F1:%.3f\tAcc.:%.3f"
            % (task, total_loss / float(counter), f1_2, f1_1, f1_0, macro_f1, acc)
        )

        if task == "test":
            self.logger.info("Handoff Test Metrics:\tWS F1\tF1\tUS F1\tMacro F1\tAcc.")
            self.logger.info(
                "Metrics %s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f"
                % (task, f1_2, f1_1, f1_0, macro_f1, acc)
            )

            self.logger.info(
                "\n" + classification_report(label_list, pre_label_list, digits=4)
            )
            self.logger.info("\n" + str(confusion_matrix(label_list, pre_label_list)))

        return total_loss / float(counter), f1_2, f1_1, f1_0, macro_f1, acc


if __name__ == "__main__":
    pass
