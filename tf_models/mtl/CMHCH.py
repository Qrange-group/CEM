#!/usr/bin/env python
# -*- coding: utf-8 -*-


from cProfile import label
import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)
sys.path.insert(0, prodir)
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or any {'0', '1', '3'}
from Network import Network
from tf_models.layers.attention import *
from tf_models.layers.transformer import *


class CMHCH(Network):
    def __init__(
        self,
        memory=0,
        vocab=None,
        config_dict=None,
        **kwargs
    ):
        Network.__init__(self, memory=memory, vocab=vocab)
        self.model_name = self.__class__.__name__
        self.logger.info("Model Name: {}".format(self.model_name))

        self.classification_dim = config_dict["classification_dim"]
        self.state_dim = config_dict["state_dim"]
        self.num_class = config_dict["num_class"]
        self.rnn_dim = config_dict["rnn_dim"]
        self.dense_dim = config_dict["dense_dim"]
        self.attention_size = config_dict["attention_size"]

        self.num_heads = config_dict["num_heads"]
        self.ff_dim = config_dict["ff_dim"]  # hidden size

        self.lr = config_dict["learning_rate"]
        self.l2_reg_lambda = config_dict["l2_reg_lambda"]
        self.weight_decay = config_dict["weight_decay"]
        self.warmup_steps = config_dict["warmup_steps"]
        self.loss_lambda = config_dict["loss_lambda"]

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
        self.senti_y = tf.compat.v1.placeholder(
            tf.float32, [None, self.dia_max_len, 3], name="senti_y"
        )
        self.score_y = tf.compat.v1.placeholder(tf.float32, [None, 3], name="score_y")

        self.dropout_keep_prob = tf.compat.v1.placeholder(
            tf.float32, name="dropout_keep_prob"
        )
        self.logger.info("setup placeholders.")

    def _inference(self):
        """
        encode and interact for prediction
        """
        # [B, T, S_len, E_dim]
        self.embedded = tf.nn.embedding_lookup(self.word_embeddings, self.input_x1)

        if self.dropout_keep_prob != 1:
            self.embedded = tf.nn.dropout(
                self.embedded, rate=1 - self.dropout_keep_prob
            )

        # reshape
        # [B * T, S_len, E_dim]
        self.embedded_reshaped = tf.reshape(
            self.embedded, shape=[-1, self.sent_max_len, self.embedding_dim]
        )
        self.sent_len_reshape = tf.reshape(self.sent_len, shape=[-1])
        # self.embedded.shape           # (?, 50, 64, 200)
        # self.embedded_reshaped.shape  # (?, 64, 200)
        # self.sent_len_reshape.shape   # (?,)

        with tf.name_scope("utterance_encoder"):
            self.utter_encode_output, self.utter_encode_state = self._bidirectional_rnn(
                self.embedded_reshaped, self.sent_len_reshape, rnn_type="lstm"
            )
            with tf.compat.v1.variable_scope("utter_attention"):
                _, self.alphas = attention(
                    self.utter_encode_output,
                    attention_size=self.attention_size,
                    return_alphas=True,
                )

            # reduce with attention vector
            self.utter_attened = tf.reduce_sum(
                self.utter_encode_output * tf.expand_dims(self.alphas, -1), 1
            )
            # self.utter_encode_output.shape            (?, sent_len, 256)
            # (self.utter_encode_output * tf.expand_dims(self.alphas, -1)).shape (?, sent_len, 256)
            # self.utter_attened.shape                  (?, 256)
            self.sent_encoder_attened_concat = tf.concat(
                [self.utter_attened, self.utter_encode_state], axis=-1
            )
            # self.sent_encoder_attened_intersection = tf.layers.dense(self.sent_encoder_attened_concat, 4 * self.rnn_dim, activation=tf.nn.tanh, use_bias=True)
            # self.sent_encoder_attened_reshape = tf.reshape(self.sent_encoder_attened_intersection, shape=[-1, self.dia_max_len, 4 * self.rnn_dim])
            self.sent_encoder_attened_intersection = tf.layers.dense(
                self.sent_encoder_attened_concat,
                self.rnn_dim,
                activation=tf.nn.tanh,
                use_bias=True,
            )
            self.sent_encoder_attened_reshape = tf.reshape(
                tf.concat(
                    [
                        self.sent_encoder_attened_intersection,
                        self.sent_encoder_attened_concat,
                    ],
                    axis=-1,
                ),
                shape=[-1, self.dia_max_len, 5 * self.rnn_dim],
            )

            # self.sent_encoder_attened_reshape = tf.reshape(self.utter_attened,
            #                                                shape=[-1, self.dia_max_len, 2 * self.rnn_dim])
            # self.sent_encoder_state_reshape = tf.reshape(self.utter_encode_state,
            #                                              shape=[-1, self.dia_max_len, 2 * self.rnn_dim])
            # # self.sent_encoder_attened_reshape.shape   (?, dia_max_len, 256)
            # self.sent_encoder_attened_concat = tf.concat(
            #     [self.sent_encoder_attened_reshape, self.sent_encoder_state_reshape], axis=-1)

            if self.dropout_keep_prob != 1:
                # [B * T, 2 * H]
                self.sent_encoder_attened_reshape = tf.nn.dropout(
                    self.sent_encoder_attened_reshape, rate=1 - self.dropout_keep_prob
                )

        with tf.name_scope("dialogue_mask"):
            # [B] => [B, T]
            self.dia_seq_mask = tf.sequence_mask(
                self.dia_len, maxlen=self.dia_max_len, dtype=tf.float32
            )
            # [B, T] => [B, T, H]
            self.customer_mask = tf.tile(
                tf.expand_dims(self.role_list, axis=-1),
                multiples=[1, 1, self.dia_max_len],
            )
            self.agent_mask = tf.tile(
                tf.expand_dims(tf.add(tf.negative(self.role_list), 1), axis=-1),
                multiples=[1, 1, self.dia_max_len],
            )
            self.dia_att_mask = tf.tile(
                tf.expand_dims(self.dia_seq_mask, axis=-1),
                multiples=[1, 1, self.dia_max_len],
            )

        with tf.name_scope("universal_context"):
            # match context
            self.cross_match = tf.matmul(
                self.sent_encoder_attened_reshape,
                tf.transpose(self.sent_encoder_attened_reshape, [0, 2, 1]),
            )
            self.cross_match_upper = tf.matrix_band_part(
                self.cross_match, num_lower=0, num_upper=-1
            )
            self.cross_match_sim_one_direct = self.cross_match - self.cross_match_upper
            self.encode_match = tf.concat(
                [self.sent_encoder_attened_reshape, self.cross_match_sim_one_direct],
                axis=-1,
            )

        with tf.name_scope("mhch_transform"):
            self.mhch_enc_dense = tf.keras.layers.Dense(
                units=self.dense_dim, activation="relu"
            )(self.encode_match)
            if self.dropout_keep_prob != 1:
                self.mhch_enc_dense = tf.nn.dropout(
                    self.mhch_enc_dense, rate=1 - self.dropout_keep_prob
                )

        with tf.name_scope("ssa_transform"):
            self.ssa_enc_dense = tf.keras.layers.Dense(
                units=self.dense_dim, activation="relu"
            )(self.encode_match)
            if self.dropout_keep_prob != 1:
                self.ssa_enc_dense = tf.nn.dropout(
                    self.ssa_enc_dense, rate=1 - self.dropout_keep_prob
                )

        with tf.name_scope("ssa2mhch_context"):
            self.ssa2mhch_matrix = tf.matmul(
                self.mhch_enc_dense, tf.transpose(self.ssa_enc_dense, perm=[0, 2, 1])
            )
            paddings = tf.ones_like(self.ssa2mhch_matrix) * (-(2 ** 32) + 1)
            self.ssa2mhch_matrix_dia_masked = tf.where(
                tf.equal(self.dia_att_mask, 0), paddings, self.ssa2mhch_matrix
            )
            self.ssa2mhch_matrix_customer = tf.where(
                tf.equal(self.customer_mask, 0),
                paddings,
                self.ssa2mhch_matrix_dia_masked,
            )
            self.ssa2mhch_alpha = tf.nn.softmax(self.ssa2mhch_matrix_customer)
            # [B, T, Dense_dim]
            self.ssa2mhch_atted_vec = tf.matmul(self.ssa2mhch_alpha, self.ssa_enc_dense)
            self.ssa2mhch_concat = tf.concat(
                [self.ssa2mhch_atted_vec, self.mhch_enc_dense], -1
            )
            self.ssa2mhch_final_vec = tf.keras.layers.Dense(
                units=self.dense_dim, activation="relu"
            )(self.ssa2mhch_concat)
            # self.ssa2mhch_final_vec.shape (?, dia_max_len, 256)

        with tf.name_scope("mhch2ssa_context"):
            self.mhch2ssa_matrix = tf.matmul(
                self.ssa_enc_dense, tf.transpose(self.mhch_enc_dense, perm=[0, 2, 1])
            )
            # position weights
            self.N, self.T = (
                tf.shape(self.mhch2ssa_matrix)[0],
                tf.shape(self.mhch2ssa_matrix)[1],
            )  # dynamic
            self.position_ind = tf.tile(
                tf.expand_dims(tf.range(self.T), 0), [self.N, 1]
            )  # [N, T]
            self.dial_position_weights = tf.nn.softmax(
                tf.cast(self.position_ind, tf.float32)
            )
            self.mhch2ssa_matrix = self.mhch2ssa_matrix * tf.expand_dims(
                self.dial_position_weights, -1
            )

            paddings = tf.ones_like(self.mhch2ssa_matrix) * (-(2 ** 32) + 1)
            self.mhch2ssa_matrix_dia_masked = tf.where(
                tf.equal(self.dia_att_mask, 0), paddings, self.mhch2ssa_matrix
            )

            # w/o Role difference
            self.mhch2ssa_alpha = tf.nn.softmax(self.mhch2ssa_matrix_dia_masked)
            self.mhch2ssa_atted_vec = tf.matmul(
                self.mhch2ssa_alpha, self.mhch_enc_dense
            )

            self.mhch2ssa_final_vec = self.ssa_enc_dense + self.mhch2ssa_atted_vec
            self.mhch2ssa_final_vec = self.ln(self.mhch2ssa_final_vec, scope="mhch2ssa")

        with tf.name_scope("ssa_context"):
            with tf.variable_scope("ssa_context_encoding"):
                self.ssa_context_customer = self.multihead_attention(
                    queries=self.mhch2ssa_final_vec,
                    keys=self.mhch2ssa_final_vec,
                    values=self.mhch2ssa_final_vec,
                    key_masks=self.dia_seq_mask,
                    causality=True,
                    num_heads=self.num_heads,
                    dropout_rate=1 - self.dropout_keep_prob,
                )
                # [B, T, Dense_dim]
                self.ssa_ff_customer = self.ff(
                    self.ssa_context_customer, num_units=[self.ff_dim, self.dense_dim]
                )
                self.ssa_ff_customer_distri = tf.keras.layers.Dense(
                    units=3, activation="softmax"
                )(self.ssa_ff_customer)

        with tf.name_scope("mhch_context"):
            with tf.variable_scope("mhch_context_encoding"):
                # [B, T, H], [B, H]
                cells = tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.BasicLSTMCell(self.rnn_dim, state_is_tuple=True),
                    output_keep_prob=self.dropout_keep_prob,
                )
                self.mhch_context_ff, _ = tf.nn.dynamic_rnn(
                    cell=cells,
                    inputs=self.ssa2mhch_final_vec,
                    sequence_length=self.dia_len,
                    dtype=tf.float32,
                )
                # self.mhch_context_ff = tf.matmul(, self.mhch_context_ff_rnn)
                # self.ssa2mhch_final_vec (?, dia_max_len, 256)
                # self.mhch_context_ff    (?, dia_max_len, 128)

        with tf.name_scope("ssa_dis_combination"):
            # Trainable parameters
            w_ssa = tf.Variable(tf.random.normal([self.dense_dim, self.attention_size]))
            b_ssa = tf.Variable(tf.random.normal([self.attention_size]))
            u_ssa = tf.Variable(tf.random.normal([self.attention_size]))
            # [B, T, A]
            v_ssa = tf.tanh(tf.matmul(self.ssa_ff_customer, w_ssa) + b_ssa)
            # [B, T]
            self.vu_ssa = tf.tensordot(v_ssa, u_ssa, axes=1, name="vu")
            paddings = tf.ones_like(self.vu_ssa) * (-(2 ** 32) + 1)
            # sequence mask
            self.vu_ssa_masked = tf.where(
                tf.equal(self.dia_seq_mask, 0), paddings, self.vu_ssa
            )
            # role mask
            self.vu_ssa_customer_masked = tf.where(
                tf.equal(self.role_list, 0), paddings, self.vu_ssa_masked
            )
            self.weights_local = tf.nn.softmax(self.vu_ssa_customer_masked)
            self.ssa_combine_vec = tf.reduce_sum(
                self.ssa_ff_customer_distri * tf.expand_dims(self.weights_local, -1), 1
            )

        with tf.name_scope("output"):
            # [B, T, nb_classes]
            self.main_logits = tf.keras.layers.Dense(units=2, activation="softmax")(
                self.mhch_context_ff
            )
            self.senti_logits = self.ssa_ff_customer_distri
            self.score_logits = self.ssa_combine_vec
            if not self.is_only_cf:
                self.main_logits = self.main_logits * tf.expand_dims(
                    tf.concat(
                        [
                            tf.expand_dims(self.score_logits[:, 0], axis=-1),
                            tf.expand_dims(self.score_logits[:, 2], axis=-1),
                        ],
                        -1,
                    ),
                    1,
                )

            # [B, T]
            self.output = tf.argmax(self.main_logits, axis=-1)
            self.senti = tf.argmax(self.senti_logits, axis=-1)
            # [B]
            self.pre_score = tf.argmax(self.score_logits, axis=-1)

            self.proba = self.main_logits

        self.score_correct_prediction = tf.equal(
            tf.argmax(self.score_y, -1), self.pre_score
        )
        self.score_acc = tf.reduce_mean(
            tf.cast(self.score_correct_prediction, tf.float32)
        )
        tf.compat.v1.summary.scalar("metrics/score_acc", self.score_acc)
        self.logger.info("network inference.")

    def _compute_loss(self):
        """
        The loss function by adding the AutomaticWeight
        """

        def nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                weight = tf.concat(
                    [
                        tf.pow(tf.expand_dims((1 - probs)[:, :, 1], -1), 0.5),
                        tf.pow(tf.expand_dims(probs[:, :, 0], -1), 0.5),
                    ],
                    -1,
                )
                losses = -tf.reduce_sum(
                    weight * labels * tf.math.log(probs + epsilon), -1
                )
            return losses

        # masked main loss
        self.cross_entropy_main = nll_loss(self.main_logits, self.main_y)
        self.mask_cross_entropy = self.dia_seq_mask * self.cross_entropy_main
        # reduction
        self.scale_cross_entropy = tf.reduce_sum(self.mask_cross_entropy, -1) / tf.cast(
            self.dia_len, tf.float32
        )
        self.main_loss = tf.reduce_mean(self.scale_cross_entropy)

        self.score_loss = tf.reduce_mean(
            -tf.reduce_sum(self.score_y * tf.math.log(self.score_logits + 1e-9), -1), -1
        )

        self.cost_loss_simulator = tf.reduce_mean(self.main_logits[:, 1])

        # Combine loss
        if self.is_only_ssa:
            self.loss = self.main_loss + self.loss_lambda * self.score_loss
        else:
            self.loss = (
                self.main_loss
                + self.loss_lambda * self.score_loss
                + 0.01 * self.cost_loss_simulator
            )
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
