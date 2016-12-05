import os
import argparse

import numpy as np
import tensorflow as tf
import chess

def make_board_tensor(epd_sym):
    def conv(epd):
        b = chess.Board()
        b.set_epd(epd)


class LogisticRegression(object):
    def __init__(self, datafile, num_features, batch_size, min_after_dequeue, use_bias, learning_rate, use_board_vector, float_type, discount):
        self._datafile = datafile
        self._num_features = num_features
        self._batch_size = batch_size
        self._min_after_dequeue = min_after_dequeue
        self._use_bias = use_bias
        self._learning_rate = learning_rate
        self._float_type = tf.float64 if float_type == "float64" else tf.float32
        self._discount = discount
        self._use_board_vector = use_board_vector

        self._build_params()
        self._build_train_model()

    def _build_params(self):
        self._W = tf.Variable(dtype=self._float_type, initial_value=np.zeros([self._num_features]))
        if self._use_bias:
            self._b = tf.Variable(dtype=self._float_type, initial_value=0.)
        save_d = {}
        save_d["weights"] = self._W
        if self._use_bias:
            save_d["bias"] = self._b
        self._saver = tf.train.Saver(save_d)

    def _get_features(self):
        file_q = tf.train.string_input_producer([self._datafile])
        if self._use_board_vector:
            reader = tf.TextLineReader()
            _, row = reader.read(file_q)
            record_defaults = [[""], [0], [-1]]
            epd, result, steps = tf.decode_csv(row, record_defaults)
            board = make_board_tensor(epd)
            result = make_result_tensor(result)
            steps = make_steps_tensor(steps)
        else:
            reader = tf.TFRecordReader()
            _, example = reader.read(file_q)
            features = tf.parse_single_example(
                example,
                features={
                    "board": tf.FixedLenFeature([], tf.string),
                    "steps": tf.FixedLenFeature([], tf.int64),
                    "label": tf.FixedLenFeature([], tf.int64)
                }
            )
            board_, steps, label_ = features["board"], features["steps"], features["label"]
            board = tf.decode_raw(board_, tf.float64)
            board = tf.cast(board, self._float_type)
            board.set_shape([self._num_features])
            steps = tf.cast(steps, self._float_type)
            label = tf.cast((label_ + 1) / 2, self._float_type)

            return board, steps, label

    def _get_scores(self, board_b):
        scores_b = tf.reduce_sum((board_b * self._W), 1)
        return scores_b

    def _build_train_model(self):
        board, steps, label = self._get_features()
        board_b, steps_b, label_b = tf.train.shuffle_batch(
            [board, steps, label],
            batch_size=self._batch_size,
            capacity=self._min_after_dequeue + 3 * self._batch_size,
            min_after_dequeue=self._min_after_dequeue
        )

        scores_b = self._get_scores(board_b)
        if self._use_bias:
            scores_b += self._use_bias

        logloss = tf.nn.sigmoid_cross_entropy_with_logits(scores_b, label_b)
        costs = tf.pow(tf.cast(self._discount, self._float_type), steps_b) * logloss
        self._cost = tf.reduce_mean(costs)
        self._opt = tf.train.GradientDescentOptimizer(self._learning_rate).minimize(self._cost)

    def init(self, sess, prev_checkpoint=None):
        variables = tf.global_variables()
        sess.run(tf.variables_initializer(variables))
        if prev_checkpoint:
            self._saver.restore(prev_checkpoint)

    def run_for_batches(self, sess, num_batches):
        self._coord = tf.train.Coordinator()
        self._threads = tf.train.start_queue_runners(sess=sess, coord=self._coord)

        for step in range(num_batches):
            if step % 100 == 0:
                _, cost = sess.run([self._opt, self._cost])
                print "Step: %s, cost: %s" % (step + 1, cost)
            else:
                sess.run([self._opt])

    def save(self, sess, path):
        self._saver.save(sess, path)

    def close(self, sess):
        self._coord.request_stop()
        self._coord.join(self._threads)

def main(args):
    model = LogisticRegression(datafile=args.datafile, num_features=args.num_features, batch_size=args.batch_size,
                               min_after_dequeue=args.min_after_dequeue, use_bias=args.use_bias, learning_rate=args.learning_rate,
                               use_board_vector=args.board_vector, float_type=args.float_type, discount=args.discount)
    with tf.Session() as sess:
        model.init(sess, prev_checkpoint=args.prev_checkpoint)
        i = 0
        while True:
            i += 1
            model.run_for_batches(sess, args.checkpoint_freq)
            model.save(sess, os.path.join(args.checkpoint_folder, "model%s.ckpt" % i))
        model.close(sess)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run logistic regression on the board positions dataset.")
    parser.add_argument("--datafile", type=str, help="The board positions dataset", required=True)
    parser.add_argument("--num_features", type=int, help="The length of the feature vector")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size in training.")
    parser.add_argument("--min_after_dequeue", type=int, required=True, help="The buffer size.")
    parser.add_argument("--use_bias", type=bool, default=False, help="Use a bias in the model.")
    parser.add_argument("--prev_checkpoint", type=str, default=None, help="Checkpoint to resume from.")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate.")
    parser.add_argument("--checkpoint_folder", type=str, required=True, help="Folder in which to produce checkpoints.")
    parser.add_argument("--checkpoint_freq", type=int, required=True, help="How often to save checkpoints.")
    parser.add_argument("--board_vector", type=bool, default=False, help="If True, the model uses generates a board representation from the epd; num_features is ignored.")
    parser.add_argument("--float_type", type=str, default="float64", help="The float type to use, float32 or float64.")
    parser.add_argument("--discount", type=float, default=1., help="How much to discount early board positions.")

    main(parser.parse_args())
