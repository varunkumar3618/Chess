import time
import os

import numpy as np
import tensorflow as tf
import chess
import chess.pgn

from game import play
from ops import affine_layer
from agents import MTDAgent, HumanAgent, RandomAgent

FEATS_LEN = 2 * (8 * 8 * 6 + 3)

class Model(object):
    def __init__(self, sess, model_path, summary_path, checkpoint_path, restore=False, scope="model",
                 lamda=0.7, alpha=0.01, hidden_size=1024, depth=4):
        self._model_path = model_path
        self._summary_path = summary_path
        self._checkpoint_path = checkpoint_path
        self._sess = sess
        self._scope = scope
        self._hidden_size = hidden_size
        self._depth = depth

        with tf.variable_scope(scope):
            self._global_step = tf.Variable(0, trainable=False, name="global_step")

            # lambda decay
            self._lamda = tf.maximum(lamda, tf.train.exponential_decay(0.9, self._global_step, \
                30000, 0.96, staircase=True), name="lambda")
            # learning rate decay
            self._alpha = tf.maximum(alpha, tf.train.exponential_decay(0.1, self._global_step, \
                40000, 0.96, staircase=True), name="alpha")

            tf.scalar_summary('lambda', lamda)
            tf.scalar_summary('alpha', alpha)

            self._board_t = tf.placeholder(tf.float32, [None, FEATS_LEN], name="board")
            self._v_next = tf.placeholder(tf.float32, [None, 1], name="n_next")

            h1 = affine_layer(
                input_t=self._board_t,
                input_dim=FEATS_LEN,
                output_dim=self._hidden_size,
                w_init=tf.random_normal_initializer(dtype=tf.float32, stddev=0.02),
                b_init=tf.constant_initializer(0., dtype=tf.float32),
                activation=tf.sigmoid,
                name="affine_1"
            )
            h2 = affine_layer(
                input_t=h1,
                input_dim=self._hidden_size,
                output_dim=1,
                w_init=tf.random_normal_initializer(dtype=tf.float32, stddev=0.02),
                b_init=tf.constant_initializer(0., dtype=tf.float32),
                activation=tf.sigmoid,
                name="affine_2"
            )

            self._v = h2
            # watch the individual value predictions over time
            tf.scalar_summary('v_next', tf.reduce_sum(self._v_next))
            tf.scalar_summary('v', tf.reduce_sum(self._v))

            delta = tf.reduce_sum(self._v_next - self._v, name="delta")

            with tf.variable_scope("game"):
                game_step = tf.Variable(tf.constant(0.0), name="game_step", trainable=False)
                game_step_op = game_step.assign_add(1.0)

                delta_sum = tf.Variable(tf.constant(0.0), name="delta_sum", trainable=False)
                delta_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
                delta_sum_op = delta_sum.assign_add(delta)
                delta_avg_op = delta_sum / tf.maximum(game_step, 1.0)
                delta_avg_ema_op = delta_avg_ema.apply([delta_avg_op])

                tf.scalar_summary("game/delta_avg", delta_avg_op)
                tf.scalar_summary("game/delta_avg_ema", delta_avg_ema.average(delta_avg_op))

                # reset per-game monitoring variables
                self._reset_op = game_step.assign(0.0)

            # increment global step: we keep this as a variable so it's saved with checkpoints
            global_step_op = self._global_step.assign_add(1)

            # get gradients of output V wrt trainable variables (weights and biases)
            tvars = tf.trainable_variables()
            grads = tf.gradients(self._v, tvars)

            # watch the weight and gradient distributions
            for grad, var in zip(grads, tvars):
                tf.histogram_summary(var.name, var)
                tf.histogram_summary(var.name + "/gradients/grad", grad)

            # for each variable, define operations to update the var with delta,
            # taking into account the gradient as part of the eligibility trace
            apply_gradients = []
            with tf.variable_scope("apply_gradients"):
                for grad, var in zip(grads, tvars):
                    with tf.variable_scope("trace"):
                        trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name="trace")
                        trace_op = trace.assign((lamda * trace) + grad)
                        tf.histogram_summary(var.name + "/traces", trace)

                    # grad with trace = alpha * delta * e
                    grad_trace = alpha * delta * trace_op
                    tf.histogram_summary(var.name + "/gradients/trace", grad_trace)

                    grad_apply = var.assign_add(grad_trace)
                    apply_gradients.append(grad_apply)

            with tf.control_dependencies([
                global_step_op,
                game_step_op,
                delta_sum_op,
                delta_avg_ema_op
            ]):
                self._train_op = tf.group(*apply_gradients, name="train")

        self._summaries_op = tf.merge_all_summaries()
        self._saver = tf.train.Saver(max_to_keep=1)
        self._sess.run(tf.global_variables_initializer())

        if restore:
            self._restore()

    def _restore(self):
        pass

    def _get_output(self, board_t):
        return self._sess.run(self._v, feed_dict={ self._board_t: board_t })

    def play(self, humanAsWhite=True):
        if humanAsWhite:
            play(HumanAgent(), MTDAgent(self, self._depth))
        else:
            play(MTDAgent(self, self._depth), HumanAgent())

    def test(self, episodes=100):
        td_agent = MTDAgent(self, self._depth)
        random_agent = RandomAgent()

        wins, losses, draws = 0, 0, 0
        for episode in range(episodes):
            td_agent.begin_game()
            random_agent.begin_game()
            if episode % 2 == 0:
                result = play(td_agent, random_agent)
                if result == "1-0":
                    wins += 1
                elif result == "0-1":
                    losses += 1
                else:
                    draws += 1
            else:
                result = play(random_agent, td_agent)
                if result == "1-0":
                    losses += 1
                elif result == "0-1":
                    wins += 1
                else:
                    draws += 1

            print("[Episode %s] Wins: %s, losses: %s, draws: %s." % (episode, wins, losses, draws))

    def _get_pgn_files(self):
        for file in os.listdir("data/"):
            if ".pgn" in file:
                yield os.path.join("data", file)

    def _get_boards(self, pgn_file):
        with open(pgn_file, "r") as f:
            offsets = []
            for offset, _ in chess.pgn.scan_headers(f):
                offsets.append(offset)

            for offset in offsets:
                f.seek(offset)
                game_node = chess.pgn.read_game(f)
                game_result = game_node.headers["Result"]
                boards = []
                while len(game_node.variations) > 0:
                    boards.append(game_node.board())
                    game_node = game_node.variations[0]
                boards.append(game_node.board())

                yield game_result, boards

    def pretrain(self):
        tf.train.write_graph(self._sess.graph_def, self._model_path, "td_chess.pb", as_text=False)
        summary_writer = tf.train.SummaryWriter("{0}{1}".format(self._summary_path, int(time.time()), self._sess.graph_def))

        validation_interval = 1000
        episodes = 5000

        pgn_files = self._get_pgn_files()

        episode = 0
        for pgn_file in pgn_files:
            for game_result, boards in self._get_boards(pgn_file):
                episode += 1
                board = boards[0]
                board_t = self._extract_features(board)
                game_step = 0

                for i in range(1, len(boards)):
                    board = boards[i]
                    board_next_t = self._extract_features(board)
                    v_next_t = self._get_output(board_next_t)
                    self._sess.run(self._train_op, feed_dict={ self._board_t: board_t, self._v_next: v_next_t})
                    board_t = board_next_t
                    game_step += 1

                if game_result == "1-0":
                    result = 1
                    winner_str = "Winner: White"
                elif game_result == "0-1":
                    result = 0
                    winner_str = "Winner: Black"
                else:
                    result = 0.5
                    winner_str = "Draw"
                _, global_step, summaries, _ = self._sess.run([
                    self._train_op,
                    self._global_step,
                    self._summaries_op,
                    self._reset_op
                ], feed_dict={ self._board_t: board_t, self._v_next: np.array([[result]], dtype="float32") })
                summary_writer.add_summary(summaries, global_step=global_step)

                print("Game %d/%d (%s) in %d turns" % (episode, episodes, winner_str, game_step))
                self._saver.save(self._sess, self._checkpoint_path + "checkpoint", global_step=global_step)

        summary_writer.close()
        self.test(episodes=1000)

    def train(self):
        tf.train.write_graph(self._sess.graph_def, self._model_path, "td_chess.pb", as_text=False)
        summary_writer = tf.train.SummaryWriter("{0}{1}".format(self._summary_path, int(time.time()), self._sess.graph_def))

        validation_interval = 1000
        episodes = 5000

        agent = MTDAgent(self, self._depth)
        for episode in range(episodes):
            agent.begin_game()
            if episode != 0 and episode % validation_interval == 0:
                self.test(episodes=100)

            board = chess.Board()
            board_t = self._extract_features(board)

            game_step = 0
            while not board.is_game_over():
                move = agent.get_move(board)
                board.push(move)

                board_next_t = self._extract_features(board)
                v_next = self._get_output(board_next_t)
                self._sess.run(self._train_op, feed_dict={ self._board_t: board_t, self._v_next: v_next})

                board_t = board_next_t
                game_step += 1

            if board.result() == "1-0":
                result = 1
                winner_str = "Winner: White"
            elif board.result() == "0-1":
                result = 0
                winner_str = "Winner: Black"
            else:
                result = 0.5
                winner_str = "Draw"
            _, global_step, summaries, _ = self._sess.run([
                self._train_op,
                self._global_step,
                self._summaries_op,
                self._reset_op
            ], feed_dict={ self._board_t: board_t, self._v_next: np.array([[result]], dtype="float32") })
            summary_writer.add_summary(summaries, global_step=global_step)

            print("Game %d/%d (%s) in %d turns" % (episode, episodes, winner_str, game_step))
            self._saver.save(self._sess, self._checkpoint_path + "checkpoint", global_step=global_step)

        summary_writer.close()
        self.test(episodes=1000)

    def _extract_features(self, board):
        board_t = np.zeros((2, 8, 8, 6), dtype="float32")
        for rank in range(8):
            for file in range(8):
                piece = board.piece_at(8 * rank + file)
                if piece is not None:
                    piece_color, piece_type = piece.color, piece.piece_type
                    piece_color_key = 1 if piece_color is chess.WHITE else 0
                    board_t[piece_color_key][rank][file][piece_type-1] = 1

        props_t = np.zeros((2, 3), dtype="float32")
        for color in [chess.WHITE, chess.BLACK]:
            color_key = 1 if piece_color is chess.WHITE else 0
            if board.turn == color:
                props_t[color_key][0] = 1
            if board.has_kingside_castling_rights(color):
                props_t[color_key][1] = 1
            if board.has_queenside_castling_rights(color):
                props_t[color_key][2] = 1

        return np.expand_dims(np.concatenate((board_t.reshape((2, -1)), props_t), axis=1).flatten(), axis=0)

    def evaluate(self, board):
        return self._get_output(self._extract_features(board))
