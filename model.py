import time

import numpy as np
import tensorflow as tf
import chess

from game import play
from ops import affine_layer

class Model(object):
    def __init__(self, sess, model_path, summary_path, checkpoint_path, restore=False, scope="model",
                 lamda=0.7, alpha=0.01, hidden_size=1024):
        self._model_path = model_path
        self._summary_path = summary_path
        self._checkpoint_path = checkpoint_path
        self._sess = sess
        self._scope = scope
        self._hidden_size = hidden_size

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

            self._board_t = tf.placeholder(tf.float32, [1, 2*8*8*6], name="board")
            self._v_next = tf.placeholder(tf.float32, [1, 1], name="n_next")

            h1 = affine_layer(
                input_t=self._board_t,
                input_dim=2*8*8*6,
                output_dim=self._hidden_size,
                w_init=0,
                b_init=0,
                activation=tf.sigmoid,
                name="affine_1"
            )
            h2 = affine_layer(
                input_t=h1,
                input_dim=self._hidden_size,
                output_dim=1,
                w_init=0,
                b_init=0,
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
        self._sess.run(tf.initialize_all_variables())

        if restore:
            self._restore()

    def _restore(self):
        pass

    def _get_output(self, board_t):
        return self.sess.run(self._v, feed_dict={ self._board_t: board_t })

    def play(self, humanAsWhite=True):
        if humanAsWhite:
            play(HumanAgent(), TDAgent(self))
        else:
            play(TDAgent(self), HumanAgent())

    def test(self, episodes=100):
        td_agent = TDAgent(self)
        random_agent = RandomAgent()

        wins, losses, draws = 0, 0, 0
        for episode in range(episodes):
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


    def train(self):
        tf.train.write_graph(self._sess.graph_def, self._model_path, "td_chess.pb", as_text=False)
        summary_writer = tf.train.SummaryWriter("{0}{1}".format(self._summary_path, int(time.time()), self._sess.graph_def))

        validation_interval = 1000
        episodes = 5000

        agent = TDAgent(self)
        for episode in range(episodes):
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
                ], feed_dict={ self._board_t: board_t, self._v_next: np.array([result], dtype="float32") })
                summary_writer.add_summary(summaries, global_step=global_step)


                winner_str = "White" if result == 1 else ()
                print("Game %d/%d (%s) in %d turns" % (episode, episodes, winner_str, game_step))
                self._saver.save(self._sess, self._checkpoint_path + "checkpoint", global_step=global_step)

        summary_writer.close()
        self.test(episodes=1000)
