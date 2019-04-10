#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Network:
    DATA = 144
    TEST = 40
    TRAIN = DATA - TEST

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.sequence = tf.placeholder(tf.float32, [self.TRAIN], name="sequence")
            # self.label = tf.placeholder(tf.float32, [self.TRAIN], name="label")
            # TODO: Create RNN cell according to args.rnn_cell (RNN, LSTM and GRU should be supported,
            # using BasicRNNCell, BasicLSTMCell and GRUCell from tf.nn.rnn_cell module),
            # with dimensionality of args.rnn_cell_dim. Store the cell in `rnn_cell`.
            num_units = args.rnn_cell_dim
            if args.rnn_cell == 'RNN':
                rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units)
            elif args.rnn_cell == 'LSTM':
                rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
            else:
                rnn_cell = tf.nn.rnn_cell.GRUCell(num_units)
                

            #initial_state = state = tf.zeros([1, rnn_cell.state_size])

            state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32) # TODO: Create zero state using rnn_cell.zero_state call. Use batch size 1.
            predictions, loss = [], 0
            dense = tf.layers.Dense(1)  # TODO: Create a dense layer object using tf.layers.Dense, with 1 output unit.
            #print("initial loss ", loss)
            #inputs = self.sequence
            #input = tf.placeholder(tf.float32, [1,1]
            
            #bos = tf.constant(0.0, tf.float32, [1])        # prepend beginning of seq
            #inputs = tf.concat([bos, self.sequence], axis=0)
            # print(inputs.get_shape().as_list())
            
            # TRAIN
            x = tf.constant(0.0, tf.float32)
            for i in range(0, self.TRAIN):
                _x = tf.reshape(x, [1,1]) # (batch, x), one batch
                #y = tf.reshape(inputs[i], [1,1])
                #print(i, "x", x)
                #print("y", y)
                # TODO: Call rnn_cell (the input should be 0.0 on first step and self.sequence[i - 1] otherwise).
                # Note that rnn_cell assumes the input is a batch of vectors, so you need to produce the
                # input with [1, 1] shape.
                #output, state = tf.nn.dynamic_rnn(rnn_cell, x, initial_state=state)
                output, state = rnn_cell(_x, state)
                # Then compute current prediction, by using `dense` layer, and append the scalar prediction
                # (i.e., with shape []) to `predictions`.
                _y = dense(output)
                _y = tf.reshape(_y, ())
                predictions.append(_y)
                # Also add mean square error of the prediction and self.sequence[i] to the loss.
                # loss += tf.squared_difference(y, _y)
                y = tf.reshape(self.sequence[i], [1,1])                
                loss += (y -_y) * (y -_y)
                #print(loss)
                #print("y", y)
                x = y
            # predictions.append(tf.constant(0.0, tf.float32))
            loss = loss / float(self.TRAIN)
            #print('total loss:', loss)
            #print('predictions', predictions)
            
            # TEST, get predictions of test set
            for i in range(self.TEST):
                x = predictions[-1]
                _x = tf.reshape(x, [1,1]) # last prediction
                
                # TODO: Call rnn_cell, the input should be the latest prediction. Generate a new
                # prediction using the `dense` layer and append it to `predictions`.
                output, state = rnn_cell(_x, state)
                _y = dense(output)
                _y = tf.reshape(_y, ())                
                #print('_y predicted', _y)
                predictions.append(_y)
            
            # TODO: Generate `self.predictions` tensor (instead of Python list), use `tf.stack`.
            self.predictions = tf.stack(predictions)
            #print(self.predictions)
            
            # Training 
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = tf.contrib.summary.scalar("train/loss", loss)
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.prediction_image = tf.placeholder(tf.uint8, [self.DATA, self.DATA, 3])
                self.prediction_gold = tf.placeholder(tf.float32, [self.TEST], name="prediction_gold")
                self.prediction_loss = tf.losses.mean_squared_error(self.prediction_gold, self.predictions[self.TRAIN:])
                self.summaries["prediction"] = [tf.contrib.summary.scalar("prediction/loss", self.prediction_loss),
                                                tf.contrib.summary.image("prediction", tf.expand_dims(self.prediction_image, 0))]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, sequence):
        self.session.run([self.training, self.summaries["train"]], {self.sequence: sequence})

    def predict(self, sequence):
        return self.session.run(self.predictions, {self.sequence: sequence})

    def prediction_summary(self, gold, predictions):
        min_value = min(np.min(gold), np.min(predictions))
        max_value = max(np.max(gold), np.max(predictions))
        def y(x):
            return int(self.DATA - 1 - (self.DATA - 1) * (x - min_value) / (max_value - min_value))

        prediction_image = np.full([self.DATA, self.DATA, 3], 255, dtype=np.uint8)
        for i in range(self.DATA):
            prediction_image[y(gold[i]), i] = [0, 0, 255] if i < self.TRAIN else [0, 255, 0]
            prediction_image[y(predictions[i]), i] = [255, 0, 0]

        loss, _ = self.session.run([self.prediction_loss, self.summaries["prediction"]],
                         {self.predictions: predictions, self.prediction_gold: gold[self.TRAIN:], self.prediction_image: prediction_image})
        return loss

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=10, type=int, help="RNN cell dimension.")
    parser.add_argument("--steps_per_epoch", default=100, type=int, help="Number of training steps per epoch.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
                  for key, value in sorted(vars(args).items()))).replace("/", "-")
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    with open("C:\\Users\\PC\\Downloads\\test_Conv\\Tests\\Airlanes\\international-airline-passengers (1).tsv", "r") as data_file:
        data = [float(line.split("\t")[1]) for line in data_file.readlines()[1:]]
        assert(len(data) == Network.DATA)
        data = np.array(data, dtype=np.float32)
        data -= np.min(data)
        data /= np.max(data)
        print('train data/n', data[:Network.TRAIN])
        print('test data/n', data[:Network.TRAIN])
		
    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train and predict
    for epoch in range(args.epochs):
        for step in range(args.steps_per_epoch):
            network.train(data[:Network.TRAIN])
            
        predictions = network.predict(data[:Network.TRAIN])
        loss = network.prediction_summary(data, predictions)
        print("{:.2g}".format(loss))
        # TODO: Print network.prediction_loss for each epoch, using "{:.2g}" format.
        