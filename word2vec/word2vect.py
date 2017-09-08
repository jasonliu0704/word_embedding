__author__ = "Jiacheng Liu"

# define all the hyper parameter
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

from process_data import process_data
import utils

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128 # dimension of the word embedding vectors
SKIP_WINDOW = 1 # the context window
NUM_SAMPLED = 64    # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100000
WEIGHTS_FLD = 'processed/'
SKIP_STEP = 2000


class SkipGramWord2Vec:
    def __init__(self):

    def construct_graph(self):

        # input layer, conter words
        with tf.name_scope("input_layer"):
            # input words are scalar value which are indices of each word in dictionary
            # we didn't use one hot encoding here for space and computation saving
            center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name="center_words")

        # hidden layer
        # save this layer's weight for word vector embedding look up
        with tf.name_scope("hidden_layer"):
            # weights are normalized
            word_embedding_weights = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0), name="word_embedding_weights")
            # no activation and biase used here
            # create word embedding look up using word_embedding_weights
            self.word_embedding = tf.nn.embedding_lookup(word_embedding_weights, center_words, name="word_embedding")

        # output-NCE(noise contrastive estimation) layer
        # we don't need to output inference since we only use word embedding to look up word
        # but we still need to do inference for loss funciton
        with tf.name_scope("output_layer"):
            # TODO need to investigate the shape and std for the uniform distribution
            nce_weight = tf.Varaible(tf.trancated_normal([VOCAB_SIZE, EMBED_SIZE], stddev=1.0/(EMBED_SIZE ** 0.5)), name="nce_weight")
            nce_bias = td.Variable(tf.zeros([VOCAB_SIZE]), name="nce_biase")

            # prepare context word(gorund truth) for object function
            context_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE,1], nane="context_words")

            # define object funciton, here we use noise contrastive estimation
            nce_loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=bce_bias, labels=context_words,
            input=self.word_embedding, num_samped=NUM_SAMPLED, num_classes=VOCAB_SIZE), name="nce_loss")

        # define optimizer
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(nce_loss)

        # training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            total_loss = 0.0
            write = tf.summary.FileWrite("./graphs", sess.graph)
            for index in range(NUM_TRAIN_STEPS):
                center_w, context_w = next(batch_gen)
                total_loss += sess.run([nce_loss, optimizer], feed_dict={center_words: center_w, context_words: context_w})
                if(index+1) % SKIP_STEP == 0:
                    print("Average loss at step {}: {:5.1f}".format(index, total_loss / SKIP_STEP))
                    total_loss = 0.0
            writer.close()
def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)

if __name__ == '__main__':
    main()
