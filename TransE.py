# -*- coding:utf-8 -*-
"""
@author:Wenqiang Liu
@file:TransE.py
@time:2017/11/720:39
"""
from ReadData import DataSet
import tensorflow as tf
import random
import numpy as np
class TransE(object):
    def __init__(self, learning_rate, batch_size, num_epoch, margin, embedding_dimension, dissimilarity, evaluate_size):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.margin = margin
        self.embedding_dimension = embedding_dimension
        self.dissimilarity = dissimilarity
        self.evaluate_size = evaluate_size
    def inference(self, id_triplets_positive):
        d_positive = self.get_dissimilarity(id_triplets_positive)

        return d_positive
    def loss(self, d_positive):
        margin = tf.constant(self.margin, tf.float32, shape=[self.batch_size], name="margin")
        loss = tf.reduce_sum(tf.nn.relu(d_positive), name='max_margin_loss')
        return loss
    def train (self, loss):
        tf.summary.scalar(loss.op.name, loss)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss)
        return train_op
    def get_dissimilarity(self, id_triplets):
        # get embedding from the graph
        with tf.variable_scope('embedding', reuse=True):
            embedding_entity = tf.get_variable(name='entity')
            embedding_relation = tf.get_variable(name='relation')

        # normalize the entity embeddings
        embedding_head = tf.nn.embedding_lookup(embedding_entity, id_triplets[:, 0])
        embedding_relation = tf.nn.embedding_lookup(embedding_relation, id_triplets[:, 1])
        embedding_tail = tf.nn.embedding_lookup(embedding_entity, id_triplets[:, 2])

        if self.dissimilarity == 'L2':
            dissimilarity = tf.sqrt(tf.reduce_sum(tf.square(embedding_head + embedding_relation- embedding_tail), axis=1))
        else:  # default: L1
            dissimilarity = tf.reduce_sum(tf.abs(embedding_head + embedding_relation - embedding_tail), axis=1)
        return dissimilarity
    def evaluation(self, id_triplets_predict_head):
        # get one single validate triplet and do evaluation
        prediction_head = self.get_dissimilarity(id_triplets_predict_head)

        return prediction_head