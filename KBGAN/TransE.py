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
	def __init__(self, learning_rate, batch_size, num_epoch, margin, embedding_dimension, dissimilarity, evaluate_size,num_negative):
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.num_epoch = num_epoch
		self.margin = margin
		self.embedding_dimension = embedding_dimension
		self.dissimilarity = dissimilarity
		self.evaluate_size = evaluate_size
		self.num_negative = num_negative
	def inference(self, id_triplets_positive, id_triplets_negative):
		d_positive = self.get_dissimilarity(id_triplets_positive)
		d_negative = self.get_dissimilarity(id_triplets_negative)

		return d_positive, d_negative

	# 定义生成器生成负采样
	def generater_negative_sampling(self, batch_negative):
		id_triplets_negative = tf.Variable(tf.zeros(1,3))
		negative_prob = tf.Variable([0.12])
		for index in range(self.batch_size):
			negative_set = batch_negative[index,:]
			d_negative = self.get_dissimilarity(negative_set)
			negative_log_prob = tf.nn.softmax(d_negative)
			id_negative = np.arange(self.num_negative)
			negative_sampling = np.random.choice(id_negative)
			print(negative_sampling)
			print(negative_log_prob)
			id_triplets_negative = tf.concat([id_triplets_negative, negative_set[negative_sampling]], 0)
			# id_triplets_negative = np.concatenate((id_triplets_negative, negative_set[negative_sampling]), axis=0)
			# id_triplets_negative.append(negative_set[negative_sampling])
			negative_prob=tf.concat([negative_prob, negative_log_prob[negative_sampling]], axis=0)
		id_triplets_negative = id_triplets_negative.reshape(self.batch_size,3)
		negative_prob = negative_prob.reshape(self.batch_size, 1)
		return id_triplets_negative, negative_prob

	# 定义生成器
	def generator(self, negative_prob, id_triplets_negative):
		d_negative = self.get_dissimilarity(id_triplets_negative)
		reward_basline = tf.constant(tf.reduce_sum(d_negative),tf.float32,shape=[self.batch_size],name="reward_basline")
		reward = tf.add(d_negative,reward_basline)

		g_loss = tf.reduce_mean(tf.log(negative_prob)*reward, name='generator_loss')
		return g_loss
	# 定义判别器
	def discriminator(self, d_positive, d_negative):
		margin = tf.constant(self.margin, tf.float32, shape=[self.batch_size], name="margin")
		d_loss = tf.reduce_sum(tf.nn.relu(margin+d_positive-d_negative), name='discriminator_loss')
		return d_loss


	def discriminator_train (self, loss):
		tf.summary.scalar(loss.op.name, loss)
		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		discriminator_train_op = optimizer.minimize(loss)
		return discriminator_train_op
	def generator_train (self, loss):
		tf.summary.scalar(loss.op.name, loss)
		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		generator_train_op = optimizer.minimize(loss)
		return generator_train_op
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
	def evaluation(self, id_triplets_predict_head, id_triplets_predict_tail):
		# get one single validate triplet and do evaluation
		prediction_head = self.get_dissimilarity(id_triplets_predict_head)
		prediction_tail = self.get_dissimilarity(id_triplets_predict_tail)

		return prediction_head, prediction_tail