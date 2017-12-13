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
	def __init__(self, learning_rate, batch_size, num_epoch, margin, embedding_dimension, dissimilarity, evaluate_size,num_negative,noise_dim):
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.num_epoch = num_epoch
		self.margin = margin
		self.embedding_dimension = embedding_dimension
		self.dissimilarity = dissimilarity
		self.evaluate_size = evaluate_size
		self.num_negative = num_negative
		self.noise_dim = noise_dim

	# 定义一个可以生成m*n阶随机矩阵的函数，该矩阵的元素服从均匀分布，随机生成的z就为生成器的输入
	def sample_Z(self, m, n):
		return np.random.uniform(-1., 1., size=[m, n])

	def variable_init(self,size):
		in_dim = size[0]

		# 计算随机生成变量所服从的正态分布标准差
		w_stddev = 1. / tf.sqrt(in_dim / 2.)
		return tf.random_normal(shape=size, stddev=w_stddev)
	# 定义生成器
	def generator(self,z,id_triplets_positive):
		G_W1 = tf.Variable(self.variable_init([self.num_negative * self.noise_dim, self.num_negative * 100]),dtype=tf.float32)
		G_b1 = tf.Variable(tf.zeros(shape=[self.num_negative * 100]), dtype=tf.float32)
		G_W2 = tf.Variable(self.variable_init([self.num_negative * 100, self.num_negative * self.embedding_dimension]),dtype=tf.float32)
		G_b2 = tf.Variable(tf.zeros(shape=[self.num_negative * self.embedding_dimension]), dtype=tf.float32)
		theta_G = [G_W1, G_W2, G_b1, G_b2]  # 梯度下降的参数范围
		# 第一层先计算 y=z*G_W1+G-b1,然后投入激活函数计算G_h1=ReLU（y）,G_h1 为第二次层神经网络的输出激活值
		G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
		# 以下两个语句计算第二层传播到第三层的激活结果，第三层的激活结果是含有784个元素的向量，该向量转化28×28就可以表示图像
		G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
		G_prob = tf.nn.softmax(G_log_prob)
		embedding_head, embedding_relation, embedding_tail = self.get_embedding(id_triplets_positive)
		embedding_head = tf.tile(embedding_head, [1, self.num_negative])
		if self.dissimilarity == 'L2':
			dissimilarity = tf.sqrt(tf.reduce_sum(tf.square(embedding_head - G_prob), axis=1))
		else:  # default: L1
			dissimilarity = tf.reduce_sum(tf.abs(embedding_head - G_prob), axis=1)
		g_loss = tf.reduce_sum(tf.abs(dissimilarity), axis=0)
		generator_train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(g_loss)
		return generator_train_op, g_loss

	# 定义判别器
	def discriminator(self, id_triplets, postive):
		# 计算E=(h+r-t)
		dissimilarity = self.get_dissimilarity(id_triplets)
		if postive ==True: #负样例
			G_prob = tf.nn.sigmoid(dissimilarity)
		else: #负样例
			G_prob =tf.nn.sigmoid(-dissimilarity)
		G_log_prob = tf.log(G_prob)
		# 输出G_log_prob是构建损失函数
		return G_log_prob

	def get_embedding(self, id_triplets):
		# get embedding from the graph
		with tf.variable_scope('embedding', reuse=True):
			embedding_entity = tf.get_variable(name='entity')
			embedding_relation = tf.get_variable(name='relation')

		# normalize the entity embeddings
		embedding_head = tf.nn.embedding_lookup(embedding_entity, id_triplets[:, 0])
		embedding_relation = tf.nn.embedding_lookup(embedding_relation, id_triplets[:, 1])
		embedding_tail = tf.nn.embedding_lookup(embedding_entity, id_triplets[:, 2])
		return embedding_head,embedding_relation,embedding_tail
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