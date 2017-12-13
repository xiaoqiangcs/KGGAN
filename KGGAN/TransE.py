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
	def generator(self,noise_Z):
		# 根据噪声Z进行多层感知机操作，生成num_negative个负样例
		G_W1 = tf.Variable(self.variable_init([self.num_negative * self.noise_dim, self.num_negative * 100]),dtype=tf.float32)
		G_b1 = tf.Variable(tf.zeros(shape=[self.num_negative * 100]), dtype=tf.float32)
		G_W2 = tf.Variable(self.variable_init([self.num_negative * 100, self.num_negative * self.embedding_dimension]),dtype=tf.float32)
		G_b2 = tf.Variable(tf.zeros(shape=[self.num_negative * self.embedding_dimension]), dtype=tf.float32)
		theta_G = [G_W1, G_W2, G_b1, G_b2]  # 梯度下降的参数范围
		# 第一层先计算 y=z*G_W1+G-b1,然后投入激活函数计算G_h1=ReLU（y）,G_h1 为第二次层神经网络的输出激活值
		G_h1 = tf.nn.relu(tf.matmul(noise_Z, G_W1) + G_b1)
		# 以下两个语句计算第二层传播到第三层的激活结果，第三层的激活结果是含有784个元素的向量，该向量转化28×28就可以表示图像
		G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
		G_prob = tf.nn.softmax(G_log_prob)
		return G_prob, theta_G

	# 定义判别器
	def discriminator(self, id_triplets, postive, eps=1e-20):
		# 计算E=(h+r-t)
		dissimilarity = self.get_dissimilarity(id_triplets)
		if postive ==True: #负样例
			D_prob = tf.nn.sigmoid(dissimilarity)
		else: #负样例
			D_prob =tf.nn.sigmoid(-dissimilarity)
		D_log_prob = tf.log(D_prob+eps)
		# 输出G_log_prob是构建损失函数
		return D_log_prob

	def get_negative_embedding(self, G_prob, num_entity):
		# 根据生成的实体，从图中获取离该节点最近的点的距离，并把该样例作为负样例get embedding from the graph
		batch_negative = []  # 定义一个存储负样例的矩阵，矩阵大小为(batch_size *(num_negative*3))
		generator_samples= tf.reshape(G_prob,shape=[self.batch_size,self.num_negative,self.embedding_dimension])
		loss_sum = 0.0
		with tf.variable_scope('embedding', reuse=True):
			embedding_entity = tf.get_variable(name='entity')
		embedding_entity_extend = tf.tile(embedding_entity,[self.num_negative,1])
		for batch_index in range(self.batch_size):
			generator_index_samples=generator_samples[batch_index,:]
			generator_index_samples_extend = tf.reshape(tf.tile(generator_index_samples,[1,num_entity]),shape=[self.batch_size*num_entity,self.embedding_dimension])
			if self.dissimilarity == 'L2':
				dissimilarity = tf.sqrt(tf.reduce_sum(tf.square(generator_index_samples_extend - embedding_entity_extend), axis=1))
			else:  # default: L1
				dissimilarity = tf.reduce_sum(tf.abs(generator_index_samples_extend - embedding_entity_extend), axis=1)
			dissimilarity_extend = tf.reshape(dissimilarity,shape=[self.batch_size,num_entity])
			batch_negative.append(tf.argmin(dissimilarity_extend,axis=1))
			loss_sum += tf.reduce_sum(tf.reduce_max(dissimilarity_extend,axis=1))
		return batch_negative,loss_sum
	def get_negative_sampling(self, G_prob, num_entity):
		# 根据生成的实体，从图中获取离该节点最近的点的距离，并把该样例作为负样例get embedding from the graph
		batch_negative = []  # 定义一个存储负样例的矩阵，矩阵大小为(batch_size *(num_negative*3))
		generator_samples= tf.reshape(G_prob,shape=[self.batch_size,self.num_negative,self.embedding_dimension])
		loss_sum = 0.0
		with tf.variable_scope('embedding', reuse=True):
			embedding_entity = tf.get_variable(name='entity')
		embedding_entity_extend = tf.tile(embedding_entity,[self.num_negative,1])
		for batch_index in range(self.batch_size):
			generator_index_samples=generator_samples[batch_index,:]
			generator_index_samples_extend = tf.reshape(tf.tile(generator_index_samples,[1,num_entity]),shape=[self.batch_size*num_entity,self.embedding_dimension])
			if self.dissimilarity == 'L2':
				dissimilarity = tf.sqrt(tf.reduce_sum(tf.square(generator_index_samples_extend - embedding_entity_extend), axis=1))
			else:  # default: L1
				dissimilarity = tf.reduce_sum(tf.abs(generator_index_samples_extend - embedding_entity_extend), axis=1)
			dissimilarity_extend = tf.reshape(dissimilarity,shape=[self.batch_size,num_entity])
			batch_negative.append(tf.argmin(dissimilarity_extend,axis=1))
			loss_sum += tf.reduce_sum(tf.reduce_max(dissimilarity_extend,axis=1))
		return batch_negative,loss_sum
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