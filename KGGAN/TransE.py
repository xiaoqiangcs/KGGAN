# -*- coding:utf-8 -*-
"""
@author:Wenqiang Liu
@file:TransE.py
@time:2017/11/720:39
"""
from ReadData import DataSet
import tensorflow as tf
import random
from tensorflow.contrib.layers.python.layers import batch_norm
import numpy as np
class TransE(object):
	def __init__(self, learning_rate, batch_size, num_epoch, margin, embedding_dimension, dissimilarity, evaluate_size,num_negative,noise_dim,num_type,entityid_to_typeid):
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.num_epoch = num_epoch
		self.margin = margin
		self.embedding_dimension = embedding_dimension
		self.dissimilarity = dissimilarity
		self.evaluate_size = evaluate_size
		self.num_negative = num_negative
		self.noise_dim = noise_dim
		self.num_type = num_type
		self.entityid_to_typeid = entityid_to_typeid
	# 定义一个可以生成m*n阶随机矩阵的函数，该矩阵的元素服从均匀分布，随机生成的z就为生成器的输入
	def sample_Z(self, m, n):
		return np.random.uniform(-1., 1., size=[m, n])

	def variable_init(self,size):
		in_dim = size[0]

		# 计算随机生成变量所服从的正态分布标准差
		w_stddev = 1. / tf.sqrt(in_dim / 2.)
		return tf.random_normal(shape=size, stddev=w_stddev)

	def batch_normal(self,input):
		return batch_norm(input, epsilon=1e-5, decay=0.9, scale=True,updates_collections=None)
	# 定义生成器
	def generator(self,noise_Z, type_y_dim):
		# 根据噪声Z进行多层感知机操作，生成num_negative个负样例
		noise_Z = tf.concat([noise_Z,type_y_dim],1)
		G_W1 = tf.Variable(self.variable_init([(self.noise_dim+self.num_type), 64]),
						   name="gen_w1", dtype=tf.float32)
		G_b1 = tf.Variable(tf.zeros(shape=[64]), name="gen_b1", dtype=tf.float32)
		G_W2 = tf.Variable(self.variable_init([64, 128]),
						   name="gen_w2", dtype=tf.float32)
		G_b2 = tf.Variable(tf.zeros(shape=[128]), name="gen_b2", dtype=tf.float32)

		G_W3 = tf.Variable(self.variable_init([128, self.embedding_dimension]),name="gen_w3",dtype=tf.float32)
		G_b3 = tf.Variable(tf.zeros(shape=[self.embedding_dimension]), name="gen_b3",dtype=tf.float32)
		theta_G = [G_W1, G_W2, G_b1, G_b2]  # 梯度下降的参数范围
		# 第一层先计算 y=z*G_W1+G-b1,然后投入激活函数计算G_h1=ReLU（y）,G_h1 为第二次层神经网络的输出激活值
		G_h1 = tf.nn.relu(self.batch_normal(tf.matmul(noise_Z, G_W1) + G_b1))
		# 以下两个语句计算第二层传播到第三层的激活结果，第三层的激活结果是含有784个元素的向量，该向量转化28×28就可以表示图像
		G_h2 = tf.nn.relu(self.batch_normal(tf.matmul(G_h1, G_W2) + G_b2))
		G_log_prob = tf.matmul(G_h2, G_W3) + G_b3
		G_prob = tf.nn.softmax(G_log_prob)
		G_prob = tf.reshape(G_prob,[self.batch_size,-1])
		return G_prob, theta_G
	def get_negative_embedding(self, G_prob):
		# 根据生成的实体，从图中获取离该节点最近的点的距离，并把该样例作为负样例get embedding from the graph
		generator_samples = tf.reshape(G_prob,shape=[self.batch_size*self.num_negative,self.embedding_dimension])
		with tf.variable_scope('embedding', reuse=True):
			embedding_entity = tf.get_variable(name='dis_entity')
		embedding_entity_extend =tf.transpose(embedding_entity, [1, 0])
		dissimilarity = tf.matmul(generator_samples,embedding_entity_extend)
		batch_negative = tf.reshape(tf.argmin(dissimilarity, axis=1),[self.batch_size,self.num_negative])
		loss_sum = tf.reduce_sum(tf.abs(tf.reduce_min(dissimilarity,axis=1)))
		return batch_negative,loss_sum

	def get_negative_sampling(self, batch_negative, id_triplets_positive,random_dim):
		# 根据生成的候选负样实体h'或者t',构建候选负样例(h',r,t)或者(h',r,t')
		batch_negative_extend = tf.reshape(batch_negative, [-1, 1])
		triplets_positive = tf.reshape(tf.tile(id_triplets_positive, [1, self.num_negative]), [-1, 3])
		get_head = tf.multiply((1-tf.gather(random_dim, [0], axis=1)),tf.gather(triplets_positive, [0], axis=1))+tf.multiply((tf.gather(random_dim, [0], axis=1)),batch_negative_extend)
		get_relation = tf.gather(triplets_positive, [1], axis=1)
		get_tail= tf.multiply((1-tf.gather(random_dim, [1], axis=1)),tf.gather(triplets_positive, [2], axis=1))+tf.multiply((tf.gather(random_dim, [1], axis=1)),batch_negative_extend)
		batch_negative_sample = tf.concat([get_head,get_relation,get_tail],axis=1)
		g_loss_fake = self.discriminator(batch_negative_sample, postive=False)
		batch_negative_sample = tf.reshape(batch_negative_sample, [self.batch_size, -1])
		return batch_negative_sample, g_loss_fake
	# 定义判别器
	def discriminator(self, id_triplets, postive, eps=1e-20):
		# 计算E=(h+r-t)
		dissimilarity = self.get_dissimilarity(id_triplets)
		if postive ==True: #负样例
			D_prob = tf.nn.sigmoid(7-dissimilarity)
		else: #负样例
			D_prob =tf.nn.sigmoid(dissimilarity-7)
		D_log_prob = tf.log(D_prob+eps)
		D_loss = tf.reduce_sum(D_log_prob)
		# 输出G_log_prob是构建损失函数
		return D_loss
	def get_discriminator_smooth(self,id_triplets,batch_negative_sample):
		# 计算E=(h-h')
		id_triplets_head = tf.reshape(tf.tile(tf.gather(id_triplets,[0],axis=1),[1,self.num_negative]),[-1,1])
		id_triplets_tail = tf.reshape(tf.tile(tf.gather(id_triplets, [2], axis=1), [1, self.num_negative]), [-1, 1])
		negative_triplets_head = tf.gather(tf.reshape(batch_negative_sample,[-1,3]),[0],axis=1)
		negative_triplets_tail = tf.gather(tf.reshape(batch_negative_sample, [-1, 3]), [2], axis=1)
		entity_type = tf.Variable(np.array(sorted(self.entityid_to_typeid.items())))
		type_positve_head = tf.gather(tf.reshape(tf.nn.embedding_lookup(entity_type, id_triplets_head),[-1,2]),[1],axis=1)
		type_positve_tail = tf.gather(tf.reshape(tf.nn.embedding_lookup(entity_type, id_triplets_tail),[-1,2]),[1],axis=1)
		type_negative_head = tf.gather(tf.reshape(tf.nn.embedding_lookup(entity_type, negative_triplets_head),[-1,2]),[1],axis=1)
		type_negative_tail = tf.gather(tf.reshape(tf.nn.embedding_lookup(entity_type, negative_triplets_tail),[-1,2]),[1],axis=1)
		headequal_positive_negative = tf.cast(tf.equal(type_positve_head,type_negative_head),dtype=tf.float32)
		tailequal_positive_negative = tf.cast(tf.equal(type_positve_tail, type_negative_tail), dtype=tf.float32)
		dissimilarity_positve = tf.multiply(headequal_positive_negative,self.get_dissimilarity_head(id_triplets_head,negative_triplets_head))
		dissimilarity_negative = tf.multiply(tailequal_positive_negative ,self.get_dissimilarity_head(id_triplets_tail, negative_triplets_tail))
		d_loss = tf.reduce_sum(dissimilarity_positve) + tf.reduce_sum(dissimilarity_negative)
		return d_loss
	def discriminator_train(self, id_triplets,batch_negative_sample, eps=1e-20):
		# 计算E=(h+r-t)
		d_loss_positive = self.discriminator(id_triplets,postive=True)
		d_loss_negative = self.discriminator(id_triplets=tf.reshape(batch_negative_sample,[-1,3]),postive=False)
		d_loss_distance = -(d_loss_positive + d_loss_negative)
		d_loss_smooth = self.get_discriminator_smooth(id_triplets,batch_negative_sample)
		d_loss = d_loss_distance+d_loss_smooth
		return d_loss
	def get_dissimilarity_head(self, id_positve_triplets,id_negatvie_triplets):
		# get embedding from the graph
		with tf.variable_scope('embedding', reuse=True):
			embedding_entity = tf.get_variable(name='dis_entity')

		# normalize the entity embeddings
		embedding_positive_head = tf.nn.embedding_lookup(embedding_entity, id_positve_triplets)
		embedding_negative_head = tf.nn.embedding_lookup(embedding_entity, id_negatvie_triplets)

		if self.dissimilarity == 'L2':
			dissimilarity = tf.sqrt(tf.reduce_sum(tf.square(embedding_positive_head - embedding_negative_head), axis=1))
		else:  # default: L1
			dissimilarity = tf.reduce_sum(tf.abs(embedding_positive_head - embedding_negative_head), axis=1)
		return dissimilarity
	def get_dissimilarity(self, id_triplets):
		# get embedding from the graph
		with tf.variable_scope('embedding', reuse=True):
			embedding_entity = tf.get_variable(name='dis_entity')
			embedding_relation = tf.get_variable(name='dis_relation')

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