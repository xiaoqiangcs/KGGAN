# -*- coding:utf-8 -*-
"""
@author:Wenqiang Liu
@file:ReadData.py
@time:2017/11/214:53
"""
import random
import pandas as pd
import numpy as np
from numpy.random import binomial
from os import path
import tensorflow as tf

class DataSet(object):
	def __init__(self, data_dir, negative_sampling):
		self.data_dir = data_dir  # 数据的存储路径
		self.negative_sampling = negative_sampling  # 负采样

		self.entity_to_id = {}  # 字典. key:entity value：id
		self.id_to_entity = {}  # 字典. key:id value：entity
		self.relation_to_id = {}  # 字典. key:relation value：id
		self.id_to_relation = {}  # 字典. key:id value：relation
		self.entityid_to_typeid = {}  #字典.key:entityid value: typeid
		# 训练集Set: {(id_head, id_relation, id_tail), ...}
		self.triplets_train_pool = set()
		self.type_train_pool = {} #字典. key: type value:id
		# 训练集合列表：[(id_head, id_relation, id_tail), ...]
		self.triplets_train = []
		self.triplets_validate = []  #验证集
		self.triplets_test = []  #测试集

		self.num_entity = 0   #entity的数量
		self.num_relation = 0 #relation的数据量
		self.num_triplets_train = 0 #训练集的数量
		self.num_triplets_validate = 0# 验证集的数据量
		self.num_triplets_test = 0 #测试集的数量
		self.num_type = 0# type 类型的数量
		# for reducing false negative labels
		self.relation_dist = {}  # {relation, (head_per_tail, tail_per_head)}

		# load train, validate and test files
		self.load_data()

	def load_data(self):
		# read the entity_to_id file
		print('loading entities...')
		entity_to_id_df = pd.read_csv(path.join(self.data_dir,'entity2id.txt'),sep='\t')
		self.entity_to_id = dict(
			zip(entity_to_id_df['entity'], entity_to_id_df['id']))
		self.id_to_entity = dict(
			zip(entity_to_id_df['id'], entity_to_id_df['entity']))

		self.num_entity = len(self.entity_to_id)
		print('got {} entities'.format(self.num_entity))

		# read the relation_to_id file
		print('loading relations...')
		relation_to_id_df = pd.read_csv(path.join(self.data_dir,'relation2id.txt'),sep='\t')
		self.relation_to_id = dict(
			zip(relation_to_id_df['relation'], relation_to_id_df['id']))
		self.id_to_relation = dict(
			zip(relation_to_id_df['id'], relation_to_id_df['relation']))
		self.num_relation = len(self.relation_to_id)
		print('got {} relations'.format(self.num_relation))

		# read the train file
		print('loading train triplets...')
		triplets_train_df = pd.read_csv(path.join(self.data_dir,'train.txt'),sep='\t')
		self.triplets_train = list(zip([self.entity_to_id[head] for head in triplets_train_df['head']],[self.relation_to_id[relation] for relation in triplets_train_df['relation']],[self.entity_to_id[tail] for tail in triplets_train_df['tail']]))
		self.num_triplets_train = len(self.triplets_train)
		print(
			'got {} triplets from training set'.format(
				self.num_triplets_train))
		# construct the train triplets pool
		self.triplets_train_pool = set(self.triplets_train)

		if self.negative_sampling == 'bern':
			self.set_bernoulli(triplets_train_df)
		else:
			print('do not need to calculate hpt & tph...')

		# read the validate file
		print('loading validate triplets...')
		triplets_validate_df = pd.read_csv(path.join(self.data_dir,'valid.txt'),sep='\t')
		self.triplets_validate = list(zip(
			[self.entity_to_id[head] for head in triplets_validate_df['head']],
			[self.relation_to_id[relation] for relation in triplets_validate_df['relation']],
			[self.entity_to_id[tail] for tail in triplets_validate_df['tail']]
		))
		self.num_triplets_validate = len(self.triplets_validate)
		print('got {} triplets from validation set'.format(self.num_triplets_validate))

		# read the test file
		print('loading test triplets...')
		triplets_test_df = pd.read_csv(path.join(self.data_dir, 'test.txt'),sep='\t')
		self.triplets_test = list(zip(
			[self.entity_to_id[head] for head in triplets_test_df['head']],
			[self.relation_to_id[relation] for relation in triplets_test_df['relation']],
			[self.entity_to_id[tail] for tail in triplets_test_df['tail']]
		))
		self.num_triplets_test = len(self.triplets_test)
		print('got {} triplets from test set'.format(self.num_triplets_test))

		# read the type constrain file
		print('loading type constrain file...')
		type_train_df = pd.read_csv(path.join(self.data_dir, 'entity2type.txt'), sep='\t')
		type_train_pool_set = set(list(type_train_df['type']))
		self.type_train_pool = dict(zip(type_train_pool_set,np.arange(len(type_train_pool_set))))
		self.entityid_to_typeid=dict(zip(
							[self.entity_to_id[entity] for entity in type_train_df["entity"] ],
							[self.type_train_pool[type] for type in type_train_df["type"]]
		))
		self.num_type=len(self.type_train_pool)
	def set_bernoulli(self, triplets_train_df):
		print('calculating hpt & tph for reducing negative false labels...')
		grouped_relation = triplets_train_df.groupby(
			'relation', as_index=False)
		# calculate head_per_tail and tail_per_head after group by relation
		n_to_one = grouped_relation.agg({
			'head': lambda heads: heads.count(),
			'tail': lambda tails: tails.nunique()
		})
		# one_to_n dataframe, columns = ['relation', 'head', 'tail']
		one_to_n = grouped_relation.agg({
			'head': lambda heads: heads.nunique(),
			'tail': lambda tails: tails.count()
		})
		relation_dist_df = pd.DataFrame({
			'relation': n_to_one['relation'],
			# element-wise division
			'head_per_tail': n_to_one['head'] / n_to_one['tail'],
			'tail_per_head': one_to_n['tail'] / one_to_n['head']
		})
		self.relation_dist = dict(zip(
			[self.relation_to_id[relation] for relation in relation_dist_df['relation']],
			zip(
				relation_dist_df['head_per_tail'],
				relation_dist_df['tail_per_head']
			)
		))

	def next_batch_train(self, batch_size,num_negative):
		# construct positive batch
		batch_positive = random.sample(self.triplets_train, batch_size)
		#构建样例的 Type矩阵，作为生成器Y的输入
		type_binomial = np.random.binomial(1, 0.5, batch_size * num_negative)
		type_y = tf.reshape(type_binomial, [batch_size * num_negative, 1])
		type_y_dim = tf.to_int32(tf.concat([type_y, 1 - type_y], 1))
		type_head_entity =[self.entityid_to_typeid[batch_positive[index][0]] for index in range(batch_size)]
		type_tail_entity =[self.entityid_to_typeid[batch_positive[index][2]] for index in range(batch_size)]
		type_head_onehot = tf.one_hot(type_head_entity,depth=self.num_type,on_value=1,off_value=0)
		tail_tail_onehot = tf.one_hot(type_tail_entity, depth=self.num_type, on_value=1, off_value=0)
		type_head_onehot_extend = tf.reshape(tf.tile(type_head_onehot,[1,num_negative]),[-1,self.num_type])
		tail_tail_onehot_extend = tf.reshape(tf.tile(tail_tail_onehot,[1,num_negative]),[-1,self.num_type])
		y_dim = tf.multiply(tf.gather(type_y_dim, [0], axis=1),type_head_onehot_extend) + tf.multiply(tf.gather(type_y_dim, [1], axis=1),tail_tail_onehot_extend)
		return batch_positive, type_y_dim, y_dim

	def next_batch_validate(self, batch_size):
		batch_validate = random.sample(self.triplets_validate, batch_size)

		return batch_validate

	def next_batch_eval(self, triplet_evaluate):
		# construct two batches for head and tail prediction
		batch_predict_head = [triplet_evaluate]
		# replacing head
		id_heads_corrupted = set(self.id_to_entity.keys())
		id_heads_corrupted.remove(
			triplet_evaluate[0])  # remove the golden head
		batch_predict_head.extend(
			[(head, triplet_evaluate[1], triplet_evaluate[2]) for head in id_heads_corrupted])

		batch_predict_tail = [triplet_evaluate]
		# replacing tail
		id_tails_corrupted = set(self.id_to_entity.keys())
		id_tails_corrupted.remove(
			triplet_evaluate[2])  # remove the golden tail
		batch_predict_tail.extend(
			[(triplet_evaluate[0], triplet_evaluate[1], tail) for tail in id_tails_corrupted])

		return batch_predict_head, batch_predict_tail
if __name__ == '__main__':
	dataset = DataSet("../data/", negative_sampling="bern")