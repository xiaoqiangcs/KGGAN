# -*- coding:utf-8 -*-
"""
@author:Wenqiang Liu
@file:Main.py
@time:2017/11/99:30
"""
from ReadData import DataSet
from TransE import TransE
import tensorflow as tf
import random
import time
import argparse
import math
def run_training(args):
	dataset = DataSet(data_dir=args.data_dir,
					  negative_sampling=args.negative_sampling)

	model = TransE(learning_rate = args.learning_rate,
				   batch_size = args.batch_size,
				   num_epoch = args.num_epoch,
				   margin = args.margin,
				   embedding_dimension = args.embedding_dimension,
				   dissimilarity = args.dissimilarity,
				   evaluate_size = args.evaluate_size)
	#construct the training graph
	graph_transe = tf.Graph()
	with graph_transe.as_default():
		print("constructing the traing graph")
		with tf.variable_scope("input"):
			id_triplets_positive = tf.placeholder(dtype=tf.int32, shape=[model.batch_size, 3], name='triplets_positive')
			id_triplets_predict_head = tf.placeholder(dtype=tf.int32, shape=[None, 3],
													  name='triplets_predict_head')
		# embedding table
		bound = 6 / math.sqrt(model.embedding_dimension)
		with tf.variable_scope("embedding"):
			embedding_entity = tf.get_variable(name='entity', initializer=tf.random_uniform(
				shape=[dataset.num_entity, model.embedding_dimension], minval=-bound, maxval=bound))
			embedding_relation = tf.get_variable(name='relation', initializer=tf.random_uniform(
				shape=[dataset.num_relation, model.embedding_dimension], minval=-bound, maxval=bound))
		with tf.name_scope('normalization'):
			normalize_relation_op = embedding_relation.assign(tf.clip_by_norm(embedding_relation, clip_norm=1, axes=1))
			normalize_entity_op = embedding_entity.assign(tf.clip_by_norm(embedding_entity, clip_norm=1, axes=1))

		# ops into scopes, convenient for TensorBoard's Graph visualization
		with tf.name_scope("inference"):
			d_positive = model.inference(id_triplets_positive)
		with tf.name_scope('loss'):
			# model train loss
			loss = model.loss(d_positive)
		with tf.name_scope('optimization'):
			# model train operation
			train_op = model.train(loss)
		with tf.name_scope('evaluation'):
			# model evaluation
			predict_head = model.evaluation(id_triplets_predict_head)
		print('graph constructing finished')
		# initilize op
		init_op = tf.global_variables_initializer()
		# merge all the summaries
		merge_summary_op = tf.summary.merge_all()
		# saver op to save or restore all variables
		saver = tf.train.Saver()
	# open a session and run the training graph
	session_config = tf.ConfigProto(log_device_placement=True)
	session_config.gpu_options.allow_growth = True
	with tf.Session(graph=graph_transe, config=session_config) as sess:
		#run the initial operation
		print("initializing all variables...")
		sess.run(init_op)
		print("all variables initialized")
		# normalize relation embeddings after initialization
		sess.run(normalize_relation_op)
		# op to write logs to tensorboard
		summary_writer = tf.summary.FileWriter(args.log_dir, graph=sess.graph)
		num_batch = dataset.num_triplets_train // model.batch_size
		# training
		print("star training ...")
		start_total = time.time()
		for epoch in range(model.num_epoch):
			loss_epoch = 0.0
			start_train = time.time()
			for batch in range(num_batch):
				# normalize entity embeddings before every batch
				sess.run(normalize_entity_op)
				batch_positive= dataset.next_batch_train(model.batch_size)
				feed_dict_train = {
					id_triplets_positive: batch_positive
				}
				_,loss_batch,summary = sess.run([train_op, loss, merge_summary_op],feed_dict=feed_dict_train)
				loss_epoch += loss_batch
				# write tensorboard logs
				summary_writer.add_summary(summary, global_step=epoch * num_batch + batch)
				# print an overview every 10 batches
				if (batch + 1) % 100 == 0 or (batch + 1) == num_batch:
					print('epoch {}, batch {}, loss: {}'.format(epoch, batch, loss_batch))
			end_train = time.time()
			print('epoch {}, mean batch loss: {:.3f}, time elapsed last epoch: {:.3f}s'.format(epoch,loss_epoch / num_batch,end_train - start_train))
			# save a checkpoint every epoch
			save_path = saver.save(sess, args.save_dir + 'model.ckpt')
			print('model save in file {}'.format(save_path))
			# evaluate the model every 5 epochs
			if (epoch + 1) % 5 == 0:
				run_evaluation(sess,predict_head,model,dataset,id_triplets_predict_head)
		end_total = time.time()
		print('total time elapsed: {:.3f}s'.format(end_total - start_total))
		print('training finished')
		print('Begian Test')
		start_total = time.time()
		run_test(sess,predict_head,model,dataset,id_triplets_predict_head)
		end_total = time.time()
		print('total time elapsed: {:.3f}s'.format(end_total - start_total))
		print('Test finished')
def run_evaluation(sess,predict_head,model,dataset,id_triplets_predict_head):
	print('evaluating the current model...')
	start_eval = time.time()
	batch_validate =random.sample(dataset.triplets_validate, model.evaluate_size)
	feed_dict_eval = {
		id_triplets_predict_head: batch_validate,
	}
	prediction_head= sess.run([predict_head], feed_dict=feed_dict_eval)
	loss = sum(sum(prediction_head))
	end_eval = time.time()
	print('All loss: {:.3f}, mean loss: {:.3f}'.format(loss, loss / model.evaluate_size))
	print('time elapsed last evaluation: {:.3f}s'.format(end_eval - start_eval))
	print('back to training...')
def run_test(sess,predict_head,model,dataset,id_triplets_predict_head):
	print('evaluating the current model...')
	start_eval = time.time()
	batch_validate = random.sample(dataset.triplets_test, dataset.num_triplets_test)
	feed_dict_eval = {
		id_triplets_predict_head: batch_validate,
	}
	prediction_head = sess.run([predict_head], feed_dict=feed_dict_eval)
	loss = sum(sum(prediction_head))
	end_eval = time.time()
	print('All loss: {:.3f}, mean loss: {:.3f}'.format(loss, loss / model.evaluate_size))
	print('time elapsed last evaluation: {:.3f}s'.format(end_eval - start_eval))
	print('back to training...')
def main():
	parser = argparse.ArgumentParser()
	print("Beig")
	# dataset args
	parser.add_argument(
		'--data_dir',
		type=str,
		default='data/',
		help='dataset directory'
	)
	parser.add_argument(
		'--negative_sampling',
		type=str,
		default='bern',
		help='negative sampling method, unif or bern'
	)

	# model args
	parser.add_argument(
		'--learning_rate',
		type=float,
		default=0.001,
		help='initial learning rate'
	)
	parser.add_argument(
		'--batch_size',
		type=int,
		default=300,
		help='mini batch size for SGD'
	)
	parser.add_argument(
		'--num_epoch',
		type=int,
		default=400,
		help='number of epochs'
	)
	parser.add_argument(
		'--margin',
		type=float,
		default=1,
		help='margin of a golden triplet and a corrupted one'
	)
	parser.add_argument(
		'--embedding_dimension',
		type=int,
		default=150,
		help='dimension of entity and relation embeddings'
	)
	parser.add_argument(
		'--dissimilarity',
		type=str,
		default='L1',
		help='using L1 or L2 distance as dissimilarity'
	)
	parser.add_argument(
		'--evaluate_size',
		type=int,
		default=500,
		help='the size of evaluate triplets, max is 50000'
	)

	# tensorflow args
	parser.add_argument(
		'--log_dir',
		type=str,
		default='data/log/',
		help='tensorflow log files directory, for tensorboard'
	)
	parser.add_argument(
		'--save_dir',
		type=str,
		default='data/ckpt/',
		help='tensorflow checkpoint directory, for variable save and restore'
	)

	args = parser.parse_args()
	print('args: {}'.format(args))
	print(args.evaluate_size)
	run_training(args=args)
if __name__ == '__main__':
	main()