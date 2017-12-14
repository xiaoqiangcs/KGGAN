# -*- coding:utf-8 -*-
# 实现论文KBGAN: Adversarial Learning for Knowledge Graph Embeddings
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

    model = TransE(learning_rate=args.learning_rate,
                   batch_size=args.batch_size,
                   num_epoch=args.num_epoch,
                   margin=args.margin,
                   embedding_dimension=args.embedding_dimension,
                   dissimilarity=args.dissimilarity,
                   evaluate_size=args.evaluate_size,
                   num_negative=args.num_negative,
                   noise_dim =args.noise_dim)
    #construct the training graph
    num_negative = args.num_negative # 负采样的个数
    noise_dim = args.noise_dim #潜在变量Latent的维度大小
    graph_transe = tf.Graph()
    with graph_transe.as_default():
        print("constructing the traing graph")
        with tf.variable_scope("Generator_Sampling"):
            noise_Z = tf.placeholder(tf.float32, shape=[model.batch_size, num_negative * noise_dim])
            # 输入随机噪声z而输出生成样本
        with tf.variable_scope("input"):
            id_triplets_positive = tf.placeholder(dtype=tf.int64, shape=[model.batch_size, 3], name='triplets_positive')
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
        with tf.name_scope("generator"):
            G_prob, theta_G = model.generator(noise_Z)
            batch_negative, g_loss_sum = model.get_negative_embedding(G_prob,dataset.num_entity)
            batch_negative_sample, g_loss_fake = model.get_negative_sampling(batch_negative, id_triplets_positive)
        with tf.name_scope("loss"):
            #生成器的损失函数为两部分，即E[log(-D(x))]和生成的负样例与所有entity最小距离的和
            g_loss = g_loss_fake + g_loss_sum
            g_solver = tf.train.AdagradOptimizer(learning_rate=args.learning_rate).minimize(g_loss,var_list=theta_G)
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
    with tf.Session(graph=graph_transe) as sess:
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
            g_loss_epoch = 0.0
            start_train = time.time()
            for batch in range(num_batch):
                # normalize entity embeddings before every batch
                sess.run(normalize_entity_op)
                batch_positive = dataset.next_batch_train(model.batch_size)
                noise_sample = model.sample_Z(model.batch_size, num_negative * noise_dim)
                feed_dict_generator = {
                    noise_Z: noise_sample,
                    id_triplets_positive: batch_positive
                }
                _,g_loss_batch = sess.run([g_solver, g_loss], feed_dict=feed_dict_generator)
                g_loss_epoch += g_loss_batch
                # write tensorboard logs
                #summary_writer.add_summary(summary, global_step=epoch * num_batch + batch)
                # print an overview every 10 batches
                if (batch + 1) % 100 == 0 or (batch + 1) == num_batch:
                    print('epoch {}, batch {}, g_loss: {}'.format(epoch, batch,g_loss_batch))
            end_train = time.time()
            print('epoch {},mean batch g_loss: {:.3f}, time elapsed last epoch: {:.3f}s'.format(epoch, g_loss_epoch / num_batch, end_train - start_train))
            # save a checkpoint every epoch
            save_path = saver.save(sess, args.save_dir + 'model.ckpt')
            print('model save in file {}'.format(save_path))
            # evaluate the model every 5 epochs
            # if (epoch + 1) % 5 == 0:
            # 	run_evaluation(sess,predict_head,predict_tail,model,dataset,id_triplets_predict_head,id_triplets_predict_tail)
        end_total = time.time()
        print('total time elapsed: {:.3f}s'.format(end_total - start_total))
        print('training finished')
        print('Begian Test')
        start_total = time.time()
        # run_test(sess,predict_head,predict_tail,model,dataset,id_triplets_predict_head,id_triplets_predict_tail)
        end_total = time.time()
        print('total time elapsed: {:.3f}s'.format(end_total - start_total))
        print('Test finished')
def run_evaluation(sess,predict_head,predict_tail, model,dataset,id_triplets_predict_head,id_triplets_predict_tail):
    print('evaluating the current model...')
    start_eval = time.time()
    rank_head = 0
    rank_tail = 0
    hit10_head = 0
    hit1_head = 0
    hit10_tail = 0
    hit1_tail  = 0
    evaluate_index=0
    for triplet in random.sample(dataset.triplets_validate, model.evaluate_size):
        batch_predict_head, batch_predict_tail = dataset.next_batch_eval(triplet)
        feed_dict_eval = {
            id_triplets_predict_head: batch_predict_head,
            id_triplets_predict_tail: batch_predict_tail
        }
        # rank list of head and tail prediction
        prediction_head, prediction_tail = sess.run([predict_head, predict_tail], feed_dict=feed_dict_eval)
        rank_head_current = prediction_head.argsort().argmin()
        rank_head += rank_head_current
        if rank_head_current < 10:
            hit10_head += 1
        if rank_head_current==1:
            hit1_head +=1
        rank_tail_current = prediction_tail.argsort().argmin()
        rank_tail += rank_tail_current
        if rank_tail_current < 10:
            hit10_tail += 1
        if rank_tail_current==1:
            hit1_tail +=1
        evaluate_index+=1
        if evaluate_index% 1000 ==0 :
            print("evaluating the current:{:d}".format(evaluate_index))
    rank_head_mean = rank_head // model.evaluate_size
    hit10_head /= model.evaluate_size
    hit1_head /= model.evaluate_size
    rank_tail_mean = rank_tail // model.evaluate_size
    hit10_tail /= model.evaluate_size
    hit1_tail /= model.evaluate_size
    end_eval = time.time()
    print('head prediction mean rank: {:d}, hit@10: {:.3f}%, hit@1: {:.3f}%'.format(rank_head_mean, hit10_head * 100, hit1_head * 100))
    print('tail prediction mean rank: {:d}, hit@10: {:.3f}%, hit@1: {:.3f}%'.format(rank_tail_mean, hit10_tail * 100, hit1_tail * 100))
    print('time elapsed last evaluation: {:.3f}s'.format(end_eval - start_eval))
    print('back to training...')
def run_test(sess,predict_head,predict_tail, model,dataset,id_triplets_predict_head,id_triplets_predict_tail):
    print('evaluating the current model...')
    start_eval = time.time()
    rank_head = 0
    rank_tail = 0
    hit10_head = 0
    hit1_head = 0
    hit10_tail = 0
    hit1_tail = 0
    evaluate_index = 0
    for triplet in random.sample(dataset.triplets_test, dataset.num_triplets_test):
        batch_predict_head, batch_predict_tail = dataset.next_batch_eval(triplet)
        feed_dict_eval = {
            id_triplets_predict_head: batch_predict_head,
            id_triplets_predict_tail: batch_predict_tail
        }
        # rank list of head and tail prediction
        prediction_head, prediction_tail = sess.run([predict_head, predict_tail], feed_dict=feed_dict_eval)
        rank_head_current = prediction_head.argsort().argmin()
        rank_head += rank_head_current
        if rank_head_current < 10:
            hit10_head += 1
        if rank_head_current==1:
            hit1_head +=1
        rank_tail_current = prediction_tail.argsort().argmin()
        rank_tail += rank_tail_current
        if rank_tail_current < 10:
            hit10_tail += 1
        if rank_tail_current==1:
            hit1_tail +=1
        evaluate_index += 1
        if evaluate_index % 1000 == 0:
            print("evaluating the size of Test datasets:{:d}".format(evaluate_index))
    rank_head_mean = rank_head // dataset.num_triplets_test
    hit10_head /= dataset.num_triplets_test
    hit1_head /= dataset.num_triplets_test
    rank_tail_mean = rank_tail // dataset.num_triplets_test
    hit10_tail /= dataset.num_triplets_test
    hit1_tail /= dataset.num_triplets_test
    end_eval = time.time()
    print('head prediction mean rank: {:d}, hit@10: {:.3f}%, hit@1: {:.3f}%'.format(rank_head_mean, hit10_head * 100, hit1_head * 100))
    print('tail prediction mean rank: {:d}, hit@10: {:.3f}%, hit@1: {:.3f}%'.format(rank_tail_mean, hit10_tail * 100, hit1_tail * 100))
    print('time elapsed last evaluation: {:.3f}s'.format(end_eval - start_eval))
def main():
    parser = argparse.ArgumentParser()
    print("Begain")
    # dataset args
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../data/',
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
        default=0.01,
        help='initial learning rate'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=150,
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
        default=1.0,
        help='margin of a golden triplet and a corrupted one'
    )
    parser.add_argument(
        '--embedding_dimension',
        type=int,
        default=100,
        help='dimension of entity and relation embeddings'
    )
    parser.add_argument(
        '--dissimilarity',
        type=str,
        default='L2',
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
    parser.add_argument(
        '--num_negative',
        type=int,
        default=20,
        help='tensorflow checkpoint directory, for variable save and restore'
    )
    parser.add_argument(
        '--reward_decay',
        type=int,
        default=20,
        help='tensorflow checkpoint directory, for variable save and restore'
    )
    parser.add_argument(
        '--noise_dim',
        type=int,
        default=16,
        help='tensorflow checkpoint directory, for variable save and restore'
    )
    args = parser.parse_args()
    print('args: {}'.format(args))
    print(args.evaluate_size)
    run_training(args=args)
if __name__ == '__main__':
    main()