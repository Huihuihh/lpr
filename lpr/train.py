#!/usr/bin/env python3
#
# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import random
import os
import argparse
import numpy as np
import tensorflow as tf
from lpr.trainer import CTCUtils, inference, InputData, LPRVocab


def parse_args():
  parser = argparse.ArgumentParser(description='Perform training of a model')
  parser.add_argument('--init_checkpoint', default=None, help='Path to checkpoint')
  parser.add_argument('--input_shape', default=(24, 94, 3), help='Input shape')
  parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
  parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
  parser.add_argument('--opt_type', default='Adam', help='The optimization algorithm to use')
  parser.add_argument('--grad_noise_scale', default=0.001, help='Grad noise scale')
  parser.add_argument('--steps', type=int, default=250000, help='Steps')
  parser.add_argument('--apply_basic_aug', default=False, action='store_true', help='Apply basic aug')
  parser.add_argument('--apply_stn_aug', action='store_true', help='Apply stn aug')
  parser.add_argument('--apply_blur_aug', action='store_true', default=False, help='Apply blur aug')
  parser.add_argument('--train_file_list_path', help='Train file list path', default='Synthetic_Chinese_License_Plates/train')
  parser.add_argument('--eval_file_list_path', help='Eval file list path', default='Synthetic_Chinese_License_Plates/val')
  parser.add_argument('--rnn_cells_num', default=128, help='Rnn cells num')
  parser.add_argument('--need_to_save_log', action='store_true', help='Need to save log')
  parser.add_argument('--model_dir', default='./model', help='Model dir')
  parser.add_argument('--display_iter', default=100, type=int, help='Display iter')
  parser.add_argument('--save_checkpoints_steps', default=1000, type=int, help='Save checkpoints steps')
  parser.add_argument('--need_to_save_weights', action='store_true', help='Need to save weights')
  parser.add_argument('--gpu_memory_fraction', default=0.8, type=float, help='Gpu memory fraction')
  return parser.parse_args()

# pylint: disable=too-many-locals, too-many-statements
def train(args):
  vocab, r_vocab, num_classes = LPRVocab.create_vocab(args.train_file_list_path,
                                                      args.eval_file_list_path,
                                                      False,
                                                      False)

  CTCUtils.vocab = vocab
  CTCUtils.r_vocab = r_vocab

  input_train_data = InputData(batch_size=args.batch_size,
                               input_shape=args.input_shape,
                               file_list_path=args.train_file_list_path,
                               apply_basic_aug=args.apply_basic_aug,
                               apply_stn_aug=args.apply_stn_aug,
                               apply_blur_aug=args.apply_blur_aug)


  graph = tf.Graph()
  with graph.as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    input_data, input_labels = input_train_data.input_fn()

    prob = inference(args.rnn_cells_num, input_data, num_classes)
    prob = tf.transpose(prob, (1, 0, 2))  # prepare for CTC

    data_length = tf.fill([tf.shape(prob)[1]], tf.shape(prob)[0])  # input seq length, batch size
    ctc = tf.py_func(CTCUtils.compute_ctc_from_labels, [input_labels], [tf.int64, tf.int64, tf.int64])
    ctc_labels = tf.to_int32(tf.SparseTensor(ctc[0], ctc[1], ctc[2]))

    predictions = tf.to_int32(
      tf.nn.ctc_beam_search_decoder(prob, data_length, merge_repeated=False, beam_width=10)[0][0])
    tf.sparse_tensor_to_dense(predictions, default_value=-1, name='d_predictions')
    tf.reduce_mean(tf.edit_distance(predictions, ctc_labels, normalize=False), name='error_rate')

    loss = tf.reduce_mean(
      tf.nn.ctc_loss(inputs=prob, labels=ctc_labels, sequence_length=data_length, ctc_merge_repeated=True), name='loss')

    learning_rate = tf.train.piecewise_constant(global_step, [150000, 200000],
                                                [args.learning_rate, 0.1 * args.learning_rate,
                                                 0.01 * args.learning_rate])
    opt_loss = tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, args.opt_type,
                                               args.grad_noise_scale, name='train_step')

    tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1000, write_version=tf.train.SaverDef.V2, save_relative_paths=True)

  conf = tf.ConfigProto(allow_soft_placement=True)
  conf.gpu_options.allow_growth = True
  conf.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction

  session = tf.Session(graph=graph, config=conf)
  coordinator = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

  session.run('init')

  if args.init_checkpoint:
    tf.logging.info('Initialize from: ' + args.init_checkpoint)
    saver.restore(session, args.init_checkpoint)
  else:
    lastest_checkpoint = tf.train.latest_checkpoint(args.model_dir)
    if lastest_checkpoint:
      tf.logging.info('Restore from: ' + lastest_checkpoint)
      saver.restore(session, lastest_checkpoint)

  writer = None
  if args.need_to_save_log:
    writer = tf.summary.FileWriter(args.model_dir, session.graph)

  graph.finalize()


  for i in range(args.steps):
    curr_step, curr_learning_rate, curr_loss, curr_opt_loss = session.run([global_step, learning_rate, loss, opt_loss])

    if i % args.display_iter == 0:
      if args.need_to_save_log:

        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='train/loss',
                                                              simple_value=float(curr_loss)),
                                             tf.Summary.Value(tag='train/learning_rate',
                                                              simple_value=float(curr_learning_rate)),
                                             tf.Summary.Value(tag='train/optimization_loss',
                                                              simple_value=float(curr_opt_loss))
                                             ]),
                           curr_step)
        writer.flush()

      tf.logging.info('Iteration: ' + str(curr_step) + ', Train loss: ' + str(curr_loss))

    if ((curr_step % args.save_checkpoints_steps == 0 or curr_step == args.steps)
        and args.need_to_save_weights):
      saver.save(session, args.model_dir + '/model.ckpt-{:d}.ckpt'.format(curr_step))

  coordinator.request_stop()
  coordinator.join(threads)
  session.close()


def main(_):
  args = parse_args()
  train(args)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
