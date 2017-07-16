#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from input_helpers import InputHelper
import load_word_2vec_model
# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("eval_filepath", "/home/rathee/projects/quora_similar_questions/data/siamese_train_final.tsv", "Evaluate on this data (Default: None)")
tf.flags.DEFINE_string("model", "/mnt/data_18_4/rathee_output_dir/deep-siamese-text-similarity/runs/1492186449/checkpoints/model-448000", "Load trained model checkpoint (Default: None)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.eval_filepath==None or FLAGS.model==None :
    print("Eval or Vocab filepaths are empty.")
    exit()

w2v, model_dict, index_to_word = load_word_2vec_model.get_model_embeddings()
# load data and map id-transform based on training time vocabulary
inpH = InputHelper()
x1_test,x2_test,ids = inpH.getTestDataSet(FLAGS.eval_filepath, 30, model_dict)

wr = open('submissions_train.csv', 'w')
print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = FLAGS.model
print checkpoint_file
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
        input_x2 = graph.get_operation_by_name("input_x2").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/distance").outputs[0]

        #emb = graph.get_operation_by_name("embedding/W").outputs[0]
        #embedded_chars = tf.nn.embedding_lookup(emb,input_x)
        # Generate batches for one epoch
        batches = inpH.batch_iter(list(zip(x1_test,x2_test,ids)), 2*FLAGS.batch_size, 1, shuffle=False)
        # Collect the predictions here
        all_predictions = []
        all_d=[]
        for db in batches:
            x1_dev_b,x2_dev_b,ids_batch = zip(*db)
            batch_predictions = sess.run(predictions, {input_x1: x1_dev_b, input_x2: x2_dev_b, dropout_keep_prob: 1.0})
            #print(batch_predictions)
            d = np.copy(batch_predictions)
            d[d>=0.6]=999.0
            d[d<0.6]=1
            d[d>1.0]=0
            x = d.astype(np.int64)
            print len(ids_batch), len(batch_predictions)
            for i in range(len(ids_batch)) :
               wr.write(ids_batch[i] + ',' + str(1 - batch_predictions[i]) + '\n')
wr.close()
