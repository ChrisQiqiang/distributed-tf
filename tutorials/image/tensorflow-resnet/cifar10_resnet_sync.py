from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
#from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import resnet_model

#from batchsizemanager import BatchSizeManager
import cifar10
from tensorflow.python.client import timeline

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
#                           """Directory where to write event logs """
#                           """and checkpoint.""")
# added by faye
#tf.app.flags.DEFINE_string('ps_hosts', '10.2.99.2:2222', 'Comma-separated list of hostname:port pairs')
#tf.app.flags.DEFINE_string('worker_hosts', '10.2.99.3:2222,10.2.99.4:2222',
#                    'Comma-separated list of hostname:port pairs')
#tf.app.flags.DEFINE_string('job_name', 'ps', 'job name: worker or ps')
#tf.app.flags.DEFINE_integer('task_id', 0, 'Index of task within the job')
#tf.app.flags.DEFINE_integer('sync', 0, 'Whether synchronization: 0 or 1')

tf.logging.set_verbosity(tf.logging.INFO)

INITIAL_LEARNING_RATE = 0.32       # Initial learning rate.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

updated_batch_size_num = 28
_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5
_WEIGHT_DECAY = 2e-4

def train():
    global updated_batch_size_num
    global passed_info
    global shall_update
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    print ('PS hosts are: %s' % ps_hosts)
    print ('Worker hosts are: %s' % worker_hosts)
    issync = FLAGS.sync
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == "worker":
        time.sleep(10)
    is_chief = (FLAGS.task_index == 0)
    if is_chief:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

    # modified by faye
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster
            )):
        global_step = tf.get_variable(
                    'global_step', [],
                    initializer=tf.constant_initializer(0), trainable=False)
        decay_steps = 50000*350.0/FLAGS.batch_size
        batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')
        images, labels = cifar10.distorted_inputs(batch_size)
        print('zx0')
        print(images.get_shape().as_list())
#       print (str(tf.shape(images))+ str(tf.shape(labels)))
        re = tf.shape(images)[0]
        network = resnet_model.cifar10_resnet_v2_generator(FLAGS.resnet_size, _NUM_CLASSES)
        inputs = tf.reshape(images, [-1, _HEIGHT, _WIDTH, _DEPTH])
#            labels = tf.reshape(labels, [-1, _NUM_CLASSES])
        labels = tf.one_hot(labels, 10, 1, 0)
        logits = network(inputs, True)
        print(logits.get_shape())
        cross_entropy = tf.losses.softmax_cross_entropy(
            logits=logits, 
            onehot_labels=labels)
        loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        opt = tf.train.GradientDescentOptimizer(lr)

        # Track the moving averages of all trainable variables.
        exp_moving_averager = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = exp_moving_averager.apply(tf.trainable_variables())

    # added by faye
        #grads = opt.compute_gradients(loss)
        grads0 = opt.compute_gradients(loss) 
        grads = [(tf.scalar_mul(tf.cast(batch_size/FLAGS.batch_size, tf.float32), grad), var) for grad, var in grads0]
        if issync == 1:
            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=len(worker_hosts),
#                   replica_id=FLAGS.task_id,
                total_num_replicas=len(worker_hosts),
                variable_averages=exp_moving_averager,
                variables_to_average=variables_to_average)
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
            chief_queue_runners = opt.get_chief_queue_runner()
            init_tokens_op = opt.get_init_tokens_op()
        else:
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        train_op = tf.group(apply_gradient_op, variables_averages_op)
        sv = tf.train.Supervisor(is_chief=is_chief,
                                    logdir=FLAGS.train_dir,
                                    init_op=tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()),
                                    summary_op=None,
                                    global_step=global_step,
#                                     saver=saver,
                                    saver=None,
                                    recovery_wait_secs=1,
                                    save_model_secs=60)

        tf.logging.info('%s Supervisor' % datetime.now())
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=FLAGS.log_device_placement)

   	    # Get a session.
        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
#	    sess.run(tf.global_variables_initializer())

        # Start the queue runners.
        if is_chief and issync == 1:
            sess.run(init_tokens_op)

            #queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
            #sv.start_queue_runners(sess, queue_runners)

            sv.start_queue_runners(sess, [chief_queue_runners])
        else:
             sv.start_queue_runners(sess=sess)
            #sess.run(init_tokens_op)

        #"""Train CIFAR-10 for a number of steps."""
        step = 0
        g_step = 0
        #time0 = time.time()
        batch_size_num = FLAGS.batch_size
#            for step in range(FLAGS.max_steps):
        while g_step <= FLAGS.max_steps:
            start_time = time.time()
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            if step <= 5:
                batch_size_num = FLAGS.batch_size
                num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size_num
                decay_steps_num = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
                _, loss_value, g_step = sess.run([train_op, loss, global_step], feed_dict={batch_size: batch_size_num},  options=run_options, run_metadata=run_metadata)
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                if step % 1 == 0:
                    duration = time.time() - start_time
                    num_examples_per_step = batch_size_num
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)
                    format_str = ('%s: step %d (global_step %d), loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    tf.logging.info(format_str % (datetime.now(), step, g_step, loss_value, examples_per_sec, sec_per_batch))             
                step += 1
        # end of while
        sv.stop()
        # end of with

def main(argv=None):
    cifar10.maybe_download_and_extract()
    train()

if __name__ == '__main__':
    tf.app.run()
