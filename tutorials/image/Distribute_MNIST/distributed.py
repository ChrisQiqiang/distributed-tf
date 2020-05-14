# encoding:utf-8
import math
import tempfile
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
IMAGE_PIXELS = 28
# 定义默认训练参数和数据路径
flags.DEFINE_string('data_dir', '/tmp/mnist-data', 'Directory  for storing mnist data')
flags.DEFINE_integer('hidden_units', 100, 'Number of units in the hidden layer of the NN')
flags.DEFINE_integer('train_steps', 10000, 'Number of training steps to perform')
flags.DEFINE_integer('batch_size', 32, 'Training batch size ')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
# 定义分布式参数
# 参数服务器parameter server节点
flags.DEFINE_string('ps_hosts', '10.2.99.2:2222', 'Comma-separated list of hostname:port pairs')
flags.DEFINE_string('worker_hosts', '10.2.99.3:2222,10.2.99.4:2222,10.2.99.5:2222,10.2.99.6:2222',
                    'Comma-separated list of hostname:port pairs')

# 设置job name参数
flags.DEFINE_string('job_name', None, 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task_index', None, 'Index of task within the job')
# 选择异步并行，同步并行
flags.DEFINE_integer("issync", None, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

FLAGS = flags.FLAGS


def main(unused_argv):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print 'job_name : %s' % FLAGS.job_name
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print 'task_index : %d' % FLAGS.task_index

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')
    issync = FLAGS.issync

    # 创建集群
    num_worker = len(worker_spec)
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == 'worker':
        time.sleep(10)

    is_chief = (FLAGS.task_index == 0)
    # worker_device = '/job:worker/task%d/cpu:0' % FLAGS.task_index
    with tf.device(tf.train.replica_device_setter(
	    worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster
    )):
        global_step = tf.Variable(0, name='global_step', trainable=False)  # 创建纪录全局训练步数变量

        hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                                                stddev=1.0 / IMAGE_PIXELS), name='hid_w')
        hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name='hid_b')

        sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10],
                                               stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name='sm_w')
        sm_b = tf.Variable(tf.zeros([10]), name='sm_b')

        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
        y_ = tf.placeholder(tf.float32, [None, 10])

        hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
        hid = tf.nn.relu(hid_lin)

        y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
        cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

	# Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

	grads = opt.compute_gradients(cross_entropy)

	if issync == 1:
                syn_opt = tf.train.SyncReplicasOptimizer(opt,
                                        replicas_to_aggregate=num_worker,
                                        #replica_id=FLAGS.task_index,
                                        total_num_replicas=num_worker)
                                        #use_locking=True)
                train_op = syn_opt.apply_gradients(grads, global_step=global_step)
                init_token_op = syn_opt.get_init_tokens_op()
                chief_queue_runner = syn_opt.get_chief_queue_runner()
	else:
		train_op = opt.apply_gradients(grads, global_step=global_step)

        #train_step = opt.minimize(cross_entropy, global_step=global_step)

        # 生成本地的参数初始化操作init_op
        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()

        sv = tf.train.Supervisor(is_chief=is_chief,
				 logdir=train_dir,
				 init_op=init_op,
				 recovery_wait_secs=1,
                                 global_step=global_step)

        if is_chief:
            print 'Worker %d: Initializing session...' % FLAGS.task_index
        else:
            print 'Worker %d: Waiting for session to be initialized...' % FLAGS.task_index
        
	sess = sv.prepare_or_wait_for_session(server.target)
        print 'Worker %d: Session initialization  complete.' % FLAGS.task_index

        if is_chief and issync == 1:
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_token_op)
        else:
            sv.start_queue_runners(sess=sess)

        time_begin = time.time()
        print 'Training begins @ %s' % datetime.now()

        local_step = 0
        while True:
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            train_feed = {x: batch_xs, y_: batch_ys}

            _, step, loss_value = sess.run([train_op, global_step, cross_entropy], feed_dict=train_feed)
            local_step += 1

            now = time.time()
	    # added by faye
	    if step % 10 == 0:
            	print '%s: Worker %d: training step %d done (global step:%d, cross entropy=%.2f)' % (datetime.now(), FLAGS.task_index, local_step, step, loss_value)

            if step >= FLAGS.train_steps:
                break

        time_end = time.time()
        print 'Training ends @ %f' % time_end
        train_time = time_end - time_begin
        print 'Training elapsed time:%f s' % train_time

        val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        val_xent = sess.run(accuracy, feed_dict=val_feed)
        print 'After %d training step(s), validation accuracy = %f' % (FLAGS.train_steps, val_xent)
    sess.close()

if __name__ == '__main__':
    tf.app.run()
