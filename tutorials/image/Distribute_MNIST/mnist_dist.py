from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

tf.app.flags.DEFINE_string('ps_hosts', '10.2.59.6:2222', 'Comma-separated list of hostname:port pairs')
tf.app.flags.DEFINE_string('worker_hosts', '10.2.59.2:2222,10.2.59.3:2222,10.2.59.4:2222,10.2.59.5:2222,10.2.99.2:2222,10.2.99.3:2222,10.2.99.4:2222,10.2.99.5:2222',
                    'Comma-separated list of hostname:port pairs')
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("sync", 0, "Whether synchronization")

FLAGS = tf.app.flags.FLAGS

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    issync = FLAGS.sync

    num_worker = len(worker_hosts)

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    print("Cluster job: %s, task_index: %d, target: %s" % (FLAGS.job_name, FLAGS.task_index, server.target))
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        
	is_chief = (FLAGS.task_index == 0)

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # Build model ...
            mnist = input_data.read_data_sets("data", one_hot=True)
            
            # Create the model
            x = tf.placeholder(tf.float32, [None, 784])
            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))
            y = tf.matmul(x, W) + b

            # Define loss and optimizer
            y_ = tf.placeholder(tf.float32, [None, 10])
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

	    global_step = tf.get_variable(
        	'global_step', [],
        	initializer=tf.constant_initializer(0), trainable=False)

            #train_op = tf.train.AdagradOptimizer(0.01).minimize(
            #    cross_entropy, global_step=global_step)

	    opt = tf.train.AdagradOptimizer(0.01)

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

            # Test trained model
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.initialize_all_variables()

        # Create a "Supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="/opt/tensor",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver = saver,
                                 global_step=global_step,
                                 save_model_secs=600)

        # The supervisor takes care of session initialization and restoring from
        # a checkpoint.
        sess = sv.prepare_or_wait_for_session(server.target)

        # Start queue runners for the input pipelines (if ang).
        #sv.start_queue_runners(sess)

    	if is_chief and issync == 1:
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_token_op)
        else:
            sv.start_queue_runners(sess=sess)

        # Loop until the supervisor shuts down (or 2000 steps have completed).
        step = 0
        while not sv.should_stop() and step < 2000:
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, loss_value = sess.run([train_op, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
	    g_step = int(tf.train.global_step(sess, global_step))
	    print("local step %d in task %d: global step = %d, loss= %.2f" % (step, FLAGS.task_index, g_step, loss_value))
	    step += 1

        print("done.")
        if FLAGS.task_index != 0:
            print("accuracy: %f" % sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                             y_: mnist.test.labels}))


if __name__ == "__main__":
    tf.app.run()
