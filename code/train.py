
import json
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATASET = input_data.read_data_sets("data/", one_hot=True)
OUT = "models/mnist"

batch_size = 128
num_steps = 10000
start = time.time()

# input
x = tf.placeholder(tf.float32, [None, 784], "x")

# weight
W = tf.Variable(tf.zeros([784, 10]))
# bias
b = tf.Variable(tf.zeros([10]))
# test_data * W + b
y = tf.matmul(x, W) + b
sm = tf.nn.softmax(y)

# cross entropy (loss function)
y_ = tf.placeholder(tf.float32, [None, 10], "y")
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_), name="loss")

# train step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# evaluating the model
correct_prediction = tf.equal(tf.argmax(sm, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

# init
saver = tf.train.Saver()
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)

    # training
    for step in xrange(num_steps):
        batch_data, batch_labels = DATASET.train.next_batch(batch_size)
        
        error, ts, acc = session.run([loss, train_step, accuracy], 
            feed_dict={x: batch_data, y_: batch_labels})       

    save_path = saver.save(session, OUT)

    with open('metrics/train.json', 'w') as outfile:
        json.dump({ "took" : (time.time() - start) / 1000 }, outfile)
