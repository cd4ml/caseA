
import json
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATASET = input_data.read_data_sets("data/", one_hot=True)
OUT = "models/mnist"

init = tf.global_variables_initializer()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('models/mnist.meta')
    saver.restore(sess, tf.train.latest_checkpoint('models/'))

    graph = tf.get_default_graph()
    print("Model restored.")

    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")

    acc = accuracy.eval(feed_dict={ x: DATASET.test.images, y: DATASET.test.labels })
    
    with open('metrics/eval.json', 'w') as outfile:
        json.dump({ "accuracy" : str(acc) }, outfile)
