# -*- coding: utf-8 -*-
"""
用Bi-LSTM 进行分类
"""
 
import tensorflow as tf
import numpy as np
 
 
#加载测试数据的读写工具包，加载测试手写数据，目录MNIST_data是用来存放下载网络上的训练和测试数据的。
#可以修改 “/tmp/data”，把数据放到自己要存储的地方
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("/home/archlab/tanw/JOB/BiRNN/data", one_hot=True)
 
 
#设置了训练参数
learning_rate = 0.01
max_samples = 400000
batch_size =128 
display_step = 10
 
#MNIST图像尺寸为28*28，输入n_input 为28，
#同时n_steps 即LSTM的展开步数，也为28.
#n_classes为分类数目
n_input = 28
n_steps = 28
n_hidden =128 #LSTM的hidden是什么结构
n_classes =10
 
x=tf.placeholder("float",[None,n_steps,n_input])
y=tf.placeholder("float",[None,n_classes])
 
#softmax层的weights和biases 
#双向LSTM有forward 和backwrad两个LSTM的cell，所以wights的参数数量为2*n_hidden
weights = {
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}
#定义了Bidirectional LSTM网络的生成
#
 
def BiRNN(x,weights,biases):
    print("x shape:", x.shape)
    x = tf.transpose(x,[1,0,2]) 
    x = tf.reshape(x,[-1,n_input])
    print(x.shape)
    x = tf.split(x,n_steps)
    
    #修改添加了作用域
    with tf.variable_scope('forward'):
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden ,forget_bias=1.0)
    with tf.variable_scope('backward'):
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden ,forget_bias=1.0)
    with tf.variable_scope('birnn'):
        outputs,_,_ =tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype=tf.float32)
    #outputs = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1],weights['out'])+biases['out']
 
pred =BiRNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits =pred,labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
 
init = tf.global_variables_initializer()
 
with tf.Session() as sess:
    sess.run(init)
    step =1 
    while step* batch_size <max_samples :
        batch_x ,batch_y =mnist.train.next_batch(batch_size)
        batch_x =batch_x.reshape((batch_size,n_steps,n_input))
        
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if step % display_step ==0:
            acc = sess.run(accuracy , feed_dict={x:batch_x,y:batch_y})
            loss = sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            print("Iter" + str(step*batch_size)+",Minibatch Loss = "+\
                  "{:.6f}".format(loss)+", Training Accuracy = "+ \
                  "{:.5f}".format(acc))
        step+=1
        
    print("Optimization Finished!")
    
    test_len = 10000
    test_data = mnist.test.images[:test_len].reshape((-1,n_steps,n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:",
          sess.run(accuracy,feed_dict={x:test_data,y:test_label}))
