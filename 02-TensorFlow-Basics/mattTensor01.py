import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_data = np.linspace(0.0,10.0,1000000)
noise = np.random.normal(0,200,len(x_data))

b = 5
y_true =  (20 * x_data ) + 5 + noise
my_data = pd.concat([pd.DataFrame(data=x_data,columns=['X Data']),pd.DataFrame(data=y_true,columns=['Y'])],axis=1)

batch_size = 8

m = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])

#GRAPH
y_model = m*xph + b

error = tf.reduce_sum(tf.square(yph-y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batches = 1000
    for i in range(batches):
        rand_ind = np.random.randint(len(x_data),size=batch_size)
        feed = {xph:x_data[rand_ind],yph:y_true[rand_ind]}
        sess.run(train,feed_dict=feed)
    model_m,model_b = sess.run([m,b])

print(model_m)
print(model_b)
y_hat = x_data * model_m + model_b
my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')
# my_data.plot(kind='scatter',x='X Data',y='Y')
plt.plot(x_data,y_hat,'r')
plt.show()
