import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_data = np.linspace(0.0,10.0,10000)
noise = np.random.randn(len(x_data))

y_true =  (0.5 * x_data ) + 5 + noise

my_data = pd.concat([pd.DataFrame(data=x_data,columns=['X Data']),pd.DataFrame(data=y_true,columns=['Y'])],axis=1)
# my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')
# plt.show()
feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

x_train, x_eval, y_train, y_eval = train_test_split(x_data,y_true,test_size=0.3, random_state = 101)


input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=4,num_epochs=None,shuffle=True)
train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=4,num_epochs=1000,shuffle=False)
eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_eval},y_eval,batch_size=4,num_epochs=1000,shuffle=False)

estimator.train(input_fn=input_func,steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_func,steps=1000)
eval_metrics = estimator.evaluate(input_fn=eval_input_func,steps=1000)

print("train metrics: {}".format(train_metrics))
print("eval metrics: {}".format(eval_metrics))

predNums=np.linspace(10,100,10)

input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':predNums},shuffle=False)
list(estimator.predict(input_fn=input_fn_predict))

predictions = []# np.array([])
for x in estimator.predict(input_fn=input_fn_predict):
    predictions.append(x['predictions'])

my_data.sample(n=2500).plot(kind='scatter',x='X Data',y='Y')
plt.plot(np.linspace(0,10,10),predictions,'*r')
plt.show()
