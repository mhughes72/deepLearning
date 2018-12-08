import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

housing = pd.read_csv('cal_housing_clean.csv')

print(housing.head())
cols_to_norm = ['totalRooms', 'totalBedrooms', 'population', 'households', 'medianIncome', 'medianHouseValue']
housing[cols_to_norm] = housing[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

num_housingMedianAge = tf.feature_column.numeric_column('housingMedianAge')
age_buckets = tf.feature_column.bucketized_column(num_housingMedianAge, boundaries=[20,30,40,50])
num_totalRooms = tf.feature_column.numeric_column('totalRooms')
num_totalBedrooms = tf.feature_column.numeric_column('totalBedrooms')
num_population = tf.feature_column.numeric_column('population')
num_households = tf.feature_column.numeric_column('households')
num_medianIncome = tf.feature_column.numeric_column('medianIncome')

feat_cols = [age_buckets, num_totalRooms, num_totalBedrooms, num_population, num_households, num_medianIncome]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

y_true =  housing['medianHouseValue']
x_data = housing.drop('medianHouseValue',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_true, test_size=0.3, random_state = 101)

input_func = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)

model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)
model.train(input_fn=input_func,steps=1000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=x_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)

results = model.evaluate(eval_input_func)
print(results)

pred_input_func = tf.estimator.inputs.pandas_input_fn(
      x=x_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)

predictions = model.predict(pred_input_func)

# dnn_model.train(input_fn=input_func,steps=1000)

# my_data.sample(n=2500).plot(kind='scatter',x='X Data',y='Y')
plt.plot(np.linspace(0,10,10),predictions,'*r')
plt.show()

# housing['housingMedianAge'].hist(bins=20)
# plt.show()
