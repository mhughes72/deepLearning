import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

census = pd.read_csv('census_data.csv')


gender = pd.get_dummies(census['gender'],drop_first=True)
census.drop(['gender'],axis=1,inplace=True)
census = pd.concat([census,gender],axis=1)

income_bracket = pd.get_dummies(census['income_bracket'],drop_first=True)
census.drop(['income_bracket'],axis=1,inplace=True)
census = pd.concat([census,income_bracket],axis=1)

print(census.columns)

age = tf.feature_column.numeric_column('age')
age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70])
# workclass = tf.feature_column.numeric_column('workclass')
workclass_bucket = tf.feature_column.categorical_column_with_vocabulary_list('workclass',['State-gov',
                                    'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov',
                                    '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])
education_bucket = tf.feature_column.categorical_column_with_vocabulary_list('education',['Bachelors', ' HS-grad',
                                    '11th', 'Masters', '9th', 'Some-college',
                                    'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
                                    '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
education_num = tf.feature_column.numeric_column('education_num')
marital_status = tf.feature_column.categorical_column_with_vocabulary_list('marital_status',['Never-married',
                                    'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated',
                                    'Married-AF-spouse', 'Widowed'])
occupation = tf.feature_column.categorical_column_with_vocabulary_list('occupation',['Adm-clerical',
                                    'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service',
                                    'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing',
                                    'Machine-op-inspct', 'Tech-support', '?', 'Protective-serv', 'Armed-Forces',
                                    'Priv-house-serv'])
relationship = tf.feature_column.categorical_column_with_vocabulary_list('relationship',['Not-in-family',
                                    'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])
race = tf.feature_column.categorical_column_with_vocabulary_list('race',['White', 'Black',
                                    'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
gender = tf.feature_column.numeric_column(' Male')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')
native_country = tf.feature_column.categorical_column_with_vocabulary_list('race',['United-States', 'Cuba',
                                    'Jamaica', 'India', '?', 'Mexico', 'South', 'Puerto-Rico', 'Honduras', 'England',
                                    'Canada', 'Germany', 'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia',
                                    'Cambodia', 'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
                                    'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'China', 'Japan',
                                    'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
                                    'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands'])
income_bracket = tf.feature_column.numeric_column(' >50K')

feat_cols = ['age', 'workclass', 'education', 'education_num', 'marital_status',
       'occupation', 'relationship', 'race', 'gender', 'capital_gain',
       'capital_loss', 'hours_per_week', 'native_country', 'income_bracket']

print(census.columns)
#
# print(census['gender'])

X_data = census.drop(' >50K',axis=1)
y_data = census[' >50K']

x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state = 101)


print (x_train)
print(y_test)
input_func = tf.estimator.inputs.pandas_input_fn(x=x_train,y=x_test ,batch_size=10,num_epochs=1000,shuffle=True)
model = tf.estimator.DNNRegressor(hidden_units=[1,1,1],feature_columns=feat_cols)
model.train(input_fn=input_func,steps=25000)

predict_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)

pred_gen = model.predict(predict_input_func)
predictions = list(pred_gen)
# print(predictions)

# census['income_bracket'].hist(bins=20)
# plt.show()
