#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

url1 = 'https://raw.githubusercontent.com/lsharples1/Data_Science_Competition/main/Simulated_Data_Train.csv?token=AO4LMYK744PF4GAEXCQHBZDAPSKSM'
train = pd.read_csv(url1)

url2 = 'https://raw.githubusercontent.com/lsharples1/Data_Science_Competition/main/Simulated_Data_Test.csv'
test = pd.read_csv(url2)

url3 = 'https://raw.githubusercontent.com/lsharples1/Data_Science_Competition/main/Simulated_Data_Validation.csv?token=AO4LMYMGZVCQVZJ6P6TBBS3APSKXS'
validate = pd.read_csv(url3)


y_Train = train['Default_ind']
x_Train = train
x_Train.drop('Default_ind', axis='columns',inplace=True)
x_Train['uti_card_50plus_pct'].fillna((x_Train['uti_card_50plus_pct'].mean()), inplace=True) 
x_Train['rep_income'].fillna((x_Train['rep_income'].mean()), inplace=True) 
x_Train = pd.get_dummies(x_Train, prefix_sep='_', drop_first=False)
scaler = preprocessing.StandardScaler().fit(x_Train)
x_TrainScale = scaler.transform(x_Train)
smote = SMOTE()

y_Test = test['Default_ind']
x_Test = test
x_Test.drop('Default_ind', axis='columns',inplace=True)
x_Test = pd.get_dummies(x_Test, prefix_sep='_', drop_first=False)
x_Test['uti_card_50plus_pct'].fillna((x_Test['uti_card_50plus_pct'].mean()), inplace=True) 
x_Test['rep_income'].fillna((x_Test['rep_income'].mean()), inplace=True) 
scaler2 = preprocessing.StandardScaler().fit(x_Test)
x_TestScale = scaler2.transform(x_Test)
y_Val = validate['Default_ind']
x_Val = validate
x_Val.drop('Default_ind', axis='columns',inplace=True)
x_Val = pd.get_dummies(x_Val, prefix_sep='_', drop_first=False)
x_Val['uti_card_50plus_pct'].fillna((x_Val['uti_card_50plus_pct'].mean()), inplace=True) 
x_Val['rep_income'].fillna((x_Val['rep_income'].mean()), inplace=True) 
scaler3 = preprocessing.StandardScaler().fit(x_Val)
x_ValScale = scaler3.transform(x_Val)




model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.L2(0.01), activation='relu'))
model.add(tf.keras.layers.Dense(12, kernel_regularizer=tf.keras.regularizers.L2(0.01), activation='relu'))
model.add(tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.L2(0.01), activation='sigmoid'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_Train, y_Train, epochs=30)
score = model.evaluate(x_Train, y_Train, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.summary())

predictions = np.argmax(model.predict(x_Test), axis=-1)
print(predictions)
r_squared = r2_score(y_Test, predictions)
print(y_Test[0])
print(x_Test)

print(classification_report(y_Test, predictions))

# summarize the first 5 cases
plt.scatter(y_Test, predictions)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.plot(np.unique(y_Test), np.poly1d(np.polyfit(y_Test, predictions, 1))(np.unique(y_Test)))

plt.text(0.6, 0.5, 'R-squared = %0.2f' % r_squared)
plt.show()
adj = [0]
aicList = [0]
y = [y_Test[0]]
z = [predictions[0]]
def plot(classifier, x, y, title): 
    class_names = ["Not Defaulted", "Defaulted"]
    disp = plot_confusion_matrix(classifier, x, y,
                                 display_labels=class_names,
                                 cmap=plt.cm.PuRd, values_format = ''
                                   )
    disp.ax_.set_title(title)
    plt.show()
 


# In[2]:


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.inspection import permutation_importance
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE

url1 = 'https://raw.githubusercontent.com/lsharples1/Data_Science_Competition/main/Simulated_Data_Train.csv?token=AO4LMYK744PF4GAEXCQHBZDAPSKSM'
train = pd.read_csv(url1)

url2 = 'https://raw.githubusercontent.com/lsharples1/Data_Science_Competition/main/Simulated_Data_Test.csv'
test = pd.read_csv(url2)

url3 = 'https://raw.githubusercontent.com/lsharples1/Data_Science_Competition/main/Simulated_Data_Validation.csv?token=AO4LMYMGZVCQVZJ6P6TBBS3APSKXS'
validate = pd.read_csv(url3)


y_Train = train['Default_ind']
x_Train = train
x_Train.drop('Default_ind', axis='columns',inplace=True)
x_Train['uti_card_50plus_pct'].fillna((x_Train['uti_card_50plus_pct'].mean()), inplace=True) 
x_Train['rep_income'].fillna((x_Train['rep_income'].mean()), inplace=True) 
x_Train = pd.get_dummies(x_Train, prefix_sep='_', drop_first=False)
scaler = preprocessing.StandardScaler().fit(x_Train)
x_TrainScale = scaler.transform(x_Train)
smote = SMOTE()

y_Test = test['Default_ind']
x_Test = test
x_Test.drop('Default_ind', axis='columns',inplace=True)
x_Test = pd.get_dummies(x_Test, prefix_sep='_', drop_first=False)
x_Test['uti_card_50plus_pct'].fillna((x_Test['uti_card_50plus_pct'].mean()), inplace=True) 
x_Test['rep_income'].fillna((x_Test['rep_income'].mean()), inplace=True) 
scaler2 = preprocessing.StandardScaler().fit(x_Test)
x_TestScale = scaler2.transform(x_Test)
y_Val = validate['Default_ind']
x_Val = validate
x_Val.drop('Default_ind', axis='columns',inplace=True)
x_Val = pd.get_dummies(x_Val, prefix_sep='_', drop_first=False)
x_Val['uti_card_50plus_pct'].fillna((x_Val['uti_card_50plus_pct'].mean()), inplace=True) 
x_Val['rep_income'].fillna((x_Val['rep_income'].mean()), inplace=True) 
scaler3 = preprocessing.StandardScaler().fit(x_Val)
x_ValScale = scaler3.transform(x_Val)


model = GaussianNB()

# fit the model with the training data
model.fit(x_Train,y_Train)

# predict the target on the train dataset
predict_train = model.predict(x_Train)
print('Target on train data',predict_train) 

# Accuray Score on train dataset
accuracy_train = accuracy_score(y_Train,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = model.predict(x_Test)
print('Target on test data',predict_test) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_Test,predict_test)
print('accuracy_score on test dataset : ', accuracy_test)


print("R2")
print(r2_score(y_Test, predict_test))

print(classification_report(y_Test, predict_test))

def plot(classifier, x, y, title): 
    class_names = ["Not Defaulted", "Defaulted"]
    disp = plot_confusion_matrix(classifier, x, y,
                                 display_labels=class_names,
                                 cmap=plt.cm.PuRd, values_format = ''
                                   )
    disp.ax_.set_title(title)
    plt.show()
    
plot(model, x_Test, y_Test, "Naive Bayes Model-scaled")

imps = permutation_importance(model, x_Test, y_Test)
print(imps.importances_mean)


# In[ ]:




