import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
dataset = pd.read_csv("C:/Users/HP/OneDrive/Desktop/data/1_deepLearning_files/Churn_Modelling.csv")
#%%
# first 3 columns won't affect the output, better not take them
x = dataset.iloc[:,3:13]
y = dataset.iloc[:,13]
#%%

# creating dummy variables for gender and geography.
x = pd.get_dummies(x, columns=['Gender','Geography'],drop_first=True)
#%%
# splitting dataset into train and test set.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#%%
# Feature scaling: 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#%%
# importing libraries for neural network
import tensorflow

#%%
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#%%
#initializing the ANN
classifier = Sequential()
#%%
# adding input layer and the first hidden layer(unit).
# Here, weight initialization technique is 'he_uniform'/he_normal
# and we used 'relu' as activation function.
classifier.add(Dense(units=10,kernel_initializer='he_normal',activation='relu',input_dim=11))
#let's add dropout to randomly select units*probability neurons.
classifier.add(Dropout=0.3)
#%%
# adding second hidden layer.
classifier.add(Dense(units=20,kernel_initializer='he_normal',activation='relu'))
classifier.add(Dropout=0.4)
#%%
# adding third hidden layer.
classifier.add(Dense(units=15,kernel_initializer='he_normal',activation='relu'))
classifier.add(Dropout=0.2)
#%%
#adding the output layer.
# for output, we need to convert ans to 0 to 1, so we used sigmoid activation function.
# glorort weight initilization method works well for sigmoid activation function.
classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))
#%%
#compiling the ANN
# since we have 0/1 as output (binary), we used binary_crossentropy as loss function.
# and we used our best optimizer-adam.
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#%%
model_history = classifier.fit(x_train,y_train,validation_split=0.33,batch_size=10,epochs=100)
#%%
# prediction on the test data
y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)

#%%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#%%
#calculate the accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred,y_test)
score
#%%
"""
1. First tried with different 2 hidden layers each with 6 neurons and weight as he_uniform.
2. Then added 3rd hidden layer and changed the no of neurons in each to 10,20, 15 resp.
3. Also changed the weight init method from he-uniform to normal.
4. Finally addded a dropout probablity to randomly activate/deactivate neurons.
5. Got Y-pred, confusion matrix, accuracy on training, validation and test dataset.
"""





