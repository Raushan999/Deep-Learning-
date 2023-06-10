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
# importing libraries for hyperparamter tuning
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense,Activation,Embedding,Flatten,LeakyReLU,BatchNormalization
from keras.activations import relu, sigmoid
#%%
# here we will use grid search for finding the best set of parameteres.
# IN grid search we provide a list of parameters and the model is trained on 
# all possible combinations and give the best set of parameters.
from sklearn.model_selection import GridSearchCV
#%%
# Function for gridsearch on all possible combinations of layer and activation function.
def create_model(layers,activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0: # Input Layer.
            model.add(Dense(nodes,input_dim=x_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else: # Hidden Layers.
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
    # output layer.
    model.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model
#%%
# calling the classifier
model = KerasClassifier(build_fn=create_model,verbose=0)

#%%
# list of layers,activation,bath-size..
hidden_layers = [[20],[40,20],[12,21,11]]
activations = ['sigmoid','relu']
param_grid=dict(layers=hidden_layers,activation=activations,batch_size=[100,140],epochs=[20])
grid=GridSearchCV(estimator=model, param_grid=param_grid,cv=5) 

#%%
# fitting the training set on all the combinations of model.
grid_result=grid.fit(x_train,y_train)
#%%
print(grid_result.best_score_,grid_result.best_params_)
##  0.8518749952316285 {'activation': 'relu', 'batch_size': 100, 'epochs': 20, 'layers': [40, 20]}
#%%
# test set prediction
y_pred  = grid.predict(x_test)
y_pred = (y_pred>0.5)
#%%
# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
#%%
#printing the accuracy for test data.
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print(score)
#%%
