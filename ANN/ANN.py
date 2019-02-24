import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Bank_Data.csv")
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values
dataset.head(3)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 =LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])
X

onehotencoder = OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X

X= X[:,1:]#To avoid the dummy variable Trap
X

#splitting the data set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim =6,init='uniform',activation ='relu',input_dim=11))
#adding the second hidden layer
classifier.add(Dense(output_dim =6,init='uniform',activation ='relu'))

#adding the output layer
classifier.add(Dense(output_dim =1,init='uniform',activation ='sigmoid'))

#compliling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#a gradient descent techinique is used  using the keyword adam
#loss - 
#metrics-

#fitting the ANN to training set
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)

#Making the predictions and evaluating the model
#predicting the test reults
y_pred =classifier.predict(X_test)
y_pred = (y_pred>0.5)
y_pred

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm
