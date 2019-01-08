
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Social_Network_Ads.csv')

X=dataset.iloc[:,[2,3]].values#independent variables
y=dataset.iloc[:,4].values#dependent variables
 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)
# X_train is training data randomly selected 75% of the given data,X_test = remaining 25%of data,
# y_train is the predicted value(result i.e., Yes or no) wrt X_train values, y_test is predicted value(result i.e., Yes or no) wrt X_test values

from sklearn.preprocessing import StandardScaler#StandardScaler is a class
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)#age is in 2 digits and salary is a 6digit no. therefore, we scale them to compare efficiently
#fit-->creating an object model for fitting your data wheras the tansform is going to transform the data to the object you have created.
X_test=sc_x.transform(X_test)

#DecisionTree Suffers the Overfitting that is why it gives less error 
#Fitting Decision Tree Classification to Training Set
from sklearn.tree import DecisionTreeClassifier #DecisionTreeClassifier is a class 
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


#Visualizing the Training Data
from matplotlib.colors import ListedColormap
X_set, y_set = X_test,y_test # can use X_train and y_train also
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step=0.01), np.arange(start = X_set[:,1].min() -1 , stop=X_set[:,1].max() + 1, step=0.01))

plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.xlim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):
plt.scatter(X_set[y_set == j,0], X_set[y_set == j, 1], c= ListedColormap(('red', 'green'))(i), label = j)
plt.title('DecisionTree(Training Set)')
plt.xlabel('Age')
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
#Visualizing the Test Data
X_set, y_set = X_test,y_test # can use X_train and y_train also
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step=0.01), np.arange(start = X_set[:,1].min() -1 , stop=X_set[:,1].max() + 1, step=0.01))

plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.xlim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
    plt.title('DecisionTree(Testset)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()




