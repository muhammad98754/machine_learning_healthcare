#Deep learning algorithm to predict the stage of breast cancer

import numpy as np   # tools for numerical computing
import pandas as pd  #  for data manipulation and analysis.
import matplotlib.pyplot as plt  # enabling data visualization and plotting capabilities.

from sklearn import preprocessing #machine learning library
from subprocess import check_output

#read the datafile
data=pd.read_csv("1000_Companies.csv")
print(data.head()) #printing first 5 rows of dataset


#important
"""the variable in the dataset called 'diagnosis' has values M and B.
               if value is M then the cell is cancer cell and not otherwise.
               this variable value is what we are going to predict"""
#cleaning and modifying data (unnecessary columns we can remove which has no role in final output)
#data=data.drop("id",axis=1)
#data=data.drop("unnamed: 32",axis=1)

#mapping cancer cells to numeric value 1 and non cancer cells to value 0
data["diagnosis"] = data["diagnosis"].map({'M':1,"B":0})

#scale the dataset
datas=pd.DataFrame(preprocessing.scale(data.iloc[:,1:32]))
datas.columns=list(data.iloc[:,1:32].columns)
datas["diagnosis"] = data["diagnosis"]

#getting rid of the variable "diagnosis" which we want to predict
data_drop=datas.drop("diagnosis",axis=1)
X=data_drop.values

#create a feed forward neural network with3 hidden layers
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input
from keras.optimizers import SGD

model=Sequential() #basically a feed forward neural network
model.add(Dense(128,activation='relu',input_dim=np.shape(X)[1]))
model.add(Dropout(0.25))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='binary crossentropy',optimizer=sgd,metrices=['accuracy'])

#fit and test the model by randomly splitting it
#67% data for training and 33% for data validation

model.fit(X,datas['diagnosis'],batch_size=5,epochs=10,validation_split=0.33)

#cross validation analysis
from sklearn.model_selection import StratifiedKFold
#fix random seed for reproducibility
seed=3
np.random.seed(seed)
# K fold cross validation (k=2)
k=2
kfold = StratifiedKFold(n_splits=2,shuffle=True,random_state=seed)
cvscores = []
Y=datas['diagnosis']

for train,test  in kfold.split(X,Y):
    #fit the model
    model.fit(X[train],Y[train],epochs=10,batch_size=10,verbose=0)
    #evaluate the model
    scores=model.evaluate(X[test],Y[test],verbose=0)
    #print scores from each cross validation run
    print("%s:%.2f%%" %(model.metrics_names[1],scores[1]*100))
    cvscores.append(scores[1]*100)
