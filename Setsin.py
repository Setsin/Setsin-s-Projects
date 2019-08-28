# Setsin-s-Projects
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 02:05:01 2019

@author: Setsin
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:, 13].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Partie 2 - Construire le réseau de neuronne

#Preparation des modules de keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initiation
classifier = Sequential ()

# Ajouter la couche d'entree et une couche cachee
classifier.add(Dense(units=6, activation="relu",
                     kernel_initializer="uniform", input_dim=11 ))


#Ajouter une couche cachee

classifier.add(Dense(units=6, activation="relu",
                     kernel_initializer="uniform"))

#Ajouter la couche de sortie

classifier.add(Dense(units=1, activation="sigmoid",
                     kernel_initializer="uniform"))
# Compiler le réseau de neurones

classifier.compile(optimizer="adam", loss="binary_crossentropy", 
                   metrics=['accuracy'])

# Entrainer le réseau de neurones

classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Predire une observation seule

"""Pays : France
Score de crédit : 600
Genre : Masculin
Âge : 40 ans
Durée depuis entrée dans la banque : 3 ans
Balance : 60000 €
Nombre de produits : 2
Carte de crédit ? Oui
Membre actif ? : Oui
Salaire estimé : 50000 €"""

new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000,
                                               2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    
    classifier = Sequential ()
    classifier.add(Dense(units=6, activation="relu",
                     kernel_initializer="uniform", input_dim=11 ))
    classifier.add(Dense(units=6, activation="relu",
                     kernel_initializer="uniform"))
    classifier.add(Dense(units=1, activation="sigmoid",
                     kernel_initializer="uniform"))
    classifier.compile(optimizer="adam", loss="binary_crossentropy", 
                   metrics=['accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
precisions = cross_val_score(estimator=classifier, X=X_train,y=y_train, cv=10)

moyenne = precisions.mean()
ecart_type = precisions.std()

#  Importation des modules

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Fonction de construction
def build_classifier(optimizer):
    
    classifier = Sequential ()
    classifier.add(Dense(units=6, activation="relu",
                     kernel_initializer="uniform", input_dim=11 ))
    classifier.add(Dense(units=6, activation="relu",
                     kernel_initializer="uniform"))
    classifier.add(Dense(units=1, activation="sigmoid",
                     kernel_initializer="uniform"))
    classifier.compile(optimizer="adam", loss="binary_crossentropy", 
                   metrics=['accuracy'])
    
    return classifier

#k-fold cross validation
    classifier = KerasClassifier(build_fn=build_classifier)
    parameters = {"batch_size":[25, 32],
                  "epochs": [200, 500],
                  "optimizer":["adam", "rmsprop"]}
    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=parameters,
                               scoring="accuracy",
                               cv=10)
    grid_search = grid_search.fit(X_train, y_train)
    
    best_parameters = grid_search.best_params_
    best_precision = grid_search.best_score_
    
