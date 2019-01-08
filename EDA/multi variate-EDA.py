#Multi Variant

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("D:/Data Science/Data")

titanic_train = pd.read_csv("titanic_train.csv")

#EDA
titanic_train.shape
titanic_train.info()

sns.FacetGrid(titanic_train, row="Survived", col="Sex").map(sns.countplot, "Pclass")
sns.FacetGrid(titanic_train, row="Survived", col="Sex").map(sns.kdeplot, "Fare")
sns.FacetGrid(titanic_train, row="Survived", col="Sex").map(sns.kdeplot, "Age")

sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived").map(sns.kdeplot, "Age").add_legend()

sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived").map(plt.scatter, "Parch", "SibSp").add_legend()

sns.FacetGrid(titanic_train, row="Survived", col="Sex").map(plt.scatter, "Pclass", "SibSp", "Parch")
