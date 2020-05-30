import pandas as pd
import numpy as np
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

titanic_df = pd.read_csv("train (1).csv")


titanic_df["Sex"].replace(to_replace="male",value=1,inplace=True)
titanic_df["Sex"].replace(to_replace="female",value=0,inplace=True)

dummie=pd.get_dummies(titanic_df["Embarked"])
titanic_df = pd.concat([titanic_df,dummie],axis=1)
titanic_df.drop(["Embarked"], inplace=True, axis=1)

titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df["C"]=titanic_df["C"].astype(np.int64)
titanic_df["Q"]=titanic_df["Q"].astype(np.int64)
titanic_df["S"]=titanic_df["S"].astype(np.int64)
titanic_df["Fare"]=titanic_df["Fare"].astype(np.int64)
titanic_df["Age"]=titanic_df["Age"].astype(int)

X = titanic_df[["Pclass","Sex","Age","SibSp","Parch","Fare","C","Q","S"]]
Y = titanic_df["Survived"]

titanic_df.drop(["Cabin"],inplace=True,axis=1)


log_reg = LogisticRegression()

X_eğitim, X_test, Y_eğitim, Y_test =  train_test_split(X, Y, test_size=0.2, random_state=112)

log_reg.fit(X_eğitim, Y_eğitim)

egitim_dogruluk = log_reg.score(X_eğitim, Y_eğitim)
test_dogruluk = log_reg.score(X_test, Y_test)
print('One-vs-rest', '-'*20,
      'Modelin eğitim verisindeki doğruluğu : {:.2f}'.format(egitim_dogruluk),
      'Modelin test verisindeki doğruluğu   : {:.2f}'.format(test_dogruluk), sep='\n')

# eğitim ve test verilerindeli doğruluk yaklaşık yüzde 80. Tatmin edici mi? çok fazla değil ama yine de iyi sayılır.

