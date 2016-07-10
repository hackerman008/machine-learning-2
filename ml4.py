#program to classify iris data set using a KNN classifier and check the accuracy

from sklearn.datasets import load_iris
import numpy as np
import random
from sklearn import metrics
from scipy.spatial import distance
#from sklearn import tree
#from sklearn import neighbors


def euc(a,b):
	return distance.euclidean(a,b)

class ScrappyKNN():
	"""docstring for scrappyknn"""
	def fit(self,X_train,y_train):
		#super(scrappyknn, self).__init__()
		self.X_train=X_train
		self.y_train=y_train

	def predict(self,X_test):
		predictionss=[]
		for row in X_test:
			label=self.closest(row)
			predictionss.append(label)
		return predictionss

	def closest(self,row):
		distance1=euc(row,self.X_train[0])
		distance_index=0
		for i in range(1,len(X_train)):
			temp_distance=euc(row,self.X_train[i])
			if temp_distance<distance1:
				distance1=temp_distance
				distance_index=i
		return self.y_train[distance_index]


iris=load_iris()

#load data in X and target value in y
X=iris.data
y=iris.target

#import library to split the data
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.5)

clf=ScrappyKNN()
#print(type(clf))
clf.fit(X_train,y_train)
predictions=clf.predict(X_test)
score=metrics.accuracy_score(y_test,predictions)
print("accurace=%f"%score)


#print(test_data)
#print(test_target)
#print(clf.predict(test_data[0::]))
#print(type(clf))

#print(X_test)
#print(y_test)
#print(predictions)
