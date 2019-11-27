from scipy.io import loadmat;
import os;
from os import listdir;
from os.path import join, isfile;
import re;
from sklearn.svm import SVC;
import pandas as pd;
from sklearn.metrics import accuracy_score;
import numpy as np;


## Stores the data required to run SVM
class Data():
	
	def __init__(self):
		self.test = {};
		self.train = {};

	## Load data from all the files in Data directory
	def loadData(self):
		dataDirectory = "Data";
		dataFiles = [f for f in listdir(dataDirectory) if isfile(join(dataDirectory, f))];
		for file in dataFiles:
			loaded = loadmat(join(dataDirectory, file));
			file1 = file.replace("X", "Matrix").replace("y", "Labels");
			typeFile = re.split("_|\.", file1)[:2];
			if typeFile[1] == "train":
				self.train[typeFile[0]] = pd.DataFrame(loaded[file[0]+"_"+typeFile[1]]);
			elif typeFile[1] == "test":
				self.test[typeFile[0]] = pd.DataFrame(loaded[file[0]+"_"+typeFile[1]]);


## SV Classifier
def SVClassification(data, kernel, degree, coef0, gamma):
	numOfClasses = len(data.train["Labels"].columns);
	predictions = [];
	for classType in range(numOfClasses):
		clt = SVC(kernel=kernel, degree=degree, coef0=coef0, gamma=gamma);
		clt.fit(data.train["Matrix"], data.train["Labels"][classType]);
		prediction = clt.predict(data.test["Matrix"]);
		predictions.append(prediction);


	overallAccuracy = 0;

	for x in range(len(predictions[0])):
		positivePredictions = 0;
		anyValueOne = 0;
		for y in range(len(predictions)):
			if predictions[y][x] == data.test["Labels"].T.to_numpy()[y][x] and predictions[y][x] == 1:
				positivePredictions += 1;
			if predictions[y][x] == 1 or data.test["Labels"].T.to_numpy()[y][x] == 1:
				anyValueOne += 1;
		accuracy = positivePredictions/anyValueOne;
		overallAccuracy += accuracy;

	return overallAccuracy/len(predictions[0]);


def main():
	allData = Data();
	allData.loadData();

	# SVM with polynomial kernel, degree=2, coef0=1. (gamma=1 is ignored by SVC function)
	print("Accuracy for SVM with polynomial kernel, degree 2: " + str(SVClassification(allData, 'poly', 2, 1, 1)));

	# SVM with Gaussian kernel (kernel='rbf'), gamma=0.125 (because parameter/sigma=2, gamma=1/(2*(sigma)^2). sigma=2 => gamma=0.125
	# degree=1 and coef0=1 is ignored by SVM with rbf kernel
	print("Accuracy for SVM with rbf kernel, parameter 2: " + str(SVClassification(allData, 'rbf', 1, 1, 0.125)));
	

if __name__ == '__main__':
	main();
