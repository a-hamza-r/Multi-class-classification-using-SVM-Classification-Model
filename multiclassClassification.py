from scipy.io import loadmat;
import os;
from os import listdir;
from os.path import join, isfile;
import re;
from sklearn.svm import SVC;
import pandas as pd;
from sklearn.metrics import accuracy_score;
import numpy as np;

class Data():
	
	def __init__(self):
		self.test = {};
		self.train = {};

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

def SVClassification(data, kernel, degree, coef0, gamma):
	numOfClasses = len(data.train["Labels"].columns);
	predictions = [];
	for classType in range(numOfClasses):
		clt = SVC(kernel=kernel, degree=degree, coef0=coef0, gamma=gamma);
		clt.fit(data.train["Matrix"], data.train["Labels"][classType]);
		prediction = clt.predict(data.test["Matrix"]);
		predictions.append(prediction);

	print(accuracy_score(np.array(predictions), data.test["Labels"].T.to_numpy()));
	print(np.array(predictions));
	print(data.test["Labels"].T.to_numpy());
	T = 0
	P = 0
	for x in range(len(predictions[0])):
		for y in range(len(predictions)):
			if predictions[y][x] == data.test["Labels"].T.to_numpy()[y][x] and predictions[y][x] == 1:
				T += 1;
			if predictions[y][x] == 1 or data.test["Labels"].T.to_numpy()[y][x] == 1:
				P += 1;

	print(T/P)


def main():
	allData = Data();
	allData.loadData();
	SVClassification(allData, 'poly', 2, 1, 1);
	SVClassification(allData, 'rbf', 1, 1, 0.125);
	
	## ask about parameters of both kernels
	## ask about the accuracy score

if __name__ == '__main__':
	main();
