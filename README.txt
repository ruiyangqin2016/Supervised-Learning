1. Requirement: 
		python3.7.1
2. Compile environment:
		Windows10
3. Setup: 
	1) on terminal, type in commends line by line
		pip3 install numpy
		pip3 install pandas
		pip3 install matplotlib
		pip3 install sklearn
		pip3 install pydotplus
	2) open */assignment1/titanic_train
		double click graphviz-2.38.msi
		choose default setting
		after installed, go to C:\Program Files (x86)\Graphviz2.38\bin
		add C:\Program Files (x86)\Graphviz2.38\bin to Path in environment variable
4. Running:
	1) python prelearning.py
		This is use to decide correlations between attributes. I leave the source code under */assignment1. Anyone who wants to see the correlations just need to open the source code and adjust the csv file and column numbers at data.iloc[;,x]. 'x' is column number
	2) My first analysis to Wine quality dataset in the folder 
		'wine_quality'. Once you go to this folder, you can do by 
		python DecisionTree.py
		python Boosting.py
		python KNN.py
		python NeuralNet.py
		python SVM.py
		Please open each python file and uncomment the functions that you want to execute
	3) My second group of codes in under folder 'heartDisease.' Please 
		using the same statement to run the five algorithms and please remember the uncomment whatever you wish to execute.