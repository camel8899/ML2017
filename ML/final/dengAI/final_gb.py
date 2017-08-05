from __future__ import print_function
from __future__ import division
import os
import pandas as pd
import numpy as np
import random as rand
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score,KFold
import statsmodels.api as sm
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from datetime import datetime
rand.seed(datetime.now())
from warnings import filterwarnings
filterwarnings('ignore')

base_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(base_path,'data')
weight_path = os.path.join(base_path,'weight')
#########################################
########   Utility Function     #########
#########################################


def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    
    # select features we want
    '''
    features = ['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_dew_point_temp_k', 
                 'station_avg_temp_c', 
                 'station_min_temp_c',
                 'station_max_temp_c',
                 'reanalysis_min_air_temp_k',
                 'reanalysis_relative_humidity_percent'

                 ]
    df = df[features]
	'''
    df.drop('week_start_date', axis=1, inplace=True)
    
    # fill missing values
    df.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)
    
    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']
    
    return sj, iq

def gradient_boosting(train_data,val_data):
	params = {'n_estimators': 800, 'max_depth': 5, 'min_samples_split': 3,
	'learning_rate': 0.01, 'loss': 'ls'}
	clf = ensemble.GradientBoostingRegressor(**params)
	train_label = train_data['total_cases']
	train_feat = train_data.drop('total_cases',axis = 1)
	clf.fit(train_feat, train_label)
	predictions = clf.predict(train_feat)
	mae = eval_measures.meanabs(predictions, train_label)
	print("Training MAE: %.4f" % mae)

	val_label = val_data['total_cases']
	val_feat = val_data.drop('total_cases',axis = 1)
	val_predictions = clf.predict(val_feat)
	mae = eval_measures.meanabs(val_predictions, val_label)
	print("Validation MAE: %.4f" % mae)
	
	return clf
	
	

#####################################
######      Main Function      ######           
#####################################

def main():

### parsing and Data pre-processing
	# load the provided data
	train_features_path = os.path.join(data_path,'dengue_features_train.csv')
	train_labels_path = os.path.join(data_path,'dengue_labels_train.csv')
	train_features = pd.read_csv(train_features_path,index_col=[0,1,2])
	train_labels = pd.read_csv(train_labels_path,index_col=[0,1,2])
	# Seperate data for San Juan
	sj_train_features = train_features.loc['sj']
	sj_train_labels = train_labels.loc['sj']
	# Separate data for Iquitos
	iq_train_features = train_features.loc['iq']
	iq_train_labels = train_labels.loc['iq']

	# Remove 'week_start_date' string.
	sj_train_features.drop('week_start_date', axis=1, inplace=True)
	iq_train_features.drop('week_start_date', axis=1, inplace=True)

	#find NaN in data be unsatisfying and eliminate those ddata
	sj_train_features.fillna(method='ffill', inplace=True)
	iq_train_features.fillna(method='ffill', inplace=True)

	### pre-processing data
	sj_train, iq_train = preprocess_data(train_features_path, labels_path = train_labels_path)
	#print(sj_train.describe())
	#print(iq_train.describe())
	
	choose = rand.sample(range(0,sj_train.shape[0]-1),800)
	val = [i for i in range(sj_train.shape[0]) if i not in choose]
	sj_train_subtrain = sj_train.ix[choose]
	sj_train_subtest = sj_train.ix[val]    

	clf_sj = gradient_boosting(sj_train_subtrain,sj_train_subtest)

	choose = rand.sample(range(0,iq_train.shape[0]-1),400)
	val = [i for i in range(iq_train.shape[0]) if i not in choose]
	iq_train_subtrain = iq_train.ix[choose]
	iq_train_subtest = iq_train.ix[val]    

	#Use K-fold to create cross validation data
	
	kf = KFold(n_splits=12)	
	sj_score = []
	for train_index, test_index in kf.split(sj_train):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X_train,X_test = sj_train.ix[train_index], sj_train.ix[test_index]
		predictions = clf_sj.predict(X_test.drop('total_cases',axis = 1)).astype(int)
		'''
		for i in range(predictions.shape[0]-1,3,-1):
			predictions.ix[i] = predictions.ix[i-4]
		'''
		sj_score.append(eval_measures.meanabs(predictions, X_test.total_cases))
	#print(sj_score)	
	print("Mean of {} cross validation of sj_score is {} (+/- {})".format(kf.get_n_splits(sj_train)
																,np.mean(sj_score)
																,np.std(sj_score)))
	iq_score = []
	ans = [0 for i in range(len(iq_train['total_cases']))]
	print(len(ans))
	for train_index, test_index in kf.split(iq_train):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X_train,X_test = iq_train.ix[train_index], iq_train.ix[test_index]
		clf_iq = gradient_boosting(X_train,X_test)
		predictions = clf_iq.predict(X_test.drop('total_cases',axis = 1)).astype(int)
		for idx,index in enumerate(test_index):
			ans[index] = predictions[idx]	
		'''
		for i in range(predictions.shape[0]-1,0,-1):
			predictions.ix[i] = predictions.ix[i-1]
		'''
		iq_score.append(eval_measures.meanabs(predictions, X_test.total_cases))
	#print(iq_score)
	print("Mean of {} cross validation of iq_score is {} (+/- {})".format(kf.get_n_splits(iq_train)
																,np.mean(iq_score)
																,np.std(iq_score)))
	print(ans)
	figs, axes = plt.subplots(nrows=2, ncols=1)
	
	# plot sj
	#sj_train['fitted'] = sj_best_model.fittedvalues
	#sj_train.fitted.plot(ax=axes[0], label="Predictions")
	SJ_predictions = clf_sj.predict(sj_train.drop('total_cases',axis = 1)).astype(int)
	'''
	for i in range(SJ_predictions.shape[0]-1,3,-1):
			SJ_predictions.ix[i] = SJ_predictions.ix[i-4]
	'''
	axes[0].plot(SJ_predictions, label="Predictions")
	sj_train.total_cases.plot(ax=axes[0], label="Actual")

	# plot iq
	#iq_train['fitted'] = iq_best_model.fittedvalues
	#iq_train.fitted.plot(ax=axes[1], label="Predictions")
	IQ_predictions = clf_iq.predict(iq_train.drop('total_cases',axis = 1)).astype(int)
	'''
	for i in range(IQ_predictions.shape[0]-1,0,-1):
			IQ_predictions.ix[i] = IQ_predictions.ix[i-1]
	'''
	axes[1].plot(IQ_predictions, label="Predictions")
	iq_train.total_cases.plot(ax=axes[1], label="Actual")

	plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
	plt.legend()
	plt.show()

	test_features_path = os.path.join(data_path,'dengue_features_test.csv')
	sj_test, iq_test = preprocess_data(test_features_path)
	sj_predictions = clf_sj.predict(sj_test).astype(int)	
	iq_predictions = clf_iq.predict(iq_test).astype(int)	
	sample_path = os.path.join(data_path,'submission_format.csv')
	submission = pd.read_csv(sample_path,index_col=[0, 1, 2])
	submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
	submission.to_csv("./data/benchmark_gb.csv")
	
if __name__ == '__main__':
	main()
