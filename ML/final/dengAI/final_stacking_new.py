from __future__ import print_function
from __future__ import division
import os
import pandas as pd
import numpy as np
import random as rand
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn import ensemble
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import ExtraTreesRegressor as ETR
from sklearn.ensemble import BaggingRegressor as BR
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression as LR
import xgboost as xgb
from datetime import datetime
rand.seed(datetime.now())
from warnings import filterwarnings
filterwarnings('ignore')

#from warnings import filterwarnings
#filterwarnings('ignore')

base_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(base_path,'data')
weight_path = os.path.join(base_path,'weight')
rand.seed(1)
#########################################
########   Utility Function     #########
#########################################


def preprocess_data(data_path, labels_path=None):
	# load data and set index to city, year, weekofyear
	df = pd.read_csv(data_path, index_col=[0, 1, 2])


	# fill missing values
	df.drop(['week_start_date','ndvi_ne'], axis=1, inplace=True)
	df.fillna(method='ffill', inplace=True)

	# add labels to dataframe
	if labels_path:
		labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
		df = df.join(labels)
	# separate san juan and iquitos
	sj = df.loc['sj']
	copy = sj.copy()
	if labels_path:
		copy.drop('total_cases',axis = 1,inplace = True)
	copy1 = copy.shift(1)
	new_column = [ name+str(1) for name in copy1.columns]
	copy1.columns = new_column
	copy2 = copy.shift(2)
	new_column = [ name+str(2) for name in copy2.columns]
	copy2.columns = new_column
	copy3 = copy.shift(3)
	new_column = [ name+str(3) for name in copy3.columns]
	copy3.columns = new_column
	
	sj = pd.concat([sj,copy1,copy2,copy3],axis = 1)
		
	sj.drop(sj.index[[0,1,2]],inplace = True)
	iq = df.loc['iq']
	copy = iq.copy()
	if labels_path:
		copy.drop('total_cases',axis = 1,inplace = True)
	copy1 = copy.shift(1)
	new_column = [ name+str(1) for name in copy1.columns]
	copy1.columns = new_column
	copy2 = copy.shift(2)
	new_column = [ name+str(2) for name in copy2.columns]
	copy2.columns = new_column
	copy3 = copy.shift(3)
	new_column = [ name+str(3) for name in copy3.columns]
	copy3.columns = new_column
	iq = pd.concat([iq,copy1,copy2,copy3],axis = 1)
	iq.drop(iq.index[[0,1,2]],inplace = True)    
	
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
	#print("Training MAE: %.4f" % mae)

	val_label = val_data['total_cases']
	val_feat = val_data.drop('total_cases',axis = 1)
	val_predictions = clf.predict(val_feat)
	mae = eval_measures.meanabs(val_predictions, val_label)
	#print("Validation MAE: %.4f" % mae)
	
	return clf


def get_best_model(train, test, city):

	if city == 'sj':
	    # Step 1: specify the form of the model
	    model_formula = "total_cases ~ 1 + " \
	                    "reanalysis_specific_humidity_g_per_kg + " \
	                    "reanalysis_dew_point_temp_k + " \
	                    "station_min_temp_c + " \
	                    "station_avg_temp_c " 
	                    #"reanalysis_min_air_temp_k + " \
	                    
	elif city == 'iq':
		model_formula = "total_cases ~ 1 + " \
	                    "reanalysis_specific_humidity_g_per_kg + " \
	                    "reanalysis_dew_point_temp_k + " \
	                    "station_min_temp_c + " \
	                    "station_avg_temp_c + " \
						"reanalysis_min_air_temp_k "
	grid = 10 ** np.arange(-10, -3, dtype=np.float64)
                    
	best_alpha = []
	best_score = 1000
	    
	# Step 2: Find the best hyper parameter, alpha
	for alpha in grid:
		model = smf.glm(formula=model_formula,data=train,family=sm.families.NegativeBinomial(alpha=alpha))
		results = model.fit()
		predictions = results.predict(test).astype(int)
	score = eval_measures.meanabs(predictions, test.total_cases)

	if score < best_score:
			best_alpha = alpha
			best_score = score
	   
    # Step 3: refit on entire dataset
	full_dataset = pd.concat([train, test])
	model = smf.glm(formula=model_formula,
					data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

	fitted_model = model.fit()
	return fitted_model
#####################################
######      Main Function      ######           
#####################################

def main():

### parsing and Data pre-processing
	# load the provided data
	train_features_path = os.path.join(data_path,'dengue_features_train.csv')
	train_labels_path = os.path.join(data_path,'dengue_labels_train.csv')
	
	### pre-processing data
	sj_train, iq_train = preprocess_data(train_features_path, labels_path = train_labels_path)
	#print(sj_train.describe())
	#print(iq_train.describe())
	
	###Define the xgb parameters	
	xgb_params = {
	'eta': 0.05,
	'max_depth': 5,
	'subsample': 0.7,
	'colsample_bytree': 0.7,
	'objective': 'reg:linear',
	'eval_metric': 'rmse',
	'silent': 1
	}
	num_boost_rounds = 1000
	##Use K-fold to create cross validation data
	kf = KFold(n_splits=6)	
	
	##Do the stacking by adding 5 dataframes 'negbi', 'gb', 'xgb','adaboost','extratree' ,'bagging'which store the training prediction
	sj_train = sj_train.assign(xgb = 0)
	sj_train = sj_train.assign(etr = 0)
	sj_train = sj_train.assign(br = 0)

	loop = 1
	for train_index, val_index in kf.split(sj_train): #The index will be split into [train_index] and [val_index] 
		X_train,X_val = sj_train.ix[train_index], sj_train.ix[val_index]
		
		###(1)xgboost method
		dtrain = xgb.DMatrix(X_train.drop(['total_cases','xgb','etr','br'],axis = 1), X_train['total_cases'])
		dval = xgb.DMatrix(X_val.drop(['total_cases','xgb','etr','br'],axis = 1))
		sj_xgb_model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)
		predictions_xgb = sj_xgb_model.predict(dval).astype(int)
		
		###(2)Extra tree regressor method
		sj_etr_model = ETR(n_estimators = 2000,  max_depth = 3,criterion = 'mae')
		sj_etr_model.fit(X_train.drop(['total_cases','xgb','etr','br'],axis = 1), X_train['total_cases'])
		predictions_etr = sj_etr_model.predict(X_val.drop(['total_cases','xgb','etr','br'], axis = 1))

		###(3) Bagging Regressor method
		sj_br_model = BR(n_estimators = 100,max_features = 0.6, max_samples = 0.6)
		sj_br_model.fit(X_train.drop(['total_cases','xgb','etr','br'],axis = 1), X_train['total_cases'])
		predictions_br = sj_br_model.predict(X_val.drop(['total_cases','xgb','etr','br'], axis = 1))

		###Store the result in sj_train  predictions_neg -> 'negbi', predictions_gb -> 'gb'
		print("Adding the result of the predictions to sj training data({}/{})".format(loop,6))
		for idx,index in enumerate(val_index):
			sj_train['xgb'].ix[index] = predictions_xgb[idx]
			sj_train['etr'].ix[index] = predictions_etr[idx]
			sj_train['br'].ix[index] = predictions_br[idx]
		loop += 1	
		
	iq_train = iq_train.assign(xgb = 0)
	iq_train = iq_train.assign(etr = 0)
	iq_train = iq_train.assign(br = 0)

	loop = 1
	for train_index, val_index in kf.split(iq_train):
		X_train,X_val = iq_train.ix[train_index], iq_train.ix[val_index]
		###(1)xgb method
		dtrain = xgb.DMatrix(X_train.drop(['total_cases','xgb','etr','br'],axis = 1), X_train['total_cases'])
		dval = xgb.DMatrix(X_val.drop(['total_cases','xgb','etr','br'],axis = 1))
		iq_xgb_model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)
		predictions_xgb = iq_xgb_model.predict(dval).astype(int)
		
		###(2)Extra tree regressor method
		iq_etr_model = ETR(n_estimators = 2000,  max_depth = 3, criterion = 'mae')
		iq_etr_model.fit(X_train.drop(['total_cases','xgb','etr','br'],axis = 1), X_train['total_cases'])
		predictions_etr = iq_etr_model.predict(X_val.drop(['total_cases','xgb','etr','br'], axis = 1))

		###(3) Bagging Regressor method
		iq_br_model = BR(n_estimators = 800, max_features = 0.6, max_samples = 0.6)
		iq_br_model.fit(X_train.drop(['total_cases','xgb','etr','br'],axis = 1), X_train['total_cases'])
		predictions_br = iq_br_model.predict(X_val.drop(['total_cases','xgb','etr','br'], axis = 1))

		###Store the result in iq_train predictions_neg -> 'negbi', predictions_gb -> 'gb'
		print("Adding the result of the predictions to iq training data({}/{})".format(loop,6))
		for idx,index in enumerate(val_index):
			iq_train['xgb'].ix[index] = predictions_xgb[idx]
			iq_train['etr'].ix[index] = predictions_etr[idx]
			iq_train['br'].ix[index] = predictions_br[idx]			
		loop += 1
	
	###Now the training data looks like [feature, total_cases, negbi, gb, xgb]

	##Accessing testing data
	test_features_path = os.path.join(data_path,'dengue_features_test.csv')
	sj_test, iq_test = preprocess_data(test_features_path)
	##Like training, add 'xgb' 'br' 'etr'  to the testing dataframe
	sj_test = sj_test.assign(xgb = 0)
	sj_test = sj_test.assign(etr = 0)
	sj_test = sj_test.assign(br = 0)
	##(1)xgb prediction
	dtest = xgb.DMatrix(sj_test.drop(['xgb', 'etr','br'],axis = 1))
	sj_predictions_xgb = sj_xgb_model.predict(dtest).astype(int)
	###(2)extra tree regressor method
	sj_predictions_etr = sj_etr_model.predict(sj_test.drop(['xgb', 'etr','br'],axis = 1)).astype(int)
	###(3)bagging regressor method
	sj_predictions_br = sj_br_model.predict(sj_test.drop(['xgb', 'etr','br'],axis = 1)).astype(int)

	print("Adding predictions as features to sj testing data...")
	for i in range(len(sj_test['xgb'])): #Add the prediction to the corresponding column 	
		sj_test['xgb'].ix[i] = sj_predictions_xgb[i]
		sj_test['etr'].ix[i] = sj_predictions_etr[i]
		sj_test['br'].ix[i] = sj_predictions_br[i]

	##Same process as city iq
	iq_test = iq_test.assign(xgb = 0)
	iq_test = iq_test.assign(etr = 0)
	iq_test = iq_test.assign(br = 0)
	
	###(1)xgb prediction	
	dtest = xgb.DMatrix(iq_test.drop(['xgb', 'etr','br'],axis = 1))
	iq_predictions_xgb = iq_xgb_model.predict(dtest).astype(int)
	###(2)extra tree regressor method
	iq_predictions_etr = iq_etr_model.predict(sj_test.drop(['xgb', 'etr','br'],axis = 1)).astype(int)
	###(3)bagging regressor method
	iq_predictions_br = iq_br_model.predict(sj_test.drop(['xgb', 'etr','br'],axis = 1)).astype(int)

	print("Adding predictions as features to iq testing data...")
	for i in range(len(iq_test['xgb'])):	
			iq_test['xgb'].ix[i] = iq_predictions_xgb[i]
			iq_test['etr'].ix[i] = iq_predictions_etr[i]
			iq_test['br'].ix[i] = iq_predictions_br[i]

	##use new information to run a linear regression
	'''
	print("Building linear regression model...")
	#Now the linear regression model uses (X = [features, negbi, gb, xgb, abr, etr, br], y = total_cases )to train(fit)
	sj_lr = LR()
	sj_lr.fit(sj_train.drop('total_cases',axis = 1),sj_train['total_cases'])
	iq_lr = LR()
	iq_lr.fit(iq_train.drop('total_cases',axis = 1),iq_train['total_cases'])
	
	#Calculate the k-fold validation error
	sj_score = []
	for train_index, val_index in kf.split(sj_train):
		X_train,X_val = sj_train.ix[train_index], sj_train.ix[val_index]
		train_predict = np.array(sj_lr.predict(X_val.drop('total_cases',axis = 1))).astype(int)
		sj_score.append(eval_measures.meanabs(train_predict, X_val.total_cases))
	print("Mean of {} cross validation of sj_score is {} (+/- {})".format(kf.get_n_splits(sj_train)
																,np.mean(sj_score),np.std(sj_score)))
	
	iq_score = []
	for train_index, val_index in kf.split(iq_train):
		X_train,X_val = iq_train.ix[train_index], iq_train.ix[val_index]
		train_predict = np.array(iq_lr.predict(X_val.drop('total_cases',axis = 1))).astype(int)
		iq_score.append(eval_measures.meanabs(train_predict, X_val.total_cases))
	print("Mean of {} cross validation of iq_score is {} (+/- {})".format(kf.get_n_splits(iq_train)
																,np.mean(iq_score),np.std(iq_score)))
	
	##Use the model sj_lr and iq_lr trained before to predict the testing data
	print("Predicting testing data...")
	sj_predictions = sj_lr.predict(sj_test)
	iq_predictions = iq_lr.predict(iq_test)
	'''
	sj_predictions = []
	iq_predictions = []
	for i in range(len(sj_test['br'])):
		sj_predictions.append(( #sj_test['negbi'].ix[i]+\
									#sj_test['gb'].ix[i]+\
									#sj_test['abr'].ix[i]+\
									sj_test['xgb'].ix[i]+\
									sj_test['etr'].ix[i]+\
									sj_test['br'].ix[i])/3)

	for i in range(len(iq_test['br'])):
		iq_predictions.append(( #iq_test['negbi'].ix[i]+\
								#iq_test['gb'].ix[i]+\
								#iq_test['abr'].ix[i]+\
								iq_test['xgb'].ix[i]+\
							iq_test['etr'].ix[i]+\
							iq_test['br'].ix[i])/3)



	sj_predictions = np.round(sj_predictions).astype(int)
	iq_predictions = np.round(iq_predictions).astype(int)
	
	print("Creating submit file...")
	##Use submission_format as template to write the answer
	sample_path = os.path.join(data_path,'submission_format.csv')
	submission = pd.read_csv(sample_path,index_col=[0, 1, 2])
	submission.total_cases = np.concatenate([[28],[25],[34],sj_predictions,[7],[6],[8],iq_predictions])
	submission.to_csv("./data/stacking_final_new.csv")
	
	'''
	##plotting but not update haha
	figs, axes = plt.subplots(nrows=2, ncols=1)
	
	# plot sj
	sj_train['fitted'] = sj_neg_model.fittedvalues
	sj_train.fitted.plot(ax=axes[0], label="Predictions")
	SJ_predictions = sj_neg_model.predict(sj_train).astype(int)
	for i in range(SJ_predictions.shape[0]-1,3,-1):
			SJ_predictions.ix[i] = SJ_predictions.ix[i-4]
	SJ_predictions.plot(ax=axes[0], label="Predictions")
	sj_train.total_cases.plot(ax=axes[0], label="Actual")

	# plot iq
	#iq_train['fitted'] = iq_neg_model.fittedvalues
	#iq_train.fitted.plot(ax=axes[1], label="Predictions")
	IQ_predictions = iq_neg_model.predict(iq_train).astype(int)
	for i in range(IQ_predictions.shape[0]-1,0,-1):
			IQ_predictions.ix[i] = IQ_predictions.ix[i-1]
	IQ_predictions.plot(ax=axes[1], label="Predictions")
	iq_train.total_cases.plot(ax=axes[1], label="Actual")

	plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
	plt.legend()
	plt.show()
	'''

if __name__ == '__main__':
	main()
