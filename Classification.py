#Author: Louis Gomez
#Health and AI Lab

'''
SCRIPT DESCRIPTION:

This file is used to perform classification using the leave one patient out framework. We also
output the AUROC (with confidence intervals), the AUPRC and accuracy
Files saved are the results and feature file in pickle format for use in makinf plots and other evaluations
performed in the paper like model calibration and signal relevance

Inputs: data - inout file is the feature extraction data
        output - location pickle files are stored
        
'''


import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import sys
import pickle
import xgboost as xgb
from scipy import stats
import itertools

from joblib import Parallel, delayed
from sklearn.model_selection import ParameterSampler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import classification_report,average_precision_score
from DeLong import delong_roc_variance

def create_hyperparameters(n):
    max_depth = np.linspace(2,7,(7-2+1),dtype=int)
    min_child_weight = np.linspace(2,6,(6-2+1),dtype=int)
    gamma = np.linspace(1,4,(4-1+1),dtype=int)
    learning_rate = [0.1,0.01]

    params_grid = {"max_depth":max_depth,"min_child_weight":min_child_weight,
               "gamma":gamma,"learning_rate":learning_rate}
    
    param_list = list(ParameterSampler(params_grid, n_iter=n,random_state=10))
    return param_list

def delong_ci(ground_truth,predictions,alpha = 0.95):
    #https://github.com/RaulSanchezVazquez/roc_curve_with_confidence_intervals/blob/master/auc_delong_xu.py
    y_true = np.array(ground_truth)
    y_score = np.array(predictions)
    auc,auc_var = delong_roc_variance(y_true,y_score)

    auc_std = np.sqrt(auc_var)
    # Confidence Interval
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    lower_upper_ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    lower_upper_ci[lower_upper_ci > 1] = 1

    print('ROC AUC: %s, Conf.' % auc) 
    print('Confidence Interval: %s (95%% confidence)' % str(lower_upper_ci))

def drop_nan_first(X_train,X_test):
    '''
    This function drops the nan values in the test (and subsequently train data) and then
    performs feature selection
    '''

    #Drop features that are nans
    X_test = X_test.dropna(axis = 1, how = "all").copy()  #If a column are all nan drop
    #if all features are dropped, we of-course cannot make a prediction
    if X_test.empty:
        print("All features dropped")
        return None,None

    test_features_present = X_test.columns.to_list()
    X_train = X_train[test_features_present].copy()

    return X_train,X_test

def new_labels(event_labels,classification_task):

  """Takes in a list of event labels and reconfigures it"""
  if classification_task == "A":
      for num in range(0,len(event_labels)):
          if event_labels[num] in [0,1,2]:
              pass
          elif event_labels[num] == 3:
              event_labels[num] = 2
          elif event_labels[num] in [4,5]:
              event_labels[num] = 3
          else:
              pass

  elif classification_task == "B":
      for num in range(0,len(event_labels)):
          if event_labels[num] in [0,1]:
              event_labels[num] = 0
          elif event_labels[num] in [2,3,4,5]:
              event_labels[num] = 1
          else:
              pass
  elif classification_task == "C":
      for num in range(0,len(event_labels)):
          if event_labels[num] in [0,1,2,3]:
              event_labels[num] = 0
          elif event_labels[num] in [4,5]:
              event_labels[num] = 1
          else:
              pass

  return event_labels

def setup():

    X = pd.read_csv(data)

    if classification_task == "A":
        task = "(1) vs (2,3)"
    elif classification_task == "B":
        task = "(0,1) vs (2,3,4,5)"
    elif classification_task == "C":
        task = "(0,1,2,3) vs (4,5)"
    print("Classification Task is: ", task)
    #label_map = {}
    old_labels = X.Event.to_numpy()
    X["Event"] = new_labels(old_labels,classification_task) #assign the new binned labels

    if classification_task == "A":
        X = X[(X["Event"] == 1) | (X["Event"] == 2)].copy()
    else:
        X = X[(X["Event"] == 0) | (X["Event"] == 1)].copy()

    label = X.Event.to_numpy()
    mode = stats.mode(label)[0]

    #assign the minority class as the positive class label
    label = [0 if n == mode else 1 for n in label]
    X["Event"] = label
    global uid
    uid = X["UID"].to_list()
    Y = X["Event"].to_numpy() #labels
    X.drop(columns = ["Event","UID"], inplace = True)
    X = X.reset_index(drop = True)

    return X,Y,uid

def CV_lopo(config,X_train,y_train):

    featurenames = list(X_train.columns)
    keep = featurenames
  
    temp_num = list(y_train)
    sum_neg_samples = temp_num.count(0)
    sum_pos_samples = temp_num.count(1)
    scale = sum_neg_samples / sum_pos_samples
    config["scale_pos_weight"] = scale

    model = xgb.XGBClassifier(objective='binary:logistic',max_depth=config["max_depth"],
                            learning_rate=config["learning_rate"],gamma=config["gamma"],
                            min_child_weight=config["min_child_weight"],scale_pos_weight=scale,n_estimators=50,
                            random_state = 10)

    sfs = SequentialFeatureSelector(model,cv=2,n_features_to_select=0.2,scoring='roc_auc',n_jobs=1)
    sfs.fit(X_train,y_train)
    ind = sfs.get_support()
    keep = []
    
    for n in range(0,len(ind)):
        if ind[n] == True:
            keep.append(featurenames[n])
    X_train = sfs.transform(X_train)
    
    dtrain = xgb.DMatrix(X_train,label = y_train,feature_names = keep)
    cv_results = xgb.cv(config,dtrain,num_boost_round=100,seed = 10,nfold = 5, stratified = True,shuffle = True,
                            early_stopping_rounds = 10,metrics = {"auc"},verbose_eval = False)
    
    mean_auc = cv_results['test-auc-mean'].max()
    boost_rounds = cv_results['test-auc-mean'].argmax()
    return keep,mean_auc,boost_rounds,config

def LOPO(X,Y,uid):

    cons_prediction_prob,cons_predictions,cons_labels = [],[],[]
    pid_new = []
    cons_output,feat_imp_store = {},{}

    pid,Atime = [],[]
    for n in uid:
        temp = n.split("_")
        pid.append(int(temp[0]))
        Atime.append(temp[1])

    assert len(pid) == len(Atime)
    u_pid = list(set(pid))
    X.insert(0,"PID",pid)
    X.insert(1,"Label",Y)

    for pid in u_pid:
        print(pid)
        min_auc = 0.5
        selected_info = ()

        data = X.copy(deep = True)
        label = Y.copy()

        train = data[data["PID"] != pid].copy()
        test = data[data["PID"] == pid].copy()

        y_train = train["Label"].to_numpy()
        y_test = test["Label"].to_numpy()

        X_train = train.drop(columns = ["PID","Label"])
        X_test = test.drop(columns = ["PID","Label"])
        X_train,X_test = drop_nan_first(X_train,X_test)

        #tune parameters and select parameters
        configs = create_hyperparameters(n=240)
        results = Parallel(n_jobs = 170)(delayed(CV_lopo)(config,X_train,y_train) for config in configs)

        for result in results:
            keep,mean_auc,boost_rounds,config = result
            if mean_auc > min_auc:
                min_auc = mean_auc
                selected_info = (keep,boost_rounds,config)

        #test
        features,rounds,config = selected_info
        X_train = X_train[features].copy()
        X_test = X_test[features].copy()
        dtrain = xgb.DMatrix(X_train,label = y_train,feature_names = features)
        dtest = xgb.DMatrix(X_test,label = y_test,feature_names = features)

        config["objective"] = "binary:logistic"
        config["nthread"] = 1

        model = xgb.train(config,dtrain,num_boost_round=rounds,verbose_eval=True)
        pred = model.predict(dtest,validate_features = True)

        prediction = [1 if x > 0.5 else 0 for x in pred]
        cons_prediction_prob.append(pred)
        cons_predictions.append(prediction)
        cons_labels.append(y_test)

        #for each pid, store the collection of valid uids
        pid_new = []
        for id in uid:
            temp = id.split("_")[0]
            if pid == temp:
                pid_new.append(id)

        '''Feature Importance'''
        #this returns a dictionary with the feature names as keys and value as scores
        feature_dict = model.get_score(importance_type = "gain")
        feature_dict = dict(itertools.islice(feature_dict.items(), 5)) #select 5 top performing features
        feat_imp_store.update({pid:(feature_dict,features)})
        cons_output.update({pid: (prediction,y_test,pred,pid_new)})

    return cons_predictions,cons_labels,cons_prediction_prob,cons_output,feat_imp_store


if __name__ == "__main__":

    if len(sys.argv) != 6:
        print("Error incomplete input: features_condition_window.pickle, window_size, output-folder, sah/ich expr task cv type")
        sys.exit(0)

    data = sys.argv[1]
    output = sys.argv[2]
    dataset_type = sys.argv[3]
    classification_task = str(sys.argv[4]) #A = UWS/MCS-
    experiment = str(sys.argv[5])
    print("Classification Task: ",classification_task)
    print("Experiment: ",experiment)

    X,Y,uid = setup()

    cons_predictions,cons_labels,cons_prediction_prob,cons_output,feature_imp_store = LOPO(X,Y,uid)
    cons_predictions = [i for j in cons_predictions for i in j]
    cons_labels = [i for j in cons_labels for i in j]
    pos_prob = [i for j in cons_prediction_prob for i in j]

    print(classification_report(cons_labels, cons_predictions, target_names = ["0","1"]))
    delong_ci(cons_labels,pos_prob)

    #get AUPRC scores here
    print("AUPRC: ",round(average_precision_score(cons_labels,pos_prob),2))

    
    #For printing assessments to file
    file = output+dataset_type+ "_"+ classification_task +"_"+ experiment + ".pickle"
    with open(file,"wb") as f:
        pickle.dump(cons_output,f,pickle.HIGHEST_PROTOCOL)

    file = "Experiments/Features/" + dataset_type + "_"+classification_task +"_" + experiment + "_features.pickle"
    with open(file,"wb") as f:
        pickle.dump(feature_imp_store,f,pickle.HIGHEST_PROTOCOL)
    
    print("work done")

    