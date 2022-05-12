
#Author: Louis Gomez
#Health and AI Lab

'''
SCRIPT DESCRIPTION:

This script is used to perform some of the other results we provide in the paper

Inputs: data - this is the csv file of the extracted time windows
        output
        dataset
        experiment - A (hopsital), B(ICU), C(Neuro-ICU) subsets

'''
import pandas as pd
from collections import Counter
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_curve

mpl.rcParams['axes.linewidth'] = 1.2

def unpack_pickle(pickled_file,cv_type):
    '''
    This function is used to unpack the pickled results file
    into a datafram with a sample per row and it associated labels
    (true/predicted) and prediction probability

    Note here the predicted probability is the probability of the sample
    belonging to the label 1 class
    '''
    log = pd.read_pickle(pickled_file)
    pred_label,true_label,pred_prob,id = [],[],[],[]

    for key,val in log.items():
        pred,true,pr_prob,ids = val
        id.extend(ids)

        pred_label.extend(pred)
        true_label.extend(true)
        pred_prob.extend(pr_prob)

    results = {"UID":id,"Pred_label":pred_label,"ytrue":true_label,"ypred":pred_prob}
    df = pd.DataFrame(data = results)

    return df

#Code to perform Signal Relevance
def Signal_Relevance(features):
    '''features is a pickle file outputtted from running the classification code'''

    top5 ,freq = [],[]
    features = pd.read_pickle(features)

    for key,val in features.items():
        variables,feature_names = val
        variables = [x.split("_")[0] for x in variables]
        feature_names = [x.split("_")[0] for x in feature_names]

        signals = list(set(variables))
        feature_names = list(set(feature_names))

        freq.extend(feature_names)
        top5.extend(signals)

    freq_dict = Counter(freq)
    top5_dict = Counter(top5)

    relevance_dict = {}
    for key,val in freq_dict.items():
        num = top5_dict[key]
        denom = val

        if denom == 0:
            relevance_dict[key] = np.nan
        else:
            relevance_dict[key] = np.round(num/denom,2)

    print(relevance_dict)

#code to perform model calibration
def Model_Calibration(result_pickle):
    '''Framework code to plot model calibration results: calibration curves, ICI and E_max for
       one experiment in a classification task
    '''
    
    def m_calib(data,cv_type):
        df = unpack_pickle(data,cv_type)
        ytrue = df["ytrue"]
        ypred = df["ypred"]
        
        Y = ytrue
        X = ypred
        
        assert len(Y) == len(X)
        lowess = sm.nonparametric.lowess
        print("here")
        
        z = lowess(Y, X)
        X1 = z[:,0] #sorted predicted probabailities
        Y1 = z[:,1] #coresponding 
        lowess = sm.nonparametric.lowess
        Ycal = lowess(Y,X,xvals = X1)
        
        return X1,Y1,Ycal

    X,Y,ycal = m_calib(result_pickle)
    ICI = np.nanmean(np.abs(ycal-X))
    E_max = np.max(np.abs(ycal-X))

    fig, ax = plt.subplots(1, 1,figsize = (5,3),dpi = 150)
    ax.plot([0, 1], [0, 1],linestyle = "dashed",lw = 1.5,color = "gray",alpha = 0.4)
    ax.set_ylim((-0.03,1.03))
    ax.set_xlim((-0.03,1.03))

    ax.plot(X,Y,label = "ICI: " + str(ICI) + ", $E_{max}$: " + str(E_max))
    ax.set_xlabel("Predicted Probabaility",fontsize =12,fontweight='bold')
    ax.set_ylabel("Observed Probabaility",fontsize=12,fontweight='bold')
    plt.close()

def Roc_Curve(result_pickle):
    '''Framework code to plot roc curves'''

    def unpack_roc(group):
        fpr,tpr,thr = [],[],[]

        for data in group:
            f,t,th = roc_curve(data["True_label"],data["Pred_prob"])
            fpr.append(f)
            tpr.append(t)
            thr.append(th)
  
        return fpr,tpr,thr
    
    df = unpack_pickle(result_pickle)
    fpr,tpr,_ = unpack_roc([df])

    fig, ax = plt.subplots(1, 1,figsize = (5,3),dpi = 150)
    ax.plot([0, 1], [0, 1],linestyle = "dashed",lw = 1.5,color = "gray",alpha = 0.4)
    ax.set_ylim((-0.03,1.03))
    ax.set_xlim((-0.03,1.03))

    #default roc and confidence interval value

    ax.plot(fpr,tpr)
    ax.set_ylabel('Sensitivity',fontsize = 12,fontweight='bold')
    ax.set_xlabel('1 - Specificity',fontsize = 12,fontweight='bold')
    plt.close()






