
#Author: Louis Gomez
#Health and AI Lab

'''
SCRIPT DESCRIPTION:

This script is used to perform feature extraction acorss all time series measures.

Inputs: data - this is the csv file of the extracted time windows
        output
        dataset
        experiment - A (hopsital), B(ICU), C(Neuro-ICU) subsets

'''
import pandas as pd
import numpy as np
import sys
import pickle
import pywt
import scipy.stats as stats

from entropy import sample_entropy,higuchi_fd
from tsfel import feature_extraction
import nolds

np.random.seed(20)

from tsfeatures import stability,nonlinearity

def compute_features(features,coeff):
    '''This is the function used to compute all the time series measures'''
    comp_features = []

    for feature in features:
        if feature == "cv":
            comp_features.append(stats.variation(coeff,nan_policy = "omit"))
        elif feature == "ent":
            comp_features.append(sample_entropy(coeff))
        elif feature == "hig":
            comp_features.append(higuchi_fd(coeff))
        elif feature == "abs":
            comp_features.append(np.sum(coeff**2))
        elif feature == "mean":
            comp_features.append(np.nanmean(coeff))
        elif feature == "med":
            comp_features.append(feature_extraction.median_diff(coeff))
        elif feature == "stab":
            comp_features.append(stability(coeff)['stability'])
        elif feature == "hur":
            comp_features.append(nolds.hurst_rs(coeff,fit = "poly"))
        elif feature == "skew":
            comp_features.append(stats.skew(coeff))
        elif feature == "kurt":
            comp_features.append(stats.kurtosis(coeff))
        elif feature == "sd":
            comp_features.append(np.nanstd(coeff))
        elif feature == "lin":
            comp_features.append(nonlinearity(coeff)['nonlinearity'])
        elif feature == "rms":
            comp_features.append(np.sqrt(np.mean(coeff**2)))
        elif feature == "mabs":
            comp_features.append(feature_extraction.mean_abs_deviation(coeff))
        elif feature == "range":
            comp_features.append(np.max(coeff) - np.min(coeff))
        elif feature == "iqr":
            comp_features.append(stats.iqr(coeff))
        elif feature == "max":
            comp_features.append(np.max(coeff))
        elif feature == "min":
            comp_features.append(np.min(coeff))
        else:
            print("Feature not present")
            sys.exit(0)

    return comp_features

def compute_global(list_coeff,col,features):
  names = []
  n = 1
  for coeff in list_coeff:
      computed_features = compute_features(features,coeff)

      for ft in features:
          new_names = col + "_" +ft + "_" + str(n)
          names.append(new_names)
      n = n + 1

  result = pd.DataFrame(data = [computed_features], columns = names)
  return result

def wavelets(df,features):

    '''Function to perform DWT, then compute time series measures'''
    df.reset_index(drop = True, inplace = True)
    event_label = df.at[0,"Event"]
    uid = df.at[0,"UID"]

    #drop all columns with all nans
    df.dropna(axis = 1, how = "all", inplace = True)

    #drop event and UID columns since we dont need then right now
    df.drop(columns = ["Event","UID"], inplace = True)

    #interpolate for missigness that may be left in the middle of the time series as
    #this affects the DWT
    #note that 80% of the time window is present when performing this operation
    df.interpolate(limit_area="inside",axis = 0,inplace = True)

    #perform wavelet decomposition operations here
    cols = df.columns.to_list()
    db4 = pywt.Wavelet('db4')

    results = pd.DataFrame()
    for col in cols:
        t_series = df[col].to_numpy(copy = True)

        #remove outstanding nan value on the beginning or end
        ind_toremove = np.argwhere(np.isnan(t_series))
        t_series = np.delete(t_series,ind_toremove)

        if np.nanvar(t_series) == 0:
            db4 = pywt.Wavelet('db1')
    
        ca2,_,_ = pywt.wavedec(t_series,db4,level = 2)
        temp_result = compute_global([ca2],col,features)
        results = pd.concat([results,temp_result], axis = 1)

    results = results.reset_index(drop = True)
    results.insert(0,"UID",uid)
    results.insert(1,"Event",event_label)
    return results

def main():
  #read in data and get assessments
    df = pd.read_csv(data)
    try:
        df.rename(columns = {"Unnamed: 0": "PID"}, inplace = True)
    except:
        pass

    if dataset == "sah":
      df.drop(columns = ["Time"], inplace = True) #for sah
    elif dataset == "ich":
      df.drop(columns = ["DateTime"],inplace = True) #for ich
      if experiment == "A":
          df.rename(columns = {"SPO2":"SPO2%"},inplace = True)
      else:
          df.rename(columns = {"SPO2":"SPO2%","EtCO2":"CO2EX"},inplace = True)

    else:
      print("invalid dataset type")
      sys.exit(0)

    pid = df["PID"].tolist()
    times = df["Event_Time"].to_list()
    tags = [str(pid[i])+"_"+str(times[i]) for i in range(len(pid))]

    #created a new unique ID (UID) that combines PID and event_time
    df["UID"] = tags
    df.drop(columns = ["PID","Event_Time"], inplace = True)
    df_feature = df.groupby(by = ["UID"], as_index = False).apply(wavelets,features = features)
    df_feature = df_feature.reset_index(drop = True)
    df_feature = df_feature.replace(np.inf,np.nan)

    if dataset == "sah":
        name = "features_wavelet_SAH_"+ experiment + ".csv"

    else:
        name = "features_wavelet_ICH_"+ experiment + ".csv"

    df_feature.to_csv(output + name, index = False)
    print(output + name)

if __name__ == "__main__":

    if len(sys.argv) == 5:
      data = sys.argv[1] #This is the data after time windows have been extracted
      output = sys.argv[2]
      dataset = sys.argv[3] #name of dataset
      experiment = sys.argv[4] #A or B or C

      features = ["stab","ent","abs","range","sd","rms","mean","cv","iqr","med","hig","hur","lin","skew","mabs","kurt"]
      main()
    else:
      print("Invalid input arguements")
      print("python extracted-data, window, outputdir")
