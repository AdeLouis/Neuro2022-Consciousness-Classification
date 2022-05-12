
#Author: Louis Gomez
#Health and AI Lab
'''
SCRIPT DESCRIPTION:

This file is used to extract the specificed time windows from the dataset. The final results
are all the time wndows appended together column-wise in a resulting csv file for all patients

Inputs: conscioussfn - command following file
        inputdir - Folder that contains the data-files. Each data file is in the form [pid,timestamp,v1,v2...vn]
        outputdir - Output folder
        first - initial time index
        second - end time index. note that first is bigger than second
        dataset - SAH or ICH. Different pre-processing based on the way the dataset is saved
        expec-Len: expected length of the extracted window
        experiment - A (hospital), B(ICU), C(Neuro-ICU) subset
'''

import pandas as pd
import glob as glob
import numpy as np
import sys
import pickle

from dateutil.parser import parse
from datetime import timedelta

def clean(df,del_labels=[]):

    for label in del_labels:
    	try:
        	df = df.drop(labels=label, axis=1)
    	except:
        	continue
    return df

def to_keep(df,labels):

    for col in df.columns[1:]:
        if col not in labels:
            df = df.drop(labels = col,axis = 1)
        else:
            pass
    return df

def extract_windows(df,time,event,PID,first,second,elen,dataset):
    '''
    This function is used to extract time window based on the time slice index
    provided.
    Df - dataframe
    time - the time of consciousness assessment in CF file
    event - label
    window - how large the extracted window should be
    PID - simply the identifier of the patient file
    first, second - time slices. First is ALWAYS bigger than second
    '''

    if dataset == "sah":
        left = time - (first*60)
        right = time - (second * 60)
        dfextract = df.loc[(df["Time"] >= left) & (df["Time"] < right)].copy()
    else:
        left = time - timedelta(minutes = first)
        right = time - timedelta(minutes = second)
        dfextract = df.loc[(df["DateTime"] >= left) & (df["DateTime"] < right)].copy()

    if len(dfextract) != elen:
        return None

    column_count = dfextract.count().to_dict()
    l = len(dfextract)
    cols_to_drop = []

    for name,val in column_count.items():
        if val < (round(0.80 * l)):
            cols_to_drop.append(name)

    #if the number of entries in the variable column is less than 80%, drop it
    dfextract.drop(columns = cols_to_drop, inplace = True)
    dfextract.dropna(axis = 1, how = "all", inplace = True)

    if len(dfextract.columns.tolist()) == 1:
        return None

    #Operations to insert three new columns
    Event = np.repeat(event,len(dfextract))
    pid = np.repeat(PID,len(dfextract))
    Event_Time = np.repeat(time,len(dfextract))
    dfextract.insert(0,"PID",pid)
    dfextract.insert(1,"Event",Event)
    dfextract.insert(2,"Event_Time",Event_Time)
    return dfextract

def main_sah():

    df_bcd = pd.read_csv(CF_file)
    data = None

    for _, row in df_bcd.iterrows():

        PID = int(row.PID)
        time = int(row.timeafterbleed)
        event = int(row.Event)
        filename = "Patient_" + str(PID) + ".csv"
        del_labels = ["GLU","LAC","LGR","LPR","PGR","PYR","GLU-panel"]

        try:
            df = pd.read_csv(inputdir + filename, sep = ',')
            df = clean(df,del_labels)
            df.rename(columns = {"DateTime":"Time"}, inplace = True)
        except Exception as e:
            #print(str(PID) + "   EXCEPTION:     " + str(e))
            continue

        df = to_keep(df,cols_to_keep)

        if len(df.columns) == 1:
            continue

        extracted_data = extract_windows(df,time,event,PID,first,second,expec_len,dataset)
        if extracted_data is None:
            pass
        else:
            data = pd.concat([data,extracted_data],sort = False)

    name = "Time_windows_SAH_" + experiment + ".csv"

    #file = outputdir + name
    #with open(file,"wb") as f:
        #pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    data.to_csv(outputdir + name,index = False)

def main_ich():
    df_bcd = pd.read_csv(CF_file)
    data = None

    for _,row in df_bcd.iterrows():
        PID = int(row.PID)
        time = row.Time
        event = int(row.Event)
        filename = "patient_"+str(PID)+".csv"

        try:
            df = pd.read_csv(inputdir + filename,parse_dates = ["DateTime"])
        except Exception as e:
            #print(str(PID) + "   EXCEPTION:     " + str(e))
            continue

        #check if time is a valid entry
        try:
            time = parse(time, fuzzy = False)
        except:
            print(str(PID) + " invalid time record: ",time)
            continue

        #check if the is 0:00, these are not comple time records and have to be ignored
        if time.hour == 0 and time.minute == 0:
            continue

        #two columns - datetime and a variable columns must be at least present to be valid
        if len(df.columns.to_list()) < 2:
            continue

        df = to_keep(df,cols_to_keep)

        extracted_data = extract_windows(df,time,event,PID,first,second,expec_len,dataset)
        if extracted_data is None:
            pass
        else:
            data = pd.concat([data,extracted_data],sort = False)

    name = "Time_windows_ICH_" + str(experiment) + ".csv"
    data.to_csv(outputdir + name,index = False)

if __name__ == "__main__":

    if len(sys.argv) == 9:
        CF_file = sys.argv[1]  
        inputdir = sys.argv[2]
        outputdir = sys.argv[3]
        first = int(sys.argv[4]) #initial time index to begin time slice
        second = int(sys.argv[5]) #end time index to end time slice
        dataset = str(sys.argv[6]) #name of dataset
        expec_len = int(sys.argv[7]) #expected length of data window. 
        experiment = sys.argv[8] #A or B or C

        if dataset == "sah":
            if experiment == "A":
                cols_to_keep = ["SPO2%","RR","HR"]
            elif experiment == "B":
                cols_to_keep = ["SPO2%","RR","HR","MAP","CO2EX","TMP"]
            else:
                cols_to_keep = ["SPO2%","RR","HR","MAP","CO2EX","TMP","ICP","PbtO2","BrT"]


        elif dataset == "ich":
            if experiment == "A":
                cols_to_keep = ["SPO2","RR","HR"]
            else:
                cols_to_keep = ["SPO2","RR","HR","MAP","EtCO2","TMP",'ICP']
        else:
            print("incorrect dataset name")
            sys.exit(0)

        print("index is: ",first)
        print("index is: ",second)

        if second > first:
            print("Incorrect time slice. First should be bigger than second")
            sys.exit(0)

        if dataset == "sah":
            main_sah()
        elif dataset == "ich":
            main_ich()
        else:
            print("incorrect dataset name")
            sys.exit(0)
    else:
        print("Invalid input arguements")
        print("python file-name command following file, inputdir, outputdir, windowm first second True/False")
