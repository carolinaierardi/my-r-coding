#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:08:33 2022

@author: carolinaierardi
"""

#Script Name: importing_files.py
#Date: 04/11/2022
#Author: CMI
#Version: 1.0
#Purpose: calculate and analyse differences in global and regional measures for controls, 
          #ADHD-I and ADHD-C
#Notes: this scipt is divided in :Phenotypic data extraction 
                              # - Extracting ROIS 
                              # - Graph Theoretical Measures 
                              # - Statistical analysis 
                              # - Generating figures

# import required module
import os                                             #directory changing
import numpy as np                                    #operations involving arrays with numbers
import bct as bct                                     #brain connectivity toolbox
import pandas as pd                                   #operations with dataframes
import matplotlib.pyplot as plt                       # all plotting
from nilearn import plotting                          # brain network plotting
from scipy.stats import kruskal                       #One-Way ANOVA 
from statsmodels.stats.multitest import multipletests #for multiple correction comparison
import seaborn as sns                                 #create boxplot


#%% Getting participant general phenotypic data
os.chdir("/Users/carolinaierardi/Documents/KCL/Term 5/Computational Neuroscience/ADHD200_AAL_TCs_filtfix") #change wd

phenotypes1 = pd.read_csv("Peking_1_phenotypic.csv")                      #phenotypic data
phenotypes2 = pd.read_csv("Peking_2_phenotypic.csv")                      #phenotypic data
phenotypes3 = pd.read_csv("Peking_3_phenotypic.csv")                      #phenotypic data
phenotypes = pd.concat([phenotypes1, phenotypes2,phenotypes3], axis = 0)  #join the phenotypic csvs
phenotypes = phenotypes.reset_index(drop = True)                          #reset their index


IQ_scores = phenotypes["Full4 IQ"]                                        #get mean IQ for the sample
IQ_scores[phenotypes["Full4 IQ"] < 0] = np.nan                            #make negative values into nan

IDscontrols = phenotypes["ScanDir ID"][phenotypes.DX == 0]                #find where the participants are controls 
Gendercontr = len(np.where(phenotypes.Gender[IDscontrols.index] == 1)[0]) #get gender for controls
Agecontr = np.mean(phenotypes.Age[IDscontrols.index])                     #get mean age for controls
minAgec = np.min(phenotypes.Age[IDscontrols.index])                       #minimum age for controls
maxAgec = np.max(phenotypes.Age[IDscontrols.index])                       #maximum age for controls
IQ_controls = np.nanmean(IQ_scores[IDscontrols.index])                    #get mean IQ fro controls
IDscontrols = IDscontrols.reset_index(drop = True)                        #reorder the indeces in the table so it is iterable (will be necessary later)
nControls = len(IDscontrols)                                              #find how many participants are controls

print("There are",nControls,"controls in the sample."\
      " The group is composed of",Gendercontr,"males and the mean age is",
      Agecontr,"(",minAgec,"-",maxAgec,"). Their mean IQ is",IQ_controls) #print for easier visualisation

IDsADHDC = phenotypes["ScanDir ID"][phenotypes.DX == 1]                   #find where the participants are ADHD combined patients
GenderADHDC = len(np.where(phenotypes.Gender[IDsADHDC.index] == 1)[0])    #get gender for ADHD-C
AgeADHDC = np.mean(phenotypes.Age[IDsADHDC.index])                        #get mean age for ADHD-C
minAADHDC = np.min(phenotypes.Age[IDsADHDC.index])                        #minimum age for ADHD-C
maxAADHDC = np.max(phenotypes.Age[IDsADHDC.index])                        #maximum age for ADHD-C
IQ_ADHDC = np.nanmean(IQ_scores[IDsADHDC.index])                          #get mean IQ fro ADHD-C
IDsADHDC = IDsADHDC.reset_index(drop = True)                              #reorder the indeces in the table so it is iterable
nADHDC = len(IDsADHDC)                                                    #find how many participants are ADHD-C patients

print("There are",nADHDC,"ADHD-C patients in the sample."\
      "The group is composed of",GenderADHDC,"males and the mean age is",
      AgeADHDC,"(",minAADHDC,"-",maxAADHDC,"). Their mean IQ is",IQ_ADHDC) #print for easier visualisation
    
IDsADHDI = phenotypes["ScanDir ID"][phenotypes.DX == 3]                   #find where the participants are ADHD innatentive patients 
GenderADHDI = len(np.where(phenotypes.Gender[IDsADHDI.index] == 1)[0])    #get gender for ADHD-I
AgeADHDI = np.mean(phenotypes.Age[IDsADHDI.index])                        #get mean age for ADHD-I
minAADHDI = np.min(phenotypes.Age[IDsADHDI.index])                        #minimum age for ADHD-I
maxAADHDI = np.max(phenotypes.Age[IDsADHDI.index])                        #maximum age for ADHD-I
IQ_ADHDI = np.nanmean(IQ_scores[IDsADHDI.index])                          #get mean IQ fro ADHD-I
IDsADHDI = IDsADHDI.reset_index(drop = True)                              #reorder the indeces in the table so it is iterable
nADHDI = len(IDsADHDI)                                                    #find how many participants are ADHD-I patients

print("There are",nADHDI,"ADHD-I in the sample."\
      "The group is composed of",GenderADHDI,"males and the mean age is",
      AgeADHDI,"(",minAADHDI,"-",maxAADHDI,"). Their mean IQ is",IQ_ADHDI) #print for easier visualisation


#%% Extracting ROIs

# assign directory
directory = "/Users/carolinaierardi/Documents/KCL/Term 5/Computational Neuroscience/ADHD200_AAL_TCs_filtfix/Peking_1 copy"

#Create a function to extract the rois           
def extractROIs(rawdata):
    """ This function extracts the ROIs 
    from the object given by ADHD200 """   
    
    data_list = rawdata.readlines() #read the imported document  
    all_values = data_list[0].split() #get the amount of ROIs
    
    n = len(data_list) - 1;      #calculate its length 
    m = len(all_values) - 2      #the first two lines in the files are unnecessary
    my_data = []                 #create an empty matrix to store the separate values
    rois_ts =  np.zeros([n,m])   #matrix with 0s to be filled with timeseries

    for i in range(n): #for every timepoint
        my_data = data_list[i+1].split()    #we want to read the lines with the numbers, so start from second line, i.e i+1, 
                                            #and split them to get separate cells.
        my_data = np.delete(my_data, (0,1)) #the first and second lines are unimportant, delete them
        
        rois_ts[i,:] = my_data              #assign the values to the matrix previously created
        
    return rois_ts                          #the output is the timeseries for that data


files = []      #create an empty list to store partipants' data
TimeSeries = [] #another empty list, this time to store the timeseries
IDs = []        #and one to store the participant IDs (the loop does not load them in order)

# We want to iterate over files in the directory named above

for filename in os.listdir(directory):             #loop through all folders in the directory
    subfolder = os.path.join(directory, filename)  #get path for the subfolder

    if subfolder == "/Users/carolinaierardi/Documents/KCL/Term 5/Computational Neuroscience/ADHD200_AAL_TCs_filtfix/Peking_1 copy/.DS_Store":
        continue                                   #MACOS imports the .DS_Store files, which we will not use
    
    IDs += [int(filename)]                         #store participants' IDs
    
    for filename2 in os.listdir(subfolder):        # loop through the elements inside the subfolder
        f = os.path.join(subfolder, filename2)     #go into the subfolder we specificied
        
        if f[118] == 'f':                          #we only want one of the files in the subfolder 
            continue                               #(their names are all the same except for one letter)
                                                   #we want the filtered data
            
        elif os.path.isfile(f):                    # checking if it is a file
            print(f)                               #print for visual inspection of correct file download
            files += [open(f)]                     #store the raw data
            TimeSeries += [extractROIs(files[-1])] #we extract the ROIs using the function above on the last element of the list
            
            
TimeSeries = [IDs] + [TimeSeries]                  #the final document has two arrays: one with the order the IDs were retrieved 
                                                   #and one with the timeseries

#%% Graph theoretical measures

def Thresholds(matrix, start, stop, quantity):                      #we create a function to threshold network in steps
    
    """Produces a list thresholded matrices in steps
    with the threshold used. Accepts the fc matrix, lower bound threshold, 
    upper bound threshold and number of iterations to do the thresholding in"""
    
    steps = np.linspace(start, stop, quantity)                       #to create a list with all the threshold values
                                                                     #we want to have
    Steps = []                                                       #an empty list for the threshold values
    thresholds = []                                                  #another empty list for the thresholded networks
    
    for i in range(len(steps)):                                      #now for each thresholded value
        Steps += [steps[i]]                                          #I will first store this value
        thresholds += [bct.threshold_proportional(matrix, steps[i])] #and then the network
    
    return(Steps, thresholds)                                        #we return both the values and networks as an output

                                                                   
def GraphThMeasures(timeseries, thrmin = 1, thrmax = 1, thrn = 1):   #we create a function to calculate gt measures
    """ Calculates all graph theoretical
        measures for our analysis. 
        When no thresholding measures are entered, 
        default values will be used so there is no thresholding """
        
    fc = np.corrcoef(timeseries.T)                                   #functional connectome
                                                                     #use transpose ".T" to correlate along the correct dimension
    thr_fc = Thresholds(fc, thrmin, thrmax, thrn)                    #we want to create 4 thresholded networks with 40-50% threshold 
                                                          
   
    betw_cent = []     #set up empty list to store betweenness centrality measure for the thresholded matrix
    mfc = []           #set up empty list to store mean functional connectivity measure for the thresholded matrix 
    eff_wei = []       #set up empty list to store efficiency measure for the thresholded matrix
    pl_wei = []        #set up empty list to store path length measure for the thresholded matrix
    glb_clust_wei = [] #set up empty list to store global clustering measure for the thresholded matrix
    
    for ii in range(len(thr_fc[0])):
        
        betw_cent += [bct.betweenness_wei(thr_fc[1][ii])]                  #betweenness centrality
        mfc += [np.mean(thr_fc[1][ii])]                                      #mean connectivity
        eff_wei += [bct.efficiency_wei(thr_fc[1][ii])]                     #network efficiency
        Dwei = bct.distance_wei(thr_fc[1][ii])                             #get topological distance
        pl_wei += [bct.charpath(Dwei[0])[0]]                               #weighted characteristic path length
        glb_clust_wei += [np.mean(bct.clustering_coef_wu(thr_fc[1][ii]))]  #global clustering coefficient (mean of clustering coeffs)
    
    return(betw_cent,
           mfc,
           eff_wei, 
           pl_wei, 
           glb_clust_wei)                                                  #return the measures

                                #%%%whole-brain connectivity #%%%
#Set up the for loop to perform function for all the participants and store in separate matrices

lbthr = 0.95 #lower bound of thresholding - leave 95% of connections
upthr = 0.8  #upper bound of thresholding - leave 80% of connections
nthr = 4     #number of thresholded netwroks wanted

GraphMeasures_controls = []   #empty list for graph theoretical measures for controls
GraphMeasures_ADHDC = []      #empty list for graph theoretical measures for ADHD combined
GraphMeasures_ADHDI = []      #empty list for graph theoretical measures for ADHD combined

#look at mean connectivity - looking for outliers - look at conn mat, calculate mean conn and plot that

for ii in range(len(TimeSeries[0])):                                      #loop through all participants
    
    if  len(np.where(TimeSeries[0][ii] == IDscontrols)[0]) != 0:          #if the length of the array matching IDs in different categories is not 0...
        GraphMeasures_controls += [GraphThMeasures(TimeSeries[1][ii],thrmin = lbthr, thrmax = upthr, thrn = nthr)] #calculate Graph Theory measures and store in appropriate list
        continue                                                          #continue to next iteration
        
    elif len(np.where(TimeSeries[0][ii] == IDsADHDC)[0]) != 0:            #if the length of the array matching IDs in different categories is not 0...

        GraphMeasures_ADHDC += [GraphThMeasures(TimeSeries[1][ii],thrmin = lbthr, thrmax = upthr, thrn = nthr)]   #calculate Graph Theory measures and store in appropriate list
        continue
        
    elif len(np.where(TimeSeries[0][ii] == IDsADHDI)[0]) != 0:            #if the length of the array matching IDs in different categories is not 0...

        GraphMeasures_ADHDI += [GraphThMeasures(TimeSeries[1][ii],thrmin = lbthr, thrmax = upthr, thrn = nthr)]   #calculate Graph Theory measures and store in appropriate list


                                    #%%% DMN connectivity #%%%
        
AAL_labels = pd.read_csv('aal_labels.csv')                                  #see AAL labels
AAL_labels = AAL_labels.iloc[1:,:]                                          #select only from 2nd row onwwards (first row is unnecessary)
DMN_network = [2601, 2602, 4001, 4002, 4021, 4022, 6221, 6222, 6301, 6302]  #ROIs that belong to the DMN

labels = AAL_labels.iloc[:,0].astype(int)           #set the labels as integers to be used in the comparison            
ind_dict = dict((k,i) for i,k in enumerate(labels)) #create a dictionary with the original indices of the labels
inter = set(ind_dict).intersection(DMN_network)     #get the intersection for the DMN network and whole network
indices = [ind_dict[x] for x in inter]            #get the indeices of the intersection
indices.sort()                                      #put them in ascending order


DMNGraphMeasures_controls = []   #empty list for graph theoretical measures for controls
DMNGraphMeasures_ADHDC = []      #empty list for graph theoretical measures for ADHD combined
DMNGraphMeasures_ADHDI = []      #empty list for graph theoretical measures for ADHD combined


for iii in range(len(TimeSeries[1])):  #for every participant in the sample

    DMN_ts = TimeSeries[1][iii][:,indices]                       #get the timeseries only for the ROIs belonging to the DMN network
    
    if  len(np.where(TimeSeries[0][iii] == IDscontrols)[0]) != 0:#if the length of the array matching IDs in different categories is not 0...
        DMNGraphMeasures_controls += [GraphThMeasures(DMN_ts,thrmin = lbthr, thrmax = upthr, thrn = nthr)]   #calculate Graph Theory measures and store in appropriate list
        continue                                                 #continue to next iteration
    
    elif len(np.where(TimeSeries[0][iii] == IDsADHDC)[0]) != 0: #if the length of the array matching IDs in different categories is not 0...
        DMNGraphMeasures_ADHDC += [GraphThMeasures(DMN_ts,thrmin = lbthr, thrmax = upthr, thrn = nthr)]     #calculate Graph Theory measures and store in appropriate list
        continue                                                #continue to next iteration
    
    elif len(np.where(TimeSeries[0][iii] == IDsADHDI)[0]) != 0: #if the length of the array matching IDs in different categories is not 0...
        DMNGraphMeasures_ADHDI += [GraphThMeasures(DMN_ts,thrmin = lbthr, thrmax = upthr, thrn = nthr)]     #calculate Graph Theory measures and store in appropriate list


#The resulting lists (whole brain and DMN) contain all the participants and sub-lists with each measure for a given participant.
#The lists are separated according to the group the participant is in. For each graph theoretical measure, there are four values - one for each thresholded network.
#In the analysis section, we extract the values for a given measure for a given participant and perform comparisons at each thresholded level.
#Therefore, each of the five measures has a comparison between Controls vs. Patients and ADHD-C and ADHD-I, each done 4 times (for each threshold).

#%% Statistical analysis

#calculate anovas for difference in connectivity

#first, we need to create a function to extract only the items we want from the sublists
#piece of code inspired by: https://www.geeksforgeeks.org/python-get-first-element-of-each-sublist/
def Extract(lst, i, ii):
    """Within all sublists of lst, 
    get the ith sub-sublist and the iith item"""
    t = list(list(zip(*lst))[i])   #t will extract the ith sub-sublist for all sublists
    return list(list(zip(*t))[ii]) #will return the iith item of the sub-sublift

                                #%%% Whole-brain connectivity #%%%
                      
#Kruskal-Wallis tests                                   

measures_names = ['Between Cent','Mean Connectivity','Efficiency', 'Char Path Length', 'Glb Clustering Coeff']  #names of measures calculated (will be used later for df)

GMPatients = GraphMeasures_ADHDC + GraphMeasures_ADHDI #joint list for patient measures


W_diffPC = []    #empty list for KW for efficiency between patients and controls
W_diffADHD = []  #empty list for KW for efficiency between ADHD types


for ii in range(1, len(measures_names)):               #to loop through to each measure
    for i in range(len(GraphMeasures_controls[1][1])): #to loop through each of the thresholded networks
    
        #for each global measure (efficiency, char path length and clustering) (ii)
        #calculate the statistical difference for each thresholded network (i)
        W_diffPC += [kruskal(Extract(GraphMeasures_controls, ii, i), 
                              Extract(GMPatients, ii, i))] 
        
        #now between ADHD types
        W_diffADHD += [kruskal(Extract(GraphMeasures_ADHDC, ii, i),
                              Extract(GraphMeasures_ADHDI, ii, i))] 
        
        #correct for multiple comparison
        correction = multipletests([W_diffPC[-1][1], W_diffADHD[-1][1]], alpha = 0.05, method = 'holm')
        
        #print statements of signficance for easy access
        print("The whole-brain contrast between patients and controls for", measures_names[ii],"is K = ",
              W_diffPC[-1][0],", p = ",W_diffPC[-1][1],"before correction and p = ",
              correction[1][0],"after correction.")
        
        #print statements of signficance for easy access
        print("The whole-brain contrast between ADHD-C and ADHD-I for ", measures_names[ii],"is K = ",
              W_diffADHD[-1][0],", p = ",W_diffADHD[-1][1],"before correction and p = ",
              correction[1][1],"after correction.")


                                      #%%% DMN connectivity #%%%
                                    
        #Kruskal-Wallis tests                              

DMNGMPatients = DMNGraphMeasures_ADHDC + DMNGraphMeasures_ADHDI #joint list for patient measures

DMN_diffPC = []    #empty list for KW for measures between patients and controls
DMN_diffADHD = []  #empty list for KW for measures between ADHD types


for ii in range(1, len(measures_names)):               #to loop through to each measure (except betw centrality)
    for i in range(len(GraphMeasures_controls[1][1])): #to loop through each of the thresholded networks
    
        #for each global measure (efficiency, char path length and clustering) (ii)
        #calculate the statistical difference for each thresholded network (i)
        DMN_diffPC += [kruskal(Extract(DMNGraphMeasures_controls, ii, i), 
                              Extract(DMNGMPatients, ii, i))] 
        
        #now between ADHD types
        DMN_diffADHD += [kruskal(Extract(DMNGraphMeasures_ADHDC, ii, i),
                              Extract(DMNGraphMeasures_ADHDI, ii, i))] 
        
        #correct for multiple comparisons
        correction = multipletests([DMN_diffPC[-1][1], DMN_diffADHD[-1][1]], alpha = 0.05, method = 'holm')

        #print statements of signficance for easy access
        print("The DMN contrast between patients and controls for", measures_names[ii],"is K =",
              round(DMN_diffPC[-1][0],3),", p =",round(DMN_diffPC[-1][1],3),"bc and p =",
              round(correction[1][0],3),"ac.")

        #print statements of signficance for easy access
        print("The DMN contrast between ADHD-C and ADHD-I for ", measures_names[ii],"is K =",
              round(DMN_diffADHD[-1][0],3),", p =",round(DMN_diffADHD[-1][1],3),"bc and p =",
              round(correction[1][1],3),"ac.")

#testing the betweeness centrality for each node in the DMN network
DMN_betwPC = []   #empty list for comparisons between patients and controls
DMN_betwADHD = [] #empty list for comparisons between ADHD-C and ADHD-I

def Extract2(lst, i, ii, iii): #function (variation of the previous one to get elements of betw centr)
    """Within all sublists of lst, 
    get the ith sub-sublist and the iith item 
    and within them the iiith item"""
    
    t = list(list(zip(*lst))[i])   #t will extract the ith sub-sublist for all sublists
    tt = list(list(zip(*t))[ii])
    return list(list(zip(*tt))[iii]) #will return the iith item of the sub-sublift

def Extract0(lst, i):  #function (variation of the previous one to get p-values of betw centr)
    """Within all sublists of lst"""
    
    return list(list(zip(*lst))[i]) #will return the iith item of the sub-sublift


correction = [] #initialise list for multiple comparisons corrections

for i in range(len(GraphMeasures_controls[1][1])): #to loop through each of the thresholded networks
    for ii in range(len(DMN_network)):             #iterate through each node in the DMN matrix

        #now, we extract the first measure (centrality), in the ith thresholded network and the iith node
        #and compare between patients and controls
        DMN_betwPC += [kruskal(Extract2(DMNGraphMeasures_controls, 0, i, ii),
                              Extract2(DMNGMPatients, 0, i, ii))]
                              
        #and do the same between different kinds of ADHD
        DMN_betwADHD += [kruskal(Extract2(DMNGraphMeasures_ADHDC, 0, i, ii),
                                 Extract2(DMNGraphMeasures_ADHDI, 0, i, ii))]
        
    #we correct for multiple comparisons across all nodes and between contrasts (PC and ADHDs)
    correction += [multipletests(Extract0(DMN_betwPC[-10:],1) + Extract0(DMN_betwADHD[-10:],1), alpha = 0.05, method = 'holm')]
    
    

#%% Generating figures 

                                #%%% methods diagram

coord_aal = pd.read_excel("Table_1.xls", index_col=None, header=None) #get coordinates for AAL atlas

#1) plot the time series 
one_participant = TimeSeries[1][0]              #get the ts for the first participant (selected randomly)
plt.plot(one_participant[0:100,0]);             #plot the ts for the first 100 timepoints for the first ROI
plt.xlabel("time",fontsize=15);                 # x-axis label
plt.ylabel("connection weight",fontsize=15);    #y-axis label
plt.title("Time Course example - ROI 1")        #title for the plot


fc = np.corrcoef(one_participant.T)             #whole-brain connectivity matrix
plt.imshow(fc, interpolation='nearest')         #plot the connectivity matrix
plt.colorbar()                                  #add a colorbar
plt.title("Connectivity matrix - whole brain",fontsize=15)  #title for matrix
plt.xlabel("ROIs",fontsize=15)                              #x-axis label
plt.ylabel("ROIs",fontsize=15)                              #y-axis label

#view 2D network
fc_thr = np.array(Thresholds(fc, lbthr, lbthr, 1)[1])[0] #threshold figure for better presentation
f = plt.figure(figsize=(9, 3))                           # set-up figure
plotting.plot_connectome(fc_thr,                         # network
                          node_coords=coord_aal,         # node coordinates
                          node_color='black',            # node colors (here, uniform)
                          node_size=30,                  # node sizes (here, uniform)
                          edge_cmap='RdGy',              #change the edge colormap
                          colorbar = True,               #add a colorbar
                          figure=f) 

##DMN matrix
DMN_one_participant = (one_participant[:,indices])  #getting the ts only for ROIs in the DMN
DMNfc = np.corrcoef(DMN_one_participant.T)          #correlation matrix for DMN 
plt.imshow(DMNfc, interpolation='nearest')          #plot the connectivity matrix
plt.colorbar()                                      #add a colorbar
plt.title("Connectivity matrix - DMN",fontsize=15)  #title for matrix
plt.xlabel("ROIs",fontsize=15)                      #x-axis label
plt.ylabel("ROIs",fontsize=15)                      #y-axis label

#view DMN 2D network
f = plt.figure(figsize=(9, 3))                   # set-up figure
plotting.plot_connectome(DMNfc,                  # DMN network
                          node_coords=coord_aal.iloc[indices,:], # node coordinates
                          node_color='black',    # node colors (here, uniform)
                          node_size=30,          # node sizes (here, uniform)
                          edge_cmap='RdGy',      # change the edge colormap
                          colorbar = True,       #add a colorbar
                          figure=f)     

                                ##%%% boxplot for results

labels = ['controls']*len(IDscontrols)+['ADHDC']*len(IDsADHDC)+['ADHDI']*len(IDsADHDI) #get group labels

                                #%%%% Global measures ##
                                
wb_m = []                                                        #create empty list for whole brain measure
DMN_m = []                                                       #create empty list for DMN measure

for i in range(1,len(measures_names)):                           #loop through each measure taken
    m = [np.array(Extract(GraphMeasures_controls, i, 0)), 
           np.array(Extract(GraphMeasures_ADHDI, i, 0)), 
           np.array(Extract(GraphMeasures_ADHDC, i, 0))]         #get the measure (at thr 0.05) for all groups

    m = [item for sublist in m for item in sublist]              #flatten the list
    
    wb_m.append(m)                                               #add to main list

    DMNm = [np.array(Extract(DMNGraphMeasures_controls, i, 0)), 
           np.array(Extract(DMNGraphMeasures_ADHDI, i, 0)), 
           np.array(Extract(DMNGraphMeasures_ADHDC, i, 0))]      #get the measure (at thr 0.05) for all groups

    DMNm = [item for sublist in DMNm for item in sublist]        #flatten the list
    
    DMN_m.append(DMNm)                                           #add to main list

fig, axes = plt.subplots(2,4, figsize = (20,12))                 #create figure and axes

#create a dataframe with the measures, both whole-brain and DMN with the group as a separate column
df = pd.DataFrame(data={'WB mean conn':wb_m[0],                  #WB mean connectivity
                    'WB Efficiency': wb_m[1],                    #WB efficiency
                    'WB Characteristic PL': wb_m[2],             #WB characteristic path length
                    'WB CLustering': wb_m[3],                    #WB clustering
                    'DMN mean conn': DMN_m[0],                   #DMN mean connectivity
                    'DMN Efficiency':DMN_m[1],                   #DMN efficiency
                    'DMN Characterstic PL':DMN_m[2],             #DMN charactersitic path length
                    'DMN CLustering':DMN_m[3],                   #DMN clustering 
                    'group': labels})                            #group labels

for i,el in enumerate(list(df.columns.values)[:-1]):                          #for each column
    a = df.boxplot(el, by="group", ax=axes.flatten()[i], fontsize=18)         #plot a subplot
    a.set_title(df.columns[i], fontsize = 18)                                 #set the title for each subplot
    
fig.suptitle("Global Graph Theory measures whole brain and DMN - minimal thresholding",
             fontsize = 20)                                      #add a title over all graphs
plt.tight_layout()                                               #tight layput
plt.savefig('globalplot.png')                                    #save figure
plt.show()                                                       #show figure


                            #%%%% Regional measures ##
                            
betwDMN = []   #create an empty list with the values of betweenness centrality for each node

for i in range(len(DMNGraphMeasures_controls[0][0][0])):                #for each node in the DMN

    #extract the betweenness centrality for each participant from each group
    betw_DMN = [np.array(Extract2(DMNGraphMeasures_controls, 0, 0, i)),
                  np.array(Extract2(DMNGraphMeasures_ADHDI, 0, 0,i)), 
                  np.array(Extract2(DMNGraphMeasures_ADHDC, 0, 0, i))]
                     
    betw_DMN = [item for sublist in betw_DMN for item in sublist]      #flatten the list

    betwDMN.append(betw_DMN)                                           #add to main list
    
#make into a dataframe
betw_df = pd.DataFrame(data = {"Front Sup Med L":betwDMN[0],           #each column will correspond to a node
                               "Front Sup Med R":betwDMN[1],
                               "Cingulum Ant L":betwDMN[2],
                               "Cingulum Ant R":betwDMN[3],
                               "Cingulum Post L":betwDMN[4],
                               "Cingulum Post R":betwDMN[5],
                               "Angulum L":betwDMN[6],
                               "Angulum R":betwDMN[7],
                               "Precuneus L":betwDMN[8],
                               "Precuneus R":betwDMN[9],
                               'group': labels})               #a final column will indicate the group they belong to 
    
new_df = pd.melt(betw_df, id_vars='group',                     #tranform the dataframe from wide to long format
                 value_vars=list(betw_df.columns), var_name="value", 
                 value_name="Nodes", ignore_index=False)

sns.set(font_scale=3)                                         #set scale for legends                              
ax = sns.boxplot(data=new_df, x="value", y="Nodes", hue="group").set(title = 'Betweenness Centrality per node by groups')  #plot for each node for all groups
plt.xticks(rotation=30)                                       #rotate the x-axis ticks by 30o
sns.set(rc={'figure.figsize':(45,20)})                        #set the figure size
plt.savefig('regplot.png')                                    #save figure

                            ### END OF CODE ###   
