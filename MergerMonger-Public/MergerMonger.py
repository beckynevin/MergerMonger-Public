'''
~~~
Contains all the utilities to run MergerMonger including the code to create the LDA classification and run on SDSS galaxies.
FILL THIS IN ABOUT ALL THE UTILITIES
~~~
'''

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import pandas as pd
import seaborn as sns
import os
from os import path
from util_LDA import run_LDA, run_RFR, run_RFC, cross_term
import scipy
from util_SDSS import download_sdss_ra_dec_table, download_galaxy
from util_smelter import get_predictors
import random
from astropy.io import fits
import os
import seaborn as sns
import matplotlib.colors as colors
import time

def convert_LDA_to_pmerg(LDA):
    return 1/(1 + np.exp(-LDA))

def locate_min(a):
    smallest = min(a)
    return smallest, [index for index, element in enumerate(a)
                  if smallest == element]
    
def get_df(prefix, run, verbose=False):
    feature_dict = {i:label for i,label in zip(
                range(39),
                  ('Counter_x',
                  'Image',
                  'class label',
                  'Myr',
                  'Viewpoint',
                '# Bulges',
                   'Sep',
                   'Flux Ratio',
                  'Gini',
                  'M20',
                  'Concentration (C)',
                  'Asymmetry (A)',
                  'Clumpiness (S)',
                  'Sersic N',
                  'Shape Asymmetry (A_S)',
                  'Counter_y',
                  'Delta PA',
                            'v_asym',
                            's_asym',
                            'resids',
                            'lambda_r',
                            'epsilon',
                            'A',
                            'A_2',
                            'deltapos',
                            'deltapos2',
                            'nspax','re',
                            'meanvel','varvel','skewvel','kurtvel',
                  'meansig','varsig','skewsig','kurtsig','abskewvel','abskewsig','random'))}

    features_list = ['Gini','M20','Concentration (C)','Asymmetry (A)','Clumpiness (S)','Sersic N','Shape Asymmetry (A_S)','random']


    if run[0:12]=='major_merger':
        priors=[0.9,0.1]#[0.75,0.25]]
    else:
        if run[0:12]=='minor_merger':
            priors=[0.7,0.3]#[0.75,0.25]]
        else:
            STOP

    # Read in the the measured predictor values from the LDA table
    df = pd.io.parsers.read_csv(filepath_or_buffer = prefix + 'LDA_merged_'+str(run)+'.txt',header=[0],sep='\t')

    #Rename all of the kinematic columns (is this necessary?)
    df.rename(columns={'kurtvel':'$h_{4,V}$','kurtsig':'$h_{4,\sigma}$','lambda_r':'\lambdare',
             'epsilon':'$\epsilon$','Delta PA':'$\Delta$PA','A_2':'$A_2$',
              'varsig':'$\sigma_{\sigma}$',
             'meanvel':'$\mu_V$','abskewvel':'$|h_{3,V}|$',
             'abskewsig':'$|h_{3,\sigma}|$',
             'meansig':'$\mu_{\sigma}$',
             'varvel':'$\sigma_{V}$'},
    inplace=True)
    
    df.columns = [l for i,l in sorted(feature_dict.items())]

    df.dropna(how="all", inplace=True) # to drop the empty line at file-end
    df.dropna(inplace=True) # to drop the empty line at file-end
    if verbose:
        print(df['class label'].value_counts())
    myr=[]
    myr_non=[]
    for j in range(len(df)):
        if df[['class label']].values[j][0]==0.0:
            myr_non.append(df[['Myr']].values[j][0])
        else:
            myr.append(df[['Myr']].values[j][0])
    if verbose:
        print('myr that are considered merging', sorted(set(myr)))
        print('myr that are nonmerging', sorted(set(myr_non)))


    

    myr_non=sorted(list(set(myr_non)))
    myr=sorted(list(set(myr)))
    
    return df, myr, myr_non, features_list

def get_df_simulation_sliding_time(prefix, post_coal, run):
    """_summary_

    Args:
        post_coal (_type_): _description_
        run (_type_): _description_
        verbose (bool, optional): _description_. Defaults to True.
        plot (bool, optional): _description_. Defaults to True.
    """

    feature_dict = {i:label for i,label in zip(
                range(39),
                  ('Counter_x',
                  'Image',
                  'class label',
                  'Myr',
                  'Viewpoint',
                '# Bulges',
                   'Sep',
                   'Flux Ratio',
                  'Gini',
                  'M20',
                  'Concentration (C)',
                  'Asymmetry (A)',
                  'Clumpiness (S)',
                  'Sersic N',
                  'Shape Asymmetry (A_S)',
                  'Counter_y',
                  'Delta PA',
                              'v_asym',
                            's_asym',
                            'resids',
                            'lambda_r',
                            'epsilon',
                            'A',
                            'A_2',
                            'deltapos',
                            'deltapos2',
                            'nspax','re',
                            'meanvel','varvel','skewvel','kurtvel',
                  'meansig','varsig','skewsig','kurtsig','abskewvel','abskewsig','random'))}

    features_list = ['Gini','M20','Concentration (C)','Asymmetry (A)','Clumpiness (S)','Sersic N','Shape Asymmetry (A_S)','random']

    run_short = str(str(run.split('_include_')[0]).split('_postc')[0])
    df = pd.io.parsers.read_csv(filepath_or_buffer= prefix + 'LDA_merged_'+str(run_short)+'.txt',header=[0],sep='\t')
    
    #Rename all of the kinematic columns (is this necessary?)                                                         
    df.rename(columns={'kurtvel':'$h_{4,V}$','kurtsig':'$h_{4,\sigma}$','lambda_r':'\lambdare',
             'epsilon':'$\epsilon$','Delta PA':'$\Delta$PA','A_2':'$A_2$',
              'varsig':'$\sigma_{\sigma}$',
             'meanvel':'$\mu_V$','abskewvel':'$|h_{3,V}|$',
             'abskewsig':'$|h_{3,\sigma}|$',
             'meansig':'$\mu_{\sigma}$',
             'varvel':'$\sigma_{V}$'},
    inplace=True)

    df.columns = [l for _,l in sorted(feature_dict.items())]

    df.dropna(how="all", inplace=True) # to drop the empty line at file-end                                           
    df.dropna(inplace=True) # to drop the empty line at file-end      



    # Define the coalescence point for the different 'Image' names:



    print('time since coalescence where we are drawing the line', post_coal)
   
  
    # Maybe just go through the whole dataframe and do this yourself:
    
    df['class label'] = np.where((df['Image']=='fg3_m12') & (df['Myr'] > (2.15 +post_coal)),0, df['class label'])
    df['class label'] = np.where((df['Image']=='fg1_m13') & (df['Myr'] > (2.74 +post_coal)),0, df['class label'])
    df['class label'] = np.where((df['Image']=='fg3_m13') & (df['Myr'] > (2.59 +post_coal)),0, df['class label'])
    df['class label'] = np.where((df['Image']=='fg3_m15') & (df['Myr'] > (3.72 +post_coal)),0, df['class label'])
    df['class label'] = np.where((df['Image']=='fg3_m10') & (df['Myr'] > (9.17 +post_coal)),0, df['class label'])

    # Also have to make sure that if the post_coal is greater than 0.5 that everything less
    # that remains but is labeled as a 1
    df['class label'] = np.where((df['Image']=='fg3_m12') & (df['Myr'] < (2.15 +post_coal)),1, df['class label'])
    df['class label'] = np.where((df['Image']=='fg1_m13') & (df['Myr'] < (2.74 +post_coal)),1, df['class label'])
    df['class label'] = np.where((df['Image']=='fg3_m13') & (df['Myr'] < (2.59 +post_coal)),1, df['class label'])
    df['class label'] = np.where((df['Image']=='fg3_m15') & (df['Myr'] < (3.72 +post_coal)),1, df['class label'])
    df['class label'] = np.where((df['Image']=='fg3_m10') & (df['Myr'] < (9.17 +post_coal)),1, df['class label'])


     
    # Make sure you delete everything from before coalescence:
    
    # Get names of indexes that below to a given name and were formerly labeled as mergers and are before coal:
    indexNames = df[ (df['Image']=='fg3_m12') & (df['Myr'] <2.15) & (df['class label'] ==1) ].index
    # Delete these row indexes from dataFrame
    df.drop(indexNames , inplace=True)

    indexNames = df[ (df['Image']=='fg1_m13') & (df['Myr'] <2.74) & (df['class label'] ==1) ].index
    df.drop(indexNames , inplace=True)

    indexNames = df[ (df['Image']=='fg3_m13') & (df['Myr'] <2.59) & (df['class label'] ==1) ].index
    df.drop(indexNames , inplace=True)

    indexNames = df[ (df['Image']=='fg3_m15') & (df['Myr'] <3.72) & (df['class label'] ==1) ].index
    df.drop(indexNames , inplace=True)

    indexNames = df[ (df['Image']=='fg3_m10') & (df['Myr'] <9.17) & (df['class label'] ==1) ].index
    df.drop(indexNames , inplace=True)


    myr=[]
    myr_non=[]
    for j in range(len(df)):
        if df[['class label']].values[j][0]==0.0:
            myr_non.append(df[['Myr']].values[j][0])
        else:
            myr.append(df[['Myr']].values[j][0])


    myr_non=sorted(list(set(myr_non)))
    myr=sorted(list(set(myr)))

    return df, myr, myr_non, features_list

def load_LDA_from_simulation_sliding_time_include_coal(prefix, post_coal,  run,  verbose=True, plot=True):
    """_summary_

    Args:
        post_coal (_type_): _description_
        run (_type_): _description_
        verbose (bool, optional): _description_. Defaults to True.
        plot (bool, optional): _description_. Defaults to True.
    """
    
    if run[0:12]=='major_merger':
        priors=[0.9,0.1]#[0.75,0.25]]                                                                                 
    else:
        if run[0:12]=='minor_merger':
            priors=[0.7,0.3]#[0.75,0.25]]                                                                             
        else:
            STOP

    df, myr, myr_non, features_list = get_df_simulation_sliding_time(prefix,post_coal,run)
    

    terms_RFR, _ = run_RFC(df, features_list, verbose)
    output_LDA = run_LDA(df, priors,terms_RFR, myr, myr_non, 21,  verbose)


    return output_LDA, terms_RFR, df#, len_nonmerg, len_merg    

                                                   

def load_LDA_from_simulation_changing_priors(prefix, run, priors, verbose=False, plot=False):

    if 'postc' in run:
        print('postc in run')
        # From this get the post_coal value
        post_coal = float(str(run.split('coal_')[1]))#.split('.')[0])
        
        df, myr, myr_non, features_list = get_df_simulation_sliding_time(prefix, post_coal,run)
        
        
    else:
        df, myr, myr_non, features_list = get_df(prefix, run)


    terms_RFR, _ = run_RFC(df, features_list, verbose)
    

    output_LDA = run_LDA(df, priors,terms_RFR, myr, myr_non, 21,  verbose)

    LDA_ID = output_LDA[8]

    

    
    if plot==True:
        '''Now make a beautiful plot by first making contours from the sims'''
        '''But also reclassify all of df to compare'''


        nonmerg_gini=[]
        nonmerg_m20=[]

        merg_gini=[]
        merg_m20=[]

        nonmerg_gini_LDA=[]
        nonmerg_m20_LDA=[]

        merg_gini_LDA=[]
        merg_m20_LDA=[]


        nonmerg_C_LDA=[]
        nonmerg_A_LDA=[]
        nonmerg_S_LDA=[]
        nonmerg_n_LDA=[]
        nonmerg_A_S_LDA=[]

        merg_C_LDA=[]
        merg_A_LDA=[]
        merg_S_LDA=[]
        merg_n_LDA=[]
        merg_A_S_LDA=[]

        LDA_ID_merg = []
        LDA_ID_nonmerg = []

        indices_nonmerg = []
        indices_merg = []

        merg_p = []
        nonmerg_p = []

        for j in range(len(df)):
            if df['class label'].values[j]==0:
                nonmerg_gini.append(df['Gini'].values[j])
                nonmerg_m20.append(df['M20'].values[j])
                LDA_ID_nonmerg.append(LDA_ID[j])
                indices_nonmerg.append(j)
            if df['class label'].values[j]==1:
                merg_gini.append(df['Gini'].values[j])
                merg_m20.append(df['M20'].values[j])
                LDA_ID_merg.append(LDA_ID[j])
                indices_merg.append(j)
            
            p_merg_sim = 1/(1 + np.exp(-LDA_ID[j]))
            if LDA_ID[j]<0:#then its a nonmerger
                nonmerg_gini_LDA.append(df['Gini'].values[j])
                nonmerg_m20_LDA.append(df['M20'].values[j])
                nonmerg_C_LDA.append(df['Concentration (C)'].values[j])
                nonmerg_A_LDA.append(df['Asymmetry (A)'].values[j])
                nonmerg_S_LDA.append(df['Clumpiness (S)'].values[j])
                nonmerg_n_LDA.append(df['Sersic N'].values[j])
                nonmerg_A_S_LDA.append(df['Shape Asymmetry (A_S)'].values[j])
                nonmerg_p.append(p_merg_sim)
                
            if LDA_ID[j]>0:#then its a nonmerger
                merg_gini_LDA.append(df['Gini'].values[j])
                merg_m20_LDA.append(df['M20'].values[j])
                merg_C_LDA.append(df['Concentration (C)'].values[j])
                merg_A_LDA.append(df['Asymmetry (A)'].values[j])
                merg_S_LDA.append(df['Clumpiness (S)'].values[j])
                merg_n_LDA.append(df['Sersic N'].values[j])
                merg_A_S_LDA.append(df['Shape Asymmetry (A_S)'].values[j])
                merg_p.append(p_merg_sim)
            

        



         

        merg_m20_LDA=np.array(merg_m20_LDA)
        merg_gini_LDA=np.array(merg_gini_LDA)
        merg_C_LDA=np.array(merg_C_LDA)
        merg_A_LDA=np.array(merg_A_LDA)
        merg_S_LDA=np.array(merg_S_LDA)
        merg_n_LDA=np.array(merg_n_LDA)
        merg_A_S_LDA=np.array(merg_A_S_LDA)

        nonmerg_m20_LDA=np.array(nonmerg_m20_LDA)
        nonmerg_gini_LDA=np.array(nonmerg_gini_LDA)
        nonmerg_C_LDA=np.array(nonmerg_C_LDA)
        nonmerg_A_LDA=np.array(nonmerg_A_LDA)
        nonmerg_S_LDA=np.array(nonmerg_S_LDA)
        nonmerg_n_LDA=np.array(nonmerg_n_LDA)
        nonmerg_A_S_LDA=np.array(nonmerg_A_S_LDA)



        merg_m20=np.array(merg_m20)
        merg_gini=np.array(merg_gini)
        nonmerg_m20=np.array(nonmerg_m20)
        nonmerg_gini=np.array(nonmerg_gini)

    

    return output_LDA, terms_RFR, df

# This is for playing around with changing the number of mergers in the validation set
def load_LDA_from_simulation_changing_priors_changing_validation_set(run, prefix_frames, priors, verbose=True, plot=True):


    feature_dict = {i:label for i,label in zip(
                range(39),
                  ('Counter_x',
                  'Image',
                  'class label',
                  'Myr',
                  'Viewpoint',
                '# Bulges',
                   'Sep',
                   'Flux Ratio',
                  'Gini',
                  'M20',
                  'Concentration (C)',
                  'Asymmetry (A)',
                  'Clumpiness (S)',
                  'Sersic N',
                  'Shape Asymmetry (A_S)',
                  'Counter_y',
                  'Delta PA',
                            'v_asym',
                            's_asym',
                            'resids',
                            'lambda_r',
                            'epsilon',
                            'A',
                            'A_2',
                            'deltapos',
                            'deltapos2',
                            'nspax','re',
                            'meanvel','varvel','skewvel','kurtvel',
                  'meansig','varsig','skewsig','kurtsig','abskewvel','abskewsig','random'))}

    features_list = ['Gini','M20','Concentration (C)','Asymmetry (A)','Clumpiness (S)','Sersic N','Shape Asymmetry (A_S)','random']


    

    # Read in the the measured predictor values from the LDA table
    df = pd.io.parsers.read_csv(filepath_or_buffer='../Tables/LDA_merged_'+str(run)+'.txt',header=[0],sep='\t')

    #Rename all of the kinematic columns (is this necessary?)
    df.rename(columns={'kurtvel':'$h_{4,V}$','kurtsig':'$h_{4,\sigma}$','lambda_r':'\lambdare',
             'epsilon':'$\epsilon$','Delta PA':'$\Delta$PA','A_2':'$A_2$',
              'varsig':'$\sigma_{\sigma}$',
             'meanvel':'$\mu_V$','abskewvel':'$|h_{3,V}|$',
             'abskewsig':'$|h_{3,\sigma}|$',
             'meansig':'$\mu_{\sigma}$',
             'varvel':'$\sigma_{V}$'},
    inplace=True)
    
    df.columns = [l for i,l in sorted(feature_dict.items())]

    df.dropna(how="all", inplace=True) # to drop the empty line at file-end
    df.dropna(inplace=True) # to drop the empty line at file-end
    if verbose:
        print(df['class label'].value_counts())
    myr=[]
    myr_non=[]
    for j in range(len(df)):
        if df[['class label']].values[j][0]==0.0:
            myr_non.append(df[['Myr']].values[j][0])
        else:
            myr.append(df[['Myr']].values[j][0])
    if verbose:
        print('myr that are considered merging', sorted(set(myr)))
        print('myr that are nonmerging', sorted(set(myr_non)))



    myr_non=sorted(list(set(myr_non)))
    myr=sorted(list(set(myr)))
    
    
    

    terms_RFR, _ = run_RFC(df, features_list, verbose)


    

    output_LDA = run_LDA(run, df, priors,terms_RFR, myr, myr_non, 21,  verbose)
   
    


    

    return output_LDA, terms_RFR, df#, len_nonmerg, len_merg


def load_LDA_from_simulation(prefix, run, verbose=True, plot=True):


    df, myr, myr_non, features_list = get_df(prefix, run)
    
    

    terms_RFR, _ = run_RFC(df, features_list, verbose)
    
    if run[0:12]=='major_merger':
        priors=[0.9,0.1]#[0.75,0.25]]                                                                                 
    else:
        if run[0:12]=='minor_merger':
            priors=[0.7,0.3]#[0.75,0.25]]                                                                             
        else:
            STOP
    

    output_LDA = run_LDA( df, priors, terms_RFR, myr, myr_non, 21,  verbose)
    
    

    inputs_all = output_LDA[2]

    LDA_ID = output_LDA[8]

    

    
    if plot==True:
        '''Now make a beautiful plot by first making contours from the sims'''
        '''But also reclassify all of df to compare'''


        nonmerg_gini=[]
        nonmerg_m20=[]

        merg_gini=[]
        merg_m20=[]

        nonmerg_gini_LDA=[]
        nonmerg_m20_LDA=[]

        merg_gini_LDA=[]
        merg_m20_LDA=[]


        nonmerg_C_LDA=[]
        nonmerg_A_LDA=[]
        nonmerg_S_LDA=[]
        nonmerg_n_LDA=[]
        nonmerg_A_S_LDA=[]

        merg_C_LDA=[]
        merg_A_LDA=[]
        merg_S_LDA=[]
        merg_n_LDA=[]
        merg_A_S_LDA=[]

        LDA_ID_merg = []
        LDA_ID_nonmerg = []

        indices_nonmerg = []
        indices_merg = []

        merg_p = []
        nonmerg_p = []

        for j in range(len(df)):
            if df['class label'].values[j]==0:
                nonmerg_gini.append(df['Gini'].values[j])
                nonmerg_m20.append(df['M20'].values[j])
                LDA_ID_nonmerg.append(LDA_ID[j])
                indices_nonmerg.append(j)
            if df['class label'].values[j]==1:
                merg_gini.append(df['Gini'].values[j])
                merg_m20.append(df['M20'].values[j])
                LDA_ID_merg.append(LDA_ID[j])
                indices_merg.append(j)
            
            p_merg_sim = 1/(1 + np.exp(-LDA_ID[j]))
            if LDA_ID[j]<0:#then its a nonmerger
                nonmerg_gini_LDA.append(df['Gini'].values[j])
                nonmerg_m20_LDA.append(df['M20'].values[j])
                nonmerg_C_LDA.append(df['Concentration (C)'].values[j])
                nonmerg_A_LDA.append(df['Asymmetry (A)'].values[j])
                nonmerg_S_LDA.append(df['Clumpiness (S)'].values[j])
                nonmerg_n_LDA.append(df['Sersic N'].values[j])
                nonmerg_A_S_LDA.append(df['Shape Asymmetry (A_S)'].values[j])
                nonmerg_p.append(p_merg_sim)
                
            if LDA_ID[j]>0:#then its a nonmerger
                merg_gini_LDA.append(df['Gini'].values[j])
                merg_m20_LDA.append(df['M20'].values[j])
                merg_C_LDA.append(df['Concentration (C)'].values[j])
                merg_A_LDA.append(df['Asymmetry (A)'].values[j])
                merg_S_LDA.append(df['Clumpiness (S)'].values[j])
                merg_n_LDA.append(df['Sersic N'].values[j])
                merg_A_S_LDA.append(df['Shape Asymmetry (A_S)'].values[j])
                merg_p.append(p_merg_sim)
            




        dashed_line_x=np.linspace(-0.5,-3,100)
        

        merg_m20_LDA=np.array(merg_m20_LDA)
        merg_gini_LDA=np.array(merg_gini_LDA)
        merg_C_LDA=np.array(merg_C_LDA)
        merg_A_LDA=np.array(merg_A_LDA)
        merg_S_LDA=np.array(merg_S_LDA)
        merg_n_LDA=np.array(merg_n_LDA)
        merg_A_S_LDA=np.array(merg_A_S_LDA)

        nonmerg_m20_LDA=np.array(nonmerg_m20_LDA)
        nonmerg_gini_LDA=np.array(nonmerg_gini_LDA)
        nonmerg_C_LDA=np.array(nonmerg_C_LDA)
        nonmerg_A_LDA=np.array(nonmerg_A_LDA)
        nonmerg_S_LDA=np.array(nonmerg_S_LDA)
        nonmerg_n_LDA=np.array(nonmerg_n_LDA)
        nonmerg_A_S_LDA=np.array(nonmerg_A_S_LDA)



        merg_m20=np.array(merg_m20)
        merg_gini=np.array(merg_gini)
        nonmerg_m20=np.array(nonmerg_m20)
        nonmerg_gini=np.array(nonmerg_gini)

    

    return output_LDA, terms_RFR, df#, len_nonmerg, len_merg


# Taking the output of the LDA (trained on simulated examples),
# this code runs the classification on the full SDSS dataset

def classify(prefix, prefix_frames, run, LDA, terms_RFR, df, verbose=False):
    #~~~~~~~
    # Now bring in the SDSS galaxies!
    #~~~~~~~
    if verbose:
        print('loading up predictor value table........')
    df2 = pd.io.parsers.read_csv(prefix+'SDSS_predictors_all.txt', sep='\t')
    

    #df2 = df2[0:10000]
    
    if len(df2.columns) ==15: #then you have to delete the first column which is an empty index
        df2 = df2.iloc[: , 1:]

    

    # First, delete all rows that have weird values of n:
    #print('len before crazy values', len(df2))
    #df_filtered = df2[df2['Sersic N'] < 10]

    #df_filtered_2 = df_filtered[df_filtered['Asymmetry (A)'] > -1]

    #df2 = df_filtered_2

    # Delete duplicates:
    if verbose:
        print('len bf duplicate delete', len(df2))
    df2_nodup = df2.duplicated()
    df2 = df2[~df2_nodup]
    if verbose:
        print('len af duplicate delete', len(df2))



    
    input_singular = terms_RFR
    #Okay so this next part actually needs to be adaptable to reproduce all possible cross-terms
    crossterms = []
    ct_1 = []
    ct_2 = []
    for j in range(len(input_singular)):
        for i in range(len(input_singular)):
            if j == i or i < j:
                continue
            #input_singular.append(input_singular[j]+'*'+input_singular[i])
            crossterms.append(input_singular[j]+'*'+input_singular[i])
            ct_1.append(input_singular[j])
            ct_2.append(input_singular[i])

    inputs = input_singular + crossterms

    # Now you have to construct a bunch of new rows to the df that include all of these cross-terms
    for j in range(len(crossterms)):
        
        df2[crossterms[j]] = df2.apply(cross_term, axis=1, args=(ct_1[j], ct_2[j]))
        

    X_gal = df2[LDA[2]].values




    X_std=[]
    testing_C=[]
    testing_A=[]
    testing_Gini=[]
    testing_M20=[]
    testing_n=[]
    testing_A_S=[]

    testing_n_stat=[]
    testing_A_S_stat=[]
    testing_S_N=[]
    testing_A_R = []

    LD1_SDSS=[]
    p_merg_list=[]
    score_merg=[]
    score_nonmerg=[]

    if run[0:12]=='major_merger':
        prior_nonmerg = 0.9
        prior_merg = 0.1
    else:
        if run[0:12]=='minor_merger':
            prior_nonmerg = 0.7
            prior_merg = 0.3
            
        else:
            STOP


    if verbose:
        print('this is what its hanging on', prefix+'LDA_out_all_SDSS_predictors_'+str(run)+'.txt')
    exists = 0
    if path.exists(prefix+'LDA_out_all_SDSS_predictors_'+str(run)+'.txt'):
        exists = 1
        if verbose:
            print('Table already exists')
    else:
	   # Make a table with merger probabilities and other diagnostics:
        if verbose:
        	print('making table of LDA output for all galaxies.....')
        file_out = open(prefix+'LDA_out_all_SDSS_predictors_'+str(run)+'.txt','w')
        file_out.write('ID'+'\t'+'Classification'+'\t'+'LD1'+'\t'+'p_merg'+'\t'+'p_nonmerg'+'\t'+'Leading_term'+'\t'+'Leading_coef'+'\n')
    	#+'Second_term'+'\t'+'Second_coef'+'\n')

    most_influential_nonmerg = []
    most_influential_merg = []
    most_influential_nonmerg_c = []
    most_influential_merg_c = []

    LDA_merg_SDSS = []
    LDA_nonmerg_SDSS = []

    p_merg_merg_SDSS = []
    p_merg_nonmerg_SDSS = []
    
    

    for j in range(len(X_gal)):
        #print(X_gal[j])
        X_standardized = list((X_gal[j]-LDA[0])/LDA[1])
        X_std.append(X_standardized)
        # use the output from the simulation to assign LD1 value:
        LD1_gal = float(np.sum(X_standardized*LDA[3])+LDA[4])
        
        
        
        LD1_SDSS.append(LD1_gal)
        
        # According to my calculations, LD1 = delta_1 - delta_0 where delta is the score for each class
        # Therefore, in the probability equation p_merg = e^delta_1/(e^delta_1+e^delta_0)
        # you can sub in LD1 and instead end up with p_merg = 1/(1+e^-LD1)
        
        p_merg = 1/(1 + np.exp(-LD1_gal))
        p_nonmerg = 1/(1 + np.exp(LD1_gal))
        

        
        if LD1_gal > 0:
            merg = 1
            # select the thing that is the most positive
            max_idx = np.argmax(X_standardized*LDA[3])
            most_influential_coeff = (X_standardized*LDA[3])[0,max_idx]
            most_influential_term = LDA[2][max_idx]
            
            most_influential_merg.append(most_influential_term)
            most_influential_merg_c.append(most_influential_coeff)
            LDA_merg_SDSS.append(LD1_gal)
            p_merg_merg_SDSS.append(p_merg)
        else:
            merg = 0
            LDA_nonmerg_SDSS.append(LD1_gal)
            p_merg_nonmerg_SDSS.append(p_merg)
            # select the thing that is the most positive
            max_idx = np.argmin(X_standardized*LDA[3])
            most_influential_coeff = (X_standardized*LDA[3])[0,max_idx]
            most_influential_term = LDA[2][max_idx]
            
            most_influential_nonmerg.append(most_influential_term)
            most_influential_nonmerg_c.append(most_influential_coeff)
        if exists==1:
            pass
        else:
            file_out.write(str(df2[['ID']].values[j][0])+'\t'+str(merg)+'\t'+str(round(LD1_gal,3))+'\t'+str(round(p_merg,3))+'\t'+str(round(p_nonmerg,3))+'\t'+most_influential_term+'\t'+str(most_influential_coeff)+'\n')
        p_merg_list.append(p_merg)
                
        testing_Gini.append(df2['Gini'].values[j])
        testing_M20.append(df2['M20'].values[j])
        testing_C.append(df2['Concentration (C)'].values[j])
        testing_A.append(df2['Asymmetry (A)'].values[j])
        testing_n.append(df2['Sersic N'].values[j])
        testing_A_S.append(df2['Shape Asymmetry (A_S)'].values[j])
    if exists==1:
        pass
    else:
    	file_out.close()
    	print('finished writing out LDA file')










    if verbose:
        plt.clf()
        plt.hist(testing_Gini)
        plt.savefig('../Figures/hist_Gini.pdf')
        plt.clf()
        plt.hist(testing_M20)
        plt.savefig('../Figures/hist_M20.pdf')
        plt.clf()
        plt.hist(testing_C)
        plt.savefig('../Figures/hist_C.pdf')
        plt.clf()
        plt.hist(testing_A)
        plt.savefig('../Figures/hist_A.pdf')
        plt.clf()
        plt.hist(testing_n)
        plt.savefig('../Figures/hist_n.pdf')
        plt.clf()
        plt.hist(testing_A_S)
        plt.savefig('../Figures/hist_A_S.pdf')
        
        
    #Get numbers of which was the most influential coefficient for both mergers and nonmergers:
    set_merg = set(most_influential_merg)
    set_nonmerg = set(most_influential_nonmerg)

    list_count = []
    list_elements = []
    # Now look throw the full list to count
    for element in set_merg:
        # Now collect all the values of the coeff:
        coef_here = []
        for j in range(len(most_influential_merg_c)):
        
            if most_influential_merg[j]==element:
                coef_here.append(most_influential_merg_c[j])
        list_count.append(most_influential_merg.count(element))
        list_elements.append(element)
        
    max_idx = np.where(list_count==np.sort(list_count)[-1])[0][0]
    if verbose:
        print('top 1 merger', np.sort(list_count)[-1], list_elements[max_idx])
    first_imp_merg = list_elements[max_idx]
    try:
        sec_max_idx = np.where(list_count==np.sort(list_count)[-2])[0][0]
        if verbose:
            print('top 2 merger', np.sort(list_count)[-2], list_elements[sec_max_idx])
        second_imp_merg = list_elements[sec_max_idx]
    except IndexError:
        if verbose:
            print('no second term')
        second_imp_merg = first_imp_merg



    list_count = []
    list_elements = []
    for element in set_nonmerg:
        coef_here = []
        for j in range(len(most_influential_nonmerg_c)):
        
            if most_influential_nonmerg[j]==element:
                coef_here.append(most_influential_nonmerg_c[j])
        list_count.append(most_influential_nonmerg.count(element))
        list_elements.append(element)
        print('element', element, most_influential_nonmerg.count(element), np.mean(coef_here))
    max_idx = np.where(list_count==np.sort(list_count)[-1])[0][0]
    print('top 1 nonmerger', np.sort(list_count)[-1], list_elements[max_idx])
    first_imp_nonmerg = list_elements[max_idx]
    try:
        sec_max_idx = np.where(list_count==np.sort(list_count)[-2])[0][0]
        print('top 2 nonmerger', np.sort(list_count)[-2], list_elements[sec_max_idx])
        second_imp_nonmerg = list_elements[sec_max_idx]
    except IndexError:
        print('no second term')
        second_imp_nonmerg = first_imp_nonmerg



    sim_nonmerg_pred_1 = []
    sim_merg_pred_1 = []
    sim_nonmerg_pred_2 = []
    sim_merg_pred_2 = []


    # Will need the actual table values for the simulated galaxies to run this part
    

    for j in range(len(df)):
        if LDA[8][j]<0:#then its a nonmerger
            sim_nonmerg_pred_1.append(df[first_imp_nonmerg].values[j])
            sim_nonmerg_pred_2.append(df[second_imp_nonmerg].values[j])
            
            
        if LDA[8][j]>0:#then its a nonmerger
            sim_merg_pred_1.append(df[first_imp_merg].values[j])
            sim_merg_pred_2.append(df[second_imp_merg].values[j])


    merg_pred_1=[]
    merg_pred_2=[]
    nonmerg_pred_1=[]
    nonmerg_pred_2=[]

    merg_gini_LDA_out=[]
    merg_m20_LDA_out=[]
    merg_C_LDA_out=[]
    merg_A_LDA_out=[]
    merg_S_LDA_out=[]
    merg_n_LDA_out=[]
    merg_A_S_LDA_out=[]

    nonmerg_gini_LDA_out=[]
    nonmerg_m20_LDA_out=[]
    nonmerg_C_LDA_out=[]
    nonmerg_A_LDA_out=[]
    nonmerg_S_LDA_out=[]
    nonmerg_n_LDA_out=[]
    nonmerg_A_S_LDA_out=[]

    merg_name_list=[]
    nonmerg_name_list=[]
    #print('~~~~Mergers~~~~')

    p_merg_merg = []
    p_merg_nonmerg = []


    for j in range(len(LD1_SDSS)):
        #LDA_compare.append(float(np.sum(Xs_standardized*output_LDA[3])+output_LDA[4]))

        if LD1_SDSS[j] > 0:#merger
            p_merg_merg.append(p_merg_list[j])
            merg_pred_1.append(df2[first_imp_merg].values[j])
            merg_pred_2.append(df2[second_imp_merg].values[j])
            merg_gini_LDA_out.append(df2['Gini'].values[j])
            merg_m20_LDA_out.append(df2['M20'].values[j])
            merg_C_LDA_out.append(df2['Concentration (C)'].values[j])
            merg_A_LDA_out.append(df2['Asymmetry (A)'].values[j])
            merg_S_LDA_out.append(df2['Clumpiness (S)'].values[j])
            merg_n_LDA_out.append(df2['Sersic N'].values[j])
            merg_A_S_LDA_out.append(df2['Shape Asymmetry (A_S)'].values[j])
            merg_name_list.append(df2['ID'].values[j])
            '''print('~~~~~~')
            print('Merger', LD1_SDSS[j])
            print(df2['ID'].values[j],df2['Gini'].values[j], df2['M20'].values[j],
                  df2['Concentration (C)'].values[j],df2['Asymmetry (A)'].values[j],
                  df2['Clumpiness (S)'].values[j],df2['Sersic N'].values[j],
                  df2['Shape Asymmetry (A_S)'].values[j])
            print('~~~~~~')'''
        else:#nonmerger
            #print(df2['ID'].values[j])
            #print(X_std[j])
            #print(np.sum(X_std[j]))
            p_merg_nonmerg.append(p_merg_list[j])
            nonmerg_pred_1.append(df2[first_imp_nonmerg].values[j])
            nonmerg_pred_2.append(df2[second_imp_nonmerg].values[j])
            nonmerg_gini_LDA_out.append(df2['Gini'].values[j])
            nonmerg_m20_LDA_out.append(df2['M20'].values[j])
            nonmerg_C_LDA_out.append(df2['Concentration (C)'].values[j])
            nonmerg_A_LDA_out.append(df2['Asymmetry (A)'].values[j])
            nonmerg_S_LDA_out.append(df2['Clumpiness (S)'].values[j])
            nonmerg_n_LDA_out.append(df2['Sersic N'].values[j])
            nonmerg_A_S_LDA_out.append(df2['Shape Asymmetry (A_S)'].values[j])
            nonmerg_name_list.append(df2['ID'].values[j])
            
            
                    
    if verbose:
        # This is all about making a panel plot of weird parameters
        plt.clf()
        fig, axs = plt.subplots(2,2, figsize=(5, 5), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = -0.1, wspace=-0.1)

        axs = axs.ravel()
        counter = 0
        for p in range(len(nonmerg_name_list)):
            if counter == 4:
                break
            if p_merg_nonmerg[p] < 0.01:
                continue
                
            gal_id=nonmerg_name_list[p]
            
            try:
                im=fits.open('../imaging/out_'+str(gal_id)+'.fits')
                print('in existence', gal_id)
            except:
                #print('not in existence', gal_id)
                continue
            
            most_influential = most_influential_nonmerg[p]
            
            
            gal_prob=p_merg_nonmerg[p]
            #list_sklearn[new_min_index].predict_proba(X_std)[p]
            
            
            camera_data=(im[1].data/0.005)
            
            axs[counter].imshow(np.abs(camera_data),norm=colors.LogNorm(vmin=10**(3), vmax=10**(6.5)), cmap='afmhot')#vmin=10**(1), vmax=10**(4)norm=colors.LogNorm(vmin=10**(-1), vmax=10**(2.5))
            if str(run)=='minor_merger':
                axs[counter].annotate('p$_{\mathrm{merg, minor}} = $'+str(round(p_merg_nonmerg[p],2))+'\n'+'n =  '+str(round(nonmerg_A_LDA_out[p],2))+'\n'+str(most_influential), xycoords='axes fraction',xy=(0.05,0.8),color='white', size=10)
            else:
                axs[counter].annotate('p$_{\mathrm{merg, major}} = $'+str(round(p_merg_nonmerg[p],2))+'\n'+'n =  '+str(round(nonmerg_A_LDA_out[p],2))+'\n'+str(most_influential), xycoords='axes fraction',xy=(0.05,0.8),color='white', size=10)
            axs[counter].axis('off')
            counter+=1

        plt.tight_layout()
        plt.savefig('../Figures/panel_plot_0.0_nonmergers_SDSS_predictors_'+str(run)+'.pdf')

    if verbose:

        # First get the LDA values for the knwon and unknown mergers and nonmergers:
        '''
        LDA_ID_merg = []
        LDA_ID_nonmerg = []

        nonmerg_p = []
        merg_p = []
        for j in range(len(df)):
            p_merg_sim = 1/(1 + np.exp(-LDA[8][j]))
            if df['class label'].values[j]==0:
                LDA_ID_nonmerg.append(LDA[8][j])
            if df['class label'].values[j]==1:
                LDA_ID_merg.append(LDA[8][j])
            if LDA[8][j]<0:          
                nonmerg_p.append(p_merg_sim)
            else:
                merg_p.append(p_merg_sim)
        '''    
        nonmerg_gini=[]
        nonmerg_m20=[]

        merg_gini=[]
        merg_m20=[]

        nonmerg_gini_LDA=[]
        nonmerg_m20_LDA=[]

        merg_gini_LDA=[]
        merg_m20_LDA=[]


        nonmerg_C_LDA=[]
        nonmerg_A_LDA=[]
        nonmerg_S_LDA=[]
        nonmerg_n_LDA=[]
        nonmerg_A_S_LDA=[]

        merg_C_LDA=[]
        merg_A_LDA=[]
        merg_S_LDA=[]
        merg_n_LDA=[]
        merg_A_S_LDA=[]

        LDA_ID_merg = []
        LDA_ID_nonmerg = []

        indices_nonmerg = []
        indices_merg = []

        merg_p = []
        nonmerg_p = []

        for j in range(len(df)):
            if df['class label'].values[j]==0:
                nonmerg_gini.append(df['Gini'].values[j])
                nonmerg_m20.append(df['M20'].values[j])
                LDA_ID_nonmerg.append(LDA[8][j])
                indices_nonmerg.append(j)
            if df['class label'].values[j]==1:
                merg_gini.append(df['Gini'].values[j])
                merg_m20.append(df['M20'].values[j])
                LDA_ID_merg.append(LDA[8][j])
                indices_merg.append(j)


            p_merg_sim = 1/(1 + np.exp(-LDA[8][j]))
            if LDA[8][j]<0:#then its a nonmerger                                                                                           
                nonmerg_gini_LDA.append(df['Gini'].values[j])
                nonmerg_m20_LDA.append(df['M20'].values[j])
                nonmerg_C_LDA.append(df['Concentration (C)'].values[j])
                nonmerg_A_LDA.append(df['Asymmetry (A)'].values[j])
                nonmerg_S_LDA.append(df['Clumpiness (S)'].values[j])
                nonmerg_n_LDA.append(df['Sersic N'].values[j])
                nonmerg_A_S_LDA.append(df['Shape Asymmetry (A_S)'].values[j])
                nonmerg_p.append(p_merg_sim)

            if LDA[8][j]>0:#then its a nonmerger                                                                                           
                merg_gini_LDA.append(df['Gini'].values[j])
                merg_m20_LDA.append(df['M20'].values[j])
                merg_C_LDA.append(df['Concentration (C)'].values[j])
                merg_A_LDA.append(df['Asymmetry (A)'].values[j])
                merg_S_LDA.append(df['Clumpiness (S)'].values[j])
                merg_n_LDA.append(df['Sersic N'].values[j])
                merg_A_S_LDA.append(df['Shape Asymmetry (A_S)'].values[j])
                merg_p.append(p_merg_sim)



        merg_m20_LDA=np.array(merg_m20_LDA)
        merg_gini_LDA=np.array(merg_gini_LDA)
        merg_C_LDA=np.array(merg_C_LDA)
        merg_A_LDA=np.array(merg_A_LDA)
        merg_S_LDA=np.array(merg_S_LDA)
        merg_n_LDA=np.array(merg_n_LDA)
        merg_A_S_LDA=np.array(merg_A_S_LDA)

        merg_m20=np.array(merg_m20)
        merg_gini=np.array(merg_gini)
        nonmerg_m20=np.array(nonmerg_m20)
        nonmerg_gini=np.array(nonmerg_gini)



        
        
        #Make a histogram of LDA values for merging and nonmerging SDSS
        #galaxies and maybe overlay the LDA values for the simulated galaxies
        hist, bins = np.histogram(LDA[8], bins=50)
        
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(312)
        _ = ax.hist(LDA_merg_SDSS, label='SDSS mergers',alpha=0.75, color='#EA9010',  bins=bins)
        _ = ax.hist(LDA_nonmerg_SDSS, label='SDSS nonmergers',alpha=0.75, color='#90BE6D',  bins=bins)
        plt.axvline(x=0, color='black')
        
        plt.legend()
        ax1 = fig.add_subplot(311)
        _ = ax1.hist(LDA_ID_merg, label='Simulated mergers',alpha=0.75, color='#EFA8B8',  bins=bins)
        _ = ax1.hist(LDA_ID_nonmerg, label='Simulated nonmergers',alpha=0.75, color='#55868C',  bins=bins)
        
        plt.axvline(x=0, color='black')



        plt.legend()
        ax.set_xlabel('LD1 Value')
        if run=='major_merger':
            plt.title('Major')
        if run=='minor_merger':
            plt.title('Minor')


        ax2 = fig.add_subplot(313)
        _ = ax2.hist(p_merg_merg_SDSS, label='SDSS mergers',alpha=0.75, color='#EA9010',  bins=50, range=[0,1])
        a, b, c = ax2.hist(p_merg_nonmerg_SDSS, label='SDSS nonmergers',alpha=0.75, color='#90BE6D',  bins=50, range=[0,1])
        
        print(a)
        plt.axvline(x=0.5, color='black')
        ax2.set_ylim([0,(len(p_merg_merg_SDSS)+len(p_merg_nonmerg_SDSS))/5])
        ax2.annotate('^'+str(int(np.max(a))), xy=(0.07,0.89), xycoords='axes fraction')
        ax2.set_xlabel(r'$p_{\mathrm{merg}}$ value')

        plt.savefig('../Figures/hist_LDA_divided_'+str(run)+'.pdf')


        #Make a histogram of LDA values for merging and nonmerging SDSS
        #galaxies and maybe overlay the LDA values for the simulated galaxies
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(212)
        
        #_, bins, _ = ax.hist(LDA_ID, label='Simulation LD1', color='#FE4E00', bins=50, range = [-12,12])
        _ = ax.hist(p_merg_merg, label='SDSS mergers',alpha=0.75, color='#EA9010',  bins=50, range = [-0.1,1.1])
        _ = ax.hist(p_merg_nonmerg, label='SDSS nonmergers',alpha=0.75, color='#90BE6D',  bins=50, range = [-0.1,1.1])
        plt.axvline(x=0.5, color='black')
        
        plt.legend()
        ax1 = fig.add_subplot(211)
        _ = ax1.hist(merg_p, label='Simulated mergers',alpha=0.75, color='#EFA8B8',  bins=50, range = [-0.1,1.1])
        _ = ax1.hist(nonmerg_p, label='Simulated nonmergers',alpha=0.75, color='#55868C',  bins=50, range = [-0.1,1.1])
        
        plt.axvline(x=0.5, color='black')
        plt.legend()
        ax.set_xlabel(r'$p_{\mathrm{merg}}$')
        if run=='major_merger':
            plt.title('Major')
        if run=='minor_merger':
            plt.title('Minor')
        plt.savefig('../Figures/hist_LDA_divided_p_'+str(run)+'.pdf')
        
    print('~~~~~~~Analysis Results~~~~~~')



    print('percent nonmerg',len(nonmerg_name_list)/(len(nonmerg_name_list)+len(merg_name_list)))
    print('percent merg',len(merg_name_list)/(len(nonmerg_name_list)+len(merg_name_list)))

    print('# nonmerg',len(nonmerg_name_list))
    print('# merg',len(merg_name_list))





    print('~~~~~~~~~~~~~~~~~~~~~~~~')


    if verbose:
        # Okay sort everything by which probability grouping it is in
        ps = 3
        # Get indices that are
        indices_nonmerg = []
        for j in range(5):
            indices = np.where((np.array(p_merg_nonmerg) < 0.1*j+0.1) & (np.array(p_merg_nonmerg) > 0.1*j))[0]
            indices_nonmerg.append(indices)
        indices_merg = []
        for j in range(5):
            indices_merg.append(np.where((np.array(p_merg_merg) < 0.1*j+0.6) & (np.array(p_merg_merg) > 0.1*j+0.5))[0])
        
        print('shape of indices', np.shape(indices_merg))

        sns.set_style('dark')
        plt.clf()
        fig=plt.figure(figsize=(11,7))
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.3, wspace=0.1)


        ax1=fig.add_subplot(221)
        ax1.set_title('SDSS mergers (# = '+str(len(merg_m20_LDA_out))+')', loc='right')
        
        
        #im1 = ax1.hexbin(merg_pred_1, merg_pred_2, C=p_merg_merg, cmap='magma', vmin=0.5, vmax=1, gridsize=40)
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(merg_pred_2, merg_pred_1, p_merg_merg,statistic='mean', bins=50)
         

         
        
        
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(nonmerg_pred_2, nonmerg_pred_1, p_merg_nonmerg,statistic='mean', bins=50)
         

        min_x = np.min([xedges[0],xedgesnon[0]])
        max_x = np.max([xedges[-1], xedgesnon[-1]])
        
        min_y = np.min([yedges[0],yedgesnon[0]])
        max_y = np.max([yedges[-1],yedgesnon[-1]])
        
        im1 = ax1.imshow(np.flipud(heatmap), cmap='magma', vmin=0.5, vmax=1.0,extent=[yedges[0], yedges[-1], xedges[0],xedges[-1]])
        
        ax1.set_ylim(xedges[0], xedges[-1])
        ax1.set_xlim(yedges[0], yedges[-1])
        '''
        cs = []
        xs = []
        ys = []
        for j in range(5):
            cs.append(np.mean(np.array(p_merg_merg)[indices_merg[j]]))
            xs.append(np.mean(np.array(merg_pred_1)[indices_merg[j]]))
            ys.append(np.mean(np.array(merg_pred_2)[indices_merg[j]]))
        ax1.scatter(xs, ys, c=cs, marker='*', s=150, vmin=0.5, vmax=1, cmap='magma', edgecolor='black', zorder=100)
        '''
        plt.colorbar(im1, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        
        ax1.set_xlabel(first_imp_merg)
        ax1.set_ylabel(second_imp_merg)
        ax1.set_aspect((yedges[-1]-yedges[0])/(xedges[-1]-xedges[0]))#'equal')

        ax2=fig.add_subplot(222)
        ax2.set_title('SDSS non-mergers (# = '+str(len(nonmerg_m20_LDA_out))+')', loc='right')
        
        #sns.kdeplot(nonmerg_m20_LDA, nonmerg_gini_LDA, cmap="Blues", shade=True,shade_lowest=False)
        
        
        im2 = ax2.imshow(np.flipud(heatmapnon), cmap='viridis', vmin=0, vmax=0.5,extent=[yedgesnon[0],yedgesnon[-1],xedgesnon[0],xedgesnon[-1]])
         
        ax2.set_ylim(xedgesnon[0],xedgesnon[-1])
        ax2.set_xlim(yedgesnon[0],yedgesnon[-1])
        
        '''
        cs = []
        xs = []
        ys = []
        for j in range(5):
            cs.append(np.mean(np.array(p_merg_nonmerg)[indices_nonmerg[j]]))
            xs.append(np.mean(np.array(nonmerg_pred_1)[indices_nonmerg[j]]))
            ys.append(np.mean(np.array(nonmerg_pred_2)[indices_nonmerg[j]]))
        ax2.scatter(xs, ys, c=cs, marker='*', s=150, vmin=0, vmax=0.5, cmap='viridis', edgecolor='white', zorder=100)
        '''
        plt.colorbar(im2, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        
        ax2.set_xlabel(first_imp_nonmerg)
        ax2.set_ylabel(second_imp_nonmerg)
        ax2.set_aspect((yedgesnon[-1]-yedgesnon[0])/(xedgesnon[-1]-xedgesnon[0]))#'equal')

        ax3 = fig.add_subplot(223)
        #sim_nonmerg_pred_1
        
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(sim_merg_pred_2, sim_merg_pred_1, merg_p,statistic='mean', bins=20)
         

         
        
        
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(sim_nonmerg_pred_2, sim_nonmerg_pred_1, nonmerg_p,statistic='mean', bins=20)
         

        
        
        ax3.set_title('Simulated mergers (# = '+str(len(merg_p))+')', loc='right')
        
        #sns.kdeplot(nonmerg_m20_LDA, nonmerg_gini_LDA, cmap="Blues", shade=True,shade_lowest=False)
        
        
        im3 = ax3.imshow(np.flipud(heatmap), cmap='magma', vmin=0.5, vmax=1.0,extent=[yedges[0],yedges[-1],xedges[0],xedges[-1]])
         
        ax3.set_ylim(xedges[0],xedges[-1])
        ax3.set_xlim(yedges[0],yedges[-1])
        
        
        plt.colorbar(im3, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        
        ax3.set_xlabel(first_imp_merg)
        ax3.set_ylabel(second_imp_merg)
        ax3.set_aspect((yedges[-1]-yedges[0])/(xedges[-1]-xedges[0]))

        ax4 = fig.add_subplot(224)
        
        ax4.set_title('Simulated non-mergers (# = '+str(len(nonmerg_p))+')', loc='right')
        im4 = ax4.imshow(np.flipud(heatmapnon), cmap='viridis', vmin=0, vmax=0.5,extent=[yedgesnon[0],yedgesnon[-1],xedgesnon[0],xedgesnon[-1]])
         
        ax4.set_ylim(xedgesnon[0],xedgesnon[-1])
        ax4.set_xlim(yedgesnon[0],yedgesnon[-1])
        
        
        plt.colorbar(im4, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        
        ax4.set_xlabel(first_imp_nonmerg)
        ax4.set_ylabel(second_imp_nonmerg)
        ax4.set_aspect((yedgesnon[-1]-yedgesnon[0])/(xedgesnon[-1]-xedgesnon[0]))

        plt.savefig('../Figures/first_sec_separate_'+str(run)+'.pdf',  bbox_inches = 'tight')
        
        
        # Okay try to make more of a density-based scatter plot for the SDSS galaxies:
        
       
        '''
        plt.clf()
        fig=plt.figure(figsize=(12,5))
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)


        ax1=fig.add_subplot(121)
        ax1.set_title(str(len(merg_m20_LDA_out))+' Mergers', loc='right')
        ax1.plot(dashed_line_x, dashed_line_y, ls='--', color='black')

        
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(merg_gini_LDA_out, merg_m20_LDA_out, p_merg_merg,statistic='mean', bins=50)
         
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(nonmerg_gini_LDA_out, nonmerg_m20_LDA_out, p_merg_nonmerg,statistic='mean', bins=50)
        
        xmin = np.min([xedges[0],xedgesnon[0]])
        xmax = np.max([xedges[-1], xedgesnon[-1]])
        
        ymin = np.min([yedges[0],yedgesnon[0]])
        ymax = np.max([yedges[-1],yedgesnon[-1]])
         
        im1 = ax1.imshow(np.fliplr(np.flipud(heatmap)), cmap='magma', vmin=0.5, vmax=1.0,extent=[ymax, ymin, xmin, xmax])
         
        ax1.set_ylim(xmin, xmax)
        ax1.set_xlim(ymax, ymin)

        #im1 = ax1.hexbin(merg_m20_LDA_out, merg_gini_LDA_out, C=p_merg_merg, cmap='magma', vmin=0.5, vmax=1, gridsize=40)
        
        
        
        sns.kdeplot(merg_m20_LDA, merg_gini_LDA, color='black', thresh=0.05)
        #sns.kdeplot(merg_m20_LDA, merg_gini_LDA, color='black', thresh=0.05, ls='--')#cmap="Oranges", shade=True,thresh=0.05)
        cs = []
        xs = []
        ys = []
        for j in range(5):
            cs.append(np.mean(np.array(p_merg_merg)[indices_merg[j]]))
            xs.append(np.mean(np.array(merg_m20_LDA_out)[indices_merg[j]]))
            ys.append(np.mean(np.array(merg_gini_LDA_out)[indices_merg[j]]))
        ax1.scatter(xs, ys, c=cs, marker='*', s=150, vmin=0.5, vmax=1, cmap='magma', edgecolor='black', zorder=100)
        zero_indices = np.where(np.array(p_merg_merg)>0.999)[0]
        #im3=ax1.scatter(np.array(merg_m20_LDA_out)[zero_indices], np.array(merg_gini_LDA_out)[zero_indices],s=ps, edgecolor='None', color='red')
        
        plt.colorbar(im1, fraction=0.046)
        #for j in range(len(merg_name_list)):
        #    if merg_name_list[j]=='8309-12701':
        #        ax1.annotate(merg_name_list[j],(merg_m20_LDA_out[j], merg_gini_LDA_out[j]))

        #ax1.set_xlim([0,-3])
        #ax1.set_ylim([0.2,0.8])#ax1.set_ylim([0.3,0.8])
        ax1.set_xlabel(r'M$_{20}$')
        ax1.set_ylabel(r'Gini')
        ax1.set_aspect((ymax-ymin)/(xmax-xmin))

        ax2=fig.add_subplot(122)
        ax2.set_title(str(len(nonmerg_m20_LDA_out))+' Nonmergers', loc='right')
        ax2.plot(dashed_line_x, dashed_line_y, ls='--', color='black')

        
        
        
       
        im2 = ax2.imshow(np.fliplr(np.flipud(heatmapnon)), cmap='viridis', vmin=0, vmax=0.5,extent=[ymax, ymin, xmin, xmax])
        ax2.scatter(xs, ys, c=cs, marker='*', s=150, vmin=0.5, vmax=1, cmap='magma', edgecolor='white', zorder=99)
        
        ax2.set_ylim(xmin, xmax)
        ax2.set_xlim(ymax, ymin)
        plt.colorbar(im2, fraction=0.046)
        #im2 = ax2.hexbin(nonmerg_m20_LDA_out, nonmerg_gini_LDA_out, C=p_merg_nonmerg, cmap='viridis', vmin=0, vmax=0.5, gridsize=40)
        #im2=ax2.scatter(nonmerg_m20_LDA_out, nonmerg_gini_LDA_out, c=p_merg_nonmerg, s=ps, edgecolor='None', vmin=0, vmax=0.5, cmap='viridis_r')
        zero_indices_n = np.where(np.array(p_merg_nonmerg)<0.001)[0]
        #im3=ax2.scatter(np.array(nonmerg_m20_LDA_out)[zero_indices_n], np.array(nonmerg_gini_LDA_out)[zero_indices_n],s=ps, edgecolor='None', color='red')
        sns.kdeplot(nonmerg_m20_LDA, nonmerg_gini_LDA, color='white', thresh=0.05)#cmap="Blues", shade=True,thresh=0.05)
        
        cs = []
        xs = []
        ys = []
        for j in range(5):
            cs.append(np.mean(np.array(p_merg_nonmerg)[indices_nonmerg[j]]))
            xs.append(np.mean(np.array(nonmerg_m20_LDA_out)[indices_nonmerg[j]]))
            ys.append(np.mean(np.array(nonmerg_gini_LDA_out)[indices_nonmerg[j]]))
        im2 = ax2.scatter(xs, ys, c=cs, marker='*', s=150, vmin=0, vmax=0.5, cmap='viridis', edgecolor='white', zorder=100)
        im1 = ax1.scatter(xs, ys, c=cs, marker='*', s=150, vmin=0, vmax=0.5, cmap='viridis', edgecolor='black', zorder=99)
        
        #for j in range(len(nonmerg_gini_LDA_out)):
        #    ax2.annotate(name_list[j],(nonmerg_m20_LDA_out[j], nonmerg_gini_LDA_out[j]))

        
        #ax2.set_xlim([0,-3])
        #ax2.set_ylim([0.2,0.8])#ax1.set_ylim([0.3,0.8])
        ax2.set_xlabel(r'M$_{20}$')
        ax2.set_ylabel(r'Gini')
        ax2.set_aspect((ymax-ymin)/(xmax-xmin))

        plt.savefig('../LDA_figures/gini_m20_density_'+str(run)+'.pdf',  bbox_inches = 'tight')
        
        
        

        plt.clf()
        fig=plt.figure(figsize=(12,5))
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)


        ax1=fig.add_subplot(121)

        #ax1.set_xlim([-0.2,1])
        #ax1.set_ylim([0,6])#ax1.set_ylim([0.3,0.8])
        ax1.set_xlabel(r'A')
        ax1.set_ylabel(r'C')
        ax1.set_aspect((ymax-ymin)/(xmax-xmin))
        ax1.set_title(str(len(merg_A_S_LDA_out))+' Mergers', loc='right')
        plt.axvline(x=0.35, ls='--', color='black')
        
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(merg_A_LDA_out, merg_C_LDA_out, p_merg_merg,statistic='mean', bins=50)
        
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(nonmerg_A_LDA_out, nonmerg_C_LDA_out, p_merg_nonmerg,statistic='mean', bins=50)
        
        xmin = np.min([xedges[0],xedges[0]])
        xmax = np.max([xedges[-1], xedges[-1]])
        
        ymin = np.min([yedges[0],yedges[0]])
        ymax = np.max([yedges[-1],yedges[-1]])
         
        
        im1 = ax1.imshow(np.rot90(np.rot90(np.rot90(heatmap))), cmap='magma', vmin=0.5, vmax=1,extent=[xmin, xmax, ymin, ymax])
         
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        ax1.set_aspect((xmax-xmin)/(ymax-ymin))
        
        #im1=ax1.hexbin(merg_A_LDA_out, merg_C_LDA_out, C=p_merg_merg, vmin=0.5, vmax=1, cmap='magma', gridsize=40)
        
        plt.colorbar(im1, fraction=0.046)
        sns.kdeplot(merg_A_LDA, merg_C_LDA, color='black', thresh=0.05)#cmap="Oranges", shade=True,thresh=0.05)
        cs = []
        xs = []
        ys = []
        for j in range(5):
            cs.append(np.mean(np.array(p_merg_merg)[indices_merg[j]]))
            xs.append(np.mean(np.array(merg_A_LDA_out)[indices_merg[j]]))
            ys.append(np.mean(np.array(merg_C_LDA_out)[indices_merg[j]]))
        ax1.scatter(xs, ys, c=cs, marker='*', s=150, vmin=0.5, vmax=1, cmap='magma', edgecolor='black', zorder=100)
        #for j in range(len(merg_name_list)):
        #    if merg_name_list[j]=='8309-12701':
        #        ax1.annotate(merg_name_list[j],(merg_A_LDA_out[j], merg_C_LDA_out[j]))

        ax2=fig.add_subplot(122)

        #ax2.set_xlim([-0.2,1])
        #ax2.set_ylim([0,6])#ax1.set_ylim([0.3,0.8])
        ax2.set_xlabel(r'A')
        ax2.set_ylabel(r'C')
        
        ax2.set_title(str(len(nonmerg_A_S_LDA_out))+' Nonmergers', loc='right')
        plt.axvline(x=0.35, ls='--', color='black')
        
        
         
        
        im2 = ax2.imshow(np.rot90(np.rot90(np.rot90(heatmapnon))), cmap='viridis', vmin=0, vmax=0.5,extent=[xmin, xmax, ymin, ymax])
        
        ax2.scatter(xs, ys, c=cs, marker='*', s=150, vmin=0.5, vmax=1, cmap='magma', edgecolor='white', zorder=99)
         
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)
        #im2=ax2.hexbin(nonmerg_A_LDA_out, nonmerg_C_LDA_out, C=p_merg_nonmerg, vmin=0, vmax=0.5, cmap='viridis', gridsize=40)
        
        sns.kdeplot(nonmerg_A_LDA, nonmerg_C_LDA, color='white', thresh=0.05)#cmap="Blues", shade=True,thresh=0.05)
        cs = []
        xs = []
        ys = []
        for j in range(5):
            cs.append(np.mean(np.array(p_merg_nonmerg)[indices_nonmerg[j]]))
            xs.append(np.mean(np.array(nonmerg_A_LDA_out)[indices_nonmerg[j]]))
            ys.append(np.mean(np.array(nonmerg_C_LDA_out)[indices_nonmerg[j]]))
        ax2.scatter(xs, ys, c=cs, marker='*', s=150, vmin=0, vmax=0.5, cmap='viridis', edgecolor='white', zorder=100)
        ax1.scatter(xs, ys, c=cs, marker='*', s=150, vmin=0, vmax=0.5, cmap='viridis', edgecolor='black', zorder=99)
        plt.colorbar(im2, fraction=0.046)
        #for j in range(len(nonmerg_A_S_LDA_out)):
        #    ax2.annotate(name_list[j],(nonmerg_A_LDA_out[j], nonmerg_C_LDA_out[j]))
        ax2.set_aspect((xmax-xmin)/(ymax-ymin))
        plt.tight_layout()
        plt.savefig('../LDA_figures/C_A_density_'+str(run)+'.pdf')
        
        

        plt.clf()
        fig=plt.figure(figsize=(12,5))
        ax1=fig.add_subplot(121)

        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(merg_n_LDA_out, merg_A_S_LDA_out, p_merg_merg,statistic='mean', bins=50)
        
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(nonmerg_n_LDA_out, nonmerg_A_S_LDA_out, p_merg_nonmerg,statistic='mean', bins=50)
        
        xmin = np.min([xedges[0],xedgesnon[0]])
        xmax = np.max([xedges[-1], xedgesnon[-1]])
        
        ymin = np.min([yedges[0],yedgesnon[0]])
        ymax = np.max([yedges[-1],yedgesnon[-1]])
         
        
        im1 = ax1.imshow(np.flipud(heatmap), cmap='magma', vmin=0.5, vmax=1,extent=[ymin, ymax, xmin, xmax])
         
        ax1.set_ylim(xmin, xmax)
        ax1.set_xlim(ymin, ymax)
        #im1=ax1.hexbin(merg_A_S_LDA_out, merg_n_LDA_out, C=p_merg_merg, vmin=0.5, vmax=1, cmap='magma', gridsize=40)
        # Plot in red everything that is a probability of zero (extreme outliers)
        #im3=ax1.scatter(np.array(merg_A_S_LDA_out)[zero_indices], np.array(merg_n_LDA_out)[zero_indices],s=ps, edgecolor='None', color='red')
        #sns.kdeplot(merg_A_S_LDA, merg_n_LDA, color='red', thresh=0.05, lw=2)#, shade=True,thresh=0.05)
        sns.kdeplot(merg_A_S_LDA, merg_n_LDA, color='black', thresh=0.05)
        cs = []
        xs = []
        ys = []
        for j in range(5):
            cs.append(np.mean(np.array(p_merg_merg)[indices_merg[j]]))
            xs.append(np.mean(np.array(merg_A_S_LDA_out)[indices_merg[j]]))
            ys.append(np.mean(np.array(merg_n_LDA_out)[indices_merg[j]]))
        ax1.scatter(xs, ys, c=cs, marker='*', s=150, vmin=0.5, vmax=1, cmap='magma', edgecolor='black', zorder=100)
        
        

        plt.colorbar(im1, fraction=0.046)
        #cbar1.set_clim([0,1])
        #ax1.set_xlim([0,1])
        #ax1.set_ylim([0,4])#ax1.set_ylim([0.3,0.8])
        ax1.set_xlabel(r'$A_S$')
        ax1.set_ylabel(r'$n$')
        ax1.set_aspect((ymax-ymin)/(xmax-xmin))

        #ax1.legend(loc='lower center',
        #          ncol=2)
        ax1.set_title(str(len(merg_A_S_LDA_out))+' Mergers', loc='right')

        #ax1.annotate(str(NAME), xy=(0.03,1.05),xycoords='axes fraction',size=15)
        plt.axvline(x=0.2, ls='--', color='black')


        ax2=fig.add_subplot(122)
        
         
        
        im2 = ax2.imshow(np.flipud(heatmapnon), cmap='viridis', vmin=0, vmax=0.5,extent=[ymin, ymax, xmin, xmax])
        
        ax2.scatter(xs, ys, c=cs, marker='*', s=150, vmin=0.5, vmax=1, cmap='magma', edgecolor='white', zorder=99)
        ax2.set_ylim(xmin, xmax)
        ax2.set_xlim(ymin, ymax)
        #im2=ax2.hexbin(nonmerg_A_S_LDA_out, nonmerg_n_LDA_out, C=p_merg_nonmerg, vmin=0, vmax=0.5, cmap='viridis', gridsize=40)
        
        # Plot in red everything that is a probability of zero (extreme outliers)
        #im3=ax2.scatter(np.array(nonmerg_A_S_LDA_out)[zero_indices_n], np.array(nonmerg_n_LDA_out)[zero_indices_n],s=ps, edgecolor='None', color='red')
        
        ##color='#EA9010',color='#90BE6D'
        sns.kdeplot(nonmerg_A_S_LDA, nonmerg_n_LDA,thresh=0.05, color='white')# cmap="Blues", shade=True,thresh=0.05)
        cs = []
        xs = []
        ys = []
        for j in range(5):
            cs.append(np.mean(np.array(p_merg_nonmerg)[indices_nonmerg[j]]))
            xs.append(np.mean(np.array(nonmerg_A_S_LDA_out)[indices_nonmerg[j]]))
            ys.append(np.mean(np.array(nonmerg_n_LDA_out)[indices_nonmerg[j]]))
        ax2.scatter(xs, ys, c=cs, marker='*', s=150, vmin=0, vmax=0.5, cmap='viridis', edgecolor='white', zorder=100)
        ax1.scatter(xs, ys, c=cs, marker='*', s=150, vmin=0, vmax=0.5, cmap='viridis', edgecolor='black', zorder=99)
        
        #ax2.set_xlim([0,1])
        #ax2.set_ylim([0,4])#ax1.set_ylim([0.3,0.8])
        ax2.set_xlabel(r'$A_S$')
        #ax2.set_ylabel(r'$n$')
        plt.axvline(x=0.2, ls='--', color='black')
        ax2.set_aspect((ymax-ymin)/(xmax-xmin))
        plt.colorbar(im2, fraction=0.046)


        #ax1.legend(loc='lower center',
        #          ncol=2)
        ax2.set_title(str(len(nonmerg_A_S_LDA_out))+' Nonmergers', loc='right')
        plt.savefig('../LDA_figures/n_A_S_density_'+str(run)+'.pdf',  bbox_inches = 'tight')
        '''
        
        dashed_line_x=np.linspace(-0.5,-3,100)
        dashed_line_y=[-0.14*x + 0.33 for x in dashed_line_x]



        # Makes density of the simulated galaxies separately from the SDSS galaxies
        plt.clf()
        sns.set_style('dark')
        fig=plt.figure(figsize=(11,7))
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.3, wspace=0.1)


        ax1=fig.add_subplot(221)
        ax1.set_title('SDSS mergers (# = '+str(len(merg_m20_LDA_out))+')', loc='right')
        ax1.plot(dashed_line_x, dashed_line_y, ls='--', color='black')

         
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(merg_gini_LDA_out, merg_m20_LDA_out, p_merg_merg,statistic='mean', bins=50)
          
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(nonmerg_gini_LDA_out, nonmerg_m20_LDA_out, p_merg_nonmerg,statistic='mean', bins=50)
         
        xmin = np.min([xedges[0],xedgesnon[0]])
        xmax = np.max([xedges[-1], xedgesnon[-1]])
         
        ymin = np.min([yedges[0],yedgesnon[0]])
        ymax = np.max([yedges[-1],yedgesnon[-1]])
          
        im1 = ax1.imshow(np.fliplr(np.flipud(heatmap)), cmap='magma', vmin=0.5, vmax=1.0,extent=[ymax, ymin, xmin, xmax])
          
        ax1.set_ylim(xmin, xmax)
        ax1.set_xlim(ymax, ymin)

       
         
        plt.colorbar(im1, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax1.set_xlabel(r'$M_{20}$')
        ax1.set_ylabel(r'$Gini$')
        ax1.set_aspect((ymax-ymin)/(xmax-xmin))

        ax2=fig.add_subplot(222)
        ax2.set_title('SDSS non-mergers (# = '+str(len(nonmerg_m20_LDA_out))+')', loc='right')
        ax2.plot(dashed_line_x, dashed_line_y, ls='--', color='black')

         
         
         

        im2 = ax2.imshow(np.fliplr(np.flipud(heatmapnon)), cmap='viridis', vmin=0, vmax=0.5,extent=[ymax, ymin, xmin, xmax])
        ax2.set_ylim(xmin, xmax)
        ax2.set_xlim(ymax, ymin)
        plt.colorbar(im2, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax2.set_xlabel(r'$M_{20}$')
        ax2.set_ylabel(r'$Gini$')
        ax2.set_aspect((ymax-ymin)/(xmax-xmin))

        ax3=fig.add_subplot(223)
        ax3.set_title('Simulated mergers (# = '+str(len(merg_p))+')', loc='right')
        ax3.plot(dashed_line_x, dashed_line_y, ls='--', color='black')

          
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(merg_gini_LDA, merg_m20_LDA, merg_p,statistic='mean', bins=20)
           
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(nonmerg_gini_LDA, nonmerg_m20_LDA, nonmerg_p,statistic='mean', bins=20)
          
        xmins = np.min([xedges[0],xedgesnon[0]])
        xmaxs = np.max([xedges[-1], xedgesnon[-1]])
          
        ymins = np.min([yedges[0],yedgesnon[0]])
        ymaxs = np.max([yedges[-1],yedgesnon[-1]])
           
        im3 = ax3.imshow(np.fliplr(np.flipud(heatmap)), cmap='magma', vmin=0.5, vmax=1.0,extent=[ymaxs, ymins, xmins, xmaxs])
           
        ax3.set_ylim(xmin, xmax)
        ax3.set_xlim(ymax, ymin)

        
          
        plt.colorbar(im3, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax3.set_xlabel(r'$M_{20}$')
        ax3.set_ylabel(r'$Gini$')
        ax3.set_aspect((ymax-ymin)/(xmax-xmin))

        ax4=fig.add_subplot(224)
        ax4.set_title('Simulated non-mergers (# = '+str(len(nonmerg_p))+')', loc='right')
        ax4.plot(dashed_line_x, dashed_line_y, ls='--', color='black')

          
          
          

        im4 = ax4.imshow(np.fliplr(np.flipud(heatmapnon)), cmap='viridis', vmin=0, vmax=0.5,extent=[ymaxs, ymins, xmins, xmaxs])
        ax4.set_ylim(xmin, xmax)
        ax4.set_xlim(ymax, ymin)
        plt.colorbar(im4, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax4.set_xlabel(r'$M_{20}$')
        ax4.set_ylabel(r'$Gini$')
        ax4.set_aspect((ymax-ymin)/(xmax-xmin))

        plt.savefig('../Figures/gini_m20_separate_'+str(run)+'.pdf',  bbox_inches = 'tight')
        
        
        
        #~~~~~~~~~~~~~~~~~ Now, make this for C-A ~~~~~~~~~~~~~~~~~~~~~~~~~~
        plt.clf()
        fig=plt.figure(figsize=(11,7))
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.3, wspace=0.1)


        ax1=fig.add_subplot(221)
        ax1.set_title('SDSS mergers (# = '+str(len(merg_m20_LDA_out))+')', loc='right')
        
          
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(merg_C_LDA_out, merg_A_LDA_out, p_merg_merg,statistic='mean', bins=50)
           
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(nonmerg_C_LDA_out, nonmerg_A_LDA_out, p_merg_nonmerg,statistic='mean', bins=50)
          
        xmin = np.min([xedges[0],xedgesnon[0]])
        xmax = np.max([xedges[-1], xedgesnon[-1]])
          
        ymin = np.min([yedges[0],yedgesnon[0]])
        ymax = np.max([yedges[-1],yedgesnon[-1]])
           
        im1 = ax1.imshow(np.fliplr(np.flipud(heatmap)), cmap='magma', vmin=0.5, vmax=1.0,extent=[ymax, ymin, xmin, xmax])
           
        ax1.set_ylim(xmin, xmax)
        ax1.set_xlim(ymin, ymax)

        
          
        plt.colorbar(im1, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax1.set_xlabel(r'$A$')
        ax1.set_ylabel(r'$C$')
        ax1.set_aspect((ymax-ymin)/(xmax-xmin))

        ax2=fig.add_subplot(222)
        ax2.set_title('SDSS non-mergers (# = '+str(len(nonmerg_m20_LDA_out))+')', loc='right')
        
        im2 = ax2.imshow(np.fliplr(np.flipud(heatmapnon)), cmap='viridis', vmin=0, vmax=0.5,extent=[ymax, ymin, xmin, xmax])
        ax2.set_ylim(xmin, xmax)
        ax2.set_xlim(ymin, ymax)
        plt.colorbar(im2, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax2.set_xlabel(r'$A$')
        ax2.set_ylabel(r'$C$')
        ax2.set_aspect((ymax-ymin)/(xmax-xmin))

        ax3=fig.add_subplot(223)
        ax3.set_title('Simulated mergers (# = '+str(len(merg_p))+')', loc='right')
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(merg_C_LDA, merg_A_LDA, merg_p,statistic='mean', bins=20)
            
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(nonmerg_C_LDA, nonmerg_A_LDA, nonmerg_p,statistic='mean', bins=20)
           
        xmins = np.min([xedges[0],xedgesnon[0]])
        xmaxs = np.max([xedges[-1], xedgesnon[-1]])
           
        ymins = np.min([yedges[0],yedgesnon[0]])
        ymaxs = np.max([yedges[-1],yedgesnon[-1]])
            
        im3 = ax3.imshow(np.fliplr(np.flipud(heatmap)), cmap='magma', vmin=0.5, vmax=1.0,extent=[ymaxs, ymins, xmins, xmaxs])
            
        ax3.set_ylim(xmin, xmax)
        ax3.set_xlim(ymin, ymax)

         
           
        plt.colorbar(im3, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax3.set_xlabel(r'$A$')
        ax3.set_ylabel(r'$C$')
        ax3.set_aspect((ymax-ymin)/(xmax-xmin))

        ax4=fig.add_subplot(224)
        ax4.set_title('Simulated non-mergers (# = '+str(len(nonmerg_p))+')', loc='right')
        
           
           
           

        im4 = ax4.imshow(np.fliplr(np.flipud(heatmapnon)), cmap='viridis', vmin=0, vmax=0.5,extent=[ymaxs, ymins, xmins, xmaxs])
        ax4.set_ylim(xmin, xmax)
        ax4.set_xlim(ymin, ymax)
        plt.colorbar(im4, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax4.set_xlabel(r'$A$')
        ax4.set_ylabel(r'$C$')
        ax4.set_aspect((ymax-ymin)/(xmax-xmin))
        
        ax1.axvline(x=0.35, ls='--', color='black')
        ax2.axvline(x=0.35, ls='--', color='black')
        ax3.axvline(x=0.35, ls='--', color='black')
        ax4.axvline(x=0.35, ls='--', color='black')

        plt.savefig('../Figures/C_A_separate_'+str(run)+'.pdf',  bbox_inches = 'tight')
        
        
        #~~~~~~~~~~~~~~~~~ Now, make this for n-A_S ~~~~~~~~~~~~~~~~~~~~~~~~~~
        plt.clf()
        fig=plt.figure(figsize=(11,7))
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.3, wspace=0.1)


        ax1=fig.add_subplot(221)
        ax1.set_title('SDSS mergers (# = '+str(len(merg_m20_LDA_out))+')', loc='right')
        
          
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(merg_n_LDA_out, merg_A_S_LDA_out, p_merg_merg,statistic='mean', bins=50)
           
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(nonmerg_n_LDA_out, nonmerg_A_S_LDA_out, p_merg_nonmerg,statistic='mean', bins=50)
          
        xmin = np.min([xedges[0],xedgesnon[0]])
        xmax = np.max([xedges[-1], xedgesnon[-1]])
          
        ymin = np.min([yedges[0],yedgesnon[0]])
        ymax = np.max([yedges[-1],yedgesnon[-1]])
           
        im1 = ax1.imshow(np.fliplr(np.flipud(heatmap)), cmap='magma', vmin=0.5, vmax=1.0,extent=[ymax, ymin, xmin, xmax])
           
        ax1.set_ylim(xmin, xmax)
        ax1.set_xlim(ymin, ymax)

        
          
        plt.colorbar(im1, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax1.set_xlabel(r'$A_S$')
        ax1.set_ylabel(r'Sersic $n$')
        ax1.set_aspect((ymax-ymin)/(xmax-xmin))

        ax2=fig.add_subplot(222)
        ax2.set_title('SDSS non-mergers (# = '+str(len(nonmerg_m20_LDA_out))+')', loc='right')
        
        im2 = ax2.imshow(np.fliplr(np.flipud(heatmapnon)), cmap='viridis', vmin=0, vmax=0.5,extent=[ymax, ymin, xmin, xmax])
        ax2.set_ylim(xmin, xmax)
        ax2.set_xlim(ymin, ymax)
        plt.colorbar(im2, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax2.set_xlabel(r'$A_S$')
        ax2.set_ylabel(r'Sersic $n$')
        ax2.set_aspect((ymax-ymin)/(xmax-xmin))

        ax3=fig.add_subplot(223)
        ax3.set_title('Simulated mergers (# = '+str(len(merg_p))+')', loc='right')
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(merg_n_LDA, merg_A_S_LDA, merg_p,statistic='mean', bins=20)
            
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(nonmerg_n_LDA, nonmerg_A_S_LDA, nonmerg_p,statistic='mean', bins=20)
           
        xmins = np.min([xedges[0],xedgesnon[0]])
        xmaxs = np.max([xedges[-1], xedgesnon[-1]])
           
        ymins = np.min([yedges[0],yedgesnon[0]])
        ymaxs = np.max([yedges[-1],yedgesnon[-1]])
            
        im3 = ax3.imshow(np.fliplr(np.flipud(heatmap)), cmap='magma', vmin=0.5, vmax=1.0,extent=[ymaxs, ymins, xmins, xmaxs])
            
        ax3.set_ylim(xmin, xmax)
        ax3.set_xlim(ymin, ymax)

         
           
        plt.colorbar(im3, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax3.set_xlabel(r'$A_S$')
        ax3.set_ylabel(r'Sersic $n$')
        ax3.set_aspect((ymax-ymin)/(xmax-xmin))

        ax4=fig.add_subplot(224)
        ax4.set_title('Simulated non-mergers (# = '+str(len(merg_p))+')', loc='right')
        
           
           
           

        im4 = ax4.imshow(np.fliplr(np.flipud(heatmapnon)), cmap='viridis', vmin=0, vmax=0.5,extent=[ymaxs, ymins, xmins, xmaxs])
        ax4.set_ylim(xmin, xmax)
        ax4.set_xlim(ymin, ymax)
        plt.colorbar(im4, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax4.set_xlabel(r'$A_S$')
        ax4.set_ylabel(r'Sersic $n$')
        ax4.set_aspect((ymax-ymin)/(xmax-xmin))
        
        ax1.axvline(x=0.2, ls='--', color='black')
        ax2.axvline(x=0.2, ls='--', color='black')
        ax3.axvline(x=0.2, ls='--', color='black')
        ax4.axvline(x=0.2, ls='--', color='black')

        plt.savefig('../Figures/n_A_S_separate_'+str(run)+'.pdf',  bbox_inches = 'tight')
        

    
    # Now, for the SDSS show
    sdss, ra, dec = download_sdss_ra_dec_table(prefix)

    sns.set_style("white")
    '''
    count = 0
    for pp in range(len(df2)):
        p = pp+191#+1800#0000
        
        
        gal_id=df2[['ID']].values[p][0]
        print('p', p, 'id', gal_id)
        
        gal_prob=LD1_SDSS[p]
        print('gal_id', gal_id)
        print('LDA', gal_prob)
        print('prob', p_merg_list[p])
        index_match = np.where(sdss==gal_id)[0][0]
        print(index_match)
        if index_match:
            plot_individual(gal_id, ra[index_match], dec[index_match], p_merg_list[p], run, prefix_frames)
        else:
            continue
        count+=1
        if count > 20:
            break



    #Optional panel to plot the images of these things with their probabilities assigned


    sns.set_style("white")


    #First, an option for just plotting these individually
    #os.chdir(os.path.expanduser('/Volumes/My Book/Clone_Docs_old_mac/Backup_My_Book/My_Passport_backup/MergerMonger'))
    for p in range(len(df2)):#len(df2)):
        plt.clf()
        gal_id=df2[['ID']].values[p][0]
        
        if LD1_SDSS[p] > 0:
            gal_name='Merger'
            most_influential = most_influential_merg[p]
        else:
            gal_name='Nonmerger'
            most_influential = most_influential_nonmerg[p]
        gal_prob=LD1_SDSS[p]
        #list_sklearn[new_min_index].predict_proba(X_std)[p]
        
        try:
            im=fits.open('../imaging/out_'+str(gal_id)+'.fits')
            camera_data=(im[1].data/0.005)
        except:
            # Get RA and DEC:
            index_match = np.where(sdss==gal_id)[0][0]
            print('match', index_match)
            if index_match:
                camera_data = download_galaxy(gal_id, ra[index_match], dec[index_match], prefix_frames, 40)
            else:
                continue
            #continue
        
        
        plt.imshow(np.abs(camera_data),norm=colors.LogNorm(vmin=10**(3), vmax=10**(6.5)), cmap='afmhot')#vmin=10**(1), vmax=10**(4)norm=colors.LogNorm(vmin=10**(-1), vmax=10**(2.5))
        plt.annotate('SDSS '+str(type_gal)+'\n'+r'p$_{\mathrm{merg}} = $'+str(round(p_merg_list[p],2))+'\n'+'LD1 =  '+str(round(LD1_SDSS[p],2))+'\n'+str(most_influential), xycoords='axes fraction',xy=(0.05,0.8),color='white', size=20)
        
        plt.axis('off')
     #   plt.colorbar()
        plt.tight_layout()
        plt.savefig('../Figures/ind_gals/SDSS_'+str(type_gal)+'_'+str(run)+'_'+str(gal_id)+'.pdf')
        if p > 40:
            break
    
        
    # I would like to make a panel plot of mergers and then a panel plot of nonmergers
    plt.clf()
    fig, axs = plt.subplots(5,5, figsize=(15, 15), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = -0.1, wspace=-0.1)

    axs = axs.ravel()
    counter = 0
    for p in range(len(df2)):
        if counter == 25:
            break
        if LD1_SDSS[p] < 0:
            continue
        most_influential = most_influential_merg[counter]
        gal_id=df2[['ID']].values[p][0]
        
        gal_prob=LD1_SDSS[p]
        #list_sklearn[new_min_index].predict_proba(X_std)[p]
        
        try:
            im=fits.open('../imaging/out_'+str(gal_id)+'.fits')
            camera_data=(im[1].data/0.005)
        except:
            index_match = np.where(sdss==gal_id)[0][0]
            if index_match:
                camera_data = download_galaxy(gal_id, ra[index_match], dec[index_match], prefix_frames, 40)
            else:
                continue
            
        
        
        axs[counter].imshow(np.abs(camera_data),norm=colors.LogNorm(vmin=10**(3), vmax=10**(6.5)), cmap='afmhot')#vmin=10**(1), vmax=10**(4)norm=colors.LogNorm(vmin=10**(-1), vmax=10**(2.5))
        if str(run)=='minor_merger':
            axs[counter].annotate('p$_{\mathrm{merg, minor}} = $'+str(round(p_merg_list[p],2))+'\n'+'LD1 =  '+str(round(LD1_SDSS[p],2))+'\n'+str(most_influential), xycoords='axes fraction',xy=(0.05,0.8),color='white', size=10)
        else:
            axs[counter].annotate('p$_{\mathrm{merg, major}} = $'+str(round(p_merg_list[p],2))+'\n'+'LD1 =  '+str(round(LD1_SDSS[p],2))+'\n'+str(most_influential), xycoords='axes fraction',xy=(0.05,0.8),color='white', size=10)
        axs[counter].axis('off')
        counter+=1

    plt.tight_layout()
    plt.savefig('../Figures/panel_plot_mergers_SDSS_'+str(type_gal)+'_'+str(run)+'.pdf')


    # nonmergers
    plt.clf()
    fig, axs = plt.subplots(5,5, figsize=(15, 15), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = -0.1, wspace=-0.1)

    axs = axs.ravel()
    counter = 0
    for p in range(len(df2)):
        if counter == 25:
            break
        if LD1_SDSS[p] > 0:
            continue
        most_influential = most_influential_nonmerg[counter]
        gal_id=df2[['ID']].values[p][0]
        
        gal_prob=LD1_SDSS[p]
        #list_sklearn[new_min_index].predict_proba(X_std)[p]
        
        try:
            im=fits.open('../imaging/out_'+str(gal_id)+'.fits')
            camera_data=(im[1].data/0.005)
        except:
            # Get RA and DEC:
            index_match = np.where(sdss==gal_id)[0][0]
            if index_match:
                camera_data = download_galaxy(gal_id, ra[index_match], dec[index_match], prefix_frames, 40)
            else:
                continue
        
        
        axs[counter].imshow(np.abs(camera_data),norm=colors.LogNorm(vmin=10**(3), vmax=10**(6.5)), cmap='afmhot')#vmin=10**(1), vmax=10**(4)norm=colors.LogNorm(vmin=10**(-1), vmax=10**(2.5))
        if str(run) == 'minor_merger':
            axs[counter].annotate('p$_{\mathrm{merg, minor}} = $'+str(round(p_merg_list[p],2))+'\n'+'LD1 =  '+str(round(LD1_SDSS[p],2))+'\n'+str(most_influential), xycoords='axes fraction',xy=(0.05,0.8),color='white', size=10)
        else:
            axs[counter].annotate('p$_{\mathrm{merg, major}} = $'+str(round(p_merg_list[p],2))+'\n'+'LD1 =  '+str(round(LD1_SDSS[p],2))+'\n'+str(most_influential), xycoords='axes fraction',xy=(0.05,0.8),color='white', size=10)
        axs[counter].axis('off')
        counter+=1

    plt.tight_layout()
    plt.savefig('../Figures/panel_plot_nonmergers_SDSS_'+str(type_gal)+'_'+str(run)+'.pdf')

    '''
    # I think it'll be cool to plot a panel with probability of being a merger on one axis and examples of multiple
    # probability bins

    # Also get the CDF value

    # Define a histogram with spacing defined
    spacing = 1000 # this will be the histogram binning but also how finely sampled the CDF is
    hist = np.histogram(p_merg_list, bins=spacing)

    # Put this in continuous distribution form in order to calculate the CDF
    hist_dist = scipy.stats.rv_histogram(hist)


    plt.clf()
    fig, axs = plt.subplots(2,5, figsize=(15, 7), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = -0.2, wspace=0.1)

    axs = axs.ravel()
    counter = 0
    bin_start_array = [0,0.0001,0.001,0.01,0.1]
    bin_end_array = [0.0001,0.001,0.01,0.1,0.5]

    # Step 1 is to shuffle df right off the bat and then also shuffle p_merg to be the same indices
    # First make sure that p is one of the rows
    df_with_p = df2
    df_with_p['p_merg'] = p_merg_list
    df_with_p_shuffle = df_with_p.sample(n=len(df_with_p)) # randomly shuffle everything

    for j in range(5):
        bin_start = bin_start_array[j]#0.1*j
        bin_end = bin_end_array[j]#0.1*j+0.1
        counter = 0
        counter_i = 0
        for p in range(len(df2)):
            if counter > 1:
                break
            # go through all bins and find two examples of each
            if df_with_p_shuffle['p_merg'][p] > 0.5:
                continue
            if df_with_p_shuffle['p_merg'][p] < bin_start or df_with_p_shuffle['p_merg'][p] > bin_end:
                # then you can use this galaxy as an example
                counter_i+=1
                continue
            gal_id=df_with_p_shuffle[['ID']].values[p][0]
            
            try:
                im=fits.open('../imaging/out_'+str(gal_id)+'.fits')
                camera_data=(im[1].data/0.005)
            except:
                # Get RA and DEC:
                index_match = np.where(sdss==gal_id)[0][0]
                if index_match:
                    camera_data = download_galaxy(gal_id, ra[index_match], dec[index_match], prefix_frames, 40)
                else:
                    continue
            
            if counter == 0:
                #then its top row
                axis_number = j
                
            else:
                axis_number = j+5
                axs[axis_number].set_xlabel(str(round(bin_start,5))+' < p$_{\mathrm{merg}}$ < '+str(round(bin_end,5)))
                
            
                
            most_influential = most_influential_nonmerg[counter_i]
            # Figure out which position you need to put this in
            axs[axis_number].imshow(np.abs(camera_data),norm=colors.LogNorm(vmax=10**5.5, vmin=10**(0.5)), cmap='afmhot', interpolation='None')#vmin=10**(1), vmax=10**(4)norm=colors.LogNorm(vmin=10**(-1), vmax=10**(2.5))
            axs[axis_number].annotate('p$_{\mathrm{merg}} = $'+str(round(df_with_p_shuffle['p_merg'][p],5))+', CDF = '+str(round(hist_dist.cdf(df_with_p_shuffle['p_merg'][p]),2)), 
                    xycoords='axes fraction',xy=(0.05,0.9),xytext=(0.05, 0.9), textcoords='axes fraction',
                    bbox=dict(boxstyle="round", fc="0.9"), color='black')
            axs[axis_number].annotate('ObjID = '+str(gal_id), xycoords='axes fraction',xy=(0.05,0.02),color='white', size=10)
            axs[axis_number].set_yticklabels([])
            axs[axis_number].set_xticklabels([])
            counter+=1
            counter_i+=1
            

    plt.savefig('../Figures/probability_panel_nonmergers_SDSS_predictors_'+str(run)+'.pdf')

    plt.clf()
    fig, axs = plt.subplots(2,5, figsize=(15, 7), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = -0.2, wspace=0.1)

    bin_start_array = [0.5,0.9,0.99,0.999,0.9999]
    bin_end_array = [0.9,0.99,0.999,0.9999,1.0]

    axs = axs.ravel()
    for j in range(5):
        
        bin_start = bin_start_array[j]#0.1*j
        bin_end = bin_end_array[j]#0.1*j+0.1
        counter = 0
        counter_i = 0
        for p in range(len(df2)):
            
            if counter > 1:
                break
            # go through all bins and find two examples of each
            if df_with_p_shuffle['p_merg'][p] <0.5:
                continue
            if df_with_p_shuffle['p_merg'][p] < bin_start or df_with_p_shuffle['p_merg'][p] > bin_end:
                # then you can use this galaxy as an example
                counter_i+=1
                continue
            gal_id=df_with_p_shuffle[['ID']].values[p][0]
            try:
                im=fits.open('../imaging/out_'+str(gal_id)+'.fits')
                camera_data=(im[1].data/0.005)
            except:
                # Get RA and DEC:
                index_match = np.where(sdss==gal_id)[0][0]
                if index_match:
                    camera_data = download_galaxy(gal_id, ra[index_match], dec[index_match], prefix_frames, 40)
                else:
                    continue
            most_influential = most_influential_merg[counter_i]
            
            if counter == 0:
                #then its top row
                axis_number = j
            else:
                axis_number = j+5
                axs[axis_number].set_xlabel(str(round(bin_start,1))+' < p$_{merg}$ < '+str(round(bin_end,1)))
            
            # Figure out which position you need to put this in 
            axs[axis_number].imshow(np.abs(camera_data),norm=colors.LogNorm(vmax=10**5.5, vmin=10**(0.5)), cmap='afmhot', interpolation='None')#vmin=10**(1), vmax=10**(4)norm=colors.LogNorm(vmin=10**(-1), vmax=10**(2.5))
            axs[axis_number].annotate('p$_{\mathrm{merg}} = $'+str(round(df_with_p_shuffle['p_merg'][p],5))+', CDF = '+str(round(hist_dist.cdf(df_with_p_shuffle['p_merg'][p]),2)), 
                    xycoords='axes fraction',xy=(0.05,0.9),xytext=(0.05, 0.9), textcoords='axes fraction',
                    bbox=dict(boxstyle="round", fc="0.9"), color='black')
            axs[axis_number].annotate('ObjID = '+str(gal_id), xycoords='axes fraction',xy=(0.05,0.02),color='white', size=10)
            #axs[axis_number].axis('off')
            axs[axis_number].set_yticklabels([])
            axs[axis_number].set_xticklabels([])
            counter+=1
            counter_i+=1
            

    plt.savefig('../Figures/probability_panel_mergers_SDSS_predictors_'+str(run)+'.pdf')

    return LD1_SDSS, p_merg_list, hist_dist.cdf(p_merg_list)

def classify_changing_priors(prefix, prefix_frames, run, LDA, terms_RFR, df, priors, number_run, verbose=False, run_all=True, eliminate_flags=True):
    #~~~~~~~
    # Now bring in the SDSS galaxies!
    #~~~~~~~

    print('loading up predictor value table........')

    if eliminate_flags:
        df2 = pd.io.parsers.read_csv(prefix+'SDSS_predictors_all_flags.txt', sep='\t')
    else:
        df2 = pd.io.parsers.read_csv(prefix+'SDSS_predictors_all.txt', sep='\t')
    
    if run_all == False:
        df2 = df2[0:number_run]
    
    if len(df2.columns) ==15: #then you have to delete the first column which is an empty index
        df2 = df2.iloc[: , 1:]

    

    # First, delete all rows that have weird values of n:
    #print('len before crazy values', len(df2))
    #df_filtered = df2[df2['Sersic N'] < 10]

    #df_filtered_2 = df_filtered[df_filtered['Asymmetry (A)'] > -1]

    #df2 = df_filtered_2

    # Delete duplicates:
    print('len bf duplicate delete', len(df2))
    df2_nodup = df2.duplicated()
    df2 = df2[~df2_nodup]
    print('len af duplicate delete', len(df2))

    # Now get rid of the flagged stuff
    if eliminate_flags:
        print('original length', len(df2))
        df2 = df2[(df2['low S/N'] == 0) & (df2['outlier predictor'] == 0)]
        print('only running these', len(df2))
        


    
    input_singular = terms_RFR
    #Okay so this next part actually needs to be adaptable to reproduce all possible cross-terms
    crossterms = []
    ct_1 = []
    ct_2 = []
    for j in range(len(input_singular)):
        for i in range(len(input_singular)):
            if j == i or i < j:
                continue
            #input_singular.append(input_singular[j]+'*'+input_singular[i])
            crossterms.append(input_singular[j]+'*'+input_singular[i])
            ct_1.append(input_singular[j])
            ct_2.append(input_singular[i])

    inputs = input_singular + crossterms

    # Now you have to construct a bunch of new rows to the df that include all of these cross-terms
    for j in range(len(crossterms)):
        
        df2[crossterms[j]] = df2.apply(cross_term, axis=1, args=(ct_1[j], ct_2[j]))
        

    X_gal = df2[LDA[2]].values


    
    
    # Make a table with merger probabilities and other diagnostics:
    print('making table of LDA output for all galaxies.....')
    if eliminate_flags == True:
        file_out = open(prefix+'change_prior/LDA_out_all_SDSS_predictors_'+str(run)+'_'+str(priors)+'_flags_cut.txt','w')
    else:
        file_out = open(prefix+'change_prior/LDA_out_all_SDSS_predictors_'+str(run)+'_'+str(priors)+'.txt','w')
    file_out.write('ID'+'\t'+'Classification'+'\t'+'LD1'+'\t'+'p_merg'+'\t'+'p_nonmerg'+'\t'+'Leading_term'+'\t'+'Leading_coef'+'\n')
    #+'Second_term'+'\t'+'Second_coef'+'\n')


    

    for j in range(len(X_gal)):
        #print(X_gal[j])
        X_standardized = list((X_gal[j]-LDA[0])/LDA[1])
        # use the output from the simulation to assign LD1 value:
        LD1_gal = float(np.sum(X_standardized*LDA[3])+LDA[4])
        
       
        
        # According to my calculations, LD1 = delta_1 - delta_0 where delta is the score for each class
        # Therefore, in the probability equation p_merg = e^delta_1/(e^delta_1+e^delta_0)
        # you can sub in LD1 and instead end up with p_merg = 1/(1+e^-LD1)
        
        p_merg = 1/(1 + np.exp(-LD1_gal))
        p_nonmerg = 0#1/(1 + np.exp(LD1_gal))
       


        
        if LD1_gal > 0:
            merg = 1

            # select the thing that is the most positive
            max_idx = np.argmax(X_standardized*LDA[3])
            most_influential_coeff = (X_standardized*LDA[3])[0,max_idx]
            most_influential_term = LDA[2][max_idx]
            
        else:
            merg = 0
            # select the thing that is the most positive
            max_idx = np.argmin(X_standardized*LDA[3])
            most_influential_coeff = (X_standardized*LDA[3])[0,max_idx]
            most_influential_term = LDA[2][max_idx]
            
           
        file_out.write(str(df2[['ID']].values[j][0])+'\t'+str(merg)+'\t'+str(round(LD1_gal,3))+'\t'+str(round(p_merg,3))+'\t'+str(round(p_nonmerg,3))+'\t'+most_influential_term+'\t'+str(most_influential_coeff)+'\n')
        
    file_out.close()
        

    return

# Figuring out how to classify the simulation
def classify_simulations(prefix, prefix_frames, run, LDA, terms_RFR, df, priors, number_run, verbose=False, run_all=True, eliminate_flags=True):
    #~~~~~~~
    # Now bring in the SDSS galaxies!
    #~~~~~~~

    print('loading up predictor value table........')

    feature_dict = {i:label for i,label in zip(
                range(39),
                  ('Counter_x',
                  'Image',
                  'class label',
                  'Myr',
                  'Viewpoint',
                '# Bulges',
                   'Sep',
                   'Flux Ratio',
                  'Gini',
                  'M20',
                  'Concentration (C)',
                  'Asymmetry (A)',
                  'Clumpiness (S)',
                  'Sersic N',
                  'Shape Asymmetry (A_S)',
                  'Counter_y',
                  'Delta PA',
                            'v_asym',
                            's_asym',
                            'resids',
                            'lambda_r',
                            'epsilon',
                            'A',
                            'A_2',
                            'deltapos',
                            'deltapos2',
                            'nspax','re',
                            'meanvel','varvel','skewvel','kurtvel',
                  'meansig','varsig','skewsig','kurtsig','abskewvel','abskewsig','random'))}

    features_list = ['Gini','M20','Concentration (C)','Asymmetry (A)','Clumpiness (S)','Sersic N','Shape Asymmetry (A_S)','random']


    

    # Read in the the measured predictor values from the LDA table
    df = pd.io.parsers.read_csv(filepath_or_buffer='../Tables/LDA_merged_'+str(run)+'.txt',header=[0],sep='\t')

    #Rename all of the kinematic columns (is this necessary?)
    df.rename(columns={'kurtvel':'$h_{4,V}$','kurtsig':'$h_{4,\sigma}$','lambda_r':'\lambdare',
             'epsilon':'$\epsilon$','Delta PA':'$\Delta$PA','A_2':'$A_2$',
              'varsig':'$\sigma_{\sigma}$',
             'meanvel':'$\mu_V$','abskewvel':'$|h_{3,V}|$',
             'abskewsig':'$|h_{3,\sigma}|$',
             'meansig':'$\mu_{\sigma}$',
             'varvel':'$\sigma_{V}$'},
    inplace=True)
    
    df.columns = [l for i,l in sorted(feature_dict.items())]

    df2 = df

    

    
    input_singular = terms_RFR
    #Okay so this next part actually needs to be adaptable to reproduce all possible cross-terms
    crossterms = []
    ct_1 = []
    ct_2 = []
    for j in range(len(input_singular)):
        for i in range(len(input_singular)):
            if j == i or i < j:
                continue
            #input_singular.append(input_singular[j]+'*'+input_singular[i])
            crossterms.append(input_singular[j]+'*'+input_singular[i])
            ct_1.append(input_singular[j])
            ct_2.append(input_singular[i])

    inputs = input_singular + crossterms

    # Now you have to construct a bunch of new rows to the df that include all of these cross-terms
    for j in range(len(crossterms)):
        
        df2[crossterms[j]] = df2.apply(cross_term, axis=1, args=(ct_1[j], ct_2[j]))
        

    X_gal = df2[LDA[2]].values


    
    
    p_list_merg = []
    

    for j in range(len(X_gal)):
        #print(X_gal[j])
        X_standardized = list((X_gal[j]-LDA[0])/LDA[1])
        # use the output from the simulation to assign LD1 value:
        LD1_gal = float(np.sum(X_standardized*LDA[3])+LDA[4])
        
       
        
        # According to my calculations, LD1 = delta_1 - delta_0 where delta is the score for each class
        # Therefore, in the probability equation p_merg = e^delta_1/(e^delta_1+e^delta_0)
        # you can sub in LD1 and instead end up with p_merg = 1/(1+e^-LD1)
        
        p_merg = 1/(1 + np.exp(-LD1_gal))
        p_nonmerg = 0#1/(1 + np.exp(LD1_gal))
       


        
        if LD1_gal > 0:
            merg = 1

            # select the thing that is the most positive
            max_idx = np.argmax(X_standardized*LDA[3])
            most_influential_coeff = (X_standardized*LDA[3])[0,max_idx]
            most_influential_term = LDA[2][max_idx]
            
        else:
            merg = 0
            # select the thing that is the most positive
            max_idx = np.argmin(X_standardized*LDA[3])
            most_influential_coeff = (X_standardized*LDA[3])[0,max_idx]
            most_influential_term = LDA[2][max_idx]
            
        p_list_merg.append(p_merg)
    print('measured merger fraction', len(np.array(p_list_merg)[np.array(p_list_merg) > 0.5])/len(p_list_merg))
    print('actual merger fraction', len(df2[df2['class label']==1])/len(df2))
    

    return len(df2[df2['class label']==1])/len(df2), len(np.array(p_list_merg)[np.array(p_list_merg) > 0.5])/len(p_list_merg)


# Taking the output of the LDA (trained on simulated examples),
# this code runs the classification on the full SDSS dataset

def classify_from_flagged(prefix, prefix_frames, run, LDA, terms_RFR, df, number_run, verbose=False, all=True, cut_flagged = True):
    #~~~~~~~
    # Now bring in the SDSS galaxies!
    #~~~~~~~

    print('loading up predictor value table........')
    df2 = pd.io.parsers.read_csv(prefix+'SDSS_predictors_with_flags.txt', sep='\t')
    
    if all:
        pass
    else:
        df2 = df2[0:number_run]
    
    if len(df2.columns) ==15: #then you have to delete the first column which is an empty index
        df2 = df2.iloc[: , 1:]

    

    # First, delete all rows that have weird values of n:
    #print('len before crazy values', len(df2))
    #df_filtered = df2[df2['Sersic N'] < 10]

    #df_filtered_2 = df_filtered[df_filtered['Asymmetry (A)'] > -1]

    #df2 = df_filtered_2

    # Delete duplicates:
    if verbose:
        print('len bf duplicate delete', len(df2))
    df2_nodup = df2.duplicated()
    df2 = df2[~df2_nodup]
    if verbose:
        print('len af duplicate delete', len(df2))
        print(df2)

    if cut_flagged:# Then get rid of the entries that are flagged
        df_keep = df2[(df2['low S/N'] == 0) & (df2['outlier predictor'] == 0) & (df2['segmap']==0)]
        df2 = df_keep
        


    
    input_singular = terms_RFR
    #Okay so this next part actually needs to be adaptable to reproduce all possible cross-terms
    crossterms = []
    ct_1 = []
    ct_2 = []
    for j in range(len(input_singular)):
        for i in range(len(input_singular)):
            if j == i or i < j:
                continue
            #input_singular.append(input_singular[j]+'*'+input_singular[i])
            crossterms.append(input_singular[j]+'*'+input_singular[i])
            ct_1.append(input_singular[j])
            ct_2.append(input_singular[i])

    inputs = input_singular + crossterms

    # Now you have to construct a bunch of new rows to the df that include all of these cross-terms
    for j in range(len(crossterms)):
        
        df2[crossterms[j]] = df2.apply(cross_term, axis=1, args=(ct_1[j], ct_2[j]))
        

    X_gal = df2[LDA[2]].values

    # Creating table
    print('making table of LDA output for all galaxies.....')
    file_out = open(prefix+'LDA_out_all_SDSS_predictors_'+str(run)+'_flags.txt','w')
    file_out.write('ID'+'\t'+'Classification'+'\t'+'LD1'+'\t'+'p_merg'+'\t'
        +'Leading_term'+'\t'+'Leading_coef'+'\t'+'low S/N'+'\t'+'outlier predictor'+'\t'+'segmap'+'\n')
    #+'Second_term'+'\t'+'Second_coef'+'\n')

    

    for j in range(len(X_gal)):
        #print(j)
        #print(X_gal[j])

        # this is an array of everything standardized
        X_standardized = list((X_gal[j]-LDA[0])/LDA[1])

        # use the output from the simulation to assign LD1 value:
        LD1_gal = float(np.sum(X_standardized*LDA[3])+LDA[4])
        
        
        
        
        # According to my calculations, LD1 = delta_1 - delta_0 where delta is the score for each class
        # Therefore, in the probability equation p_merg = e^delta_1/(e^delta_1+e^delta_0)
        # you can sub in LD1 and instead end up with p_merg = 1/(1+e^-LD1)
        
        p_merg = 1/(1 + np.exp(-LD1_gal))
        p_nonmerg = 1/(1 + np.exp(LD1_gal))
        
        coeffs = (X_standardized*LDA[3])[0]
        terms = LDA[2]

        
        
        if LD1_gal > 0:
            merg = 1
            # select the coefficient that is the most positive
            # So the max index is what?
            # is the max of the standardized array times all the coefficients, so what has the largest positive value?
            # this just gives you the index to search, if selected from LDA[2] gives you the name of that term
            # and if selected from the lda[3]*x_standardized gives the additive value of this
            
            
            # Array to sort:
            arr1inds = coeffs.argsort()
            sorted_terms = terms[arr1inds[::-1]]
            sorted_coeff = coeffs[arr1inds[::-1]]

            try:
                most_influential_term = [sorted_terms[0],sorted_terms[1],sorted_terms[2]]
                most_influential_coeff = [sorted_coeff[0],sorted_coeff[1],sorted_coeff[2]]
            except IndexError:
                most_influential_term = [sorted_terms[0]]
                most_influential_coeff = [sorted_coeff[0]]
            
        else:
            merg = 0
            

            # Array to sort:
            arr1inds = coeffs.argsort()
            sorted_terms = terms[arr1inds[::-1]]
            sorted_coeff = coeffs[arr1inds[::-1]]
            try:
                most_influential_term = [sorted_terms[-1],sorted_terms[-2],sorted_terms[-3]]
                most_influential_coeff = [sorted_coeff[-1],sorted_coeff[-2],sorted_coeff[-3]]
            except IndexError:
                most_influential_term = [sorted_terms[-1]]
                most_influential_coeff = [sorted_coeff[-1]]
            
                
        file_out.write(str(df2[['ID']].values[j][0])+'\t'+str(merg)+'\t'+str(round(LD1_gal,3))+'\t'+str(round(p_merg,3))+'\t'
            +most_influential_term[0]+'\t'+str(round(most_influential_coeff[0],1))+'\t'
            +str(df2[['low S/N']].values[j][0])+'\t'+str(df2[['outlier predictor']].values[j][0])+'\t'+str(df2[['segmap']].values[j][0])+'\n')
        
    file_out.close()
    return






def classify_from_flagged_plots(prefix, prefix_frames, run, LDA, terms_RFR, df, number_run, verbose=False, all=True, cut_flagged = True):
    #~~~~~~~
    # Now bring in the SDSS galaxies!
    #~~~~~~~

    print('loading up predictor value table........')
    df2 = pd.io.parsers.read_csv(prefix+'SDSS_predictors_with_flags.txt', sep='\t')
    
    if all:
        pass
    else:
        df2 = df2[0:number_run]
    
    if len(df2.columns) ==15: #then you have to delete the first column which is an empty index
        df2 = df2.iloc[: , 1:]

    

    # First, delete all rows that have weird values of n:
    #print('len before crazy values', len(df2))
    #df_filtered = df2[df2['Sersic N'] < 10]

    #df_filtered_2 = df_filtered[df_filtered['Asymmetry (A)'] > -1]

    #df2 = df_filtered_2

    # Delete duplicates:
    if verbose:
        print('len bf duplicate delete', len(df2))
    df2_nodup = df2.duplicated()
    df2 = df2[~df2_nodup]
    if verbose:
        print('len af duplicate delete', len(df2))
        print(df2)

    if cut_flagged:# Then get rid of the entries that are flagged
        df_keep = df2[(df2['low S/N'] == 0) & (df2['outlier predictor'] == 0) & (df2['segmap']==0)]
        df2 = df_keep
        


    
    input_singular = terms_RFR
    #Okay so this next part actually needs to be adaptable to reproduce all possible cross-terms
    crossterms = []
    ct_1 = []
    ct_2 = []
    for j in range(len(input_singular)):
        for i in range(len(input_singular)):
            if j == i or i < j:
                continue
            #input_singular.append(input_singular[j]+'*'+input_singular[i])
            crossterms.append(input_singular[j]+'*'+input_singular[i])
            ct_1.append(input_singular[j])
            ct_2.append(input_singular[i])

    inputs = input_singular + crossterms

    # Now you have to construct a bunch of new rows to the df that include all of these cross-terms
    for j in range(len(crossterms)):
        
        df2[crossterms[j]] = df2.apply(cross_term, axis=1, args=(ct_1[j], ct_2[j]))
        

    X_gal = df2[LDA[2]].values

   




    X_std=[]
    testing_C=[]
    testing_A=[]
    testing_Gini=[]
    testing_M20=[]
    testing_n=[]
    testing_A_S=[]

    testing_n_stat=[]
    testing_A_S_stat=[]
    testing_S_N=[]
    testing_A_R = []

    LD1_SDSS=[]
    p_merg_list=[]
    most_influential = []
    score_merg=[]
    score_nonmerg=[]

    if run[0:12]=='major_merger':
        prior_nonmerg = 0.9
        prior_merg = 0.1
    else:
        if run[0:12]=='minor_merger':
            prior_nonmerg = 0.7
            prior_merg = 0.3
            
        else:
            STOP


    
    exists = 0
    if cut_flagged:

        if path.exists(prefix+'LDA_out_all_SDSS_predictors_'+str(run)+'_flags_cut.txt'):
            exists = 1
            print('Table already exists')
        else:
            print('Table doesnt exist')
        # Make a table with merger probabilities and other diagnostics:
        # you need to run just 'classify' first
    else:
        if path.exists(prefix+'LDA_out_all_SDSS_predictors_'+str(run)+'_flags.txt'):
            exists = 1
            print('Table already exists')
        else:
            print('Table doesnt exist')
        # Make a table with merger probabilities and other diagnostics:
        # you need to run just 'classify' first

    most_influential_nonmerg = []
    most_influential_merg = []
    most_influential_nonmerg_c = []
    most_influential_merg_c = []

    LDA_merg_SDSS = []
    LDA_nonmerg_SDSS = []

    p_merg_merg_SDSS = []
    p_merg_nonmerg_SDSS = []

    
    

    for j in range(len(X_gal)):
        #print(X_gal[j])

        # this is an array of everything standardized
        X_standardized = list((X_gal[j]-LDA[0])/LDA[1])

        X_std.append(X_standardized)
        # use the output from the simulation to assign LD1 value:
        LD1_gal = float(np.sum(X_standardized*LDA[3])+LDA[4])
        
        
        
        LD1_SDSS.append(LD1_gal)
        
        # According to my calculations, LD1 = delta_1 - delta_0 where delta is the score for each class
        # Therefore, in the probability equation p_merg = e^delta_1/(e^delta_1+e^delta_0)
        # you can sub in LD1 and instead end up with p_merg = 1/(1+e^-LD1)
        
        p_merg = 1/(1 + np.exp(-LD1_gal))
        p_nonmerg = 1/(1 + np.exp(LD1_gal))
        
        coeffs = (X_standardized*LDA[3])[0]
        terms = LDA[2]

        
        
        if LD1_gal > 0:
            merg = 1
            # select the coefficient that is the most positive
            # So the max index is what?
            # is the max of the standardized array times all the coefficients, so what has the largest positive value?
            # this just gives you the index to search, if selected from LDA[2] gives you the name of that term
            # and if selected from the lda[3]*x_standardized gives the additive value of this
            
            
            # Array to sort:
            arr1inds = coeffs.argsort()
            sorted_terms = terms[arr1inds[::-1]]
            sorted_coeff = coeffs[arr1inds[::-1]]

            most_influential_merg.append(sorted_terms[0])
            most_influential_merg_c.append(sorted_coeff[0])
            most_influential_term = sorted_terms[0]

            '''
            most_influential_merg.append([sorted_terms[0],sorted_terms[1],sorted_terms[2]])
            most_influential_merg_c.append([sorted_coeff[0],sorted_coeff[1],sorted_coeff[2]])
            most_influential_term = [sorted_terms[0],sorted_terms[1],sorted_terms[2]]
            '''
            LDA_merg_SDSS.append(LD1_gal)
            p_merg_merg_SDSS.append(p_merg)

            
        else:
            merg = 0
            
            
            # Array to sort:
            arr1inds = coeffs.argsort()
            sorted_terms = terms[arr1inds[::-1]]
            sorted_coeff = coeffs[arr1inds[::-1]]

            most_influential_nonmerg.append(sorted_terms[-1])
            most_influential_nonmerg_c.append(sorted_coeff[-1])
            most_influential_term = sorted_terms[-1]
            '''
            
            most_influential_nonmerg.append([sorted_terms[-1],sorted_terms[-2],sorted_terms[-3]])
            most_influential_nonmerg_c.append([sorted_coeff[-1],sorted_coeff[-2],sorted_coeff[-3]])
            most_influential_term = [sorted_terms[-1],sorted_terms[-2],sorted_terms[-3]]
            '''
            LDA_nonmerg_SDSS.append(LD1_gal)
            p_merg_nonmerg_SDSS.append(p_merg)
            
            

        p_merg_list.append(p_merg)
        most_influential.append(most_influential_term)
                
        testing_Gini.append(df2['Gini'].values[j])
        testing_M20.append(df2['M20'].values[j])
        testing_C.append(df2['Concentration (C)'].values[j])
        testing_A.append(df2['Asymmetry (A)'].values[j])
        testing_n.append(df2['Sersic N'].values[j])
        testing_A_S.append(df2['Shape Asymmetry (A_S)'].values[j])

        








    if verbose:
        plt.clf()
        plt.hist(testing_Gini)
        plt.savefig('../Figures/hist_Gini.pdf')
        plt.clf()
        plt.hist(testing_M20)
        plt.savefig('../Figures/hist_M20.pdf')
        plt.clf()
        plt.hist(testing_C)
        plt.savefig('../Figures/hist_C.pdf')
        plt.clf()
        plt.hist(testing_A)
        plt.savefig('../Figures/hist_A.pdf')
        plt.clf()
        plt.hist(testing_n)
        plt.savefig('../Figures/hist_n.pdf')
        plt.clf()
        plt.hist(testing_A_S)
        plt.savefig('../Figures/hist_A_S.pdf')
        
        
    #Get numbers of which was the most influential coefficient for both mergers and nonmergers:
    set_merg = set(most_influential_merg)
    set_nonmerg = set(most_influential_nonmerg)

    list_count = []
    list_elements = []
    # Now look throw the full list to count
    for element in set_merg:
        # Now collect all the values of the coeff:
        coef_here = []
        for j in range(len(most_influential_merg_c)):
        
            if most_influential_merg[j]==element:
                coef_here.append(most_influential_merg_c[j])
        list_count.append(most_influential_merg.count(element))
        list_elements.append(element)
        
    max_idx = np.where(list_count==np.sort(list_count)[-1])[0][0]
    print('top 1 merger', np.sort(list_count)[-1], list_elements[max_idx])
    first_imp_merg = list_elements[max_idx]
    try:
        sec_max_idx = np.where(list_count==np.sort(list_count)[-2])[0][0]
        print('top 2 merger', np.sort(list_count)[-2], list_elements[sec_max_idx])
        second_imp_merg = list_elements[sec_max_idx]
    except IndexError:
        print('no second term')
        second_imp_merg = first_imp_merg



    list_count = []
    list_elements = []
    for element in set_nonmerg:
        coef_here = []
        for j in range(len(most_influential_nonmerg_c)):
        
            if most_influential_nonmerg[j]==element:
                coef_here.append(most_influential_nonmerg_c[j])
        list_count.append(most_influential_nonmerg.count(element))
        list_elements.append(element)
        print('element', element, most_influential_nonmerg.count(element), np.mean(coef_here))
    max_idx = np.where(list_count==np.sort(list_count)[-1])[0][0]
    print('top 1 nonmerger', np.sort(list_count)[-1], list_elements[max_idx])
    first_imp_nonmerg = list_elements[max_idx]
    try:
        sec_max_idx = np.where(list_count==np.sort(list_count)[-2])[0][0]
        print('top 2 nonmerger', np.sort(list_count)[-2], list_elements[sec_max_idx])
        second_imp_nonmerg = list_elements[sec_max_idx]
    except IndexError:
        print('no second term')
        second_imp_nonmerg = first_imp_nonmerg



    sim_nonmerg_pred_1 = []
    sim_merg_pred_1 = []
    sim_nonmerg_pred_2 = []
    sim_merg_pred_2 = []


    # Will need the actual table values for the simulated galaxies to run this part
    

    for j in range(len(df)):
        if LDA[8][j]<0:#then its a nonmerger
            sim_nonmerg_pred_1.append(df[first_imp_nonmerg].values[j])
            sim_nonmerg_pred_2.append(df[second_imp_nonmerg].values[j])
            
            
        if LDA[8][j]>0:#then its a nonmerger
            sim_merg_pred_1.append(df[first_imp_merg].values[j])
            sim_merg_pred_2.append(df[second_imp_merg].values[j])


    merg_pred_1=[]
    merg_pred_2=[]
    nonmerg_pred_1=[]
    nonmerg_pred_2=[]

    merg_gini_LDA_out=[]
    merg_m20_LDA_out=[]
    merg_C_LDA_out=[]
    merg_A_LDA_out=[]
    merg_S_LDA_out=[]
    merg_n_LDA_out=[]
    merg_A_S_LDA_out=[]

    nonmerg_gini_LDA_out=[]
    nonmerg_m20_LDA_out=[]
    nonmerg_C_LDA_out=[]
    nonmerg_A_LDA_out=[]
    nonmerg_S_LDA_out=[]
    nonmerg_n_LDA_out=[]
    nonmerg_A_S_LDA_out=[]

    merg_name_list=[]
    nonmerg_name_list=[]
    #print('~~~~Mergers~~~~')

    p_merg_merg = []
    p_merg_nonmerg = []


    for j in range(len(LD1_SDSS)):
        #LDA_compare.append(float(np.sum(Xs_standardized*output_LDA[3])+output_LDA[4]))

        if LD1_SDSS[j] > 0:#merger
            p_merg_merg.append(p_merg_list[j])
            merg_pred_1.append(df2[first_imp_merg].values[j])
            merg_pred_2.append(df2[second_imp_merg].values[j])
            merg_gini_LDA_out.append(df2['Gini'].values[j])
            merg_m20_LDA_out.append(df2['M20'].values[j])
            merg_C_LDA_out.append(df2['Concentration (C)'].values[j])
            merg_A_LDA_out.append(df2['Asymmetry (A)'].values[j])
            merg_S_LDA_out.append(df2['Clumpiness (S)'].values[j])
            merg_n_LDA_out.append(df2['Sersic N'].values[j])
            merg_A_S_LDA_out.append(df2['Shape Asymmetry (A_S)'].values[j])
            merg_name_list.append(df2['ID'].values[j])
            '''print('~~~~~~')
            print('Merger', LD1_SDSS[j])
            print(df2['ID'].values[j],df2['Gini'].values[j], df2['M20'].values[j],
                  df2['Concentration (C)'].values[j],df2['Asymmetry (A)'].values[j],
                  df2['Clumpiness (S)'].values[j],df2['Sersic N'].values[j],
                  df2['Shape Asymmetry (A_S)'].values[j])
            print('~~~~~~')'''
        else:#nonmerger
            #print(df2['ID'].values[j])
            #print(X_std[j])
            #print(np.sum(X_std[j]))
            p_merg_nonmerg.append(p_merg_list[j])
            nonmerg_pred_1.append(df2[first_imp_nonmerg].values[j])
            nonmerg_pred_2.append(df2[second_imp_nonmerg].values[j])
            nonmerg_gini_LDA_out.append(df2['Gini'].values[j])
            nonmerg_m20_LDA_out.append(df2['M20'].values[j])
            nonmerg_C_LDA_out.append(df2['Concentration (C)'].values[j])
            nonmerg_A_LDA_out.append(df2['Asymmetry (A)'].values[j])
            nonmerg_S_LDA_out.append(df2['Clumpiness (S)'].values[j])
            nonmerg_n_LDA_out.append(df2['Sersic N'].values[j])
            nonmerg_A_S_LDA_out.append(df2['Shape Asymmetry (A_S)'].values[j])
            nonmerg_name_list.append(df2['ID'].values[j])
            
            
                    
    if verbose:
        # This is all about making a panel plot of weird parameters
        plt.clf()
        fig, axs = plt.subplots(2,2, figsize=(5, 5), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = -0.1, wspace=-0.1)

        axs = axs.ravel()
        counter = 0
        for p in range(len(nonmerg_name_list)):
            if counter == 4:
                break
            if p_merg_nonmerg[p] < 0.01:
                continue
                
            gal_id=nonmerg_name_list[p]
            
            try:
                im=fits.open('../imaging/out_'+str(gal_id)+'.fits')
                print('in existence', gal_id)
            except:
                #print('not in existence', gal_id)
                continue
            
            most_influential = most_influential_nonmerg[p]
            
            
            gal_prob=p_merg_nonmerg[p]
            #list_sklearn[new_min_index].predict_proba(X_std)[p]
            
            
            camera_data=(im[1].data/0.005)
            
            axs[counter].imshow(np.abs(camera_data),norm=colors.LogNorm(vmin=10**(3), vmax=10**(6.5)), cmap='afmhot')#vmin=10**(1), vmax=10**(4)norm=colors.LogNorm(vmin=10**(-1), vmax=10**(2.5))
            if str(run)=='minor_merger':
                axs[counter].annotate('p$_{\mathrm{merg, minor}} = $'+str(round(p_merg_nonmerg[p],2))+'\n'+'n =  '+str(round(nonmerg_A_LDA_out[p],2))+'\n'+str(most_influential), xycoords='axes fraction',xy=(0.05,0.8),color='white', size=10)
            else:
                axs[counter].annotate('p$_{\mathrm{merg, major}} = $'+str(round(p_merg_nonmerg[p],2))+'\n'+'n =  '+str(round(nonmerg_A_LDA_out[p],2))+'\n'+str(most_influential), xycoords='axes fraction',xy=(0.05,0.8),color='white', size=10)
            axs[counter].axis('off')
            counter+=1

        plt.tight_layout()
        plt.savefig('../Figures/panel_plot_0.0_nonmergers_SDSS_predictors_'+str(run)+'.pdf')

    if verbose:

        # First get the LDA values for the knwon and unknown mergers and nonmergers:
        '''
        LDA_ID_merg = []
        LDA_ID_nonmerg = []

        nonmerg_p = []
        merg_p = []
        for j in range(len(df)):
            p_merg_sim = 1/(1 + np.exp(-LDA[8][j]))
            if df['class label'].values[j]==0:
                LDA_ID_nonmerg.append(LDA[8][j])
            if df['class label'].values[j]==1:
                LDA_ID_merg.append(LDA[8][j])
            if LDA[8][j]<0:          
                nonmerg_p.append(p_merg_sim)
            else:
                merg_p.append(p_merg_sim)
        '''    
        nonmerg_gini=[]
        nonmerg_m20=[]

        merg_gini=[]
        merg_m20=[]

        nonmerg_gini_LDA=[]
        nonmerg_m20_LDA=[]

        merg_gini_LDA=[]
        merg_m20_LDA=[]


        nonmerg_C_LDA=[]
        nonmerg_A_LDA=[]
        nonmerg_S_LDA=[]
        nonmerg_n_LDA=[]
        nonmerg_A_S_LDA=[]

        merg_C_LDA=[]
        merg_A_LDA=[]
        merg_S_LDA=[]
        merg_n_LDA=[]
        merg_A_S_LDA=[]

        LDA_ID_merg = []
        LDA_ID_nonmerg = []

        indices_nonmerg = []
        indices_merg = []

        merg_p = []
        nonmerg_p = []

        for j in range(len(df)):
            if df['class label'].values[j]==0:
                nonmerg_gini.append(df['Gini'].values[j])
                nonmerg_m20.append(df['M20'].values[j])
                LDA_ID_nonmerg.append(LDA[8][j])
                indices_nonmerg.append(j)
            if df['class label'].values[j]==1:
                merg_gini.append(df['Gini'].values[j])
                merg_m20.append(df['M20'].values[j])
                LDA_ID_merg.append(LDA[8][j])
                indices_merg.append(j)


            p_merg_sim = 1/(1 + np.exp(-LDA[8][j]))
            if LDA[8][j]<0:#then its a nonmerger                                                                                           
                nonmerg_gini_LDA.append(df['Gini'].values[j])
                nonmerg_m20_LDA.append(df['M20'].values[j])
                nonmerg_C_LDA.append(df['Concentration (C)'].values[j])
                nonmerg_A_LDA.append(df['Asymmetry (A)'].values[j])
                nonmerg_S_LDA.append(df['Clumpiness (S)'].values[j])
                nonmerg_n_LDA.append(df['Sersic N'].values[j])
                nonmerg_A_S_LDA.append(df['Shape Asymmetry (A_S)'].values[j])
                nonmerg_p.append(p_merg_sim)

            if LDA[8][j]>0:#then its a nonmerger                                                                                           
                merg_gini_LDA.append(df['Gini'].values[j])
                merg_m20_LDA.append(df['M20'].values[j])
                merg_C_LDA.append(df['Concentration (C)'].values[j])
                merg_A_LDA.append(df['Asymmetry (A)'].values[j])
                merg_S_LDA.append(df['Clumpiness (S)'].values[j])
                merg_n_LDA.append(df['Sersic N'].values[j])
                merg_A_S_LDA.append(df['Shape Asymmetry (A_S)'].values[j])
                merg_p.append(p_merg_sim)



        merg_m20_LDA=np.array(merg_m20_LDA)
        merg_gini_LDA=np.array(merg_gini_LDA)
        merg_C_LDA=np.array(merg_C_LDA)
        merg_A_LDA=np.array(merg_A_LDA)
        merg_S_LDA=np.array(merg_S_LDA)
        merg_n_LDA=np.array(merg_n_LDA)
        merg_A_S_LDA=np.array(merg_A_S_LDA)

        merg_m20=np.array(merg_m20)
        merg_gini=np.array(merg_gini)
        nonmerg_m20=np.array(nonmerg_m20)
        nonmerg_gini=np.array(nonmerg_gini)



        
        
        #Make a histogram of LDA values for merging and nonmerging SDSS
        #galaxies and maybe overlay the LDA values for the simulated galaxies
        hist, bins = np.histogram(LDA[8], bins=50)
        
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(312)
        _ = ax.hist(LDA_merg_SDSS, label='SDSS mergers',alpha=0.75, color='#EA9010',  bins=bins)
        _ = ax.hist(LDA_nonmerg_SDSS, label='SDSS nonmergers',alpha=0.75, color='#90BE6D',  bins=bins)
        plt.axvline(x=0, color='black')
        
        plt.legend()
        ax1 = fig.add_subplot(311)
        _ = ax1.hist(LDA_ID_merg, label='Simulated mergers',alpha=0.75, color='#EFA8B8',  bins=bins)
        _ = ax1.hist(LDA_ID_nonmerg, label='Simulated nonmergers',alpha=0.75, color='#55868C',  bins=bins)
        
        plt.axvline(x=0, color='black')



        plt.legend()
        ax.set_xlabel('LD1 Value')
        if run=='major_merger':
            plt.title('Major')
        if run=='minor_merger':
            plt.title('Minor')


        ax2 = fig.add_subplot(313)
        _ = ax2.hist(p_merg_merg_SDSS, label='SDSS mergers',alpha=0.75, color='#EA9010',  bins=50, range=[0,1])
        a, b, c = ax2.hist(p_merg_nonmerg_SDSS, label='SDSS nonmergers',alpha=0.75, color='#90BE6D',  bins=50, range=[0,1])
        
        print(a)
        plt.axvline(x=0.5, color='black')
        ax2.set_ylim([0,(len(p_merg_merg_SDSS)+len(p_merg_nonmerg_SDSS))/5])
        ax2.annotate('^'+str(int(np.max(a))), xy=(0.07,0.89), xycoords='axes fraction')
        ax2.set_xlabel(r'$p_{\mathrm{merg}}$ value')

        plt.savefig('../Figures/hist_LDA_divided_'+str(run)+'.pdf')


        #Make a histogram of LDA values for merging and nonmerging SDSS
        #galaxies and maybe overlay the LDA values for the simulated galaxies
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(212)
        
        #_, bins, _ = ax.hist(LDA_ID, label='Simulation LD1', color='#FE4E00', bins=50, range = [-12,12])
        _ = ax.hist(p_merg_merg, label='SDSS mergers',alpha=0.75, color='#EA9010',  bins=50, range = [-0.1,1.1])
        _ = ax.hist(p_merg_nonmerg, label='SDSS nonmergers',alpha=0.75, color='#90BE6D',  bins=50, range = [-0.1,1.1])
        plt.axvline(x=0.5, color='black')
        
        plt.legend()
        ax1 = fig.add_subplot(211)
        _ = ax1.hist(merg_p, label='Simulated mergers',alpha=0.75, color='#EFA8B8',  bins=50, range = [-0.1,1.1])
        _ = ax1.hist(nonmerg_p, label='Simulated nonmergers',alpha=0.75, color='#55868C',  bins=50, range = [-0.1,1.1])
        
        plt.axvline(x=0.5, color='black')
        plt.legend()
        ax.set_xlabel(r'$p_{\mathrm{merg}}$')
        if run=='major_merger':
            plt.title('Major')
        if run=='minor_merger':
            plt.title('Minor')
        plt.savefig('../Figures/hist_LDA_divided_p_'+str(run)+'.pdf')
        
    print('~~~~~~~Analysis Results~~~~~~')



    print('percent nonmerg',len(nonmerg_name_list)/(len(nonmerg_name_list)+len(merg_name_list)))
    print('percent merg',len(merg_name_list)/(len(nonmerg_name_list)+len(merg_name_list)))

    print('# nonmerg',len(nonmerg_name_list))
    print('# merg',len(merg_name_list))





    print('~~~~~~~~~~~~~~~~~~~~~~~~')


    if verbose:
        # Okay sort everything by which probability grouping it is in
        ps = 3
        # Get indices that are
        indices_nonmerg = []
        for j in range(5):
            indices = np.where((np.array(p_merg_nonmerg) < 0.1*j+0.1) & (np.array(p_merg_nonmerg) > 0.1*j))[0]
            indices_nonmerg.append(indices)
        indices_merg = []
        for j in range(5):
            indices_merg.append(np.where((np.array(p_merg_merg) < 0.1*j+0.6) & (np.array(p_merg_merg) > 0.1*j+0.5))[0])
        
        print('shape of indices', np.shape(indices_merg))

        sns.set_style('dark')
        plt.clf()
        fig=plt.figure(figsize=(11,7))
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.3, wspace=0.1)


        ax1=fig.add_subplot(221)
        ax1.set_title('SDSS mergers (# = '+str(len(merg_m20_LDA_out))+')', loc='right')
        
        
        #im1 = ax1.hexbin(merg_pred_1, merg_pred_2, C=p_merg_merg, cmap='magma', vmin=0.5, vmax=1, gridsize=40)
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(merg_pred_2, merg_pred_1, p_merg_merg,statistic='mean', bins=50)
         

         
        
        
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(nonmerg_pred_2, nonmerg_pred_1, p_merg_nonmerg,statistic='mean', bins=50)
         

        min_x = np.min([xedges[0],xedgesnon[0]])
        max_x = np.max([xedges[-1], xedgesnon[-1]])
        
        min_y = np.min([yedges[0],yedgesnon[0]])
        max_y = np.max([yedges[-1],yedgesnon[-1]])
        
        im1 = ax1.imshow(np.flipud(heatmap), cmap='magma', vmin=0.5, vmax=1.0,extent=[yedges[0], yedges[-1], xedges[0],xedges[-1]])
        
        ax1.set_ylim(xedges[0], xedges[-1])
        ax1.set_xlim(yedges[0], yedges[-1])
        '''
        cs = []
        xs = []
        ys = []
        for j in range(5):
            cs.append(np.mean(np.array(p_merg_merg)[indices_merg[j]]))
            xs.append(np.mean(np.array(merg_pred_1)[indices_merg[j]]))
            ys.append(np.mean(np.array(merg_pred_2)[indices_merg[j]]))
        ax1.scatter(xs, ys, c=cs, marker='*', s=150, vmin=0.5, vmax=1, cmap='magma', edgecolor='black', zorder=100)
        '''
        plt.colorbar(im1, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        
        ax1.set_xlabel(first_imp_merg)
        ax1.set_ylabel(second_imp_merg)
        ax1.set_aspect((yedges[-1]-yedges[0])/(xedges[-1]-xedges[0]))#'equal')

        ax2=fig.add_subplot(222)
        ax2.set_title('SDSS non-mergers (# = '+str(len(nonmerg_m20_LDA_out))+')', loc='right')
        
        #sns.kdeplot(nonmerg_m20_LDA, nonmerg_gini_LDA, cmap="Blues", shade=True,shade_lowest=False)
        
        
        im2 = ax2.imshow(np.flipud(heatmapnon), cmap='viridis', vmin=0, vmax=0.5,extent=[yedgesnon[0],yedgesnon[-1],xedgesnon[0],xedgesnon[-1]])
         
        ax2.set_ylim(xedgesnon[0],xedgesnon[-1])
        ax2.set_xlim(yedgesnon[0],yedgesnon[-1])
        
        '''
        cs = []
        xs = []
        ys = []
        for j in range(5):
            cs.append(np.mean(np.array(p_merg_nonmerg)[indices_nonmerg[j]]))
            xs.append(np.mean(np.array(nonmerg_pred_1)[indices_nonmerg[j]]))
            ys.append(np.mean(np.array(nonmerg_pred_2)[indices_nonmerg[j]]))
        ax2.scatter(xs, ys, c=cs, marker='*', s=150, vmin=0, vmax=0.5, cmap='viridis', edgecolor='white', zorder=100)
        '''
        plt.colorbar(im2, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        
        ax2.set_xlabel(first_imp_nonmerg)
        ax2.set_ylabel(second_imp_nonmerg)
        ax2.set_aspect((yedgesnon[-1]-yedgesnon[0])/(xedgesnon[-1]-xedgesnon[0]))#'equal')

        ax3 = fig.add_subplot(223)
        #sim_nonmerg_pred_1
        
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(sim_merg_pred_2, sim_merg_pred_1, merg_p,statistic='mean', bins=20)
         

         
        
        
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(sim_nonmerg_pred_2, sim_nonmerg_pred_1, nonmerg_p,statistic='mean', bins=20)
         

        
        
        ax3.set_title('Simulated mergers (# = '+str(len(merg_p))+')', loc='right')
        
        #sns.kdeplot(nonmerg_m20_LDA, nonmerg_gini_LDA, cmap="Blues", shade=True,shade_lowest=False)
        
        
        im3 = ax3.imshow(np.flipud(heatmap), cmap='magma', vmin=0.5, vmax=1.0,extent=[yedges[0],yedges[-1],xedges[0],xedges[-1]])
         
        ax3.set_ylim(xedges[0],xedges[-1])
        ax3.set_xlim(yedges[0],yedges[-1])
        
        
        plt.colorbar(im3, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        
        ax3.set_xlabel(first_imp_merg)
        ax3.set_ylabel(second_imp_merg)
        ax3.set_aspect((yedges[-1]-yedges[0])/(xedges[-1]-xedges[0]))

        ax4 = fig.add_subplot(224)
        
        ax4.set_title('Simulated non-mergers (# = '+str(len(nonmerg_p))+')', loc='right')
        im4 = ax4.imshow(np.flipud(heatmapnon), cmap='viridis', vmin=0, vmax=0.5,extent=[yedgesnon[0],yedgesnon[-1],xedgesnon[0],xedgesnon[-1]])
         
        ax4.set_ylim(xedgesnon[0],xedgesnon[-1])
        ax4.set_xlim(yedgesnon[0],yedgesnon[-1])
        
        
        plt.colorbar(im4, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        
        ax4.set_xlabel(first_imp_nonmerg)
        ax4.set_ylabel(second_imp_nonmerg)
        ax4.set_aspect((yedgesnon[-1]-yedgesnon[0])/(xedgesnon[-1]-xedgesnon[0]))

        plt.savefig('../Figures/first_sec_separate_'+str(run)+'.pdf',  bbox_inches = 'tight')
        
        
        
        
        dashed_line_x=np.linspace(-0.5,-3,100)
        dashed_line_y=[-0.14*x + 0.33 for x in dashed_line_x]



        # Makes density of the simulated galaxies separately from the SDSS galaxies
        plt.clf()
        sns.set_style('dark')
        fig=plt.figure(figsize=(11,7))
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.3, wspace=0.1)


        ax1=fig.add_subplot(221)
        ax1.set_title('SDSS mergers (# = '+str(len(merg_m20_LDA_out))+')', loc='right')
        ax1.plot(dashed_line_x, dashed_line_y, ls='--', color='black')

         
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(merg_gini_LDA_out, merg_m20_LDA_out, p_merg_merg,statistic='mean', bins=50)
          
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(nonmerg_gini_LDA_out, nonmerg_m20_LDA_out, p_merg_nonmerg,statistic='mean', bins=50)
         
        xmin = np.min([xedges[0],xedgesnon[0]])
        xmax = np.max([xedges[-1], xedgesnon[-1]])
         
        ymin = np.min([yedges[0],yedgesnon[0]])
        ymax = np.max([yedges[-1],yedgesnon[-1]])
          
        im1 = ax1.imshow(np.fliplr(np.flipud(heatmap)), cmap='magma', vmin=0.5, vmax=1.0,extent=[ymax, ymin, xmin, xmax])
          
        ax1.set_ylim(xmin, xmax)
        ax1.set_xlim(ymax, ymin)

       
         
        plt.colorbar(im1, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax1.set_xlabel(r'$M_{20}$')
        ax1.set_ylabel(r'$Gini$')
        ax1.set_aspect((ymax-ymin)/(xmax-xmin))

        ax2=fig.add_subplot(222)
        ax2.set_title('SDSS non-mergers (# = '+str(len(nonmerg_m20_LDA_out))+')', loc='right')
        ax2.plot(dashed_line_x, dashed_line_y, ls='--', color='black')

         
         
         

        im2 = ax2.imshow(np.fliplr(np.flipud(heatmapnon)), cmap='viridis', vmin=0, vmax=0.5,extent=[ymax, ymin, xmin, xmax])
        ax2.set_ylim(xmin, xmax)
        ax2.set_xlim(ymax, ymin)
        plt.colorbar(im2, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax2.set_xlabel(r'$M_{20}$')
        ax2.set_ylabel(r'$Gini$')
        ax2.set_aspect((ymax-ymin)/(xmax-xmin))

        ax3=fig.add_subplot(223)
        ax3.set_title('Simulated mergers (# = '+str(len(merg_p))+')', loc='right')
        ax3.plot(dashed_line_x, dashed_line_y, ls='--', color='black')

          
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(merg_gini_LDA, merg_m20_LDA, merg_p,statistic='mean', bins=20)
           
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(nonmerg_gini_LDA, nonmerg_m20_LDA, nonmerg_p,statistic='mean', bins=20)
          
        xmins = np.min([xedges[0],xedgesnon[0]])
        xmaxs = np.max([xedges[-1], xedgesnon[-1]])
          
        ymins = np.min([yedges[0],yedgesnon[0]])
        ymaxs = np.max([yedges[-1],yedgesnon[-1]])
           
        im3 = ax3.imshow(np.fliplr(np.flipud(heatmap)), cmap='magma', vmin=0.5, vmax=1.0,extent=[ymaxs, ymins, xmins, xmaxs])
           
        ax3.set_ylim(xmin, xmax)
        ax3.set_xlim(ymax, ymin)

        
          
        plt.colorbar(im3, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax3.set_xlabel(r'$M_{20}$')
        ax3.set_ylabel(r'$Gini$')
        ax3.set_aspect((ymax-ymin)/(xmax-xmin))

        ax4=fig.add_subplot(224)
        ax4.set_title('Simulated non-mergers (# = '+str(len(nonmerg_p))+')', loc='right')
        ax4.plot(dashed_line_x, dashed_line_y, ls='--', color='black')

          
          
          

        im4 = ax4.imshow(np.fliplr(np.flipud(heatmapnon)), cmap='viridis', vmin=0, vmax=0.5,extent=[ymaxs, ymins, xmins, xmaxs])
        ax4.set_ylim(xmin, xmax)
        ax4.set_xlim(ymax, ymin)
        plt.colorbar(im4, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax4.set_xlabel(r'$M_{20}$')
        ax4.set_ylabel(r'$Gini$')
        ax4.set_aspect((ymax-ymin)/(xmax-xmin))

        plt.savefig('../Figures/gini_m20_separate_'+str(run)+'.pdf',  bbox_inches = 'tight')
        
        
        
        #~~~~~~~~~~~~~~~~~ Now, make this for C-A ~~~~~~~~~~~~~~~~~~~~~~~~~~
        plt.clf()
        fig=plt.figure(figsize=(11,7))
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.3, wspace=0.1)


        ax1=fig.add_subplot(221)
        ax1.set_title('SDSS mergers (# = '+str(len(merg_m20_LDA_out))+')', loc='right')
        
          
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(merg_C_LDA_out, merg_A_LDA_out, p_merg_merg,statistic='mean', bins=50)
           
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(nonmerg_C_LDA_out, nonmerg_A_LDA_out, p_merg_nonmerg,statistic='mean', bins=50)
          
        xmin = np.min([xedges[0],xedgesnon[0]])
        xmax = np.max([xedges[-1], xedgesnon[-1]])
          
        ymin = np.min([yedges[0],yedgesnon[0]])
        ymax = np.max([yedges[-1],yedgesnon[-1]])
           
        im1 = ax1.imshow(np.fliplr(np.flipud(heatmap)), cmap='magma', vmin=0.5, vmax=1.0,extent=[ymax, ymin, xmin, xmax])
           
        ax1.set_ylim(xmin, xmax)
        ax1.set_xlim(ymin, ymax)

        
          
        plt.colorbar(im1, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax1.set_xlabel(r'$A$')
        ax1.set_ylabel(r'$C$')
        ax1.set_aspect((ymax-ymin)/(xmax-xmin))

        ax2=fig.add_subplot(222)
        ax2.set_title('SDSS non-mergers (# = '+str(len(nonmerg_m20_LDA_out))+')', loc='right')
        
        im2 = ax2.imshow(np.fliplr(np.flipud(heatmapnon)), cmap='viridis', vmin=0, vmax=0.5,extent=[ymax, ymin, xmin, xmax])
        ax2.set_ylim(xmin, xmax)
        ax2.set_xlim(ymin, ymax)
        plt.colorbar(im2, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax2.set_xlabel(r'$A$')
        ax2.set_ylabel(r'$C$')
        ax2.set_aspect((ymax-ymin)/(xmax-xmin))

        ax3=fig.add_subplot(223)
        ax3.set_title('Simulated mergers (# = '+str(len(merg_p))+')', loc='right')
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(merg_C_LDA, merg_A_LDA, merg_p,statistic='mean', bins=20)
            
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(nonmerg_C_LDA, nonmerg_A_LDA, nonmerg_p,statistic='mean', bins=20)
           
        xmins = np.min([xedges[0],xedgesnon[0]])
        xmaxs = np.max([xedges[-1], xedgesnon[-1]])
           
        ymins = np.min([yedges[0],yedgesnon[0]])
        ymaxs = np.max([yedges[-1],yedgesnon[-1]])
            
        im3 = ax3.imshow(np.fliplr(np.flipud(heatmap)), cmap='magma', vmin=0.5, vmax=1.0,extent=[ymaxs, ymins, xmins, xmaxs])
            
        ax3.set_ylim(xmin, xmax)
        ax3.set_xlim(ymin, ymax)

         
           
        plt.colorbar(im3, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax3.set_xlabel(r'$A$')
        ax3.set_ylabel(r'$C$')
        ax3.set_aspect((ymax-ymin)/(xmax-xmin))

        ax4=fig.add_subplot(224)
        ax4.set_title('Simulated non-mergers (# = '+str(len(nonmerg_p))+')', loc='right')
        
           
           
           

        im4 = ax4.imshow(np.fliplr(np.flipud(heatmapnon)), cmap='viridis', vmin=0, vmax=0.5,extent=[ymaxs, ymins, xmins, xmaxs])
        ax4.set_ylim(xmin, xmax)
        ax4.set_xlim(ymin, ymax)
        plt.colorbar(im4, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax4.set_xlabel(r'$A$')
        ax4.set_ylabel(r'$C$')
        ax4.set_aspect((ymax-ymin)/(xmax-xmin))
        
        ax1.axvline(x=0.35, ls='--', color='black')
        ax2.axvline(x=0.35, ls='--', color='black')
        ax3.axvline(x=0.35, ls='--', color='black')
        ax4.axvline(x=0.35, ls='--', color='black')

        plt.savefig('../Figures/C_A_separate_'+str(run)+'.pdf',  bbox_inches = 'tight')
        
        
        #~~~~~~~~~~~~~~~~~ Now, make this for n-A_S ~~~~~~~~~~~~~~~~~~~~~~~~~~
        plt.clf()
        fig=plt.figure(figsize=(11,7))
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.3, wspace=0.1)


        ax1=fig.add_subplot(221)
        ax1.set_title('SDSS mergers (# = '+str(len(merg_m20_LDA_out))+')', loc='right')
        
          
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(merg_n_LDA_out, merg_A_S_LDA_out, p_merg_merg,statistic='mean', bins=50)
           
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(nonmerg_n_LDA_out, nonmerg_A_S_LDA_out, p_merg_nonmerg,statistic='mean', bins=50)
          
        xmin = np.min([xedges[0],xedgesnon[0]])
        xmax = np.max([xedges[-1], xedgesnon[-1]])
          
        ymin = np.min([yedges[0],yedgesnon[0]])
        ymax = np.max([yedges[-1],yedgesnon[-1]])
           
        im1 = ax1.imshow(np.fliplr(np.flipud(heatmap)), cmap='magma', vmin=0.5, vmax=1.0,extent=[ymax, ymin, xmin, xmax])
           
        ax1.set_ylim(xmin, xmax)
        ax1.set_xlim(ymin, ymax)

        
          
        plt.colorbar(im1, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax1.set_xlabel(r'$A_S$')
        ax1.set_ylabel(r'Sersic $n$')
        ax1.set_aspect((ymax-ymin)/(xmax-xmin))

        ax2=fig.add_subplot(222)
        ax2.set_title('SDSS non-mergers (# = '+str(len(nonmerg_m20_LDA_out))+')', loc='right')
        
        im2 = ax2.imshow(np.fliplr(np.flipud(heatmapnon)), cmap='viridis', vmin=0, vmax=0.5,extent=[ymax, ymin, xmin, xmax])
        ax2.set_ylim(xmin, xmax)
        ax2.set_xlim(ymin, ymax)
        plt.colorbar(im2, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax2.set_xlabel(r'$A_S$')
        ax2.set_ylabel(r'Sersic $n$')
        ax2.set_aspect((ymax-ymin)/(xmax-xmin))

        ax3=fig.add_subplot(223)
        ax3.set_title('Simulated mergers (# = '+str(len(merg_p))+')', loc='right')
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(merg_n_LDA, merg_A_S_LDA, merg_p,statistic='mean', bins=20)
            
        heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(nonmerg_n_LDA, nonmerg_A_S_LDA, nonmerg_p,statistic='mean', bins=20)
           
        xmins = np.min([xedges[0],xedgesnon[0]])
        xmaxs = np.max([xedges[-1], xedgesnon[-1]])
           
        ymins = np.min([yedges[0],yedgesnon[0]])
        ymaxs = np.max([yedges[-1],yedgesnon[-1]])
            
        im3 = ax3.imshow(np.fliplr(np.flipud(heatmap)), cmap='magma', vmin=0.5, vmax=1.0,extent=[ymaxs, ymins, xmins, xmaxs])
            
        ax3.set_ylim(xmin, xmax)
        ax3.set_xlim(ymin, ymax)

         
           
        plt.colorbar(im3, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax3.set_xlabel(r'$A_S$')
        ax3.set_ylabel(r'Sersic $n$')
        ax3.set_aspect((ymax-ymin)/(xmax-xmin))

        ax4=fig.add_subplot(224)
        ax4.set_title('Simulated non-mergers (# = '+str(len(merg_p))+')', loc='right')
        
           
           
           

        im4 = ax4.imshow(np.fliplr(np.flipud(heatmapnon)), cmap='viridis', vmin=0, vmax=0.5,extent=[ymaxs, ymins, xmins, xmaxs])
        ax4.set_ylim(xmin, xmax)
        ax4.set_xlim(ymin, ymax)
        plt.colorbar(im4, fraction=0.046, label=r'$p_{\mathrm{merg}}$')
        ax4.set_xlabel(r'$A_S$')
        ax4.set_ylabel(r'Sersic $n$')
        ax4.set_aspect((ymax-ymin)/(xmax-xmin))
        
        ax1.axvline(x=0.2, ls='--', color='black')
        ax2.axvline(x=0.2, ls='--', color='black')
        ax3.axvline(x=0.2, ls='--', color='black')
        ax4.axvline(x=0.2, ls='--', color='black')

        plt.savefig('../Figures/n_A_S_separate_'+str(run)+'.pdf',  bbox_inches = 'tight')
        

    
    # Now, for the SDSS show
    sdss, ra, dec = download_sdss_ra_dec_table(prefix)

    sns.set_style("white")
    '''
    count = 0
    for pp in range(len(df2)):
        p = pp+191#+1800#0000
        
        
        gal_id=df2[['ID']].values[p][0]
        print('p', p, 'id', gal_id)
        
        gal_prob=LD1_SDSS[p]
        print('gal_id', gal_id)
        print('LDA', gal_prob)
        print('prob', p_merg_list[p])
        index_match = np.where(sdss==gal_id)[0][0]
        print(index_match)
        if index_match:
            plot_individual(gal_id, ra[index_match], dec[index_match], p_merg_list[p], run, prefix_frames)
        else:
            continue
        count+=1
        if count > 20:
            break



    #Optional panel to plot the images of these things with their probabilities assigned


    sns.set_style("white")


    #First, an option for just plotting these individually
    #os.chdir(os.path.expanduser('/Volumes/My Book/Clone_Docs_old_mac/Backup_My_Book/My_Passport_backup/MergerMonger'))
    for p in range(len(df2)):#len(df2)):
        plt.clf()
        gal_id=df2[['ID']].values[p][0]
        
        if LD1_SDSS[p] > 0:
            gal_name='Merger'
            most_influential = most_influential_merg[p]
        else:
            gal_name='Nonmerger'
            most_influential = most_influential_nonmerg[p]
        gal_prob=LD1_SDSS[p]
        #list_sklearn[new_min_index].predict_proba(X_std)[p]
        
        try:
            im=fits.open('../imaging/out_'+str(gal_id)+'.fits')
            camera_data=(im[1].data/0.005)
        except:
            # Get RA and DEC:
            index_match = np.where(sdss==gal_id)[0][0]
            print('match', index_match)
            if index_match:
                camera_data = download_galaxy(gal_id, ra[index_match], dec[index_match], prefix_frames, 40)
            else:
                continue
            #continue
        
        
        plt.imshow(np.abs(camera_data),norm=colors.LogNorm(vmin=10**(3), vmax=10**(6.5)), cmap='afmhot')#vmin=10**(1), vmax=10**(4)norm=colors.LogNorm(vmin=10**(-1), vmax=10**(2.5))
        plt.annotate('SDSS '+str(type_gal)+'\n'+r'p$_{\mathrm{merg}} = $'+str(round(p_merg_list[p],2))+'\n'+'LD1 =  '+str(round(LD1_SDSS[p],2))+'\n'+str(most_influential), xycoords='axes fraction',xy=(0.05,0.8),color='white', size=20)
        
        plt.axis('off')
     #   plt.colorbar()
        plt.tight_layout()
        plt.savefig('../Figures/ind_gals/SDSS_'+str(type_gal)+'_'+str(run)+'_'+str(gal_id)+'.pdf')
        if p > 40:
            break
    
        
    # I would like to make a panel plot of mergers and then a panel plot of nonmergers
    plt.clf()
    fig, axs = plt.subplots(5,5, figsize=(15, 15), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = -0.1, wspace=-0.1)

    axs = axs.ravel()
    counter = 0
    for p in range(len(df2)):
        if counter == 25:
            break
        if LD1_SDSS[p] < 0:
            continue
        most_influential = most_influential_merg[counter]
        gal_id=df2[['ID']].values[p][0]
        
        gal_prob=LD1_SDSS[p]
        #list_sklearn[new_min_index].predict_proba(X_std)[p]
        
        try:
            im=fits.open('../imaging/out_'+str(gal_id)+'.fits')
            camera_data=(im[1].data/0.005)
        except:
            index_match = np.where(sdss==gal_id)[0][0]
            if index_match:
                camera_data = download_galaxy(gal_id, ra[index_match], dec[index_match], prefix_frames, 40)
            else:
                continue
            
        
        
        axs[counter].imshow(np.abs(camera_data),norm=colors.LogNorm(vmin=10**(3), vmax=10**(6.5)), cmap='afmhot')#vmin=10**(1), vmax=10**(4)norm=colors.LogNorm(vmin=10**(-1), vmax=10**(2.5))
        if str(run)=='minor_merger':
            axs[counter].annotate('p$_{\mathrm{merg, minor}} = $'+str(round(p_merg_list[p],2))+'\n'+'LD1 =  '+str(round(LD1_SDSS[p],2))+'\n'+str(most_influential), xycoords='axes fraction',xy=(0.05,0.8),color='white', size=10)
        else:
            axs[counter].annotate('p$_{\mathrm{merg, major}} = $'+str(round(p_merg_list[p],2))+'\n'+'LD1 =  '+str(round(LD1_SDSS[p],2))+'\n'+str(most_influential), xycoords='axes fraction',xy=(0.05,0.8),color='white', size=10)
        axs[counter].axis('off')
        counter+=1

    plt.tight_layout()
    plt.savefig('../Figures/panel_plot_mergers_SDSS_'+str(type_gal)+'_'+str(run)+'.pdf')


    # nonmergers
    plt.clf()
    fig, axs = plt.subplots(5,5, figsize=(15, 15), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = -0.1, wspace=-0.1)

    axs = axs.ravel()
    counter = 0
    for p in range(len(df2)):
        if counter == 25:
            break
        if LD1_SDSS[p] > 0:
            continue
        most_influential = most_influential_nonmerg[counter]
        gal_id=df2[['ID']].values[p][0]
        
        gal_prob=LD1_SDSS[p]
        #list_sklearn[new_min_index].predict_proba(X_std)[p]
        
        try:
            im=fits.open('../imaging/out_'+str(gal_id)+'.fits')
            camera_data=(im[1].data/0.005)
        except:
            # Get RA and DEC:
            index_match = np.where(sdss==gal_id)[0][0]
            if index_match:
                camera_data = download_galaxy(gal_id, ra[index_match], dec[index_match], prefix_frames, 40)
            else:
                continue
        
        
        axs[counter].imshow(np.abs(camera_data),norm=colors.LogNorm(vmin=10**(3), vmax=10**(6.5)), cmap='afmhot')#vmin=10**(1), vmax=10**(4)norm=colors.LogNorm(vmin=10**(-1), vmax=10**(2.5))
        if str(run) == 'minor_merger':
            axs[counter].annotate('p$_{\mathrm{merg, minor}} = $'+str(round(p_merg_list[p],2))+'\n'+'LD1 =  '+str(round(LD1_SDSS[p],2))+'\n'+str(most_influential), xycoords='axes fraction',xy=(0.05,0.8),color='white', size=10)
        else:
            axs[counter].annotate('p$_{\mathrm{merg, major}} = $'+str(round(p_merg_list[p],2))+'\n'+'LD1 =  '+str(round(LD1_SDSS[p],2))+'\n'+str(most_influential), xycoords='axes fraction',xy=(0.05,0.8),color='white', size=10)
        axs[counter].axis('off')
        counter+=1

    plt.tight_layout()
    plt.savefig('../Figures/panel_plot_nonmergers_SDSS_'+str(type_gal)+'_'+str(run)+'.pdf')

    '''
    # I think it'll be cool to plot a panel with probability of being a merger on one axis and examples of multiple
    # probability bins

    # Also get the CDF value

    # Define a histogram with spacing defined
    spacing = 1000 # this will be the histogram binning but also how finely sampled the CDF is
    hist = np.histogram(p_merg_list, bins=spacing)

    # Put this in continuous distribution form in order to calculate the CDF
    hist_dist = scipy.stats.rv_histogram(hist)


    plt.clf()
    fig, axs = plt.subplots(2,5, figsize=(15, 7), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = -0.2, wspace=0.1)

    axs = axs.ravel()
    counter = 0
    bin_start_array = [0,0.0001,0.001,0.01,0.1]
    bin_end_array = [0.0001,0.001,0.01,0.1,0.5]

    # Step 1 is to shuffle df right off the bat and then also shuffle p_merg to be the same indices
    # First make sure that p is one of the rows
    df_with_p = df2
    df_with_p['p_merg'] = p_merg_list
    df_with_p['most influential'] = most_influential
    df_with_p_sample = df_with_p.sample(n=len(df_with_p))


    # Now randomly shuffle :)
    df_with_p_shuffle = df_with_p_sample.sample(frac=1).reset_index(drop=True)

    print(df_with_p_shuffle)
    '''

    for j in range(5):
        bin_start = bin_start_array[j]#0.1*j
        bin_end = bin_end_array[j]#0.1*j+0.1
        counter = 0
        counter_i = 0
        for p in range(len(df2)):
            if counter > 1:
                break
            # go through all bins and find two examples of each
            
            if df_with_p_shuffle['p_merg'][p] > 0.5:
                continue
            if df_with_p_shuffle['p_merg'][p] < bin_start or df_with_p_shuffle['p_merg'][p] > bin_end:
                # then you can use this galaxy as an example
                counter_i+=1
                continue
            gal_id=df_with_p_shuffle[['ID']].values[p][0]
            
            try:
                im=fits.open('../imaging/out_'+str(gal_id)+'.fits')
                camera_data=(im[1].data/0.005)
            except:
                # Get RA and DEC:
                index_match = np.where(sdss==gal_id)[0][0]
                if index_match:
                    camera_data = download_galaxy(gal_id, ra[index_match], dec[index_match], prefix_frames, 40)
                else:
                    continue
            
            if counter == 0:
                #then its top row
                axis_number = j
                
            else:
                axis_number = j+5
                axs[axis_number].set_xlabel(str(round(bin_start,5))+' < p$_{\mathrm{merg}}$ < '+str(round(bin_end,5)))
                
            
                
            most_influential = most_influential_nonmerg[counter_i]
            # Figure out which position you need to put this in
            axs[axis_number].imshow(np.abs(camera_data),norm=colors.LogNorm(vmax=10**5, vmin=10**(0.5)), cmap='afmhot', interpolation='None')#vmin=10**(1), vmax=10**(4)norm=colors.LogNorm(vmin=10**(-1), vmax=10**(2.5))
            axs[axis_number].annotate('p$_{\mathrm{merg}} = $'+str(round(df_with_p_shuffle['p_merg'][p],5))+', CDF = '+str(round(hist_dist.cdf(df_with_p_shuffle['p_merg'][p]),2)), 
                    xycoords='axes fraction',xy=(0.05,0.9),xytext=(0.05, 0.9), textcoords='axes fraction',
                    bbox=dict(boxstyle="round", fc="0.9"), color='black')
            axs[axis_number].annotate('ObjID = '+str(gal_id), xycoords='axes fraction',xy=(0.05,0.02),color='white', size=10)
            axs[axis_number].set_yticklabels([])
            axs[axis_number].set_xticklabels([])
            counter+=1
            counter_i+=1
            

    plt.savefig('../Figures/probability_panel_nonmergers_SDSS_'+str(type_gal)+'_'+str(run)+'.pdf')

    

    plt.clf()
    fig, axs = plt.subplots(2,5, figsize=(15, 7), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = -0.2, wspace=0.1)

    bin_start_array = [0.5,0.9,0.99,0.999,0.9999]
    bin_end_array = [0.9,0.99,0.999,0.9999,1.0]

    axs = axs.ravel()
    for j in range(5):
        
        bin_start = bin_start_array[j]#0.1*j
        bin_end = bin_end_array[j]#0.1*j+0.1
        counter = 0
        counter_i = 0
        for p in range(len(df2)):
            
            if counter > 1:
                break
            # go through all bins and find two examples of each
            if df_with_p_shuffle['p_merg'][p] <0.5:
                continue
            if df_with_p_shuffle['p_merg'][p] < bin_start or df_with_p_shuffle['p_merg'][p] > bin_end:
                # then you can use this galaxy as an example
                counter_i+=1
                continue
            gal_id=df_with_p_shuffle[['ID']].values[p][0]
            try:
                im=fits.open('../imaging/out_'+str(gal_id)+'.fits')
                camera_data=(im[1].data/0.005)
            except:
                # Get RA and DEC:
                index_match = np.where(sdss==gal_id)[0][0]
                if index_match:
                    camera_data = download_galaxy(gal_id, ra[index_match], dec[index_match], prefix_frames, 40)
                else:
                    continue
            most_influential = most_influential_merg[counter_i]
            
            if counter == 0:
                #then its top row
                axis_number = j
            else:
                axis_number = j+5
                axs[axis_number].set_xlabel(str(round(bin_start,5))+' < p$_{merg}$ < '+str(round(bin_end,5)))
            
            # Figure out which position you need to put this in 
            axs[axis_number].imshow(np.abs(camera_data),norm=colors.LogNorm(vmax=10**5, vmin=10**(0.5)), cmap='afmhot', interpolation='None')#vmin=10**(1), vmax=10**(4)norm=colors.LogNorm(vmin=10**(-1), vmax=10**(2.5))
            axs[axis_number].annotate('p$_{\mathrm{merg}} = $'+str(round(df_with_p_shuffle['p_merg'][p],5))+', CDF = '+str(round(hist_dist.cdf(df_with_p_shuffle['p_merg'][p]),2)), 
                    xycoords='axes fraction',xy=(0.05,0.9),xytext=(0.05, 0.9), textcoords='axes fraction',
                    bbox=dict(boxstyle="round", fc="0.9"), color='black')
            axs[axis_number].annotate('ObjID = '+str(gal_id), xycoords='axes fraction',xy=(0.05,0.02),color='white', size=10)
            #axs[axis_number].axis('off')
            axs[axis_number].set_yticklabels([])
            axs[axis_number].set_xticklabels([])
            counter+=1
            counter_i+=1
            

    plt.savefig('../Figures/probability_panel_mergers_SDSS_'+str(type_gal)+'_'+str(run)+'.pdf')
    '''

    # Now make a panel plot that's just one of the probability bins
    size_image = 80

    plt.clf()
    fig, axs = plt.subplots(2,5, figsize=(15, 7), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = -0.2, wspace=0.1)

    axs = axs.ravel()
    bin_start = 0
    bin_end = 0.001
    
    counter = 0
    counter_i = 0

    # First, make a list of all indices in df2:
    #indices = np.linspace(0,len(df2)-1,len(df2))


    random_index_list = random.sample(range(len(df2)), len(df2))

    for p in random_index_list:#range(len(df2)):
        if counter_i > 9:
            break
        # go through all bins and find two examples of each
        
        if (df_with_p_shuffle['p_merg'][p] < bin_start) or (df_with_p_shuffle['p_merg'][p] > bin_end):
            # then you can use this galaxy as an example
            continue
        gal_id=df_with_p_shuffle[['ID']].values[p][0]
        
        
        index_match = np.where(sdss==gal_id)[0][0]
        if index_match:
            camera_data = download_galaxy(gal_id, ra[index_match], dec[index_match], prefix_frames, size_image)
            preds = get_predictors(gal_id, camera_data, prefix+'../', size_image)
            try:
                shape = np.shape(preds[0])[0]
            except TypeError:
                continue
            
        else:
            continue
        
        axis_number = counter_i

  
        # Figure out which position you need to put this in
        axs[axis_number].imshow(np.abs(camera_data),norm=colors.LogNorm(vmax=10**5, vmin=10**(0.5)), cmap='afmhot', interpolation='None')#vmin=10**(1), vmax=10**(4)norm=colors.LogNorm(vmin=10**(-1), vmax=10**(2.5))
        axs[axis_number].contour(np.fliplr(np.rot90(preds[1])), levels=[0,1], colors='yellow')
        axs[axis_number].annotate('p$_{\mathrm{merg}} = $'+str(round(df_with_p_shuffle['p_merg'][p],5))+', CDF = '+str(round(hist_dist.cdf(df_with_p_shuffle['p_merg'][p]),2)), 
                xycoords='axes fraction',xy=(0.05,0.9),xytext=(0.05, 0.9), textcoords='axes fraction',
                bbox=dict(boxstyle="round", fc="0.9", alpha=0.5), color='black')
        axs[axis_number].set_title(str(gal_id))#, xycoords='axes fraction',xy=(0.05,0.02),color='white', size=10)
        axs[axis_number].set_yticklabels([])
        axs[axis_number].set_xticklabels([])

        axs[axis_number].annotate(str(df_with_p_shuffle['most influential'][p])+'\nGini = '+str(round(preds[2],2))+' M20 = '+str(round(preds[3],2))+'\nC = '+str(round(preds[4],2))+' A = '
            +str(round(preds[5],2))+' S = '+str(round(preds[6],2))+
            '\nn = '+str(round(preds[7],2))+' A_S = '+str(round(preds[8],2)), 
                xy=(0.05, 0.07),  xycoords='axes fraction',
            xytext=(0.05, 0.07), textcoords='axes fraction',
            bbox=dict(boxstyle="round", fc="0.9", alpha=0.5), color='black')
        
        counter_i+=1
            

    plt.savefig('../Figures/probability_panel_nonmergers_low_prob_SDSS_predictors_'+str(run)+'.pdf')


    plt.clf()
    fig, axs = plt.subplots(2,5, figsize=(15, 7), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = -0.2, wspace=0.1)

    bin_start = 0.999
    bin_end = 1

    axs = axs.ravel()
    
    counter = 0
    counter_i = 0

    # First, make a list of all indices in df2:
    #indices = np.linspace(0,len(df2)-1,len(df2))


    random_index_list = random.sample(range(len(df2)), len(df2))

    for p in random_index_list:#range(len(df2)):
        
        if counter_i > 9:
            break
        # go through all bins and find two examples of each
        if (df_with_p_shuffle['p_merg'][p] < bin_start) or (df_with_p_shuffle['p_merg'][p] > bin_end):
            # then you can use this galaxy as an example
            #counter_i+=1
            continue
        gal_id=df_with_p_shuffle[['ID']].values[p][0]
        index_match = np.where(sdss==gal_id)[0][0]
        if index_match:
            camera_data = download_galaxy(gal_id, ra[index_match], dec[index_match], prefix_frames, size_image)
            preds = get_predictors(gal_id, camera_data, prefix+'../', size_image)
            try:
                shape = np.shape(preds[0])[0]
            except TypeError:
                continue
            
        else:
            continue
        
        axis_number = counter_i

  
        # Figure out which position you need to put this in
        axs[axis_number].imshow(np.abs(camera_data),norm=colors.LogNorm(vmax=10**5, vmin=10**(0.5)), cmap='afmhot', interpolation='None')#vmin=10**(1), vmax=10**(4)norm=colors.LogNorm(vmin=10**(-1), vmax=10**(2.5))
        axs[axis_number].contour(np.fliplr(np.rot90(preds[1])), levels=[0,1], colors='yellow')
        axs[axis_number].annotate('p$_{\mathrm{merg}} = $'+str(round(df_with_p_shuffle['p_merg'][p],5))+', CDF = '+str(round(hist_dist.cdf(df_with_p_shuffle['p_merg'][p]),2)), 
                xycoords='axes fraction',xy=(0.05,0.9),xytext=(0.05, 0.9), textcoords='axes fraction',
                bbox=dict(boxstyle="round", fc="0.9", alpha=0.5), color='black')
        axs[axis_number].set_title(str(gal_id))#, xycoords='axes fraction',xy=(0.05,0.02),color='white', size=10)
        axs[axis_number].set_yticklabels([])
        axs[axis_number].set_xticklabels([])

        axs[axis_number].annotate(str(df_with_p_shuffle['most influential'][p])+'\nGini = '+str(round(preds[2],2))+' M20 = '+str(round(preds[3],2))+'\nC = '+str(round(preds[4],2))+' A = '
            +str(round(preds[5],2))+' S = '+str(round(preds[6],2))+
            '\nn = '+str(round(preds[7],2))+' A_S = '+str(round(preds[8],2)), 
                xy=(0.05, 0.07),  xycoords='axes fraction',
            xytext=(0.05, 0.07), textcoords='axes fraction',
            bbox=dict(boxstyle="round", fc="0.9", alpha=0.5), color='black')
        
        counter_i+=1
            

    plt.savefig('../Figures/probability_panel_mergers_high_prob_SDSS_predictors_'+str(run)+'.pdf')
    

    return LD1_SDSS, p_merg_list, hist_dist.cdf(p_merg_list)



# This is to create a table

def classify_from_flagged_interpretive_table(prefix, run, LDA, terms_RFR, df, number_run, verbose=False, all=True, cut_flagged = True):
    #~~~~~~~
    # Now bring in the SDSS galaxies!
    #~~~~~~~

    df2 = pd.io.parsers.read_csv(prefix+'SDSS_predictors_with_flags.txt', sep='\t')
    
    if all:
        pass
    else:
        df2 = df2[0:number_run]
    
    if len(df2.columns) ==15: #then you have to delete the first column which is an empty index
        df2 = df2.iloc[: , 1:]

    

    # First, delete all rows that have weird values of n:
    #print('len before crazy values', len(df2))
    #df_filtered = df2[df2['Sersic N'] < 10]

    #df_filtered_2 = df_filtered[df_filtered['Asymmetry (A)'] > -1]

    #df2 = df_filtered_2

    # Delete duplicates:
    if verbose:
        print('len bf duplicate delete', len(df2))
    df2_nodup = df2.duplicated()
    df2 = df2[~df2_nodup]
    if verbose:
        print('len af duplicate delete', len(df2))
        print(df2)

    if cut_flagged:# Then get rid of the entries that are flagged
        df_keep = df2[(df2['low S/N'] == 0) & (df2['outlier predictor'] == 0) & (df2['segmap']==0)]
        df2 = df_keep
        


    
    input_singular = terms_RFR
    #Okay so this next part actually needs to be adaptable to reproduce all possible cross-terms
    crossterms = []
    ct_1 = []
    ct_2 = []
    for j in range(len(input_singular)):
        for i in range(len(input_singular)):
            if j == i or i < j:
                continue
            #input_singular.append(input_singular[j]+'*'+input_singular[i])
            crossterms.append(input_singular[j]+'*'+input_singular[i])
            ct_1.append(input_singular[j])
            ct_2.append(input_singular[i])

    inputs = input_singular + crossterms

    # Now you have to construct a bunch of new rows to the df that include all of these cross-terms
    for j in range(len(crossterms)):
        
        df2[crossterms[j]] = df2.apply(cross_term, axis=1, args=(ct_1[j], ct_2[j]))
        

    X_gal = df2[LDA[2]].values


 

    # Creating table
    print('making table of LDA output for all galaxies.....')
    file_out = open(prefix+'LDA_out_all_SDSS_predictors_'+str(run)+'_flags_leading_preds.txt','w')
    file_out.write('ID'+'\t'+'Classification'+'\t'+'LD1'+'\t'+'p_merg'+'\t'
        +'Leading_term_0'+'\t'+'Leading_coef_0'+'\t'+'Leading_term_1'+'\t'+'Leading_coef_1'+'\t'+'Leading_term_2'+'\t'+'Leading_coef_2'+'\t'
        +'low S/N'+'\t'+'outlier predictor'+'\t'+'segmap'+'\n')
    #+'Second_term'+'\t'+'Second_coef'+'\n')

    

    for j in range(len(X_gal)):
        print(j)
        #print(X_gal[j])

        # this is an array of everything standardized
        X_standardized = list((X_gal[j]-LDA[0])/LDA[1])

        # use the output from the simulation to assign LD1 value:
        LD1_gal = float(np.sum(X_standardized*LDA[3])+LDA[4])
        
        
        
        
        # According to my calculations, LD1 = delta_1 - delta_0 where delta is the score for each class
        # Therefore, in the probability equation p_merg = e^delta_1/(e^delta_1+e^delta_0)
        # you can sub in LD1 and instead end up with p_merg = 1/(1+e^-LD1)
        
        p_merg = 1/(1 + np.exp(-LD1_gal))
        p_nonmerg = 1/(1 + np.exp(LD1_gal))
        
        coeffs = (X_standardized*LDA[3])[0]
        terms = LDA[2]

        
        
        if LD1_gal > 0:
            merg = 1
            # select the coefficient that is the most positive
            # So the max index is what?
            # is the max of the standardized array times all the coefficients, so what has the largest positive value?
            # this just gives you the index to search, if selected from LDA[2] gives you the name of that term
            # and if selected from the lda[3]*x_standardized gives the additive value of this
            
            
            # Array to sort:
            arr1inds = coeffs.argsort()
            sorted_terms = terms[arr1inds[::-1]]
            sorted_coeff = coeffs[arr1inds[::-1]]


            most_influential_term = [sorted_terms[0],sorted_terms[1],sorted_terms[2]]
            most_influential_coeff = [sorted_coeff[0],sorted_coeff[1],sorted_coeff[2]]
        else:
            merg = 0
            

            # Array to sort:
            arr1inds = coeffs.argsort()
            sorted_terms = terms[arr1inds[::-1]]
            sorted_coeff = coeffs[arr1inds[::-1]]
            
            most_influential_term = [sorted_terms[-1],sorted_terms[-2],sorted_terms[-3]]
            most_influential_coeff = [sorted_coeff[-1],sorted_coeff[-2],sorted_coeff[-3]]

        file_out.write(str(df2[['ID']].values[j][0])+'\t'+str(merg)+'\t'+str(round(LD1_gal,3))+'\t'+str(round(p_merg,3))+'\t'
            +most_influential_term[0]+'\t'+str(round(most_influential_coeff[0],1))+'\t'
            +most_influential_term[1]+'\t'+str(round(most_influential_coeff[1],1))+'\t'
            +most_influential_term[2]+'\t'+str(round(most_influential_coeff[2],1))+'\t'
            +str(df2[['low S/N']].values[j][0])+'\t'+str(df2[['outlier predictor']].values[j][0])+'\t'+str(df2[['segmap']].values[j][0])+'\n')
        
    file_out.close()
    return

def classify_changing_priors_from_flagged(prefix, run, LDA, terms_RFR, priors, number_run, 
    verbose=False, run_all=True, cut_flagged=True):

    file_name = prefix+'change_prior/LDA_out_all_SDSS_predictors_'+str(run)+'_'+str(priors)+'_flags_cut_segmap.txt'
    if os.path.exists(file_name):
        print('already exists')
        return
    else:
        print('gonna make the file')
        #~~~~~~~
        # Now bring in the SDSS galaxies!
        #~~~~~~~
        if verbose:
            print('loading up predictor value table........')
        df2 = pd.io.parsers.read_csv(prefix+'SDSS_predictors_all_flags_plus_segmap.txt', sep='\t')
        

        #df2 = df2[0:1000]
        
        if len(df2.columns) ==15: #then you have to delete the first column which is an empty index
            df2 = df2.iloc[: , 1:]
        #print('length of df2 pre cut', len(df2))
        if cut_flagged:
            df2 = df2[(df2['low S/N']==0) & (df2['outlier predictor']==0) & (df2['segmap']==0)]
        #print('length after flagged cut', len(df2))
        #STOP
        # First, delete all rows that have weird values of n:
        #print('len before crazy values', len(df2))
        #df_filtered = df2[df2['Sersic N'] < 10]

        #df_filtered_2 = df_filtered[df_filtered['Asymmetry (A)'] > -1]

        #df2 = df_filtered_2

        # Delete duplicates:
        if verbose:
            print('len bf duplicate delete', len(df2))
        df2_nodup = df2.duplicated()
        df2 = df2[~df2_nodup]
        if verbose:

            print('len af duplicate delete', len(df2))

        if run_all:
            pass
        else:
            df2 = df2[0:number_run]


        
        input_singular = terms_RFR
        #Okay so this next part actually needs to be adaptable to reproduce all possible cross-terms
        crossterms = []
        ct_1 = []
        ct_2 = []
        for j in range(len(input_singular)):
            for i in range(len(input_singular)):
                if j == i or i < j:
                    continue
                #input_singular.append(input_singular[j]+'*'+input_singular[i])
                crossterms.append(input_singular[j]+'*'+input_singular[i])
                ct_1.append(input_singular[j])
                ct_2.append(input_singular[i])

        # Now you have to construct a bunch of new rows to the df that include all of these cross-terms
        for j in range(len(crossterms)):
            
            df2[crossterms[j]] = df2.apply(cross_term, axis=1, args=(ct_1[j], ct_2[j]))
            

        X_gal = df2[LDA[2]].values

        #start_time = time.time()
        p_merg_gal_fast = [convert_LDA_to_pmerg(float(np.sum(LDA[3]*list((x - LDA[0])/LDA[1])) + LDA[4])) for x in X_gal]
        
        df2['p_merg'] = p_merg_gal_fast
        
        df2.to_csv(file_name, sep='\t')
        
        return
        
        
        
        calc_time_fast = time.time()
        
        print('pmerg fast', pmerg_gal_fast)

        pmerg_gal_slow = []

        for j in range(len(X_gal)):
            #print(X_gal[j])
            X_standardized = list((X_gal[j]-LDA[0])/LDA[1])
            # use the output from the simulation to assign LD1 value:
            LD1_gal = float(np.sum(X_standardized*LDA[3])+LDA[4])
            
           
            
            # According to my calculations, LD1 = delta_1 - delta_0 where delta is the score for each class
            # Therefore, in the probability equation p_merg = e^delta_1/(e^delta_1+e^delta_0)
            # you can sub in LD1 and instead end up with p_merg = 1/(1+e^-LD1)
            
            p_merg = 1/(1 + np.exp(-LD1_gal))
            pmerg_gal_slow.append(p_merg)
            
                
               
            #file_out.write(str(df2[['ID']].values[j][0])+'\t'+str(round(p_merg,3))+'\n')
            
        #file_out.close()
        
        
        calc_time_slow = time.time()
        
        print('pmerg slow', pmerg_gal_slow)
        
        print('timing')
        print(" fast --- %s seconds ---" % (calc_time_fast - start_time))
        print(" slow --- %s seconds ---" % (calc_time_slow - calc_time_fast))
        
        STOP
        

    return



'''
# So the below is if you dont include the coalescence timeslot as a 'post-coal' example, 
# which I didn't end up doing in the paper
def load_LDA_from_simulation_sliding_time(post_coal,  run,  verbose=True, plot=True):


    feature_dict = {i:label for i,label in zip(
                range(39),
                  ('Counter_x',
                  'Image',
                  'class label',
                  'Myr',
                  'Viewpoint',
                '# Bulges',
                   'Sep',
                   'Flux Ratio',
                  'Gini',
                  'M20',
                  'Concentration (C)',
                  'Asymmetry (A)',
                  'Clumpiness (S)',
                  'Sersic N',
                  'Shape Asymmetry (A_S)',
                  'Counter_y',
                  'Delta PA',
                              'v_asym',
                            's_asym',
                            'resids',
                            'lambda_r',
                            'epsilon',
                            'A',
                            'A_2',
                            'deltapos',
                            'deltapos2',
                            'nspax','re',
                            'meanvel','varvel','skewvel','kurtvel',
                  'meansig','varsig','skewsig','kurtsig','abskewvel','abskewsig','random'))}

    features_list = ['Gini','M20','Concentration (C)','Asymmetry (A)','Clumpiness (S)','Sersic N','Shape Asymmetry (A\
_S)','random']


    if run[0:12]=='major_merger':
        priors=[0.9,0.1]#[0.75,0.25]]                                                                                 
    else:
        if run[0:12]=='minor_merger':
            priors=[0.7,0.3]#[0.75,0.25]]                                                                             
        else:
            STOP

    # Read in the the measured predictor values from the LDA table                                                 
    # Load in the full table and then figure out where to draw the line

    print('RUN', run)

    df = pd.io.parsers.read_csv(filepath_or_buffer='../Tables/LDA_merged_'+str(run)+'.txt',header=[0],sep='\t')

    #Rename all of the kinematic columns (is this necessary?)                                                         
    df.rename(columns={'kurtvel':'$h_{4,V}$','kurtsig':'$h_{4,\sigma}$','lambda_r':'\lambdare',
             'epsilon':'$\epsilon$','Delta PA':'$\Delta$PA','A_2':'$A_2$',
              'varsig':'$\sigma_{\sigma}$',
             'meanvel':'$\mu_V$','abskewvel':'$|h_{3,V}|$',
             'abskewsig':'$|h_{3,\sigma}|$',
             'meansig':'$\mu_{\sigma}$',
             'varvel':'$\sigma_{V}$'},
    inplace=True)

    df.columns = [l for i,l in sorted(feature_dict.items())]

    df.dropna(how="all", inplace=True) # to drop the empty line at file-end                                           
    df.dropna(inplace=True) # to drop the empty line at file-end                                                      

    # Define the coalescence point for the different 'Image' names:



    print('time since coalescence where we are drawing the line', post_coal)
   
    #a_dataframe.loc[a_dataframe["B"] > 8, "B"] = "x"
    #a_dataframe["B"] = np.where(a_dataframe["B"] > 8, "x", a_dataframe["B"])
    # Maybe just go through the whole dataframe and do this yourself:
    
    df['class label'] = np.where((df['Image']=='fg3_m12') & (df['Myr'] > (2.15 +post_coal)),0, df['class label'])
    df['class label'] = np.where((df['Image']=='fg1_m13') & (df['Myr'] > (2.74 +post_coal)),0, df['class label'])
    df['class label'] = np.where((df['Image']=='fg3_m13') & (df['Myr'] > (2.59 +post_coal)),0, df['class label'])
    df['class label'] = np.where((df['Image']=='fg3_m15') & (df['Myr'] > (3.72 +post_coal)),0, df['class label'])
    df['class label'] = np.where((df['Image']=='fg3_m10') & (df['Myr'] > (9.17 +post_coal)),0, df['class label'])

    # Also have to make sure that if the post_coal is greater than 0.5 that everything less
    # that remains but is labeled as a 1
    df['class label'] = np.where((df['Image']=='fg3_m12') & (df['Myr'] < (2.15 +post_coal)),1, df['class label'])
    df['class label'] = np.where((df['Image']=='fg1_m13') & (df['Myr'] < (2.74 +post_coal)),1, df['class label'])
    df['class label'] = np.where((df['Image']=='fg3_m13') & (df['Myr'] < (2.59 +post_coal)),1, df['class label'])
    df['class label'] = np.where((df['Image']=='fg3_m15') & (df['Myr'] < (3.72 +post_coal)),1, df['class label'])
    df['class label'] = np.where((df['Image']=='fg3_m10') & (df['Myr'] < (9.17 +post_coal)),1, df['class label'])

    # Make sure you delete everything from before coalescence:
    
    # Get names of indexes that below to a given name and were formerly labeled as mergers and are before coal:
    indexNames = df[ (df['Image']=='fg3_m12') & (df['Myr'] <2.16) & (df['class label'] ==1) ].index
    # Delete these row indexes from dataFrame
    df.drop(indexNames , inplace=True)

    indexNames = df[ (df['Image']=='fg1_m13') & (df['Myr'] <2.75) & (df['class label'] ==1) ].index
    df.drop(indexNames , inplace=True)

    indexNames = df[ (df['Image']=='fg3_m13') & (df['Myr'] <2.60) & (df['class label'] ==1) ].index
    df.drop(indexNames , inplace=True)

    indexNames = df[ (df['Image']=='fg3_m15') & (df['Myr'] <3.73) & (df['class label'] ==1) ].index
    df.drop(indexNames , inplace=True)

    indexNames = df[ (df['Image']=='fg3_m10') & (df['Myr'] <9.18) & (df['class label'] ==1) ].index
    df.drop(indexNames , inplace=True)

    
    print(df['class label'].value_counts())
    



    myr=[]
    myr_non=[]
    for j in range(len(df)):
        if df[['class label']].values[j][0]==0.0:
            myr_non.append(df[['Myr']].values[j][0])
        else:
            myr.append(df[['Myr']].values[j][0])

    if verbose:
        print('myr that are considered merging', sorted(set(myr)))
        print('myr that are nonmerging', sorted(set(myr_non)))


    
    len_nonmerg = len(myr_non)
    len_merg = len(myr)

    myr_non=sorted(list(set(myr_non)))
    myr=sorted(list(set(myr)))

    terms_RFR, reject_terms_RFR = run_RFC(df, features_list, verbose)
    output_LDA = run_LDA( df, priors,terms_RFR, myr, myr_non, 21,  verbose)



    std_mean = output_LDA[0]
    std_std = output_LDA[1]
    inputs_all = output_LDA[2]



    coeff = output_LDA[3]
    inter = output_LDA[4]
    LDA_ID = output_LDA[8]
    return output_LDA, terms_RFR, df#, len_nonmerg, len_merg
'''       