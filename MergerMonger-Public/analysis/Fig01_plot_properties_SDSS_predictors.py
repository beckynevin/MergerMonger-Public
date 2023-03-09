# Makes Figure 1 in the paper
# By plotting allowed range of predictor values
# From which I'm defining outliers

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import scipy
import scipy.stats

# Sorry, I made an annoying structure for these directories...
import sys
sys.path.insert(0,"..")
from MergerMonger import load_LDA_from_simulation
import seaborn as sns

# Step 1 is to import the dataframe you need:
print('loading up predictor value table........')

max_n = 2e4

contours = True
fontsize = 20 # I'm making the labels pretty big for the paper
                                                              
prefix = '/Users/rnevin/Documents/MergerMonger-Public/tables/simulation_classifications/'
df2_OG = pd.io.parsers.read_csv(prefix+'../sdss_classifications/SDSS_predictors_all_flags_plus_segmap.txt', sep='\t')
df2 = df2_OG

#df2 = df2[0:10000]                                                                                                    
if len(df2.columns) ==15: #then you have to delete the first column which is an empty index                      
    df2 = df2.iloc[: , 1:]

# Now, figure out how to eliminate things that are flagged:
print('length of df before flag cuts', len(df2))
df2_filtered = df2[(df2['low S/N']==0) & (df2['outlier predictor']==0) & (df2['segmap']==0)]
print('length of df after flag cuts', len(df2_filtered))


# First, delete all rows that have weird values of n:   
df_filtered = df2[df2['Sersic N'] < 10]

df_filtered_2 = df_filtered[df_filtered['Asymmetry (A)'] > -1]

df_filtered_3 = df_filtered_2[df_filtered_2['M20'] > -10]

df2 = df_filtered_3

print('length after crazy values filtered', len(df2))


# Delete duplicates:                                                                                                 
df2_nodup = df2.duplicated()
df2 = df2[~df2_nodup]


if contours:
    # Next, load in the simulated data
    run = 'major_merger'
    LDA, RFR, df_major = load_LDA_from_simulation(prefix, run, verbose=False)
    

    run = 'minor_merger'
    LDA, RFR, df_minor = load_LDA_from_simulation(prefix, run, verbose=False)     

    df_sims = pd.concat([df_major, df_minor]).reset_index()

    df2_nodup = df2_filtered.duplicated()
    df2_f = df2_filtered[~df2_nodup]
    
    sns.set_style('white')

    gal_predictors = [[0.54,-2.15,3.62,-0.04,1.49,0.13],
                      [0.69,-0.96,3.59,0.23,1.35,0.78],
                      [0.56,-1.0,3.66,0.43,0.58,0.89],
                      [0.56,-2.16,3.59,0.14,1.38,0.57],
                      [0.56,-2.07,3.53,0.02,1.47,0.40],
                      [0.58,-0.81,1.61,0.54,0.97,0.12]]
#                        [0.71,-0.65,1.29,0.54,2.36,0.37],
#                        [0.47,-0.96,1.84,0.49,0.42,0.16]]
    letters = ['A','B','C','D','E','F']
    gal_colors = ['white','white','white','white','white','white']
    #gal_colors = ['#CE7DA5','#6DD3CE','#E59500','#2176AE','#FCFC62','#B5E898']


    plt.clf()
    fig = plt.figure()                                                                                                 
    ax2=fig.add_subplot(111)                                                                                                 
    dashed_line_x=np.linspace(-0.5,-3,100)
    dashed_line_y=[-0.14*x + 0.33 for x in dashed_line_x]
    heatmap_all, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(df2['Gini'].values, df2['M20'].values, df2['Gini'].values,statistic='count', bins=100)
    ax2.plot(dashed_line_x, dashed_line_y, ls='--', color='black')

    #ax2.contour(df_major['M20'].values, df_major['Gini'].values)                                                               
    xmin = xedges[0]
    xmax = xedges[-1]
    ymin = yedges[0]
    ymax = yedges[-1]
    im2 = ax2.imshow(np.fliplr(np.flipud(heatmap_all)), cmap='Greys',extent=[ymax, ymin, xmin, xmax], norm=matplotlib.colors.LogNorm(vmax=max_n))
    
    heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(df2_f['Gini'].values, df2_f['M20'].values, df2_f['Gini'].values,statistic='count', bins=100, 
        range = [[xmin, xmax], [-3, ymax]])
    ax2.plot(dashed_line_x, dashed_line_y, ls='--', color='black')                                                                

    im2 = ax2.imshow(np.fliplr(np.flipud(heatmap)), cmap='magma_r', extent=[ymax, -3, xmin, xmax], norm=matplotlib.colors.LogNorm(vmax=max_n))#
    #extent=[ymax_c, ymin_c, xmax_c, xmin_c]
    sns.kdeplot(data=df_sims, x="M20", y = "Gini", ax = ax2, color='black') 

    for g in range(len(gal_predictors)):
        ax2.scatter(gal_predictors[g][1],gal_predictors[g][0], color = gal_colors[g], zorder = 100)
        ax2.annotate(letters[g],xycoords='data',xy=(gal_predictors[g][1],gal_predictors[g][0]), size = fontsize, color = gal_colors[g], zorder = 101)
    


    ax2.set_ylim(xmin, xmax)
    ax2.set_xlim(ymax, -3)
    ax2.set_xlabel(r'$M_{20}$', size = fontsize, labelpad = -5)
    ax2.set_ylabel(r'$Gini$', size = fontsize)
    ax2.tick_params(axis='both', which='major', labelsize = fontsize - 5, bottom = True, left = True)
    ax2.set_aspect((ymax+3)/(xmax-xmin))
    #ax2.set_aspect((ymax-ymin)/(xmax-xmin))
    ax2.annotate('Total # of galaxies = '+str(len(df2_OG))+'\n             # selected = '+str(len(df2_filtered)), 
        xy=(0.05, 1.02), size = fontsize, xycoords='axes fraction')
    plt.savefig('figures/gini_m20_contours_overplot.png', dpi=1000)

    


    fig = plt.figure()                                                                                                 
    ax2=fig.add_subplot(111)                                                                                                 
    heatmap_all, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(df2['Concentration (C)'].values, df2['Asymmetry (A)'].values, df2['Concentration (C)'].values,statistic='count', bins=100)
    xmin_c = xedges[0]
    xmax_c = xedges[-1]
    ymin_c = yedges[0]
    ymax_c = yedges[-1]
    im2 = ax2.imshow(np.fliplr((heatmap_all)), cmap='Greys',extent=[ymax_c, ymin_c, xmax_c, xmin_c], norm=matplotlib.colors.LogNorm(vmax=max_n))
    heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(df2_f['Concentration (C)'].values, df2_f['Asymmetry (A)'].values, df2_f['Concentration (C)'].values,statistic='count', bins=100, 
        range = [[xmin_c, xmax_c], [ymin_c, ymax_c]])

    im2 = ax2.imshow(np.fliplr((heatmap)), cmap='magma_r',extent=[ymax_c, ymin_c, xmax_c, xmin_c], norm=matplotlib.colors.LogNorm(vmax=max_n))
    sns.kdeplot(data=df_sims, x="Asymmetry (A)", y = "Concentration (C)", ax = ax2, color='black') 

    for g in range(len(gal_predictors)):
        ax2.scatter(gal_predictors[g][3],gal_predictors[g][2], color=gal_colors[g], zorder=100)
        ax2.annotate(letters[g],xycoords='data',xy=(gal_predictors[g][3],gal_predictors[g][2]), size = fontsize, color=gal_colors[g], zorder=101)
    

    ax2.set_ylim(xmin_c, xmax_c)
    ax2.set_xlim( ymin_c, ymax_c)
    #plt.colorbar(im2, fraction=0.046)
    ax2.set_xlabel(r'Asymmetry (A)', size = fontsize, labelpad = 0)
    ax2.set_ylabel(r'Concentration (C)', size = fontsize)
    ax2.tick_params(axis='both', which='major', labelsize = fontsize - 5, bottom = True, left = True)
    ax2.set_aspect((ymax_c-ymin_c)/(xmax_c-xmin_c))
    ax2.axvline(x=0.3, ls='--', color='black')
    plt.savefig('figures/C_A_contours_overplot.png', dpi=1000)

    fig = plt.figure()                                                                                                 
    ax2=fig.add_subplot(111)                                                                                                 
    heatmap_all, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(df2['Sersic N'].values, df2['Shape Asymmetry (A_S)'].values, df2['Sersic N'].values,statistic='count', bins=100)
    xmin_n = xedges[0]
    xmax_n = xedges[-1]
    ymin_n = yedges[0]
    ymax_n = yedges[-1]
    im1 = ax2.imshow(np.fliplr((heatmap_all)), cmap='Greys',extent=[ymax_n, ymin_n, xmax_n, xmin_n], norm=matplotlib.colors.LogNorm(vmax=max_n))
    


    heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(df2_f['Sersic N'].values, df2_f['Shape Asymmetry (A_S)'].values, df2_f['Sersic N'].values,statistic='count', bins=100,
        range = [[xmin_n, xmax_n], [ymin_n, ymax_n]])
    im2 = ax2.imshow(np.fliplr((heatmap)), cmap='magma_r',extent=[ymax_n, ymin_n, xmax_n, xmin_n], norm=matplotlib.colors.LogNorm(vmax=max_n))
    sns.kdeplot(data=df_sims, x="Shape Asymmetry (A_S)", y = "Sersic N", ax = ax2, color='black') 

    for g in range(len(gal_predictors)):
        ax2.scatter(gal_predictors[g][5],gal_predictors[g][4], color=gal_colors[g], zorder=100)
        ax2.annotate(letters[g],xycoords='data',xy=(gal_predictors[g][5],gal_predictors[g][4]), size = fontsize, color=gal_colors[g], zorder=101)
    

    ax2.set_ylim(xmin_n, xmax_n)
    ax2.set_xlim( ymin_n, ymax_n)
    #plt.colorbar(im1, fraction=0.046, pad=-0.13)#, label=r'# of galaxies in bin')
    cb = plt.colorbar(im2, fraction=0.046, pad=0.1)
    cb.set_label(label=r'# of galaxies in bin', size = fontsize)
    cb.ax.tick_params(labelsize = fontsize)
    ax2.set_xlabel(r'Shape Asymmetry ($A_S$)', size = fontsize,  labelpad = 0)
    ax2.set_ylabel(r'Sersic n', size = fontsize)
    ax2.tick_params(axis='both', which='major', labelsize = fontsize - 5, bottom = True, left = True)
    ax2.set_aspect((ymax_n-ymin_n)/(xmax_n-xmin_n))
    ax2.axvline(x=0.3, ls='--', color='black')
    plt.tight_layout()
    plt.savefig('figures/A_S_n_contours_overplot.png', dpi=1000, bbox_inches=0) 

print('all files saved to "figures/"')