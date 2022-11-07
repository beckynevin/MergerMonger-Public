#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# trying to run the f_merg_to_rule_them_all.py
# with S/N and mass binning but also z binning
# and then you're actually just investigating that the
# negative slope with z persists at all bins of S/N 
# and mass
# Using the same strat for making a complete sample as before
# (which involves binning in z and mass)
# but then requiring that each 3D bin have at least 100 galaxies?
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from matplotlib.patches import Rectangle



# path
dir = '/Users/rebeccanevin/Documents/CfA_Code/MergerMonger-dev/Tables/'


# This is to load up the mass complete table:
ack = False # option whether to use ackermann cross-matches
mass = 'log_stellar_mass_from_color'
red = 'z'
spacing_z = 0.02
complete = True
completeness = 95
nbins_mass = 7#15
nbins_S_N = 7
suffix = str(spacing_z)+'_'+str(red)+'_'+str(mass)+'_completeness_'+str(completeness)


# Check if this table ^ even exists:
if os.path.exists(dir+'all_mass_color_complete_'+str(suffix)+'.txt'):
    print('it exists! you can run the f_merg analysis')
else:
    print('missing mass table to run this analysis')


# Now import the stuff to get your LDA and predictor table
# so you can have various properties of each galaxy
type_marginalized = '_flags_cut_segmap'
type_gal = 'predictors'
run = 'major_merger'
# set this if you want to do a limited subset
num = None
savefigs = True
# set this if you want to save the result
save_df = True
JK_anyway = False
S_N_cut = True
S_N_cut_val = 50


if complete:
    table_name = dir + 'f_merg_'+str(run)+'_'+str(suffix)+'_mass_bins_'+str(nbins_mass)+'_S_N_bins_'+str(nbins_S_N)+'_flags.csv'
else:
    table_name = dir + 'f_merg_'+str(run)+'_'+str(suffix)+'_incomplete_mass_bins_'+str(nbins_mass)+'_S_N_bins_'+str(nbins_S_N)+'_flags.csv'

if os.path.exists(table_name) and save_df:
    print('table already exists do you want to oversave?')

    




df_LDA = pd.io.parsers.read_csv(filepath_or_buffer=dir+'LDA_out_all_SDSS_'+type_gal+'_'+run+'_flags.txt',header=[0],sep='\t')

# Because the df_LDA doesn't have the final flag, use the predictor table to instead clean via merging

# Run OLS with predictor values and z and stellar mass and p_merg:
df_predictors = pd.io.parsers.read_csv(filepath_or_buffer=dir+'SDSS_predictors_all_flags_plus_segmap.txt',header=[0],sep='\t')

if len(df_LDA) != len(df_predictors):
    print('these have different lengths cannot use one to flag')
    STOP

# First clean this so that there's no segmap flags
df_predictors_clean = df_predictors[(df_predictors['low S/N'] ==0) & (df_predictors['outlier predictor']==0) & (df_predictors['segmap']==0)]


clean_LDA = df_LDA[df_LDA['ID'].isin(df_predictors_clean['ID'].values)]

if complete:
    masstable = pd.io.parsers.read_csv(filepath_or_buffer=dir+'all_mass_color_complete_'+str(suffix)+'.txt',header=[0],sep='\t')
else:
    masstable = pd.io.parsers.read_csv(filepath_or_buffer=dir+'all_mass_'+str(suffix)+'.txt',header=[0],sep='\t')



if red == 'z_spec':
    masstable = masstable[masstable['logBD'] < 13]
    masstable = masstable[masstable['dBD'] < 1]
    masstable['B/T'] = 10**masstable['logMb']/(10**masstable['logMb']+10**masstable['logMd'])
    masstable = masstable[['objID',
    'z_x','z_spec',
    'logBD','log_stellar_mass_from_color','B/T']]
    
else:
    masstable = masstable[masstable['log_stellar_mass_from_color'] < 13]
    masstable = masstable[['objID',
    'z','log_stellar_mass_from_color']]




# Now merge this with LDA
#merged_1 = masstable_m.merge(clean_LDA, left_on='objID', right_on='ID')#[0:1000]# Now merging on dr8

merged_1 = masstable.merge(clean_LDA,left_on='objID', right_on='ID',# left_index=True, right_index=True,
                  suffixes=('', '_y'))
if red == 'z_spec':
    merged_1 = merged_1[['ID','z_x','z_spec','logBD','log_stellar_mass_from_color',
        'p_merg','B/T']]
else:
    merged_1 = merged_1[['ID','z','log_stellar_mass_from_color',
        'p_merg']]


#merged_1.drop(merged_1.filter(regex='_y$').columns, axis=1, inplace=True)

final_merged = merged_1.merge(df_predictors_clean, on='ID')
if red == 'z_spec':
    final_merged = final_merged[['ID','z_x','z_spec','logBD','log_stellar_mass_from_color',
        'p_merg','S/N','B/T']]
else:
    final_merged = final_merged[['ID','z','log_stellar_mass_from_color',
        'p_merg','S/N']]
print('len merged with mendel', len(final_merged))


if S_N_cut:
    final_merged = final_merged[final_merged['S/N'] < S_N_cut_val]

print('len after S/N cut')
print(len(final_merged))
STOP
# Load in the bins in z unless its the incomplete version (suffix == 'color'), 
# in that case define your own spacing
bins_z = np.load(dir+'all_mass_color_complete_'+str(suffix)+'_bins.npy')




# This is for dropping some stuff if you need it to run fast
if num:
    final_merged = final_merged.dropna()[0:num]
else:
    final_merged = final_merged.dropna()
print('length after dropping Nans', len(final_merged))





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run OLS with predictor values and z and stellar mass and p_merg:
# Now merge these two and run an OLS:



#cats, bins

cats_S_N, bins_S_N = pd.qcut(final_merged['S/N'], q=nbins_S_N, retbins=True, precision = 1)
print('cats S/N', cats_S_N)
print('bins S/N', bins_S_N)


cats_mass, bins_mass = pd.qcut(final_merged[mass], q=nbins_mass, retbins=True, precision = 1)
#df['mass_bin'] = cats_mass

centers_z = [(bins_z[x+1] - bins_z[x])/2 + bins_z[x] for x in range(len(bins_z)-1)]
centers_mass = [(bins_mass[x+1] - bins_mass[x])/2 + bins_mass[x] for x in range(len(bins_mass)-1)]
centers_S_N = [(bins_S_N[x+1] - bins_S_N[x])/2 + bins_S_N[x] for x in range(len(bins_S_N)-1)]
 


# Before you do any analysis, it's necessary to drop certain parts of the table that are not
# complete
# Is this possible to do after you save?
# What kind of criteria do I want to use to do this?
# One way could be to drop bins that have a weird distribution of masses?
# Likewise, a weird distribution of redshifts
# Maybe the mean in one of these is significantly off





if save_df:
    # make a massive plot
    plt.clf()
    fig = plt.figure(figsize = (10,5))
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    #currentAxis = plt.gca()
    ax0.scatter(final_merged[red], final_merged[mass], color='grey', s=0.1)
    ax0.set_ylabel(r'log stellar mass ($M_*$)')
    ax0.set_xlabel(r'$z$')
    
    ax1.scatter(final_merged[red], final_merged['S/N'], color='grey', s=0.1)
    ax1.set_ylabel('S/N')
    ax1.set_xlabel(r'$z$')
    
    
    
    
    # first go through and load up all of prior files
    list_of_prior_files = glob.glob(dir + 'change_prior/LDA_out_all_SDSS_predictors_'+str(run)+'_0.*'+str(type_marginalized)+'.txt')
    print('length of prior files', len(list_of_prior_files))
    if len(list_of_prior_files) ==0:
        print('there are no priors prepared')
        name_single = dir + 'LDA_out_all_SDSS_predictors_'+str(run)+'_flags.txt'
        table = pd.io.parsers.read_csv(filepath_or_buffer=name_single,header=[0],sep='\t')[['ID','p_merg']]
    else:
        table_list = []
        for p in range(len(list_of_prior_files)):
            print('p prior', p)
            prior_file = pd.io.parsers.read_csv(filepath_or_buffer=list_of_prior_files[p],header=[0],sep='\t')
            # cut it way down
            if p == 0:
                table = prior_file[['ID','p_merg']]

            else:
                table_p = prior_file[['p_merg']] # just take p_merg if not the last one
                table_p.columns = ['p_merg_'+str(p)]
                # Now stack all of these tables
                table = table.join(table_p)
        
        

    # Now that these are all joined together, identify which ones have IDs that match mergers
    
    count = {}
    flag = {}
    S_N = {}
    if red == 'z_spec':
        B_T = {}
    f_merg = {}
    f_merg_avg = {}
    f_merg_std = {}
    for i in range(len(bins_z)-1):
        bin_start_z = bins_z[i]
        bin_end_z = bins_z[i+1]
        bin_center_z = (bin_end_z - bin_start_z)/2 + bin_start_z
        print('start z ', round(bin_start_z,2), 'stop z ', round(bin_end_z,2))
        for j in range(len(bins_mass)-1):
            bin_start_mass = bins_mass[j]
            bin_end_mass = bins_mass[j+1]
            bin_center_mass = (bin_end_mass - bin_start_mass)/2 + bin_start_mass
            print('start mass ', round(bin_start_mass,2), 'stop mass ', round(bin_end_mass,2))
            
            # Okay do the second completeness before you do S/N stuff:
            df_select = final_merged[(final_merged[red] > bin_start_z) 
                        & (final_merged[red] < bin_end_z) 
                        & (final_merged[mass] > bin_start_mass) 
                        & (final_merged[mass] < bin_end_mass)]
                
            # Now figure out where the means are 
            
            med_x = np.median(df_select[red].values)
            med_y = np.median(df_select[mass].values)
            std_x = np.std(df_select[red].values)
            std_y = np.std(df_select[mass].values)
            
            
            if (((med_x - std_x) > bin_center_z) & ((med_x + std_x) > bin_center_z)) | (((med_x - std_x) < bin_center_z) & ((med_x + std_x) < bin_center_z)) | (std_x > (bin_end_z - bin_start_z)/2):
                off_in_z = True
            else:
                off_in_z = False
                
            if (((med_y - std_y) > bin_center_mass) & ((med_y + std_y) > bin_center_mass)) | (((med_y - std_y) < bin_center_mass) & ((med_y + std_y) < bin_center_mass)) | (std_y > (bin_end_mass - bin_start_mass)/2):
                off_in_mass = True
            else:
                off_in_mass = False
                
            
            
            
            
            
            # so basically, if the medians of the distribution
            # are significantly different than the zcen and masscen,
            # then throw a flag because probably incomplete
            if off_in_mass or off_in_z:
                off = 1
                #flag[centers_z[i],centers_mass[j],:] = 1
            else:
                #flag[centers_z[i],centers_mass[j],:] = 0
                
                if len(df_select) > 1000:
                    # Rectangle is expanded from lower left corner
                    #
                   ax0.scatter(df_select[red], df_select[mass], color='#52DEE5', s=0.1)
                   ax0.add_patch(
                        Rectangle((bin_start_z, bin_start_mass), 
                                bin_end_z - bin_start_z, bin_end_mass - bin_start_mass, facecolor='None', 
                                edgecolor='black')
                        )
                   ax0.annotate(f'{round(len(df_select)/1000,1)}', 
                                xy = (bin_start_z+0.005, bin_start_mass+0.005), 
                                xycoords='data', size=5)
            
            for k in range(len(bins_S_N)-1):
                bin_start_S_N = bins_S_N[k]
                bin_end_S_N = bins_S_N[k+1]
                print('start S/N', round(bin_start_S_N,1), 'stop S/N', round(bin_end_S_N,1))
                # build dataset
                
                df_select = final_merged[(final_merged[red] > bin_start_z) 
                        & (final_merged[red] < bin_end_z) 
                        & (final_merged[mass] > bin_start_mass) 
                        & (final_merged[mass] < bin_end_mass)
                        & (final_merged['S/N'] > bin_start_S_N) 
                        & (final_merged['S/N'] < bin_end_S_N)]
                if off_in_mass or off_in_z:
                    flag[centers_z[i],centers_mass[j],centers_S_N[k]] = 1
                else:
                    flag[centers_z[i],centers_mass[j],centers_S_N[k]] = 0
                    if len(df_select) > 1000:
                        # Rectangle is expanded from lower left corner
                        #
                        ax1.scatter(df_select[red], df_select['S/N'], color='#52DEE5', s=0.1)
                        ax1.add_patch(
                                Rectangle((bin_start_z, bin_start_S_N), 
                                        bin_end_z - bin_start_z, bin_end_S_N - bin_start_S_N, facecolor='None', 
                                        edgecolor='black')
                                )
                        ax1.annotate(f'{round(len(df_select)/1000,1)}', 
                                        xy = (bin_start_z+0.005, bin_start_S_N+0.005), 
                                        xycoords='data', size=5)
                   
            
                
                S_N[centers_z[i],centers_mass[j],centers_S_N[k]] = np.mean(df_select['S/N'].values)
                if red == 'z_spec':
                    B_T[centers_z[i],centers_mass[j],centers_S_N[k]] = np.mean(df_select['B/T'].values)
                
                
                count[centers_z[i],centers_mass[j],centers_S_N[k]] = len(df_select)
                
                '''
                # if the selection is > 0 then plot:
                plot_red = False
                if len(df_select) > 0 and plot_red:
                    plt.clf()
                    plt.scatter(final_merged[red].values, final_merged[mass].values, color='grey', s=0.1)
                    plt.scatter(df_select[red].values, df_select[mass].values, color='red', s=0.1)
                    plt.scatter(bin_center_z, bin_center_mass, color='black', s=1)
                    plt.scatter(np.median(df_select[red].values), np.median(df_select[mass].values), color='orange', s=1)
                
                    
                    plt.errorbar(med_x, med_y, 
                            xerr = std_x,
                            yerr = std_y,
                            color='orange')
                    
                    if off_in_mass:
                        plt.annotate('off in mass', xy=(0.03,0.93), xycoords='axes fraction')
                    if off_in_z:
                        plt.annotate('off in z', xy=(0.03,0.97), xycoords='axes fraction')
                    
                    
                    if off_in_mass or off_in_z:
                        plt.title('flagging')
                        
                    plt.show()
                '''
                df_select = df_select[['ID']] # just take this because you don't need the other deets
            

                merged = table.merge(df_select, on = 'ID')#left_on='ID', right_on='objID')
                # for each column of p_merg, calculate the the f_merg and then find the median
            

                gt = (merged > 0.5).apply(np.count_nonzero)
                
            

                #fmerg_here = len(np.where(merged['p_merg_x'] > 0.5)[0])/len(merged)
                
                #f_merg[centers_z[i]].append(fmerg_here)
                f_merg_avg[centers_z[i],centers_mass[j],centers_S_N[k]] = np.median(gt.values[1:]/len(merged))
                f_merg_std[centers_z[i],centers_mass[j],centers_S_N[k]] = np.std(gt.values[1:]/len(merged))
    plt.show()
    
   

    # find a way to put this into df format
    mass_val = []
    z_val = []
    f_merg_val = []
    f_merg_e_val = []
    count_val = []
    s_n_val = []
    flag_val = []
    if red == 'z_spec':
        b_t_val = []
    for i in range(len(bins_z)-1):
        for j in range(len(bins_mass)-1):
            for k in range(len(bins_S_N)-1):
                f_merg_val.append(f_merg_avg[centers_z[i],centers_mass[j],centers_S_N[k]])
                f_merg_e_val.append(f_merg_std[centers_z[i],centers_mass[j],centers_S_N[k]])
                flag_val.append(flag[centers_z[i],centers_mass[j],centers_S_N[k]])
                z_val.append(centers_z[i])
                mass_val.append(centers_mass[j])
                count_val.append(count[centers_z[i],centers_mass[j],centers_S_N[k]])
                s_n_val.append(centers_S_N[k])
                
    # Now make a df out of these lists
    df_fmerg = pd.DataFrame(list(zip(flag_val, mass_val, z_val, f_merg_val, f_merg_e_val, count_val, s_n_val)),columns =['flag', 'mass', 'z', 'fmerg', 'fmerg_std', 'count', 'S/N'])
    #except:
    #		df_fmerg = pd.DataFrame(list(zip(mass_val, z_val, f_merg_val, f_merg_e_val, count_val, s_n_val, b_t_val)),
    #           columns =['mass', 'z_x', 'fmerg', 'fmerg_std', 'count', 'S/N', 'B/T'])
    df_fmerg.to_csv(table_name, sep='\t')
else:
    df_fmerg = pd.read_csv(table_name, sep = '\t')

print(df_fmerg)
print('sum of counts', np.sum(df_fmerg['count'].values))
print('table name', table_name)

# Shorten everything to only include bins that have a certain number of counts
#thresh = 100
#df_fmerg = df_fmerg[df_fmerg['count'] > thresh]

# total up the count in all of these columns

centers_mass = df_fmerg.mass.unique()
centers_z = df_fmerg.z.unique()
centers_S_N = df_fmerg['S/N'].unique()

print('centers of mass', centers_mass)
print('centers z', centers_z)
print('centers S/N', centers_S_N)



print('length of z', len(centers_z))

# shorten it to only include the threshold counts
df_fmerg_cut = df_fmerg[df_fmerg['count'] > 100]
centers_mass_cut = df_fmerg_cut.mass.unique()
centers_z_cut = df_fmerg_cut.z.unique()
centers_S_N_cut = df_fmerg_cut['S/N'].unique()
print(len(centers_z_cut))

df_fmerg = df_fmerg_cut


plt.clf()
fig, axs = plt.subplots(nrows = len(centers_mass_cut), ncols=len(centers_z_cut), figsize=(15,8))




#ax = axs.ravel()

plt.subplots_adjust(hspace=0.0, wspace=0.0)
#fig.xlabel('Mass')
#fig.ylabel('S/N')


count_y = 0

for mass in centers_mass_cut:
    count_x = 0
    
    
    
    count_threshhold = 100
    
    for z in centers_z_cut:#, axs.ravel()):
        
        
        
        count = df_fmerg[(df_fmerg['z'] == z) & (df_fmerg['mass'] == mass) & (df_fmerg['count'] > count_threshhold)]['count'].values
        y = df_fmerg[(df_fmerg['z'] == z) & (df_fmerg['mass'] == mass) & (df_fmerg['count'] > count_threshhold)]['fmerg'].values
        error = df_fmerg[(df_fmerg['z'] == z) & (df_fmerg['mass'] == mass) & (df_fmerg['count'] > count_threshhold)]['fmerg_std'].values
        x = df_fmerg[(df_fmerg['z'] == z) & (df_fmerg['mass'] == mass) & (df_fmerg['count'] > count_threshhold)]['S/N'].values
        
        
            
        
        if count_x == 0:
            # then label the y axis with the S/N bin
            axs[count_y,count_x].set_ylabel(str(round(mass,2)))
        else:
            axs[count_y,count_x].set_yticklabels([])
            axs[count_y,count_x].set_yticks([])
            
        #else:
        #    axs[count_y,count_x].set_yticklabels([])
        #    axs[count_y,count_x].set_yticks([])
        
        if len(x) < 2:
            
            axs[count_y,count_x].set_xticklabels([])
            axs[count_y,count_x].set_xticks([])
            axs[count_y,count_x].set_yticklabels([])
            axs[count_y,count_x].set_yticks([])
            #axs[count_y,count_x].set_yticklabels([])
            #axs[count_y,count_x].set_yticks([])
            count_x += 1
            continue
    
        print('x',x)
        
        X = df_fmerg[(df_fmerg['z'] == z) & (df_fmerg['mass'] == mass) & (df_fmerg['count'] > count_threshhold)]['S/N']
        #Y = df_fmerg[(df_fmerg['mass'] == mass) & (df_fmerg['S/N'] == S_N)]['fmerg']
        X = sm.add_constant(X)
        axs[count_y,count_x].scatter(x, y, color='black', zorder=100, s=2)
       
        axs[count_y,count_x].errorbar(x, y, yerr = error, linestyle='None', color='black', zorder=100)
        
        
        if count_y == 6:
            # then label the y axis with the z bin
            axs[count_y,count_x].set_xlabel(str(round(z,2)))
        else:
            axs[count_y,count_x].set_xticklabels([])
            axs[count_y,count_x].set_xticks([])
        
        
        
        
        
        # Let's MCMC on this and get the slope
        mu, sigma = 0, 1 # mean and standard deviation
        
        
        # iterate
        # save slope values
        slope_list = []
        for num in range(100):

            
            Y = [y[i]+ error[i] * np.random.normal(mu, sigma, 1)  for i in range(len(y))]
            
            # fitting a line :)
            try:
                res = sm.OLS(Y, X).fit()#, missing = 'drop'
            except ValueError:
                print('trying to fit this', Y, X) 
                print('x', x)       
            #plt.scatter(x, Y, s=0.1, color='grey')
            try:
                slope_list.append(res.params[1])
            except:
                continue
            
            st, data, ss2 = summary_table(res, alpha=0.05)
            fittedvalues = data[:,2]
            axs[count_y,count_x].plot(x, fittedvalues, color='grey', alpha=0.5)#, label='OLS')
            
        #for (count, x, y) in zip(count, x, y):
        #    ax.annotate(str(count), xy = (x, y+0.07), xycoords='data', color = 'black')

       
        axs[count_y,count_x].annotate(str(round(np.mean(slope_list),3))+'+/-'+str(round(np.std(slope_list),3)), 
            xy=(0.01,0.8), xycoords='axes fraction', color='black', size=7)
        if np.mean(slope_list) + np.std(slope_list) < 0:
            conc = 'neg'
            col = 'blue'
        else:
            if np.mean(slope_list) - np.std(slope_list) > 0:
                conc = 'pos'
                col = 'red'
            else:
                conc = 'flat'
                col = 'grey'
        axs[count_y,count_x].annotate(conc, 
            xy=(0.65,0.6), xycoords='axes fraction', size = 7, color=col)      
            
        
        #ax.annotate(r'$M_{*} = $'+str(round(mass,2))+r' $M_{\odot}$', xy=(0.01,0.8), xycoords='axes fraction')
        axs[count_y,count_x].set_ylim([0,1])
        axs[count_y,count_x].set_xlim([3,35])
        #if mass == max(centers_mass):
        #    ax.set_xlabel('z')
            
        
        count_x += 1
    count_y += 1
        
    #plt.legend()
    
plt.show()



plt.clf()

# there should be 8 different redshifts
colors = ['#493843','#61988E','#A0B2A6','#CBBFBB','#EABDA8','#FF9000','#DE639A','#D33E43']
colors = ['#C5979D','#E7BBE3','#78C0E0','#449DD1','#3943B7','#150578','#0E0E52','black']
colors = ['#7D7C7A','#DEA47E','#AD2831','#800E13','#640D14','#38040E','#250902','black',
    '#7D7C7A','#DEA47E','#AD2831','#800E13','#640D14','#38040E','#250902','black',
    '#7D7C7A','#DEA47E','#AD2831','#800E13','#640D14','#38040E','#250902','black']

print('centers of mass', centers_mass)
print('centers of zs', centers_z)

fit = []
xs = []
ys = []
errors = []

plt.clf()
#fig, axs = plt.subplots(nrows = len(centers_z), figsize=(8,8))
fig = plt.figure(figsize=(10,7))
#ax = axs.ravel()

#plt.subplots_adjust(hspace=0.5)
count_x = 0

count_thresh = 100
num_subplots = 14
num_subplots_rows = 7

print('this many zs', centers_z)

color_codes = ['#46B1C9','#84C0C6','#BDCFAB',# lots of blues
               '#F5DD90','#F68E5F','#F76C5E']
color_codes = ['#004E89','#508EB1','#A0CED9',
               '#FFC482','#D1495B','#9C0D38']


for zcen in centers_z:
    print('this is the zcen', zcen)
    # Grab everything with a z value from the df
    df_select = df_fmerg[df_fmerg['z'] == zcen].dropna().reset_index()
    if len(df_select)==0:
        #count_x += 1
        continue
    # should already be sorted by mass
    #massall = df_select['mass'].values
    count = df_select['count'].values#[(massall > centers_mass[0]) & (massall < centers_mass[-1])]
    masscen = df_select['mass'].values#[(massall > centers_mass[0]) & (massall < centers_mass[-1])]
    
    
    
        
    
    
    # to do a linear regression, I'm going to need to define X and y
    #masscen_masked = np.ma.masked_where(count < 1000, masscen)
    x = masscen.reshape(-1,1)
    x = x[count > count_thresh]
    #x = np.ma.masked_where(count < 1000, x)
    
  
    x = np.unique(x)
    
    #for m in x:
        
    if len(x) < 2:
        # because don't fit a line to two data pts
        count_x += 1
        continue
    
   
    # Now go through and average f_merg values
    # add errors in quad?
    y = []
    error = []
    for m in x:
        # so for each unique mass:
        fmerg = df_select[(df_select['mass'] == m) & (df_select['count'] > count_thresh)]['fmerg']
        fmerg_std = df_select[(df_select['mass'] == m) & (df_select['count'] > count_thresh)]['fmerg_std']
        y.append(np.mean(fmerg))
        quad = []
        for f,ferr in zip(fmerg,fmerg_std):
            quad.append(ferr/f)#(ferr/f)**2)
        error.append(np.mean(fmerg)*np.max(ferr/f))#np.sqrt(np.sum(quad))*np.mean(fmerg))
    
    
    
    
    try:
        X = sm.add_constant(x)
    except ValueError:
        #count_x += 1
        continue
    
    xs.append(x)
    ys.append(y)
    #errors.append(error[count > count_thresh])
    errors.append(error)
    
    
    if count_x > 6:
        # then odd
        #lookup = {4:2,5:4,6:6,7:8} # this is for 8 total, 4 per col
        lookup = {7:2,8:4,9:6,10:8,11:10,12:12,13:14} # this is for 14 total
        number = lookup[count_x]
    else:
        number = 2 * count_x  + 1
    if count_x == 0 or count_x == num_subplots_rows:
        ax = plt.subplot(num_subplots_rows, 2, number)
    else:
        try:
            ax = plt.subplot(num_subplots_rows, 2, number, sharex=ax)
        except NameError: # this is if ax is not yet defined:
            ax = plt.subplot(num_subplots_rows, 2, number)
    ax.scatter(x, y, color='black', zorder=100, s = 3)
    ax.errorbar(x, y, yerr=error, ls='None',color='black', zorder=100, capsize = 2)
    ax.set_ylim([0,1])
    ax.set_xlim([10,12.5])
    
    if count_x == 0:
        ax.set_ylabel(r'$f_{\mathrm{merg}}$')
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    
    
    if count_x == num_subplots - 1 or count_x == num_subplots_rows - 1:
        # then label the x axis
        ax.set_xlabel(r'log stellar mass (M$_{\odot}$)')
        #plt.setp(ax.get_xticklabels(), rotation=45, color="r", ha="right", rotation_mode="anchor", visible=True)
    else:
        xticks = ax.xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)
    #else:
    #    ax.set_xticks([])
    #    ax.set_xticklabels([])
    '''
    if count_x == 0 or count_x == num_subplots_rows:
        something = 1
    else:
        yticks = ax.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
    '''
        
        
      
    #axs[count_x].errorbar(x, y, yerr = error, linestyle='None', color='black', zorder=100)
    
    ax.text(11.35, 0.85, r'$z_{\mathrm{center}}$ = '+str(round(zcen,2)), color='black', 
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))

    #ax.set_ylabel(r'z$_{\mathrm{center}}$ = '+'\n'+str(round(zcen,2)), rotation = 0, labelpad = 20, ha='center', va='center')
    # This is if you want to MCMC
    res_bb = sm.OLS(y, X, missing = 'drop').fit()#, missing = 'drop'
    try:
        _, data_bb, _ = summary_table(res_bb, alpha=0.05)
    except TypeError:
        fit.append(0)
        continue
    big_boy_fit = data_bb[:,2]
    
    fit.append(big_boy_fit)

    print('length of error', len(error))
    #, 'length after restrict', len(error[count > count_thresh]))
    #error = error[count > count_thresh].reset_index(drop=True)

    mu, sigma = 0, 1 # mean and standard deviation
    

    
    # iterate
    # save slope values
    slope_list = []
    intercept_list = []
    for num in range(100):

        try:
            Y = [y[i]+ error[i] * np.random.normal(mu, sigma, 1)  for i in range(len(y))]
        except KeyError:
            print('y', y)
            print('error', error)
            STOP
        
        
        
        res = sm.OLS(Y, X).fit()#, missing = 'drop'
        
        
        st, data, ss2 = summary_table(res, alpha=0.05)
        fittedvalues = data[:,2]

        
        #plt.scatter(x, Y, s=0.1, color=colors[color_count])
        try:
            slope_list.append(res.params[1])
            intercept_list.append(res.params[0])
        except:
            continue
        ax.plot(x, fittedvalues, color='grey', alpha=0.5)#, label='OLS')
            
        
    line = [x * np.mean(slope_list) + np.mean(intercept_list) for x in x]
      
    
    if np.mean(slope_list) + np.std(slope_list) < 0:
        conc = 'negative'
        col = 'blue'
        ax.plot(x, line, color=col, lw=3)
    else:
        if np.mean(slope_list) - np.std(slope_list) > 0:
            color_space = np.linspace(0.1,0.52,3)
            
            if np.mean(slope_list) < color_space[0] or np.mean(slope_list) > color_space[-1]:
                print('OUTSIDE RANGE')
                print('slope', np.mean(slope_list))
                print('color space', color_space)
            
            # let's find the one thats closest
            difference_array = np.absolute(color_space-np.mean(slope_list))
  
            # find the index of minimum element from the array
            index = difference_array.argmin()
   
            col = color_codes[index+3]
   
            conc = 'positive'
            #col = 'red'
            ax.plot(x, line, color=col, lw=3)
        else:
            conc = 'flat'
            col = 'grey'
            ax.plot(x, line, color=col, lw=3)
            
    ax.annotate('Slope = '+str(round(np.mean(slope_list),2))+'+/-'+str(round(np.std(slope_list),2)), 
        xy=(10.45, 0.85), xycoords='data', color='black')
    ax.annotate(conc, 
        xy=(10.45, 0.75), xycoords='data', color=col)  
    
    count_x += 1
plt.subplots_adjust(hspace=.0, wspace=.01) 
plt.show()







xs = []
ys = []
errors = []
fit = []

plt.clf()
#fig, axs = plt.subplots(nrows = len(centers_z), figsize=(8,8))
fig = plt.figure(figsize=(10,7))
#ax = axs.ravel()

#plt.subplots_adjust(hspace=0.5)
count_x = 0


num_subplots = 14
num_subplots_rows = 7

# Make the same plot but for redshift on the x axis:

for masscen in centers_mass:

    print(masscen)
    #print(str(round(bins_mass[color_count],3))+'$ < M < $'+str(round(bins_mass[color_count+1],3)))
    #color_count+=1
    #continue
    # Grab everything with a z value from the df
    df_select = df_fmerg[df_fmerg['mass'] == masscen].dropna().reset_index()
    
    if len(df_select)==0:
        #color_count+=1
        continue
    # should already be sorted by mass
    count = df_select['count'].values
    print('total count', np.sum(count))
    try:
        zcen = df_select['z'].values
    except KeyError:
        zcen = df_select['z_x'].values


    
    

    # to do a linear regression, I'm going to need to define X and y
    #masscen_masked = np.ma.masked_where(count < 1000, masscen)
    x = zcen.reshape(-1,1)
    x = x[count > count_thresh]
    #x = np.ma.masked_where(count < 1000, x)
    
    x = np.unique(x)
    
    #for m in x:
        
    if len(x) < 2:
        # because don't fit a line to two data pts
        count_x += 1
        continue
    
   
    # Now go through and average f_merg values
    # add errors in quad?
    y = []
    error = []
    for m in x:
        # so for each unique mass:
        fmerg = df_select[(df_select['z'] == m) & (df_select['count'] > count_thresh)]['fmerg']
        fmerg_std = df_select[(df_select['z'] == m) & (df_select['count'] > count_thresh)]['fmerg_std']
        y.append(np.mean(fmerg))
        quad = []
        for f,ferr in zip(fmerg,fmerg_std):
            quad.append(ferr/f)#(ferr/f)**2)
        error.append(np.mean(fmerg)*np.max(ferr/f))#np.sqrt(np.sum(quad))*np.mean(fmerg))
    
    try:
        X = sm.add_constant(x)
    except ValueError:
        continue
    
    
    xs.append(x)
    ys.append(y)
    #errors.append(error[count > count_thresh])
    errors.append(error)
    
    if count_x > 6:
        # then odd
        lookup = {7:2,8:4,9:6,10:8,11:10,12:12,13:14}
        number = lookup[count_x]
    else:
        number = 2 * count_x  + 1
    if count_x == 0 or count_x == num_subplots_rows:
        ax = plt.subplot(num_subplots_rows, 2, number)
    else:
        ax = plt.subplot(num_subplots_rows, 2, number, sharex=ax)
    ax.scatter(x, y, color='black', zorder=100, s = 3)
    ax.errorbar(x, y, yerr=error, ls='None',color='black', zorder=100, capsize = 2)
    
    ax.set_ylim([0,1])
    ax.set_xlim([0.02,0.2])
    
    if count_x == 0:
        ax.set_ylabel(r'$f_{\mathrm{merg}}$')
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
    
    if count_x == num_subplots - 2 or count_x == num_subplots_rows - 1:
        # then label the x axis
        ax.set_xlabel(r'$z$')
        #plt.setp(ax.get_xticklabels(), rotation=45, color="r", ha="right", rotation_mode="anchor", visible=True)
    else:
        xticks = ax.xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)
        xticks[1].label1.set_visible(False)
        xticks[-1].label1.set_visible(False)
    #else:
    #    ax.set_xticks([])
    #    ax.set_xticklabels([])
    '''
    if count_x == 0 or count_x == num_subplots_rows:
        something = 1
    else:
        yticks = ax.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
    '''   
        
      
    #axs[count_x].errorbar(x, y, yerr = error, linestyle='None', color='black', zorder=100)
    
    ax.text(0.125, 0.75, r'mass$_{\mathrm{center}}$ = '+str(round(masscen,2)), color='black', 
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))

    
    
   

    res_bb = sm.OLS(y, X).fit()#, missing = 'drop'
    try:
        _, data_bb, _ = summary_table(res_bb, alpha=0.05)
    except TypeError:
        fit.append(0)
        continue
    big_boy_fit = data_bb[:,2]
    fit.append(big_boy_fit)


    error = df_select['fmerg_std']
    error = error[count > count_thresh].reset_index(drop=True)

    
   
    
    # iterate
    # save slope values
    slope_list = []
    intercept_list = []
    for num in range(100):
        Y = [y[i] + error[i] * np.random.normal(mu, sigma, 1) for i in range(len(y))]

        #scaler = StandardScaler()
        #scaler.fit(X)
        #X_standardized = scaler.transform(X)


        res = sm.OLS(Y, X).fit()
        
        
        #plt.scatter(x, Y, s=0.1, color=colors[color_count])
        try:
            slope_list.append(res.params[1])
            intercept_list.append(res.params[0])
        except:
            continue
            slope_list.append(999)
        
        

        st, data, ss2 = summary_table(res, alpha=0.05)
        fittedvalues = data[:,2]
        predict_mean_se  = data[:,3]
        predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
        predict_ci_low, predict_ci_upp = data[:,6:8].T

        #print(summary_table(res, alpha=0.05))
        

        

        #plt.plot(x, fittedvalues, 'black', alpha=0.5)#, label='OLS')
        ax.plot(x, fittedvalues, color='grey', alpha=0.5)#, label='OLS')
        
            

        #plt.plot(x, predict_ci_low, 'b--')
        #plt.plot(x, predict_ci_upp, 'b--')
        #plt.plot(x, predict_mean_ci_low, 'g--')
        #plt.plot(x, predict_mean_ci_upp, 'g--')
    line = [x * np.mean(slope_list) + np.mean(intercept_list) for x in x]
   
    ax.annotate('Slope = '+str(round(np.mean(slope_list),2))+'+/-'+str(round(np.std(slope_list),2)), 
        xy=(0.025,0.85), xycoords='data', color='black')
    if np.mean(slope_list) + np.std(slope_list) < 0:
        
        color_space = np.linspace(-4.1,-0.5,3)
            
        if np.mean(slope_list) < color_space[0] or np.mean(slope_list) > color_space[-1]:
            print('OUTSIDE RANGE')
            print('slope', np.mean(slope_list))
            print('color space', color_space)
        
        # let's find the one thats closest
        difference_array = np.absolute(color_space-np.mean(slope_list))

        # find the index of minimum element from the array
        index = difference_array.argmin()

        col = color_codes[index]
        
        conc = 'negative'
        #col = 'blue'
        ax.plot(x, line, color=col, lw=3)
    else:
        if np.mean(slope_list) - np.std(slope_list) > 0:
            conc = 'positive'
            col = 'red'
            ax.plot(x, line, color=col, lw=3)
        else:
            conc = 'flat'
            col = 'grey'
            ax.plot(x, line, color=col, lw=3)
    ax.annotate(conc, 
        xy=(0.025,0.7), xycoords='data', color=col)  
        
    

    count_x += 1
plt.subplots_adjust(hspace=.0, wspace=0.03)  
plt.show()