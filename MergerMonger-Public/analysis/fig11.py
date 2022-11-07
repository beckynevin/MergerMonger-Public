#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Compare the properties of our merger samples
# i.e., major versus minor
# pre- versus post-coalescence mergers
# also compare to the non-merging sample

# There's also still code to merge with ackermann
# if you also want to compare with that sample
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# path
dir = '/Users/rnevin/CfA_Laptop/Documents/CfA_Code/MergerMonger-dev/Tables/'

type_gal = 'predictors'

suffix = 'color_complete'
add_on_binned_table = '_ackermann'#_cut_ours'
type_marginalized = '_flags_cut_segmap'
prefix = '/Users/rnevin/CfA_Laptop/Documents/CfA_Code/MergerMonger-dev/Tables/'
savefigs = True
B_T = False




# Next import our mergers

run1 = 'major_merger'
df_LDA_major = pd.io.parsers.read_csv(filepath_or_buffer=dir+'LDA_out_all_SDSS_'+type_gal+'_'+run1+'_flags.txt',header=[0],sep='\t')

run2 = 'minor_merger'
df_LDA_minor = pd.io.parsers.read_csv(filepath_or_buffer=dir+'LDA_out_all_SDSS_'+type_gal+'_'+run2+'_flags.txt',header=[0],sep='\t')

run3 = 'major_merger_prec'
df_LDA_major_prec = pd.io.parsers.read_csv(filepath_or_buffer=dir+'LDA_out_all_SDSS_'+type_gal+'_'+run3+'_flags.txt',header=[0],sep='\t')

run4 = 'major_merger_postc_include_coal_0.5'
df_LDA_major_postc = pd.io.parsers.read_csv(filepath_or_buffer=dir+'LDA_out_all_SDSS_'+type_gal+'_'+run4+'_flags.txt',header=[0],sep='\t')

run5 = 'minor_merger_prec'
df_LDA_minor_prec = pd.io.parsers.read_csv(filepath_or_buffer=dir+'LDA_out_all_SDSS_'+type_gal+'_'+run5+'_flags.txt',header=[0],sep='\t')

run6 = 'minor_merger_postc_include_coal_0.5'
df_LDA_minor_postc = pd.io.parsers.read_csv(filepath_or_buffer=dir+'LDA_out_all_SDSS_'+type_gal+'_'+run6+'_flags.txt',header=[0],sep='\t')






# Because the df_LDA doesn't have the final flag, use the predictor table to instead clean via merging

# Run OLS with predictor values and z and stellar mass and p_merg:
df_predictors = pd.io.parsers.read_csv(filepath_or_buffer=dir+'SDSS_predictors_all_flags_plus_segmap.txt',header=[0],sep='\t')

if len(df_LDA_major) != len(df_predictors):
	print('these have different lengths cannot use one to flag')
	STOP

# First clean this so that there's no segmap flags
print('length of sample before cleaning', len(df_predictors))
df_predictors_clean = df_predictors[(df_predictors['low S/N'] ==0) & (df_predictors['outlier predictor']==0) & (df_predictors['segmap']==0)]

print('length of sample after cleaning', len(df_predictors_clean))

clean_LDA_major = df_LDA_major.merge(df_predictors_clean, on='ID')
clean_LDA_minor = df_LDA_minor.merge(df_predictors_clean, on='ID')
clean_LDA_major_prec = df_LDA_major_prec.merge(df_predictors_clean, on='ID')
clean_LDA_minor_prec = df_LDA_minor_prec.merge(df_predictors_clean, on='ID')
clean_LDA_major_postc = df_LDA_major_postc.merge(df_predictors_clean, on='ID')
clean_LDA_minor_postc = df_LDA_minor_postc.merge(df_predictors_clean, on='ID')


if suffix == 'color_complete':
	masstable = pd.io.parsers.read_csv(filepath_or_buffer=dir+'all_mass_color_complete_0.03_z_log_stellar_mass_from_color_completeness_95.txt',header=[0],sep='\t')
if suffix == 'color':
	masstable = pd.io.parsers.read_csv(filepath_or_buffer=dir+'all_mass_color.txt',header=[0],sep='\t')


masstable = masstable[masstable['log_stellar_mass_from_color'] < 13]
#masstable = masstable[(masstable['log_stellar_mass_from_color'] < 13) & (masstable['log_stellar_mass_from_color'] > 11.5)]



# Now merge this with LDA
final_merged_major = masstable.merge(clean_LDA_major, left_on='objID', right_on='ID')#[0:1000]# Now merging on dr8
final_merged_minor = masstable.merge(clean_LDA_minor, left_on='objID', right_on='ID')#[0:1000]# Now merging on dr8

final_merged_major_prec = masstable.merge(clean_LDA_major_prec, left_on='objID', right_on='ID')#[0:1000]# Now merging on dr8
final_merged_minor_prec = masstable.merge(clean_LDA_minor_prec, left_on='objID', right_on='ID')#[0:1000]# Now merging on dr8
final_merged_major_postc = masstable.merge(clean_LDA_major_postc, left_on='objID', right_on='ID')#[0:1000]# Now merging on dr8
final_merged_minor_postc = masstable.merge(clean_LDA_minor_postc, left_on='objID', right_on='ID')#[0:1000]# Now merging on dr8



if B_T:
    # Now bring in the masstable that has the B/T ratios:
    masstable_m = pd.io.parsers.read_csv(filepath_or_buffer=dir+'all_mass_measurements.txt',header=[0],sep='\t')




    #masstable_m = masstable_m[masstable_m['logBD'] < 13]
    masstable_m['B/T'] = 10**masstable_m['logMb'] / (10**masstable_m['logMb'] + 10**masstable_m['logMd'])


    minor_B_T = masstable_m.merge(final_merged_minor, on='objID')
    major_B_T = masstable_m.merge(final_merged_major, on='objID')

    minor_prec_B_T = masstable_m.merge(final_merged_minor_prec, on='objID')
    major_prec_B_T = masstable_m.merge(final_merged_major_prec, on='objID')

    minor_postc_B_T = masstable_m.merge(final_merged_minor_postc, on='objID')
    major_postc_B_T = masstable_m.merge(final_merged_major_postc, on='objID')

    print(minor_B_T.columns)
'''
minor_ack = merged_ack.merge(final_merged_minor, on='objID')
major_ack = merged_ack.merge(final_merged_major, on='objID')

minor_prec_ack = merged_ack.merge(final_merged_minor_prec, on='objID')
major_prec_ack = merged_ack.merge(final_merged_major_prec, on='objID')

minor_postc_ack = merged_ack.merge(final_merged_minor_postc, on='objID')
major_postc_ack = merged_ack.merge(final_merged_major_postc, on='objID')
'''


'''
final_merged = final_merged[
    (final_merged['log_stellar_mass_from_color'] > 9) & (final_merged['log_stellar_mass_from_color'] < 12)
    & (final_merged['z'] < 0.3)
    ]
'''

thresh_merg = 0.5
thresh_merg_sc = 0.95
thresh_merg_ack = 0.95
# First make Bobby's plots from the analysis section:
# A three panel horizontal with histograms from the parent and merger sample in m_r, stellar mass, and z

density=False
merger_color = '#EF476F'
sc_merger_color ='#FFD166'
alpha=0.7

if B_T:

    merger_selection_1 = major_B_T[major_B_T['p_merg_y'] > thresh_merg]
    merger_selection_2 = major_prec_B_T[major_prec_B_T['p_merg_y'] > thresh_merg]


    plt.clf()
    fig = plt.figure(figsize=(12,7))

    property = 'B/T'
    ax0 = fig.add_subplot(131)
    _, bins = np.histogram(major_B_T[property].values, bins=50)
    #ax.hist(major_ack['r'].values, bins=bins, label='Parent', density=density, alpha=0.5, color='#546A7B', fill=False)
    ax0.hist(major_B_T[property].values, bins=bins, 
        label=f'Parent (# = {len(final_merged_major)})', 
        density=density, color='black', fill=False)
    ax0.hist(merger_selection_1[property].values, bins=bins, 
        label=f'Major (# = {len(merger_selection_1)})', 
        density=density, alpha=alpha, color=merger_color)
    ax0.hist(merger_selection_2[property].values, bins=bins, 
        label=f'Minor (# = {len(merger_selection_2)})', 
        density=density, alpha=alpha, color=sc_merger_color)#, fill=False, ls=':')
    '''
    # Also plot a mean for each
    ax0.axvline(x = np.median(final_merged_major['S/N'].values), color='black')
    # What about spanning the uncertainty in the overall?
    ax0.axvspan(np.median(final_merged_major['S/N'].values) - np.std(final_merged_major['S/N'].values), 
        np.median(final_merged_major['S/N'].values) + np.std(final_merged_major['S/N'].values),
        alpha=0.25, color='grey')

    ax0.axvline(x = np.median(merger_selection_1['S/N'].values), color=merger_color)
    ax0.axvline(x = np.median(merger_selection_2['S/N'].values), color=sc_merger_color)
    '''

    plt.legend()
    ax0.set_xlabel(property)

    property = 'log_stellar_mass_from_color_y'

    ax = fig.add_subplot(132)
    _, bins = np.histogram(major_B_T[property].values, bins=50)
    #ax.hist(major_ack['r'].values, bins=bins, label='Parent', density=density, alpha=0.5, color='#546A7B', fill=False)
    ax.hist(major_B_T[property].values, bins=bins, 
        density=density, color='black', fill=False)
    ax.hist(merger_selection_1[property].values, bins=bins, 
        density=density, alpha=alpha, color=merger_color)
    ax.hist(merger_selection_2[property].values, bins=bins, 
        density=density, alpha=alpha, color=sc_merger_color)#, fill=False, ls=':')

    ax.set_xlabel(property)

    plt.show()

merger_selection_1 = final_merged_major[final_merged_major['p_merg_y'] > thresh_merg]
merger_selection_2 = final_merged_minor[final_merged_minor['p_merg_y'] > thresh_merg]


'''
# Before you make the pdf figure, can we do a binning analysis similar to
# the appendix of Bobby's paper where we bin first at stellar mass 
# and look at how the redshifts compare for the different samples and then
# vv

mass_bins = np.linspace(9.5,13,20)
spacing_mass = mass_bins[1] - mass_bins[0]
mass_bins_centers = [x + spacing_mass/2 for x in mass_bins]
print(mass_bins)
print(mass_bins_centers)

redshift_bins = np.linspace(0,0.3,20)
spacing_z = redshift_bins[1] - redshift_bins[0]
redshift_bins_centers = [x + spacing_z/2 for x in redshift_bins]
print(redshift_bins)

S_N_bins = np.linspace(0,100,20)
spacing_S_N = S_N_bins[1] - S_N_bins[0]
S_N_bins_centers = [x + spacing_S_N/2 for x in S_N_bins]


major_merger_S_N_mean = []
major_merger_S_N_std = []

parent_S_N_mean = []
parent_S_N_std = []



for i in range(len(mass_bins)-1):
    selection = merger_selection_1[(merger_selection_1['log_stellar_mass_from_color'] > mass_bins[i]) & 
                           (merger_selection_1['log_stellar_mass_from_color'] < mass_bins[i+1])]['S/N'].values
    major_merger_S_N_mean.append(np.mean(selection))
    major_merger_S_N_std.append(np.std(selection))
    
    selection = final_merged_major[(final_merged_major['log_stellar_mass_from_color'] > mass_bins[i]) & 
                           (final_merged_major['log_stellar_mass_from_color'] < mass_bins[i+1])]['S/N'].values
    parent_S_N_mean.append(np.mean(selection))
    parent_S_N_std.append(np.std(selection))
    
major_merger_S_N_mean = []
major_merger_S_N_std = []

parent_S_N_mean = []
parent_S_N_std = []

for i in range(len(redshift_bins)-1):
    selection = merger_selection_1[(merger_selection_1['z'] > redshift_bins[i]) & 
                           (merger_selection_1['z'] < redshift_bins[i+1])]['S/N'].values
    major_merger_S_N_mean.append(np.mean(selection))
    major_merger_S_N_std.append(np.std(selection))
    
    selection = final_merged_major[(final_merged_major['z'] > redshift_bins[i]) & 
                           (final_merged_major['z'] < redshift_bins[i+1])]['S/N'].values
    parent_S_N_mean.append(np.mean(selection))
    parent_S_N_std.append(np.std(selection))
    
plt.clf()
fig = plt.figure(figsize=(7,4))
ax0 = fig.add_subplot(121)
#plt.scatter(major_merger_zs_mean, mass_bins_centers[:-1], s=0.3)
ax0.errorbar(major_merger_S_N_mean, mass_bins_centers[:-1],xerr = major_merger_S_N_std, 
             label='Major merger',
             fmt='o',capsize=5,
             color='#CBEF43')#, s=0.3)

#plt.scatter(parent_zs_mean, mass_bins_centers[:-1], label='Parent', s=0.3)
ax0.errorbar(parent_S_N_mean, mass_bins_centers[:-1],xerr = parent_S_N_std, 
             label='Parent',
             fmt='o',capsize=5,
             color='#433A3F')#, s=0.3)
ax0.set_ylabel(r'log stellar mass (M$_{\odot}$)')
ax0.set_xlabel(r'S/N')
plt.legend()

ax1 = fig.add_subplot(122)
#plt.scatter(major_merger_zs_mean, mass_bins_centers[:-1], s=0.3)
ax1.errorbar( redshift_bins_centers[:-1], major_merger_S_N_mean, yerr = major_merger_S_N_std, 
             label='Major merger',
             fmt='o',capsize=5,
             color='#CBEF43')#, s=0.3)

#plt.scatter(parent_zs_mean, mass_bins_centers[:-1], label='Parent', s=0.3)
ax1.errorbar(redshift_bins_centers[:-1], parent_S_N_mean, yerr = parent_S_N_std, 
             label='Parent',
             fmt='o',capsize=5,
             color='#433A3F')#, s=0.3)

ax1.set_xlabel('z')
ax1.set_ylabel('S/N')

plt.legend()
plt.show()



major_merger_zs_mean = []
major_merger_zs_std = []

parent_zs_mean = []
parent_zs_std = []



for i in range(len(mass_bins)-1):
    selection = merger_selection_1[(merger_selection_1['log_stellar_mass_from_color'] > mass_bins[i]) & 
                           (merger_selection_1['log_stellar_mass_from_color'] < mass_bins[i+1])]['z'].values
    major_merger_zs_mean.append(np.mean(selection))
    major_merger_zs_std.append(np.std(selection))
    
    selection = final_merged_major[(final_merged_major['log_stellar_mass_from_color'] > mass_bins[i]) & 
                           (final_merged_major['log_stellar_mass_from_color'] < mass_bins[i+1])]['z'].values
    parent_zs_mean.append(np.mean(selection))
    parent_zs_std.append(np.std(selection))
    
major_merger_mass_mean = []
major_merger_mass_std = []

parent_mass_mean = []
parent_mass_std = []

for i in range(len(redshift_bins)-1):
    selection = merger_selection_1[(merger_selection_1['z'] > redshift_bins[i]) & 
                           (merger_selection_1['z'] < redshift_bins[i+1])]['log_stellar_mass_from_color'].values
    major_merger_mass_mean.append(np.mean(selection))
    major_merger_mass_std.append(np.std(selection))
    
    selection = final_merged_major[(final_merged_major['z'] > redshift_bins[i]) & 
                           (final_merged_major['z'] < redshift_bins[i+1])]['log_stellar_mass_from_color'].values
    parent_mass_mean.append(np.mean(selection))
    parent_mass_std.append(np.std(selection))
    
plt.clf()
fig = plt.figure(figsize=(7,4))
ax0 = fig.add_subplot(121)
#plt.scatter(major_merger_zs_mean, mass_bins_centers[:-1], s=0.3)
ax0.errorbar(major_merger_zs_mean, mass_bins_centers[:-1],xerr = major_merger_zs_std, 
             label='Major merger',
             fmt='o',capsize=5,
             color='#CBEF43')#, s=0.3)

#plt.scatter(parent_zs_mean, mass_bins_centers[:-1], label='Parent', s=0.3)
ax0.errorbar(parent_zs_mean, mass_bins_centers[:-1],xerr = parent_zs_std, 
             label='Parent',
             fmt='o',capsize=5,
             color='#433A3F')#, s=0.3)
ax0.set_ylabel(r'log stellar mass (M$_{\odot}$)')
ax0.set_xlabel(r'$z$')
plt.legend()

ax1 = fig.add_subplot(122)
#plt.scatter(major_merger_zs_mean, mass_bins_centers[:-1], s=0.3)
ax1.errorbar( redshift_bins_centers[:-1], major_merger_mass_mean, yerr = major_merger_mass_std, 
             label='Major merger',
             fmt='o',capsize=5,
             color='#CBEF43')#, s=0.3)

#plt.scatter(parent_zs_mean, mass_bins_centers[:-1], label='Parent', s=0.3)
ax1.errorbar(redshift_bins_centers[:-1], parent_mass_mean, yerr = parent_mass_std, 
             label='Parent',
             fmt='o',capsize=5,
             color='#433A3F')#, s=0.3)

plt.legend()
plt.show()
    

STOP
'''

# The pdf figure comparing all of the different properties

plt.clf()
fig = plt.figure(figsize=(7,6))

ax0 = fig.add_subplot(231)
_, bins = np.histogram(final_merged_major['S/N'].values, bins=50, range = [0,100])
#ax.hist(major_ack['r'].values, bins=bins, label='Parent', density=density, alpha=0.5, color='#546A7B', fill=False)
ax0.hist(final_merged_major['S/N'].values, bins=bins, 
    label=f'Parent (# = {len(final_merged_major)})', 
    density=density, color='black', 
    weights=np.ones(len(final_merged_major['S/N'].values)) / len(final_merged_major['S/N'].values), 
    fill=False)
ax0.hist(merger_selection_1['S/N'].values, bins=bins, 
    label=f'Major (# = {len(merger_selection_1)})', 
    density=density, alpha=alpha, color=merger_color,
    weights=np.ones(len(merger_selection_1['S/N'].values)) / len(merger_selection_1['S/N'].values) )
ax0.hist(merger_selection_2['S/N'].values, bins=bins, 
    label=f'Minor (# = {len(merger_selection_2)})', 
    density=density, alpha=alpha, color=sc_merger_color,
    weights=np.ones(len(merger_selection_2['S/N'].values)) / len(merger_selection_2['S/N'].values))#, fill=False, ls=':')




# Run a KS test:
param = 'S/N'
parent = final_merged_major[param].values
samp_1 = merger_selection_1[param].values
samp_2 = merger_selection_2[param].values

print(f'COMPARING IN THIS PROPERTY {param}')
print(r'parent med = '+str(round(np.median(final_merged_major[param].values),4))+', std = '+str(round(np.std(final_merged_major[param].values),4)))
print(r'merger_selection_1 med = '+str(round(np.median(merger_selection_1[param].values),4))+', std = '+str(round(np.std(merger_selection_1[param].values),4)))
print(r'merger_selection_2 med = '+str(round(np.median(merger_selection_2[param].values),4))+', std = '+str(round(np.std(merger_selection_2[param].values),4)))


# Okay, no, you're going to make your own cdfs:
count_parent, bins_count = np.histogram(parent, bins=bins)
count_samp_1, bins_count = np.histogram(samp_1, bins=bins)
count_samp_2, bins_count = np.histogram(samp_2, bins=bins)

  
# finding the PDF of the histogram using count values
pdf_parent = count_parent / sum(count_parent)
pdf_samp_1 = count_samp_1 / sum(count_samp_1)
pdf_samp_2 = count_samp_2 / sum(count_samp_2)
  
# using numpy np.cumsum to calculate the CDF
# We can also find using the PDF values by looping and adding
cdf_parent = np.cumsum(pdf_parent)
cdf_samp_1 = np.cumsum(pdf_samp_1)
cdf_samp_2 = np.cumsum(pdf_samp_2)

'''
ax0.plot(cdf_parent, color = 'black')
ax0.plot(cdf_samp_1, color = merger_color)
ax0.plot(cdf_samp_2, color = sc_merger_color)
'''


print('cdf difference between parent and samp1')
stat, pvalue = stats.ks_2samp(cdf_parent, cdf_samp_1)
print(stat, pvalue)

print('cdf difference between parent and samp2')
stat, pvalue = stats.ks_2samp(cdf_parent, cdf_samp_2)
print(stat, pvalue)

print('cdf difference between samp1 and samp2')
stat, pvalue = stats.ks_2samp(cdf_samp_1, cdf_samp_2)
print(stat, pvalue)

'''
ax0.hist(pdf_samp_1 - pdf_parent, 
         bins=bins, label = 'Major - Parent',
         #weights=np.ones(len(bins)) / len(bins),
         color='black')
'''


ax0.legend()
ax0.set_xlabel('S/N')
ax0.set_ylabel('pdf')

ax00 = fig.add_subplot(232)
_, bins = np.histogram(final_merged_major['r'].values, bins=50, range = [12.5,19])
#ax.hist(major_ack['r'].values, bins=bins, label='Parent', density=density, alpha=0.5, color='#546A7B', fill=False)
ax00.hist(final_merged_major['r'].values, bins=bins, 
    density=density, color='black', 
    weights=np.ones(len(final_merged_major['r'].values)) / len(final_merged_major['r'].values), 
    fill=False)
ax00.hist(merger_selection_1['r'].values, bins=bins, 
    density=density, alpha=alpha, color=merger_color,
    weights=np.ones(len(merger_selection_1['r'].values)) / len(merger_selection_1['r'].values) )
ax00.hist(merger_selection_2['r'].values, bins=bins, 
    density=density, alpha=alpha, color=sc_merger_color,
    weights=np.ones(len(merger_selection_2['r'].values)) / len(merger_selection_2['r'].values))#, fill=False, ls=':')

# Run a KS test:
param = 'r'
parent = final_merged_major[param].values
samp_1 = merger_selection_1[param].values
samp_2 = merger_selection_2[param].values

print(f'COMPARING IN THIS PROPERTY {param}')
print(r'parent med = '+str(round(np.median(final_merged_major[param].values),4))+', std = '+str(round(np.std(final_merged_major[param].values),4)))
print(r'merger_selection_1 med = '+str(round(np.median(merger_selection_1[param].values),4))+', std = '+str(round(np.std(merger_selection_1[param].values),4)))
print(r'merger_selection_2 med = '+str(round(np.median(merger_selection_2[param].values),4))+', std = '+str(round(np.std(merger_selection_2[param].values),4)))


# Okay, no, you're going to make your own cdfs:
count_parent, bins_count = np.histogram(parent, bins=bins)
count_samp_1, bins_count = np.histogram(samp_1, bins=bins)
count_samp_2, bins_count = np.histogram(samp_2, bins=bins)

  
# finding the PDF of the histogram using count values
pdf_parent = count_parent / sum(count_parent)
pdf_samp_1 = count_samp_1 / sum(count_samp_1)
pdf_samp_2 = count_samp_2 / sum(count_samp_2)
  
# using numpy np.cumsum to calculate the CDF
# We can also find using the PDF values by looping and adding
cdf_parent = np.cumsum(pdf_parent)
cdf_samp_1 = np.cumsum(pdf_samp_1)
cdf_samp_2 = np.cumsum(pdf_samp_2)


print('cdf difference between parent and samp1')
stat, pvalue = stats.ks_2samp(cdf_parent, cdf_samp_1)
print(stat, pvalue)

print('cdf difference between parent and samp2')
stat, pvalue = stats.ks_2samp(cdf_parent, cdf_samp_2)
print(stat, pvalue)

print('cdf difference between samp1 and samp2')
stat, pvalue = stats.ks_2samp(cdf_samp_1, cdf_samp_2)
print(stat, pvalue)


'''
# Also plot a mean for each
ax0.axvline(x = np.median(final_merged_major['S/N'].values), color='black')
# What about spanning the uncertainty in the overall?
ax0.axvspan(np.median(final_merged_major['S/N'].values) - np.std(final_merged_major['S/N'].values), 
    np.median(final_merged_major['S/N'].values) + np.std(final_merged_major['S/N'].values),
    alpha=0.25, color='grey')

ax0.axvline(x = np.median(merger_selection_1['S/N'].values), color=merger_color)
ax0.axvline(x = np.median(merger_selection_2['S/N'].values), color=sc_merger_color)
'''

ax00.set_xlabel(r'<-- brighter   $r$   fainter -->')
#ax00.set_ylabel('Proportion')
#ax00.set_xlim([13,19])

property = 'g_minus_r'

ax = fig.add_subplot(233)
_, bins = np.histogram(final_merged_major[property].values, bins=50, range = [0,2.5])
#ax.hist(major_ack['r'].values, bins=bins, label='Parent', density=density, alpha=0.5, color='#546A7B', fill=False)
ax.hist(final_merged_major[property].values, bins=bins, 
    density=density, color='black', 
    weights=np.ones(len(final_merged_major[property].values)) / len(final_merged_major[property].values),
    fill=False)
ax.hist(merger_selection_1[property].values, bins=bins, 
    density=density, alpha=alpha, color=merger_color,
    weights=np.ones(len(merger_selection_1[property].values)) / len(merger_selection_1[property].values))
ax.hist(merger_selection_2[property].values, bins=bins, 
    density=density, alpha=alpha, color=sc_merger_color,
    weights=np.ones(len(merger_selection_2[property].values)) / len(merger_selection_2[property].values))#, fill=False, ls=':')

ax.set_xlabel(property)
if property == 'g_minus_r':
    #ax.set_xlim([0,2.5])
    ax.set_xlabel(r'color ($g-r$)    redder -->')
#ax.set_xlim([13,19])

# Run a KS test:
param = property
parent = final_merged_major[param].values
samp_1 = merger_selection_1[param].values
samp_2 = merger_selection_2[param].values

print(f'COMPARING IN THIS PROPERTY {param}')
print(r'parent med = '+str(round(np.median(final_merged_major[param].values),4))+', std = '+str(round(np.std(final_merged_major[param].values),4)))
print(r'merger_selection_1 med = '+str(round(np.median(merger_selection_1[param].values),4))+', std = '+str(round(np.std(merger_selection_1[param].values),4)))
print(r'merger_selection_2 med = '+str(round(np.median(merger_selection_2[param].values),4))+', std = '+str(round(np.std(merger_selection_2[param].values),4)))


# Okay, no, you're going to make your own cdfs:
count_parent, bins_count = np.histogram(parent, bins=bins)
count_samp_1, bins_count = np.histogram(samp_1, bins=bins)
count_samp_2, bins_count = np.histogram(samp_2, bins=bins)

  
# finding the PDF of the histogram using count values
pdf_parent = count_parent / sum(count_parent)
pdf_samp_1 = count_samp_1 / sum(count_samp_1)
pdf_samp_2 = count_samp_2 / sum(count_samp_2)
  
# using numpy np.cumsum to calculate the CDF
# We can also find using the PDF values by looping and adding
cdf_parent = np.cumsum(pdf_parent)
cdf_samp_1 = np.cumsum(pdf_samp_1)
cdf_samp_2 = np.cumsum(pdf_samp_2)


print('cdf difference between parent and samp1')
stat, pvalue = stats.ks_2samp(cdf_parent, cdf_samp_1)
print(stat, pvalue)

print('cdf difference between parent and samp2')
stat, pvalue = stats.ks_2samp(cdf_parent, cdf_samp_2)
print(stat, pvalue)

print('cdf difference between samp1 and samp2')
stat, pvalue = stats.ks_2samp(cdf_samp_1, cdf_samp_2)
print(stat, pvalue)

'''
# Also plot a mean for each
ax.axvline(x = np.median(final_merged_major['r'].values), color='black')
ax.axvspan(np.median(final_merged_major['r'].values) - np.std(final_merged_major['r'].values), 
    np.median(final_merged_major['r'].values) + np.std(final_merged_major['r'].values),
    alpha=0.25, color='grey')
ax.axvline(x = np.median(merger_selection_1['r'].values), color=merger_color)
ax.axvline(x = np.median(merger_selection_2['r'].values), color=sc_merger_color)
'''




ax1 = fig.add_subplot(234)
_, bins = np.histogram(final_merged_major['log_stellar_mass_from_color'].values, 
                       bins=50, range=[9.5,13.5])
#ax1.hist(final_merged_major['log_stellar_mass_from_color'].values, bins=bins, label='Parent', density=density, alpha=0.5, color='#546A7B', fill=False)
ax1.hist(final_merged_major['log_stellar_mass_from_color'].values, bins=bins, density=density, color='black', fill=False,
         weights=np.ones(len(final_merged_major['log_stellar_mass_from_color'].values)) / len(final_merged_major['log_stellar_mass_from_color'].values))
ax1.hist(merger_selection_1['log_stellar_mass_from_color'].values, bins=bins, density=density, alpha=alpha, color=merger_color,
         weights=np.ones(len(merger_selection_1['log_stellar_mass_from_color'].values)) / len(merger_selection_1['log_stellar_mass_from_color'].values))
ax1.hist(merger_selection_2['log_stellar_mass_from_color'].values, bins=bins, density=density, alpha=alpha, color=sc_merger_color,
         weights=np.ones(len(merger_selection_2['log_stellar_mass_from_color'].values)) / len(merger_selection_2['log_stellar_mass_from_color'].values))#, fill=False, ls=':')
ax1.set_xlabel(r'log stellar mass (M$_{\odot}$)')
#ax1.set_xlim([9.5,13.5])
ax1.set_ylabel('pdf')

param = 'log_stellar_mass_from_color'
parent = final_merged_major[param].values
samp_1 = merger_selection_1[param].values
samp_2 = merger_selection_2[param].values

print(f'COMPARING IN THIS PROPERTY {param}')
print(r'parent med = '+str(round(np.median(final_merged_major[param].values),4))+', std = '+str(round(np.std(final_merged_major[param].values),4)))
print(r'merger_selection_1 med = '+str(round(np.median(merger_selection_1[param].values),4))+', std = '+str(round(np.std(merger_selection_1[param].values),4)))
print(r'merger_selection_2 med = '+str(round(np.median(merger_selection_2[param].values),4))+', std = '+str(round(np.std(merger_selection_2[param].values),4)))


# Okay, no, you're going to make your own cdfs:
count_parent, bins_count = np.histogram(parent, bins=bins)
count_samp_1, bins_count = np.histogram(samp_1, bins=bins)
count_samp_2, bins_count = np.histogram(samp_2, bins=bins)

  
# finding the PDF of the histogram using count values
pdf_parent = count_parent / sum(count_parent)
pdf_samp_1 = count_samp_1 / sum(count_samp_1)
pdf_samp_2 = count_samp_2 / sum(count_samp_2)
  
# using numpy np.cumsum to calculate the CDF
# We can also find using the PDF values by looping and adding
cdf_parent = np.cumsum(pdf_parent)
cdf_samp_1 = np.cumsum(pdf_samp_1)
cdf_samp_2 = np.cumsum(pdf_samp_2)


print('cdf difference between parent and samp1')
stat, pvalue = stats.ks_2samp(cdf_parent, cdf_samp_1)
print(stat, pvalue)

print('cdf difference between parent and samp2')
stat, pvalue = stats.ks_2samp(cdf_parent, cdf_samp_2)
print(stat, pvalue)

print('cdf difference between samp1 and samp2')
stat, pvalue = stats.ks_2samp(cdf_samp_1, cdf_samp_2)
print(stat, pvalue)


'''
# Also plot a mean for each
ax1.axvline(x = np.median(final_merged_major['log_stellar_mass_from_color'].values), color='black')
ax1.axvspan(np.median(final_merged_major['log_stellar_mass_from_color'].values) - np.std(final_merged_major['log_stellar_mass_from_color'].values), 
    np.median(final_merged_major['log_stellar_mass_from_color'].values) + np.std(final_merged_major['log_stellar_mass_from_color'].values),
    alpha=0.25, color='grey')
ax1.axvline(x = np.median(merger_selection_1['log_stellar_mass_from_color'].values), color=merger_color)
ax1.axvline(x = np.median(merger_selection_2['log_stellar_mass_from_color'].values), color=sc_merger_color)
'''

ax2 = fig.add_subplot(235)
_, bins = np.histogram(final_merged_major['z'].values, bins=50, range = [0,0.3])
#ax2.hist(final_merged_major['z'].values, bins=bins, label='Parent', density=density, alpha=0.5, color='#546A7B', fill=False)
ax2.hist(final_merged_major['z'].values, bins=bins, density=density, color='black', fill=False,
         weights=np.ones(len(final_merged_major['z'].values)) / len(final_merged_major['z'].values))
ax2.hist(merger_selection_1['z'].values, bins=bins, density=density, alpha=alpha, color=merger_color,
         weights=np.ones(len(merger_selection_1['z'].values)) / len(merger_selection_1['z'].values))
ax2.hist(merger_selection_2['z'].values, bins=bins, density=density, alpha=alpha, color=sc_merger_color,
         weights=np.ones(len(merger_selection_2['z'].values)) / len(merger_selection_2['z'].values))#, fill=False, ls=':')
ax2.set_xlabel('redshift')
#ax2.set_xlim([0,0.3])

param = 'z'
parent = final_merged_major[param].values
samp_1 = merger_selection_1[param].values
samp_2 = merger_selection_2[param].values

print(f'COMPARING IN THIS PROPERTY {param}')
print(r'parent med = '+str(round(np.median(final_merged_major[param].values),4))+', std = '+str(round(np.std(final_merged_major[param].values),4)))
print(r'merger_selection_1 med = '+str(round(np.median(merger_selection_1[param].values),4))+', std = '+str(round(np.std(merger_selection_1[param].values),4)))
print(r'merger_selection_2 med = '+str(round(np.median(merger_selection_2[param].values),4))+', std = '+str(round(np.std(merger_selection_2[param].values),4)))


# Okay, no, you're going to make your own cdfs:
count_parent, bins_count = np.histogram(parent, bins=bins)
count_samp_1, bins_count = np.histogram(samp_1, bins=bins)
count_samp_2, bins_count = np.histogram(samp_2, bins=bins)

  
# finding the PDF of the histogram using count values
pdf_parent = count_parent / sum(count_parent)
pdf_samp_1 = count_samp_1 / sum(count_samp_1)
pdf_samp_2 = count_samp_2 / sum(count_samp_2)
  
# using numpy np.cumsum to calculate the CDF
# We can also find using the PDF values by looping and adding
cdf_parent = np.cumsum(pdf_parent)
cdf_samp_1 = np.cumsum(pdf_samp_1)
cdf_samp_2 = np.cumsum(pdf_samp_2)


print('cdf difference between parent and samp1')
stat, pvalue = stats.ks_2samp(cdf_parent, cdf_samp_1)
print(stat, pvalue)

print('cdf difference between parent and samp2')
stat, pvalue = stats.ks_2samp(cdf_parent, cdf_samp_2)
print(stat, pvalue)

print('cdf difference between samp1 and samp2')
stat, pvalue = stats.ks_2samp(cdf_samp_1, cdf_samp_2)
print(stat, pvalue)


plt.show()

STOP

plt.clf()
fig = plt.figure(figsize=(12,7))

ax0 = fig.add_subplot(241)
_, bins = np.histogram(final_merged_major['S/N'].values, bins=50, range = [0,200])
#ax.hist(major_ack['r'].values, bins=bins, label='Parent', density=density, alpha=0.5, color='#546A7B', fill=False)
ax0.hist(final_merged_major['S/N'].values, bins=bins, 
    label=f'Parent (# = {len(final_merged_major)})', 
    density=density, color='black', 
    weights=np.ones(len(final_merged_major['S/N'].values)) / len(final_merged_major['S/N'].values), 
    fill=False)
ax0.hist(merger_selection_1['S/N'].values, bins=bins, 
    label=f'Major (# = {len(merger_selection_1)})', 
    density=density, alpha=alpha, color=merger_color,
    weights=np.ones(len(merger_selection_1['S/N'].values)) / len(merger_selection_1['S/N'].values) )
ax0.hist(merger_selection_2['S/N'].values, bins=bins, 
    label=f'Minor (# = {len(merger_selection_2)})', 
    density=density, alpha=alpha, color=sc_merger_color,
    weights=np.ones(len(merger_selection_2['S/N'].values)) / len(merger_selection_2['S/N'].values))#, fill=False, ls=':')
'''
# Also plot a mean for each
ax0.axvline(x = np.median(final_merged_major['S/N'].values), color='black')
# What about spanning the uncertainty in the overall?
ax0.axvspan(np.median(final_merged_major['S/N'].values) - np.std(final_merged_major['S/N'].values), 
    np.median(final_merged_major['S/N'].values) + np.std(final_merged_major['S/N'].values),
    alpha=0.25, color='grey')

ax0.axvline(x = np.median(merger_selection_1['S/N'].values), color=merger_color)
ax0.axvline(x = np.median(merger_selection_2['S/N'].values), color=sc_merger_color)
'''

plt.legend()
ax0.set_xlabel('S/N')

property = 'g_minus_r'

ax = fig.add_subplot(242)
_, bins = np.histogram(final_merged_major[property].values, bins=50)
#ax.hist(major_ack['r'].values, bins=bins, label='Parent', density=density, alpha=0.5, color='#546A7B', fill=False)
ax.hist(final_merged_major[property].values, bins=bins, 
    density=density, color='black', 
    weights=np.ones(len(final_merged_major[property].values)) / len(final_merged_major[property].values),
    fill=False)
ax.hist(merger_selection_1[property].values, bins=bins, 
    density=density, alpha=alpha, color=merger_color,
    weights=np.ones(len(merger_selection_1[property].values)) / len(merger_selection_1[property].values))
ax.hist(merger_selection_2[property].values, bins=bins, 
    density=density, alpha=alpha, color=sc_merger_color,
    weights=np.ones(len(merger_selection_2[property].values)) / len(merger_selection_2[property].values))#, fill=False, ls=':')

ax.set_xlabel(property)
#ax.set_xlim([13,19])

'''
# Also plot a mean for each
ax.axvline(x = np.median(final_merged_major['r'].values), color='black')
ax.axvspan(np.median(final_merged_major['r'].values) - np.std(final_merged_major['r'].values), 
    np.median(final_merged_major['r'].values) + np.std(final_merged_major['r'].values),
    alpha=0.25, color='grey')
ax.axvline(x = np.median(merger_selection_1['r'].values), color=merger_color)
ax.axvline(x = np.median(merger_selection_2['r'].values), color=sc_merger_color)
'''




ax1 = fig.add_subplot(243)
_, bins = np.histogram(final_merged_major['log_stellar_mass_from_color'].values, bins=50, range=[9,13])
#ax1.hist(final_merged_major['log_stellar_mass_from_color'].values, bins=bins, label='Parent', density=density, alpha=0.5, color='#546A7B', fill=False)
ax1.hist(final_merged_major['log_stellar_mass_from_color'].values, bins=bins, density=density, color='black', fill=False,
         weights=np.ones(len(final_merged_major['log_stellar_mass_from_color'].values)) / len(final_merged_major['log_stellar_mass_from_color'].values))
ax1.hist(merger_selection_1['log_stellar_mass_from_color'].values, bins=bins, density=density, alpha=alpha, color=merger_color,
         weights=np.ones(len(merger_selection_1['log_stellar_mass_from_color'].values)) / len(merger_selection_1['log_stellar_mass_from_color'].values))
ax1.hist(merger_selection_2['log_stellar_mass_from_color'].values, bins=bins, density=density, alpha=alpha, color=sc_merger_color,
         weights=np.ones(len(merger_selection_2['log_stellar_mass_from_color'].values)) / len(merger_selection_2['log_stellar_mass_from_color'].values))#, fill=False, ls=':')
ax1.set_xlabel('log stellar mass')
ax1.set_xlim([9,13])

'''
# Also plot a mean for each
ax1.axvline(x = np.median(final_merged_major['log_stellar_mass_from_color'].values), color='black')
ax1.axvspan(np.median(final_merged_major['log_stellar_mass_from_color'].values) - np.std(final_merged_major['log_stellar_mass_from_color'].values), 
    np.median(final_merged_major['log_stellar_mass_from_color'].values) + np.std(final_merged_major['log_stellar_mass_from_color'].values),
    alpha=0.25, color='grey')
ax1.axvline(x = np.median(merger_selection_1['log_stellar_mass_from_color'].values), color=merger_color)
ax1.axvline(x = np.median(merger_selection_2['log_stellar_mass_from_color'].values), color=sc_merger_color)
'''

ax2 = fig.add_subplot(244)
_, bins = np.histogram(final_merged_major['z'].values, bins=50)
#ax2.hist(final_merged_major['z'].values, bins=bins, label='Parent', density=density, alpha=0.5, color='#546A7B', fill=False)
ax2.hist(final_merged_major['z'].values, bins=bins, density=density, color='black', fill=False,
         weights=np.ones(len(final_merged_major['z'].values)) / len(final_merged_major['z'].values))
ax2.hist(merger_selection_1['z'].values, bins=bins, density=density, alpha=alpha, color=merger_color,
         weights=np.ones(len(merger_selection_1['z'].values)) / len(merger_selection_1['z'].values))
ax2.hist(merger_selection_2['z'].values, bins=bins, density=density, alpha=alpha, color=sc_merger_color,
         weights=np.ones(len(merger_selection_2['z'].values)) / len(merger_selection_2['z'].values))#, fill=False, ls=':')
ax2.set_xlabel('redshift')
ax2.set_xlim([0,0.5])

'''
# Also plot a mean for each
ax2.axvline(x = np.median(final_merged_major['z'].values), color='black')
ax2.axvspan(np.median(final_merged_major['z'].values) - np.std(final_merged_major['z'].values), 
    np.median(final_merged_major['z'].values) + np.std(final_merged_major['z'].values),
    alpha=0.25, color='grey')
ax2.axvline(x = np.median(merger_selection_1['z'].values), color=merger_color)
ax2.axvline(x = np.median(merger_selection_2['z'].values), color=sc_merger_color)
'''

property = 'Gini'
ax3 = fig.add_subplot(245)
_, bins = np.histogram(final_merged_major[property].values, bins=50)
#ax.hist(major_ack['r'].values, bins=bins, label='Parent', density=density, alpha=0.5, color='#546A7B', fill=False)
ax3.hist(final_merged_major[property].values, bins=bins, 
    label=f'Parent (# = {len(final_merged_major)})', 
    density=density, color='black', fill=False)
ax3.hist(merger_selection_1[property].values, bins=bins, 
    label=f'Minor (# = {len(merger_selection_1)})', 
    density=density, alpha=alpha, color=merger_color)
ax3.hist(merger_selection_2[property].values, bins=bins, 
    label=f'Minor prec (# = {len(merger_selection_2)})', 
    density=density, alpha=alpha, color=sc_merger_color)#, fill=False, ls=':')

'''
# Also plot a mean for each
ax3.axvline(x = np.median(final_merged_major['S/N'].values), color='black')
# What about spanning the uncertainty in the overall?
ax3.axvspan(np.median(final_merged_major['S/N'].values) - np.std(final_merged_major['S/N'].values), 
    np.median(final_merged_major['S/N'].values) + np.std(final_merged_major['S/N'].values),
    alpha=0.25, color='grey')

ax3.axvline(x = np.median(merger_selection_1['S/N'].values), color=merger_color)
ax3.axvline(x = np.median(merger_selection_2['S/N'].values), color=sc_merger_color)
'''

ax3.set_xlabel(property)

property = 'Asymmetry (A)'

ax4 = fig.add_subplot(246)
_, bins = np.histogram(final_merged_major[property].values, bins=50)
#ax.hist(major_ack['r'].values, bins=bins, label='Parent', density=density, alpha=0.5, color='#546A7B', fill=False)
ax4.hist(final_merged_major[property].values, bins=bins, 
    density=density, color='black', fill=False)
ax4.hist(merger_selection_1[property].values, bins=bins, 
    density=density, alpha=alpha, color=merger_color)
ax4.hist(merger_selection_2[property].values, bins=bins, 
    density=density, alpha=alpha, color=sc_merger_color)#, fill=False, ls=':')

ax4.set_xlabel(property)

'''
# Also plot a mean for each
ax.axvline(x = np.median(final_merged_major['r'].values), color='black')
ax.axvspan(np.median(final_merged_major['r'].values) - np.std(final_merged_major['r'].values), 
    np.median(final_merged_major['r'].values) + np.std(final_merged_major['r'].values),
    alpha=0.25, color='grey')
ax.axvline(x = np.median(merger_selection_1['r'].values), color=merger_color)
ax.axvline(x = np.median(merger_selection_2['r'].values), color=sc_merger_color)
'''


property = 'Concentration (C)'
ax5 = fig.add_subplot(247)
_, bins = np.histogram(final_merged_major[property].values, bins=50)
#ax1.hist(final_merged_major['log_stellar_mass_from_color'].values, bins=bins, label='Parent', density=density, alpha=0.5, color='#546A7B', fill=False)
ax5.hist(final_merged_major[property].values, bins=bins, density=density, color='black', fill=False)
ax5.hist(merger_selection_1[property].values, bins=bins, density=density, alpha=alpha, color=merger_color)
ax5.hist(merger_selection_2[property].values, bins=bins, density=density, alpha=alpha, color=sc_merger_color)#, fill=False, ls=':')
ax5.set_xlabel(property)

'''
# Also plot a mean for each
ax1.axvline(x = np.median(final_merged_major['log_stellar_mass_from_color'].values), color='black')
ax1.axvspan(np.median(final_merged_major['log_stellar_mass_from_color'].values) - np.std(final_merged_major['log_stellar_mass_from_color'].values), 
    np.median(final_merged_major['log_stellar_mass_from_color'].values) + np.std(final_merged_major['log_stellar_mass_from_color'].values),
    alpha=0.25, color='grey')
ax1.axvline(x = np.median(merger_selection_1['log_stellar_mass_from_color'].values), color=merger_color)
ax1.axvline(x = np.median(merger_selection_2['log_stellar_mass_from_color'].values), color=sc_merger_color)
'''
property = 'Shape Asymmetry (A_S)'
ax6 = fig.add_subplot(248)
_, bins = np.histogram(final_merged_major[property].values, bins=50)
#ax2.hist(final_merged_major['z'].values, bins=bins, label='Parent', density=density, alpha=0.5, color='#546A7B', fill=False)
ax6.hist(final_merged_major[property].values, bins=bins, density=density, color='black', fill=False)
ax6.hist(merger_selection_1[property].values, bins=bins, density=density, alpha=alpha, color=merger_color)
ax6.hist(merger_selection_2[property].values, bins=bins, density=density, alpha=alpha, color=sc_merger_color)#, fill=False, ls=':')
ax6.set_xlabel(property)
#ax6.set_xlim([0,0.5])

'''
# Also plot a mean for each
ax2.axvline(x = np.median(final_merged_major['z'].values), color='black')
ax2.axvspan(np.median(final_merged_major['z'].values) - np.std(final_merged_major['z'].values), 
    np.median(final_merged_major['z'].values) + np.std(final_merged_major['z'].values),
    alpha=0.25, color='grey')
ax2.axvline(x = np.median(merger_selection_1['z'].values), color=merger_color)
ax2.axvline(x = np.median(merger_selection_2['z'].values), color=sc_merger_color)
'''

plt.show()

#plt.savefig('../Figures/properties_'+str(run)+'_'+str(suffix)+'.png', dpi=1000)








