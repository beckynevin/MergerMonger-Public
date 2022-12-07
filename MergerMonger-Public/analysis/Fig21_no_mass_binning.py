#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# To test if not binning by mass matters
# also to test if doing this for the
# not mass complete sample matters

# This is the fast version
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
from cv2 import ellipse2Poly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table


# path
dir = '/Users/rebeccanevin/Documents/CfA_Code/MergerMonger-dev/Tables/'




# This is to load up the mass complete table:
mass = 'log_stellar_mass_from_color'
red = 'z_spec'
spacing_z = 0.02
complete = True
completeness = 95
suffix = str(spacing_z)+'_'+str(red)+'_'+str(mass)+'_completeness_'+str(completeness)

if complete:
	add_on_binned_table = ''
	

	# Check if this table ^ even exists:
	if os.path.exists(dir+'all_mass_color_complete_'+str(suffix)+'.txt'):
		print('it exists! you can run the f_merg analysis')
	else:
		print('missing mass table to run this analysis')
else:
	add_on_binned_table = 'incomplete'
	# Check if this table ^ even exists:
	if os.path.exists(dir+'all_mass_'+str(suffix)+'.txt'):
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
mass_limit = 8




table_name = '../Tables/f_merg_no_mass_bins_'+str(run)+'_'+str(suffix)+'_'+str(add_on_binned_table)+'.csv'

if os.path.exists(table_name) and save_df:
	print('table already exists, changing save_df to False so we dont oversave')
	#save_df = False
	



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
	'logBD','log_stellar_mass_from_color']]
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
		'p_merg']]
else:
	merged_1 = merged_1[['ID','z','log_stellar_mass_from_color',
		'p_merg']]


#merged_1.drop(merged_1.filter(regex='_y$').columns, axis=1, inplace=True)

final_merged = merged_1.merge(df_predictors_clean, on='ID')
if red == 'z_spec':
	final_merged = final_merged[['ID','z_x','z_spec','logBD','log_stellar_mass_from_color',
		'p_merg','S/N']]
else:
	final_merged = final_merged[['ID','z','log_stellar_mass_from_color',
		'p_merg','S/N']]
print('len merged with mendel', len(final_merged))







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




centers_z = [(bins_z[x+1] - bins_z[x])/2 + bins_z[x] for x in range(len(bins_z)-1)]

print('~~~~~~~~~~~~~~')
print(bins_z)
print(centers_z)

# Now that you have the bins in both dimensions, make a figure of your different bins:
plt.clf()
'''
plt.scatter(df_pred_merged['z_x'].values, df_pred_merged['log_stellar_mass_from_color'].values, color='orange', s=0.2)
plt.annotate(str(len(df_pred_merged['z_x'].values)), 
	xy = (np.mean(df_pred_merged['z_x'].values), np.mean(df_pred_merged['log_stellar_mass_from_color'].values)), 
	xycoords='data', color='black')
'''
for i in range(len(bins_z)-1):
	bin_start_z = bins_z[i]
	bin_end_z = bins_z[i+1]
	df_select = final_merged[(final_merged[red] > bin_start_z) 
		& (final_merged[red] < bin_end_z) 
		& (final_merged[mass] > mass_limit)]
	plt.scatter(df_select[red].values, df_select[mass].values, 
		s=0.2)
	plt.annotate(str(len(df_select[red].values)), 
		xy = (np.mean(df_select[red].values) - 0.005, 
			np.mean(df_select[mass].values - 0.05)), 
		xycoords='data', color='black')
plt.xlabel(r'$z$')
plt.ylabel(mass)
plt.show()






# try to load up the df of mass centers:
if save_df:
	# first go through and load up all of prior files
	list_of_prior_files = glob.glob('../Tables/change_prior/LDA_out_all_SDSS_predictors_'+str(run)+'_0.*'+str(type_marginalized)+'.txt')
	print('length of prior files', len(list_of_prior_files))
	table_list = []
	for p in range(len(list_of_prior_files)):
		
		prior_file = pd.io.parsers.read_csv(filepath_or_buffer=list_of_prior_files[p],header=[0],sep='\t')
		# cut it way down
		if p == 0:
			table = prior_file[['ID','p_merg']]

		else:
			table_p = prior_file[['p_merg']] # just take p_merg if not the last one
			table_p.columns = ['p_merg_'+str(p)]
			table = table.join(table_p)
	# Now stack all of these tables
	print(table)

	# Now that these are all joined together, identify which ones have IDs that match mergers
	
	count = {}
	f_merg = {}
	f_merg_avg = {}
	f_merg_std = {}
	for i in range(len(bins_z)-1):
		bin_start_z = bins_z[i]
		bin_end_z = bins_z[i+1]
		print('start z ', bin_start_z, 'stop z ', bin_end_z)
		
		df_select = final_merged[(final_merged[red] > bin_start_z) 
					& (final_merged[red] < bin_end_z) 
					& (final_merged[mass] > mass_limit)]
		df_select = df_select[['ID']] # just take this because you don't need the other deets
			
		count[centers_z[i]] = len(df_select)


		# Now for each of these go through all of the LDA tables and match and pull a merger fraction from this :)

		
	

		# used to be in loop
		merged = table.merge(df_select, on = 'ID')#left_on='ID', right_on='objID')
		# for each column of p_merg, calculate the the f_merg and then find the median
		

		gt = (merged > 0.5).apply(np.count_nonzero)
		

		#fmerg_here = len(np.where(merged['p_merg_x'] > 0.5)[0])/len(merged)
		
		#f_merg[centers_z[i]].append(fmerg_here)
		f_merg_avg[centers_z[i]] = np.median(gt.values[1:]/len(merged))
		f_merg_std[centers_z[i]] = np.std(gt.values[1:]/len(merged))


	# find a way to put this into df format

	z_val = []
	f_merg_val = []
	f_merg_e_val = []
	count_val = []
	for i in range(len(bins_z)-1):
		
		f_merg_val.append(f_merg_avg[centers_z[i]])
		f_merg_e_val.append(f_merg_std[centers_z[i]])
		z_val.append(centers_z[i])
		count_val.append(count[centers_z[i]])
	# Now make a df out of these lists
	
	df_fmerg = pd.DataFrame(list(zip(z_val, f_merg_val, f_merg_e_val, count_val)),
           columns =['z', 'fmerg', 'fmerg_std', 'count'])
	
	df_fmerg.to_csv(table_name, sep='\t')
else:
	df_fmerg = pd.read_csv(table_name, sep = '\t')

	
df_fmerg = df_fmerg.dropna()
print(df_fmerg)


# Do a 2D regression:
X = df_fmerg[['z', 'count']] 
y = df_fmerg['fmerg']
## fit a OLS model with intercept on mass and z
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
print(est.summary())


count = df_fmerg['count'].values

max_count = max(count)
print('also max?', max_count)



plt.clf()

# there should be 8 different redshifts
colors = ['#493843','#61988E','#A0B2A6','#CBBFBB','#EABDA8','#FF9000','#DE639A','#D33E43']
colors = ['#C5979D','#E7BBE3','#78C0E0','#449DD1','#3943B7','#150578','#0E0E52','black']
colors = ['#7D7C7A','#DEA47E','#AD2831','#800E13','#640D14','#38040E','#250902','black',
	'#7D7C7A','#DEA47E','#AD2831','#800E13','#640D14','#38040E','#250902','black',
	'#7D7C7A','#DEA47E','#AD2831','#800E13','#640D14','#38040E','#250902','black']

print('centers of zs', centers_z)



plt.clf()
# Make the same plot but for redshift on the x axis:


# Grab everything with a z value from the df
df_select = df_fmerg.dropna().reset_index()

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
x = x[count > 1000]
#x = np.ma.masked_where(count < 1000, x)
X = sm.add_constant(x)

y = df_select['fmerg']

# mask y where count is less than 1000
print(f'y before, {y}')
y = y[count > 1000].reset_index(drop=True)
#np.ma.masked_where(count < 1000, y)


res_bb = sm.OLS(y, X).fit()#, missing = 'drop'
_, data_bb, _ = summary_table(res_bb, alpha=0.05)

big_boy_fit = data_bb[:,2]



error = df_select['fmerg_std']
error = error[count > 1000].reset_index(drop=True)

print(f'y after, {y}, xs {x}, errors {error}')


plt.clf()
mu, sigma = 0, 1 # mean and standard deviation

# iterate
# save slope values
slope_list = []
for num in range(100):
	Y = [y[i] + error[i] * np.random.normal(mu, sigma, 1) for i in range(len(y))]

	#scaler = StandardScaler()
	#scaler.fit(X)
	#X_standardized = scaler.transform(X)


	res = sm.OLS(Y, X).fit()
	
	#plt.scatter(x, Y, s=0.1, color=colors[color_count])
	try:
		slope_list.append(res.params[1])
	except:
		continue
		slope_list.append(999)
	
	

	st, data, ss2 = summary_table(res, alpha=0.05)
	fittedvalues = data[:,2]
	predict_mean_se  = data[:,3]
	predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
	predict_ci_low, predict_ci_upp = data[:,6:8].T

	#print(summary_table(res, alpha=0.05))
	

	

	plt.plot(x, fittedvalues, color = 'grey', alpha=0.5)#, label='OLS')
#plt.plot(x, predict_ci_low, 'b--')
#plt.plot(x, predict_ci_upp, 'b--')
#plt.plot(x, predict_mean_ci_low, 'g--')
#plt.plot(x, predict_mean_ci_upp, 'g--')
plt.scatter(x, y, color='black', label='data', zorder=100)
plt.plot(x, big_boy_fit, color='#C3423F',  zorder=100)

plt.errorbar(x, y, yerr = error, color='black',linestyle='None',  zorder=100, capsize=5)
#for (count, x, y) in zip(count, x, y):
#	plt.annotate(str(count), xy = (x, y+0.07), xycoords='data')

plt.xlabel(r'$z$ bins')
plt.ylabel(r'f$_{\mathrm{merg}}$')
if red == 'z':
	z_label = '$z_{\mathrm{phot}}$'
else:
    z_label = '$z_{\mathrm{spec}}$'
if completeness:
	plt.title(f'no mass bins, mass complete, '+z_label)
else:
    plt.title(f'no mass bins, mass incomplete, '+z_label)
#plt.legend()
#plt.annotate(str(round(res_bb.params[1],2)), 
#		xy=(0.01,0.1), xycoords='axes fraction')
plt.ylim([0,0.6])

plt.annotate(f'MCMC slope = {round(np.mean(slope_list),2)} +/- {round(np.std(slope_list),2)}', 
	xy=(0.01,0.05), xycoords='axes fraction', color='black')#colors[color_count])
if savefigs:
	plt.savefig('../Figures/no_mass_binning_'+str(run)+'_'+str(red)+'_'+str(mass)+'_'+str(complete)+'_'+str(mass_limit)+'.png', dpi=1000)
else:
	plt.show()

print('you did it!!!')