# Compares stellar masses from various different methods
# Creates mass_comparison.txt table
# Creates figure 4 
# And finally, creates all_mass_measurements.txt

from astropy.cosmology import FlatLambdaCDM 
import astropy.units as u
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import sem
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# What information do we need?
# 1. photometric redshift
# 2. g band mag
# 3. r band mag

# enter the path where the tables live:
prefix = '/Users/rnevin/Documents/MergerMonger-Public/tables/'
if os.path.exists(prefix + 'mass_comparisons.txt'):
	table = pd.io.parsers.read_csv(filepath_or_buffer=prefix + 'mass_comparisons.txt',header=[0],sep='\t')
	print(table.columns)
	# columns are:
	# objID
	# g_minus_r = the difference of the two below quantities:
	# g = po.g from the photoobj catalog, I think this is in mag?
	# r = po.g from the photoobj catalog
	# z_x = redshift from photoZ catalog
	# dr7objid
	# z_y = redshift from mendel
	# logMt = log10 of total mass from mendel
	print('length of table', len(table))
else: 
	# This is how I merge a bunch of tables together to make the above table
	# None of this is necessary if you have the above mass_comparisons.txt table saved

	type_gal = 'predictors'
	run = 'major_merger'
	suffix = 'orig'

	# First, import the mendel stuff:

	cols = ['dr7objid','z','logMt','b_logMt','B_logMt','logMb','b_logMb','B_logMb','logMd','b_logMd','B_logMd','zmin','zmax','PpS','type','dBD']
	mendel = pd.io.parsers.read_csv(filepath_or_buffer=prefix + 'mendel_table4.dat', delim_whitespace=True)
	mendel.columns = cols
	mendel_cols = mendel[['dr7objid','z','logMt','logMb','logMd','dBD']]
	#mendel = pd.DataFrame(mendel_in, columns=cols)
	mendel_select = mendel_cols[mendel_cols['dBD'] < 1]
	#dBD is in units of standard error

	mendel_select['logBD'] = np.log10(10**mendel_select['logMb'] + 10**mendel_select['logMd'])
	



	crossmatch = pd.io.parsers.read_csv(filepath_or_buffer=dir+'crossmatch_dr8_dr7_beckynevin.csv',sep=',', header=[0])
	# columns are: 'objID', 'dr7objid'

	print(crossmatch)

	merged_dr16 = mendel_select.merge(crossmatch, on='dr7objid')
	print('len merged dr16', merged_dr16)

	# Next, import the info you need to derive your own stellar masses:
	zs = pd.io.parsers.read_csv(filepath_or_buffer=dir+'z_g_r_kcorrected_beckynevin_0.csv',sep=',', header=[0])
	# columns are: 'objID', 'dr7objid'

	merged = zs.merge(merged_dr16, on='objID')
	print('len new table merged with mendel', len(merged))

	# Finally, cross-match with your sample:



	df_LDA = pd.io.parsers.read_csv(filepath_or_buffer=dir+'LDA_out_all_SDSS_'+type_gal+'_'+run+'_flags.txt',header=[0],sep='\t')

	# Because the df_LDA doesn't have the final flag, use the predictor table to instead clean via merging

	# Run OLS with predictor values and z and stellar mass and p_merg:
	df_predictors = pd.io.parsers.read_csv(filepath_or_buffer=dir+'SDSS_predictors_all_flags_plus_segmap.txt',header=[0],sep='\t')

	if len(df_LDA) != len(df_predictors):
		print('these have different lengths cannot use one to flag')
		STOP

	# First clean this so that there's no segmap flags
	df_predictors_clean = df_predictors[(df_predictors['low S/N'] ==0) & (df_predictors['outlier predictor']==0)]#& (df_predictors['segmap']==0)]

	clean_LDA = df_LDA[df_LDA['ID'].isin(df_predictors_clean['ID'].values)]

	clean_LDA = clean_LDA[['ID','p_merg']]

	table = merged.merge(clean_LDA, left_on = 'objID', right_on = 'ID')

	# For this table, you have all the photometric info, info from mendel (to check), and LDA classifications
	print('length of final table', len(table))

	'''
	print()

	plt.clf()
	plt.scatter(table['logMt'].values, table['logBD'].values, s=0.1, color='#12664F')

	xs = np.linspace(7,13,1000)
	plt.plot(xs, xs, color='#F4845F')

	plt.xlabel('logMt')
	plt.ylabel('logMb + logMd')
	plt.xlim([8,12])
	plt.ylim([8,12])

	plt.show()
	
	'''
	table.to_csv(dir+'mass_comparisons.txt', sep='\t')

'''
# Do a quick number comparison for how many of our galaxies have Mendel masses versus how many have masses from the color approach:

type_gal = 'predictors'
run = 'major_merger'
suffix = 'orig'

df_LDA = pd.io.parsers.read_csv(filepath_or_buffer=dir+'LDA_out_all_SDSS_'+type_gal+'_'+run+'_flags.txt',header=[0],sep='\t')

# Because the df_LDA doesn't have the final flag, use the predictor table to instead clean via merging

# Run OLS with predictor values and z and stellar mass and p_merg:
df_predictors = pd.io.parsers.read_csv(filepath_or_buffer=dir+'SDSS_predictors_all_flags_plus_segmap.txt',header=[0],sep='\t')

if len(df_LDA) != len(df_predictors):
	print('these have different lengths cannot use one to flag')
	STOP

# First clean this so that there's no segmap flags



df_predictors_clean = df_predictors[(df_predictors['low S/N'] ==0) & (df_predictors['outlier predictor']==0)]#& (df_predictors['segmap']==0)]

clean_LDA = df_LDA[df_LDA['ID'].isin(df_predictors_clean['ID'].values)]

clean_LDA = clean_LDA[['ID','p_merg']]

print('length of LDA', len(clean_LDA))



# First, import the mendel stuff:

cols = ['dr7objid','z','logMt','b_logMt','B_logMt','logMb','b_logMb','B_logMb','logMd','b_logMd','B_logMd','zmin','zmax','PpS','type','dBD']
mendel = pd.io.parsers.read_csv(filepath_or_buffer=dir+'mendel_table4.dat', delim_whitespace=True)
mendel.columns = cols
mendel_cols = mendel[['dr7objid','z','logMt','logMb','logMd','dBD']]
#mendel = pd.DataFrame(mendel_in, columns=cols)
mendel_select = mendel_cols[mendel_cols['dBD'] < 1]

crossmatch = pd.io.parsers.read_csv(filepath_or_buffer=dir+'crossmatch_dr8_dr7_beckynevin.csv',sep=',', header=[0])
# columns are: 'objID', 'dr7objid'

print(crossmatch)

merged_dr16 = mendel_select.merge(crossmatch, on='dr7objid')

merged = clean_LDA.merge(merged_dr16, left_on='ID',right_on='objID')
print('len Mendel merged with our sample', len(merged))




zs = pd.io.parsers.read_csv(filepath_or_buffer=dir+'z_g_r_kcorrected_beckynevin_0.csv',sep=',', header=[0])
# columns are: 'objID', 'dr7objid'

merged = clean_LDA.merge(zs, left_on='ID',right_on='objID')
print('len color merged with our sample', len(merged))

STOP
'''


# Now, for every row in the table, go through and measure the stellar mass using the color-M/L ratio from Bell 2003:
# The formula:
# log10(M/L_lambda) = a_lambda + b_lambda * (color)
# units are solar for M/L_lambda, which I think means you need the luminosity relative to the sun (so its a ratio)
# color is in magnitudes?

# for the g-r color row and the r-band filter from Table 7 of Bell+2003
a_r = -0.306
b_r = 1.097

# the updated values from Du+2019:
a_r = -0.61
b_r =  1.19

# from zibetti+2009
a_r = -0.840
b_r = 1.654

# Luminosity = flux * d^2 because its measured at the outside of a sphere of light
# so to get the luminosity in terms of solar, you will need to know the flux ratio relative to the sun
# as well as the relative distance because:
# L_gal / L_sun = (F_gal / F_sun) * (d_gal / d_sun)**2
#               = 10**(-0.4(m_gal - m_sun)) * (d_gal / d_sun)**2

sun_d = 149597870700 * u.m # distance in meters to the sun
mag_sun_apparent = -27.05#-26.7 # is this the apparent mag in r_band?
#mag_sun_apparent_r = -27.05
mag_sun_absolute_AB = 4.67

number = 0#100 # the number of galaxies you're running through the for loop
if number != 0:
	table = table[0:number]

log_stellar_mass_from_color = []
log_stellar_mass_from_color_scott = []

for i in range(len(table)):
	z = table['z_x'].values[i] # this is the photometric redshift from the sdss photoZ catalog

	gal_d = cosmo.luminosity_distance(z).to(u.m) # distance to the galaxy in meters

	# I get an answer of 1.59e8 so within a factor of 2?
	#table['g_minus_r'].values[i]
	g_r = table['g'].values[i] - table['kcorrG'].values[i] - (table['r'].values[i] - table['kcorrR'].values[i])
	#print('compare g-r OG', table['g_minus_r'].values[i])
	#print('k corrected', g_r)
	mag_gal_r = table['r'].values[i] - table['kcorrR'].values[i]

	f_ratio = 10**(-0.4*(mag_gal_r - mag_sun_apparent))
	#print('flux ratio relative to sun', f_ratio)
	#print('distance ratio relative to sun', (gal_d/ sun_d)**2)
	L_ratio = f_ratio * (gal_d/ sun_d).value**2

	#print('L ratio',L_ratio)

	#print('log stellar mass',np.log10(10**(a_r + b_r*(g_r))*L_ratio))
	log_stellar_mass_from_color.append(np.log10(10**(a_r + b_r*(g_r))*L_ratio))

	#print('mendel mass', table['logMt'].values[i]) # this is the mass from mendel

	scott = a_r + b_r*(g_r) + 2. * np.log10(gal_d.to(u.pc).value) - 2.0 + 0.4*mag_sun_absolute_AB - 0.4*mag_gal_r
	#print('scott versus me')
	#print('log M/L', a_r + b_r*(g_r))
	#print(scott, np.log10(10**(a_r + b_r*(g_r))*L_ratio))
	log_stellar_mass_from_color_scott.append(scott)

	# method from scott:

	#log_M=logML+2.*log10(LD_pc)-2.0+0.4*M_Sun-0.4*m

	#where logML is logarithm of M/L, LD is the luminosity distance in pc, 
	#M_sun and m are the Sun's absolute AB magnitude and the apparent magnitude in whichever filter you are using, r i guess. Attached is a lookup table of Solar magnitudes that I find useful for this.

	# so it basically looks like
	#log M/L = color thing
	# M/L = 10**(color thing)
	# M = L*10**color
	# log M = log L + log 10**color
	# log M = log M/L + log L



	if i > number and number !=0:
		break


table['log_stellar_mass_from_color'] = log_stellar_mass_from_color
table['stellar_mass_from_color'] = [10**x for x in log_stellar_mass_from_color]
print('length of table bf dropping nans', len(table))
table = table.dropna()
print('length of table after dropping nans', len(table))

'''
# For comparing mine to scott's
plt.clf()
if number != 0:
	plt.scatter(log_stellar_mass_from_color, log_stellar_mass_from_color_scott)

	plt.plot(log_stellar_mass_from_color, log_stellar_mass_from_color)
else:
	plt.scatter(log_stellar_mass_from_color, log_stellar_mass_from_color_scott, s=0.5, c=table['z_x'].values, vmax=0.5)

	plt.plot(log_stellar_mass_from_color, log_stellar_mass_from_color, color='black')
#plt.colorbar(label='Photometric Z')
plt.xlabel('log mass color me')
plt.ylabel('log mass color Scott')
plt.show()
'''

spacing = 0.2#0.25
ranges = np.arange(7, 13, spacing)# did start at 8.5

mass_med = []
mass_med_log = []
mass_std = []
mass_std_log = []
mass_std_log_relative = []
mass_se = []

# Make bins
for i in range(len(ranges)-1):
	bin_start = ranges[i]
	bin_end = ranges[i+1]
	print('start', bin_start, 'stop', bin_end)
	# build dataset
	df = table[(table['logMt'] > bin_start) & (table['logMt'] < bin_end)]
	#print('df for this range', bin_start, bin_end)
	#print(df)
	#print(df['log_stellar_mass_from_color'].values)

	# Now for each of these go through all of the LDA tables and match and pull a merger fraction from this :)

	
	# Okay go through all of these, load them up, and match to the object IDs:
	#mass_med[round(bin_start+(bin_end-bin_start)/2,2)] = []
	med = np.nanmean(df['log_stellar_mass_from_color'].values)
	mass_med_log.append(med)
	mass_med.append(np.mean(df['stellar_mass_from_color'].values))
	#print('median stat', stat)

	#mass_std[round(bin_start+(bin_end-bin_start)/2,2)] = []
	y = np.mean(df['stellar_mass_from_color'].values.astype('float32'))
	
	dy = np.nanstd(df['stellar_mass_from_color'].values.astype('float32')) # / np.median(df['stellar_mass_from_color'].values)
	
	sem_log = sem(df['log_stellar_mass_from_color'].values)

	print('std log', np.nanstd(df['log_stellar_mass_from_color'].values))
	print('se log', sem_log)

	mass_se.append(sem_log)
	mass_std.append(dy)
	mass_std_log.append(np.nanstd(df['log_stellar_mass_from_color'].values))
	mass_std_log_relative.append(0.434 * (dy/y)) # 0.434*gotta do this to get the log error, 
	#see https://faculty.washington.edu/stuve/log_error.pdf

	print('doing this for not log')
	print('dy', dy)
	print('dy se', sem(df['stellar_mass_from_color'].values))
	print('y', y)
	print(0.434*dy/y)

	

	
	

'''
print(mass_med)
print(mass_std)
labels, data = mass_med.keys(), mass_med.values()
print(labels, data)
'''


#table = table[np.isfinite(table['logMt'])]
print(table)

plt.clf()
if number != 0:
	#plt.scatter(table['logMt'].values[0:number+2], log_stellar_mass_from_color_scott, color='#102542', s=0.5)
	
	# Estimate the 2D histogram
	nbins = 100
	try:
		H, xedges, yedges = np.histogram2d(table['logBD'].values[0:number+2], 
			table['log_stellar_mass_from_color'].values[0:number+2],bins=nbins)
	except:
		print(table['logMt'].values[0:number+2])
		print(table['log_stellar_mass_from_color'].values[0:number+2])
	# H needs to be rotated and flipped
	H = np.rot90(H)
	H = np.flipud(H)
	 
	# Mask zeros
	Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
	 
	# Plot 2D histogram using pcolor
	
	plt.pcolormesh(xedges,yedges,Hmasked)

	#plt.hist2d(table['logMt'].values[0:number+2], log_stellar_mass_from_color_scott,
	#		bins=100)
	

	y = [x for x in table['logBD'].values[0:number+2]] # straight line for comparison

	plt.plot(table['logBD'].values[0:number+2], y, color='#CDD7D6', zorder=50)
else:
	
	#plt.scatter(table['logMt'].values, log_stellar_mass_from_color, s=0.5,  color='#102542')#vmax=0.5)

	# Estimate the 2D histogram
	nbins = 100
	print(table['logBD'].values)
	print(table['log_stellar_mass_from_color'].values)
	STOP
	H, xedges, yedges = np.histogram2d(table['logBD'].values, 
		table['log_stellar_mass_from_color'].values,bins=nbins)
	 
	# H needs to be rotated and flipped
	H = np.rot90(H)
	H = np.flipud(H)
	 
	# Mask zeros
	Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
	 
	# Plot 2D histogram using pcolor
	
	plt.pcolormesh(xedges,yedges,Hmasked)

	plt.plot(table['logBD'].values, table['logBD'].values, color='#CDD7D6', zorder=50)
	# a better color?: #706C61 nickel

ranges_plot = [x + spacing for x in ranges]
ranges_plot_log = [10**x for x in ranges_plot]
print(np.shape(ranges_plot[:-1]), np.shape(mass_med))
plt.scatter(ranges_plot[:-1], mass_med_log, color='#FF9000', zorder=100, s=10)
three_sig = [3*x for x in mass_std_log]
standard_error = [1.96*x for x in mass_se] # 95% confidence interval
print('one sigma', mass_std)
print('means', mass_med)
print('three sigma', three_sig)
print('95 CI', standard_error)
plt.errorbar(ranges_plot[:-1], mass_med_log, yerr = mass_std_log, 
	color='#FF9000', linestyle='None', zorder=100, capsize=5)

#plt.errorbar(ranges_plot[:-1], mass_med_log, yerr = standard_error, 
#	color='#F87060', linestyle='None', zorder=100, capsize=5)
#plt.colorbar(label='Photometric Z')
#plt.colorbar()
plt.xlabel('log mass Mendel (B+D)')
plt.ylabel('log mass color')
plt.ylim([7,13])
plt.xlim([7,13])
plt.savefig('../Figures/comparing_masses_Zibetti.png', dpi=1000)


plt.clf()
notlogMt = [10**x for x in table['logMt'].values]
if number != 0:
	#plt.scatter(notlogMt[0:number+2], table['stellar_mass_from_color'].values, color='#102542', s=0.5)
	#plt.errorbar(mass_med, mass_)
	# Estimate the 2D histogram
	nbins = 100
	try:
		H, xedges, yedges = np.histogram2d(notlogMt[0:number+2], 
			table['stellar_mass_from_color'].values[0:number+2],bins=nbins)
	except:
		print(notlogMt[0:number+2])
		print(table['stellar_mass_from_color'].values[0:number+2])
	# H needs to be rotated and flipped
	H = np.rot90(H)
	H = np.flipud(H)
	 
	# Mask zeros
	Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
	 
	# Plot 2D histogram using pcolor
	
	plt.pcolormesh(xedges,yedges,Hmasked)

	#plt.hist2d(table['logMt'].values[0:number+2], log_stellar_mass_from_color_scott,
	#		bins=100)
	

	y = [x for x in notlogMt[0:number+2]] # straight line for comparison

	#plt.plot(notlogMt[0:number+2], y, color='#CDD7D6', zorder=50)
else:
	
	plt.scatter(notlogMt, table['stellar_mass_from_color'].values, s=0.5,  color='#102542')#vmax=0.5)

	# Estimate the 2D histogram
	nbins = 100
	H, xedges, yedges = np.histogram2d(notlogMt, 
		table['stellar_mass_from_color'].values,bins=nbins)
	 
	# H needs to be rotated and flipped
	H = np.rot90(H)
	H = np.flipud(H)
	 
	# Mask zeros
	Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
	 
	# Plot 2D histogram using pcolor
	
	#plt.pcolormesh(xedges,yedges,Hmasked)

	

	plt.plot(notlogMt, notlogMt, color='#CDD7D6', zorder=50)
	# a better color?: #706C61 nickel

ranges_plot = [x + spacing for x in ranges]
plt.errorbar(ranges_plot_log[:-1], mass_med, yerr = mass_std, 
	color='#F87060', linestyle='None', zorder=100)


plt.xlabel('mass Mendel')
plt.ylabel('mass color')
plt.ylim([10**7,10**12])
plt.xlim([10**7,10**12])
plt.savefig('../Figures/comparing_masses_notlog.png', dpi=1000)


# save this table
#table['log_stellar_mass_from_color_scott'] = log_stellar_mass_from_color_scott
print(table)
table.to_csv(dir+'all_mass_measurements.txt', sep='\t')

