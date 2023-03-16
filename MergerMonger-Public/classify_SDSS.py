'''
~~~
The wrapper for creating the LDA_out* tables for the full SDSS dataset
These tables already exist, so there's no need to run the classification again
The purpose of this is to show the mechanics behind the classification
~~~
'''
import MergerMonger as MM
import matplotlib.pyplot as plt
import os

run = 'major_merger'

'''
# Here's another option:
run = 'minor_merger_postc_include_coal_0.5'
# run_parent is required if you're running a post-coalescence classification and you
# want to include the snapshot of coalescence, see below
run_parent = 'minor_merger_postc'
'''
# Where the tables live
prefix = '/Users/rnevin/Documents/MergerMonger-Public/tables/'

verbose = True

if verbose:
	print(str(os.getcwd())+'../frames/')

LDA,RFR, df = MM.load_LDA_from_simulation(prefix, run, verbose=verbose)

# Some other options:
#LDA, RFR, df = MM.load_LDA_from_simulation_sliding_time(0.5, run_parent, verbose=verbose)
#LDA, RFR, df = MM.load_LDA_from_simulation_sliding_time_include_coal(prefix, 0.5, run_parent, verbose=verbose)

if verbose:
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Output from LDA~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	print('inputs', LDA[2])
	print('coefficients', LDA[3])
	print('intercept', LDA[4])
	print('accuracy, precision, and recall for simulated galaxies [5-7]', LDA[5], LDA[6], LDA[7])
	print('Standardized means LDA[0]', LDA[0])
	print('standardized stds LDA[1]', LDA[1])
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	print('~~~~~~~~~~~~~~~~~Output from RFR~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	print(RFR)
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	


# The output of this is in the format:
# 0 = standardized means on all of the coefficients
# 1 = standardized stds
# 2 = inputs
# 3 = coefficients
# 4 = intercept
# 5 = Accuracy
# 6 = Precision
# 7 = Recall
# 8 = LDA values of all simulated galaxies
# 9 = myr
# 10 = myr_non
# 11 = covariance matrix
# 12 = means of all classes



plt.clf()
plt.hist(LDA[8], bins=50, alpha=0.5)

plt.xlabel("LD1")
plt.axvline(x=0)

plt.title('LDA values from the simulation')
plt.show()

    

type_gal = 'predictors'
verbose='yes'


# Okay the below are a couple of options for how to run the classification
# All of them save a table, most of them have an option to only run on a limited number of galaxies
# to save time
'''
# This classification makes a larger interpretive table
LDA, p_merg, CDF = MM.classify_from_flagged_interpretive_table(prefix, 
	run, 
	LDA, 
	RFR, 
	df, 
	100, # only run on 100 galaxies
	verbose=False, 
	all = True, 
	cut_flagged = False)
'''

# Other options include this that has a flag option to only classify the non-flagged galaxies
LDA, p_merg, CDF = MM.classify_from_flagged('../Tables/','../frames/', run, LDA, RFR, df, 100, 
	verbose=True, all = True, cut_flagged = False)

# No extra bells and whistles:
#LDA, p_merg, CDF = classify('../Tables/','../frames',type_gal, run, LDA, RFR, df, verbose=False)
    

# Now plot the LDA values from the classified galaxies
plt.clf()
plt.hist(LDA, bins=50, alpha=0.5)

plt.xlabel("LD1")
plt.axvline(x=0)

plt.title('LDA values from the classified galaxies')
plt.show()
