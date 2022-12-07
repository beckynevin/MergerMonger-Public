# This code makes figure 14 in the paper
# (the prior plot, or how the posterior varies with the prior)

# import modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

mass_complete = True
thresh = 0.5
type_marginalized = '_flags_cut_segmap_test'

# prefix where all tables go to die
prefix = '/Users/rnevin/CfA_Laptop/Documents/CfA_Code/MergerMonger-dev/Tables/'


table = pd.io.parsers.read_csv(prefix + 'LDA_out_all_SDSS_predictors_all_classifications_flags.txt', sep = '\t')
# get rid of flagged galaxies
table = table[(table['low S/N'] ==0) & (table['outlier predictor']==0) & (table['segmap']==0)]


table = pd.io.parsers.read_csv(prefix + 'LDA_out_all_SDSS_predictors_all_priors_percentile_all_classifications_flags.txt', sep = '\t')
print(table.columns)


print('length photometrically clean sample', len(table))
# also load in the mass completeness table:
mass = 'log_stellar_mass_from_color'
red = 'z'
spacing_z = 0.02
completeness = 95
suffix = str(spacing_z)+'_'+str(red)+'_'+str(mass)+'_completeness_'+str(completeness)

# Check if this table ^ even exists:
masstable = pd.io.parsers.read_csv(filepath_or_buffer = prefix + 'all_mass_color_complete_'+str(suffix)+'.txt',header=[0],sep='\t')[['ID']]

# One option is to additionally cross-match the above with 
if mass_complete:
    table = table.merge(masstable, on='ID')
    print('length of mass complete', len(table))


# Now do the merger fractions
merger_type_list = ['major_merger','major_merger_early','major_merger_late','major_merger_prec','major_merger_postc_include_coal_0.5','major_merger_postc_include_coal_1.0',
    'minor_merger','minor_merger_early','minor_merger_late','minor_merger_prec','minor_merger_postc_include_coal_0.5','minor_merger_postc_include_coal_1.0']

merger_type_list = ['major_merger_prec','minor_merger_prec']
label_list = ['Major merger prec', 'Minor merger prec']

merger_type_list = ['major_merger','minor_merger']
label_list = ['Major merger','Minor merger']
color_list = ['#022B3A','#1F7A8C']
color_list = ['#ED254E','#F9DC5C']


####### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Everything below is to answer Joe's question
# about minor merger contamination in the major 
# merger set.
# First, I'm looking at all minor mergers
# with a p_merg value above 0.5, looking at the
# p_merg,maj values for these galaxies
# and seeing if they are all also mergers.
table_priors = pd.io.parsers.read_csv(prefix + 'LDA_out_all_SDSS_predictors_all_priors_percentile_all_classifications_flags.txt', sep = '\t')
print(table_priors.columns)

# First select all galaxies with p_merg_stat_50_minor_merger > 0.5
minor = table_priors[table_priors['p_merg_stat_50_minor_merger'] > 0.5]
print(len(minor) / len(table_priors))

plt.clf()
plt.scatter(minor['p_merg_stat_50_minor_merger'].values, 
    minor['p_merg_stat_50_major_merger'].values, s=0.1)
plt.xlabel('p minor')
plt.ylabel('p major')
plt.show()


####### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This part goes through and looks at the clean and
# not clean merger fractions when we use different
# threshold values

# !!!!!!!!!! MESSY
table = table#table_priors

for i, merger in enumerate(merger_type_list):
    col = 'p_merg_stat_50_'+merger
    
    print(merger)
    thresh_list = np.linspace(0.05,0.95,100)


    for merger_other in merger_type_list:
        if merger == merger_other:
            continue
        
        print(f'after removing all galaxies that have a higher probability of being a {merger_other}')
        col_compare = 'p_merg_stat_50_'+merger_other
        
        total_merger_fraction = []
        clean_merger_fraction = []
    
        for thresh in thresh_list:
            tot_frac = len(table[(table[col] > thresh)])/len(table)
            frac = len(table[(table[col] > thresh) & (table[col] > table[col_compare])])/len(table)
            #print('thresh', round(thresh,2), round(frac,2))
            total_merger_fraction.append(tot_frac)
            clean_merger_fraction.append(frac)
        plt.scatter(thresh_list, total_merger_fraction, 
            color = color_list[i], label=merger+' total', s=3)
        plt.scatter(thresh_list, clean_merger_fraction, 
            color = color_list[i], label=merger+' clean', marker='s', s=3)
plt.axvline(x=0.5, color='black')
plt.legend()
plt.xlabel(r'$p_{\mathrm{merg}}$ threshold')
plt.ylabel('Merger fraction')
plt.show()
         
####### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


for i, merger in enumerate(merger_type_list):
    col = 'p_merg_stat_50_'+merger
    print('all merger fraction')
    print(merger)
    print(round(len(table[table[col] > 0.5])/len(table),2))
    
    for merger_other in merger_type_list:
        if merger == merger_other:
            continue
        else:
            print(f'after removing all galaxies that have a higher probability of being a {merger_other}')
            col_compare = 'p_merg_stat_50_'+merger_other
            print(round(len(table[(table[col] > 0.5) & (table[col] > table[col_compare])])/len(table),2))
            
    # Also go through and find 
    # first go through and load up all of prior files
    list_of_prior_files = glob.glob(prefix + 'change_prior/LDA_out_all_SDSS_predictors_'+str(merger)+'_0.*'+str(type_marginalized)+'.txt')
    print('length of prior files', len(list_of_prior_files))
    print(list_of_prior_files)
    
    if len(list_of_prior_files) ==0:
        print('there are no priors prepared')
        name_single = prefix + 'LDA_out_all_SDSS_predictors_'+str(merger)+'_flags.txt'
        table = pd.io.parsers.read_csv(filepath_or_buffer=name_single,header=[0],sep='\t')[['ID','p_merg']]
    else:
        prior_list = []
        
        for p in range(len(list_of_prior_files)):
            print('p', p)
            
            prior_file = pd.io.parsers.read_csv(filepath_or_buffer=list_of_prior_files[p],header=[0],sep='\t')
            
            prior_list.append(float(str(list_of_prior_files[p].split(merger+'_')[1]).split('_flags')[0]))
           
            
            # cut it way down
            if p == 0:
                prior = prior_file[['ID','p_merg']]

            else:
                prior_p = prior_file[['p_merg']] # just take p_merg if not the last one
                prior_p.columns = ['p_merg_'+str(p)]
                # Now stack all of these tables
                prior = prior.join(prior_p)
        
        
        
        
        df_select = table[['ID']] # just take this because you don't need the other deets
            

        merged = df_select.merge(prior, on = 'ID')#left_on='ID', right_on='objID')
        # for each column of p_merg, calculate the the f_merg and then find the median
        
        
        gt = (merged > 0.5).apply(np.count_nonzero)
        
    

        #fmerg_here = len(np.where(merged['p_merg_x'] > 0.5)[0])/len(merged)
        
        #f_merg[centers_z[i]].append(fmerg_here)
        f_out_list = gt.values[1:]/len(merged)
        print('f_out_list', f_out_list)
        print('f_merg from priors', np.median(gt.values[1:]/len(merged)))
        print('pm', np.std(gt.values[1:]/len(merged)))
        
        overall_f_merg_median = np.median(gt.values[1:]/len(merged))
        overall_f_merg_std = np.std(gt.values[1:]/len(merged))
        
        plt.scatter(prior_list, f_out_list, color = color_list[i], 
                    label = f'{label_list[i]} ($f_{{\mathrm{{merg}}}} = ${round(overall_f_merg_median,2)}$\pm${round(overall_f_merg_std,2)})')
        
        y_up = overall_f_merg_median + overall_f_merg_std# for x in prior_list]
        y_down = overall_f_merg_median - overall_f_merg_std# for x in prior_list]
        plt.fill_betweenx([y_down,y_up], 0.0, 0.55, color = color_list[i], alpha=.3)
        plt.axhline(y = overall_f_merg_median, color = color_list[i])
        #plt.fill_between(prior_list, y_down, y_up, color = color_list[i], alpha=0.5)

plt.xlabel(r'Prior probability, $\pi_i$')
plt.ylabel(r'Posterior probability, $f_{\mathrm{merg}}$')
plt.legend()
#plt.xlim([0.05,0.5])
#plt.ylim([0.05,0.5])


plt.show()
STOP 

# Is there a way to do this but to make sure its the greatest of all?

merger_type_list = ['major_merger','minor_merger']
merger_type_list = ['major_merger_early','major_merger_late','major_merger_postc_include_coal_1.0']
merger_type_list = ['minor_merger_early','minor_merger_late','minor_merger_postc_include_coal_1.0']

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

for merger in merger_type_list:
    col = 'p_merg_'+merger
    print('all merger fraction')
    print(merger)
    df_merg = table[table[col] > thresh]
    
    
    print(round(len(df_merg)/len(table),2))
    
    # Now go through and drop all columns that 
    
    for merger_other in merger_type_list:
        if merger == merger_other:
            continue
        else:
            col_compare = 'p_merg_'+merger_other
            
            df_merg = df_merg[df_merg[col] > df_merg[col_compare]].dropna(how='any')
            print('fraction', round(len(df_merg)/len(table),2), 'after dropping', merger_other)
            
            