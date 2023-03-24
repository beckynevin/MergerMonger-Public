# A folder to hold all of the tables
So because Github storage is limited and some tables are >50 Mb, I haven't saved all of the tables here. If they aren't here, you can find them on Zenodo: <a href="https://zenodo.org/record/7438610#.ZB4HNOzMJ45">https://zenodo.org/record/7438610#.ZB4HNOzMJ45</a>

There are a couple of different types of tables that are either already here or that you'll want to grab from the Zenodo link and stash in this folder:
1) The simulation_classifications/ folder contains all of the LDA_merged_*.txt tables, which have the predictor values for all of the different merger classifications. These tables are useful if you want to re-create hte LDA classification. They are also used if you want to re-run any of the SDSS classifications, as they are a required import to do so.
2) The sdss_classifications/ folder contains the classifications and predictor values tables for the SDSS galaxies - these include Table_1.txt, Table_4_*.txt, and Table_5.txt from Zenodo.

These tables are described in more detail in Section 4.1 of the <a href="">accompanying paper</a>.

Table_1.txt provides the imaging predictor values and photometric flags for all SDSS photometric galaxies used in this study. Flags have a value of 1 when activated. The table is described in Table 1 of the paper. This table as known as 'SDSS_predictors_all_flags_plus_segmap.txt' in the code of this repo.

Table_4_*.txt provide the LD1, p_merg, CDF, and explanatory top three leading predictors and coefficients for all classified galaxies in the study. There are multiple versions of Table 4 for each different merger classification. This is described in more detail in Table 4 of the paper. These tables are known as 'LDA_out_all_SDSS_predictors_{merger name}_flags.txt' in the code.

Table_5.txt gives the 16th, 50th, and 84th percentile p_merg values for the marginalized analysis for all different classifications. It also provides the CDF value that corresponds to the 50th percentile value. Again, more detail is given in the caption to Table 5 of the paper. This table is named 'LDA_out_all_SDSS_predictors_all_priors_percentile_all_classifications_flags.txt' in the code.
