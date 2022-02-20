## Limits of Organism-specific training for Linear B-cell epitope prediction

This repository contains the source code and data used for the manuscript 
"_Estimating the Limits of Organism-Specific Training for Epitope Prediction_", 
by J. Ashford, A. Ekárt and F. Campelo.

Notes:
- The data generation, modelling, analysis and plotting routines are under 
`/code`. All steps except modelling were implemented in R; modelling was 
done in Python (scripts and environment under `/code/02_modelling/`).

- All data is located under `/data`, including:
  - the base data for each organism (`01_training.rds` and `02_holdout.rds`)
  - heterogeneous data for each organism (`df_heterogeneous.rds`)
  - baseline performance estimates extracted from <https://academic.oup.com/bioinformatics/article/37/24/4826/6325084> (`analysis.rds`)
  - results for each organism (`/data/Results/...`). The .CSV files containing the resulting predictions have been compressed to comply with Github's file and repository size limits. Please expand these files prior to running the scripts in `/code`.
  
If there are any queries, please contact me at <f.campelo@aston.ac.uk> or <fcampelo@gmail.com>.