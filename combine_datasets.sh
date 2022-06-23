
python3 combine_df.py -p1 data/edu_climate_fallacies.csv -p2 data/hand_picked_fallacies_cleaned_sa.csv -po data/all_combined/temp.csv

python3 combine_df.py -p1 data/all_combined/temp.csv -p2 data/kialo_climate_sa.csv -po data/all_combined/temp.csv
python3 combine_df.py -p1 data/all_combined/temp.csv -p2 data/kialo_non_climate_sa.csv -po data/all_combined/temp.csv
python3 combine_df.py -p1 data/all_combined/temp.csv -p2 data/non_fallacies_sa.csv -po data/all_combined/all_arguments.csv
