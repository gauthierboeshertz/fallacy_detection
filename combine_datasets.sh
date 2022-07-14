
echo Combining datasets with every subset
python3 combine_df.py -p1 data/edu_climate_fallacies.csv -p2 data/hand_picked_fallacies_cleaned_sa.csv -po data/all_combined/temp.csv
python3 combine_df.py -p1 data/all_combined/temp.csv -p2 data/kialo_climate_sa.csv -po data/all_combined/temp.csv
python3 combine_df.py -p1 data/all_combined/temp.csv -p2 data/kialo_non_climate_sa.csv -po data/all_combined/temp.csv
python3 combine_df.py -p1 data/all_combined/temp.csv -p2 data/non_fallacies_sa.csv -po data/all_combined/all_arguments.csv
python3 split_csv.py --path data/all_combined/all_arguments.csv


echo Combining datasets to train on everything but climate and then test on climate, name of folder is climate_test
python3 combine_df.py -p1 data/edu_fallacies_sa.csv -p2 data/hand_picked_fallacies_cleaned_sa.csv -po data/climate_test/temp.csv
python3 combine_df.py -p1 data/climate_test/temp.csv -p2 data/kialo_non_climate_sa.csv -po data/climate_test/temp.csv
python3 combine_df.py -p1 data/climate_test/temp.csv -p2 data/non_fallacies_sa.csv -po data/climate_test/non_climate.csv
python3 combine_df.py -p1 data/kialo_climate_sa.csv -p2 data/climate_fallacies_sa.csv -po data/climate_test/climate_test.csv
python3 split_csv.py --path data/climate_test/non_climate.csv --splits [0.8,0.2]


echo Combining what we mined on kialo 
python3 combine_df.py -p1 data/non_fallacies_sa.csv -p2 data/hand_picked_fallacies_cleaned_sa.csv -po data/our_fallacies/non_climate.csv
python3 split_csv.py --path data/our_fallacies/non_climate.csv --splits [0.8,0.1,0.1]


echo Combining the paper edu fallacies with good arguments from point 3  
python3 combine_df.py -p1 data/non_fallacies_sa.csv -p2 data/edu_fallacies_sa_ds.csv -po data/paper_fallacies/non_climate.csv
python3 split_csv.py --path data/paper_fallacies/non_climate.csv --splits [0.8,0.1,0.1]


echo Combining what we mined on kialo and testing on all climate data
python3 combine_df.py -p1 data/non_fallacies_sa.csv -p2 data/hand_picked_fallacies_cleaned_sa.csv -po data/our_fallacies_climate_test/non_climate.csv
python3 combine_df.py -p1 data/kialo_climate_sa.csv -p2 data/climate_fallacies_sa.csv -po data/our_fallacies_climate_test/climate_test.csv
python3 split_csv.py --path data/our_fallacies_climate_test/non_climate.csv --splits [0.8,0.2]


echo Combining the paper edu fallacies with good arguments from point 3  and testing on all climate data
python3 combine_df.py -p1 data/non_fallacies_sa.csv -p2 data/edu_fallacies_sa.csv -po data/paper_fallacies_climate_test/non_climate.csv
python3 combine_df.py -p1 data/kialo_climate_sa.csv -p2 data/climate_fallacies_sa.csv -po data/paper_fallacies_climate_test/climate_test.csv
python3 split_csv.py --path data/paper_fallacies_climate_test/non_climate.csv --splits [0.8,0.2]


