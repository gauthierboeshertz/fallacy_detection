
echo data stats of the combined datasets

echo Train
python3 data_stats.py -p data/all_combined/all_arguments_train.csv
echo Val
python3 data_stats.py -p data/all_combined/all_arguments_val.csv

echo Test
python3 data_stats.py -p data/all_combined/all_arguments_test.csv



echo data stats of the climate tests

echo Train
python3 data_stats.py -p data/climate_test/non_climate_train.csv
echo Val
python3 data_stats.py -p data/climate_test/non_climate_val.csv
echo Test
python3 data_stats.py -p data/climate_test/climate_test.csv


echo data stats of the paper fallacies on same domain
echo Train
python3 data_stats.py -p data/paper_fallacies/non_climate_train.csv
echo Val
python3 data_stats.py -p data/paper_fallacies/non_climate_val.csv
echo Test
python3 data_stats.py -p data/paper_fallacies/non_climate_test.csv


echo data stats of the paper fallacies on climate tests
echo Train
python3 data_stats.py -p data/paper_fallacies_climate_test/non_climate_train.csv
echo Val
python3 data_stats.py -p data/paper_fallacies_climate_test/non_climate_val.csv
echo Test
python3 data_stats.py -p data/paper_fallacies_climate_test/climate_test.csv


echo data stats of the reduced  paper fallacies on climate tests
echo Train
python3 data_stats.py -p data/paper_fallacies_climate_test/non_climate_ds_train.csv
echo Val
python3 data_stats.py -p data/paper_fallacies_climate_test/non_climate_ds_val.csv
echo Test
python3 data_stats.py -p data/paper_fallacies_climate_test/climate_test.csv



echo data stats of our fallacies on same domain
echo Train
python3 data_stats.py -p data/our_fallacies/non_climate_train.csv
echo Val
python3 data_stats.py -p data/our_fallacies/non_climate_val.csv
echo Test
python3 data_stats.py -p data/our_fallacies/non_climate_test.csv


echo data stats of our fallacies on climate tests
echo Train
python3 data_stats.py -p data/our_fallacies_climate_test/non_climate_train.csv
echo Val
python3 data_stats.py -p data/our_fallacies_climate_test/non_climate_val.csv
echo Test
python3 data_stats.py -p data/our_fallacies_climate_test/climate_test.csv