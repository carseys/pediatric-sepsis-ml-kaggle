# pediatric-sepsis-ml-kaggle

<hackathon-img>
  <source media="(prefers-color-scheme: dark)" srcset="readme-imgs/tumai_image.jpg">
  <source media="(prefers-color-scheme: light)" srcset="readme-imgs/phems_online_hackathon_image.jpg">
  <img alt="Hackathon Cover Image" src="default-image.png">
</hackathon-img>


# Early Detection of Pediatric Sepsis ML Modeling

A set of ML which predict which patients will develop sepsis based on data about their health. For this project, I preprocessed data, developed models, and visualized model efficacy.

This model was developed for the TUM.ai x PHEMS Online Kaggle Challenge ([link](https://www.kaggle.com/competitions/phems-hackathon-early-sepsis-prediction/l)), January-February 2025.

Packages used: pandas, numpy, collections, scikit-learn, xgboost, tqdm, matplotlib, os, joblib, datetime.

## Use
* Install required packages based on `requirements.txt`.
* Follow examples in `models.ipynb` for using processing functions and modeling processed data.
* The biggest challenge was addressing the sparse nature of the data. See `data_visualizations.ipynb` to see how sparse at different points.
* See `processing_data.py` for classes and functions to process data before splitting for model training.
* See `modeling.py` for function splitting data for model training and function for developing upload file to kaggle.
* The file `processing-and-model-development.ipynb` is less organized and was used for developing functions for this project.

## Further Description:

### Preprocessing:
* Removing duplicate entries.
* Adding 'uid' to all data, a universal ID, based on date/time stamp and person ID.
* For the patient demographics: converting birthdays and visit start dates into datetime to judge age in months at every patient interaction point (via datetimes per interaction points) as well as possibility of adding variable for length of stay (most recent interaction from visit code minus visit start date).
* For measurement meds data: removing duplicates, replacing unreasonable measurements with NaN (eg >46 degrees C body temperature).
* For the drugs exposure data, one hot encoding drugs and routes. Removing columns for drugs not present in test data.
* For the measurement labs data, replacing unreasonable measurements with NaN, removing rows that only have NaN (this removed 1063 rows including some duplicates) and combining rows from the same timestamp for one patient to have one row per patient per instance of time.
* For the devices data, one hot encoding devices.
* For the sepsis label data, removing rows that have no timestamp.
* Outer joining tables based on uid for explanatory variables with time-based measurements.
* Left joining with sepsis data as left table, with explanatory variables as right.
* Re-adding person ids to all rows based on UID.
* Adding age in months to all rows based on timedate present in UID.
* Adding gender to all rows based on person_id within UID.
* Encoding categorical data : drugsexposure (drugs, routes), devices, uid, gender.
* Removing columns with <10000 values in training data. Removing the same columns in test data.
* Gaussian imputation of values for blood pressure(systolic and diastolic), body temperature, hematocrit based on expected value by age. Based Gaussian distributions on healthy measurement ranges found online.
* Splitting the labeled train data into training and test for training/validation purposes.


### Modeling
* Random Forest Model! Adjusted class weights to favor prediction of sepsis label.
* Gradient Boost Ensemble Model! Adjusted class weights to favor prediction of sepsis label.
* HistGradientBoostClassifier! Adjusted class weights to favor prediction of sepsis label.
* See modeling in models.ipynb or use these models via the .pkl files.
