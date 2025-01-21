# pediatric-sepsis-ml-kaggle


Early Detection of Pediatric Sepsis ML Modeling

A "model type" which predicts which patients will develop sepsis based on D-dimensional data about their health. For this project, I preprocessed data, developed a model, and visualized model efficacy.

This model was developed for the TUM.ai x PHEMS Online Kaggle Challenge, January 2025.

Packages used: pandas, numpy, collections, scikit-learn, tqdm, seaborn, matplotlib, joblib, datetime.

## Further Description:
Preprocessing:
* Removing duplicate entries.
* Adding 'uid' to all data, a universal ID, based on date/time stamp and person ID.
* For the patient demographics: converting birthdays and visit start dates into datetime to judge age in months at every patient interaction point (via datetimes per interaction points) as well as possibility of adding variable for length of stay (most recent interaction from visit code minus visit start date)
* For measurement meds data: removing duplicates, replacing unreasonable (>46 degrees C) body temperature measurements with NaN
* For the drugs exposure data, combining rows that represent the same patient/doctor interaction to have only one row per interaction, e.g. there could be four rows for one interaction, each showing a different drug administered with the corresponding administration method. The new row would list all four drugs and all administration methods used.
* For the measurement labs data, removing rows that only have NaN (this removed 1063 rows including some duplicates) and combining rows from the same timestamp for one patient to have one row per instance of time knowing lab measurements about a patient (only one such row).
* Outer joining tables based on uid for explanatory variables with time-based measurements.
* Outer joining with patient demographics based on visit occurrence ID.
* Left joining with sepsis data as left table, with explanatory variables as right

Previously done:
* Encoding categorical data.
* Splitting the labeled train data into training and test. The provided test data isn't labeled because it is used to score for the leaderboard/competition. Thus the training data is split into train and test so that it is still possible to test the model with labels known to evaluate performance and prevent overfitting.
