import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, f1_score
from sklearn.metrics import classification_report 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from xgboost import XGBClassifier
from sklearn.impute import KNNImputer
import random


from collections import Counter
from datetime import datetime
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


class initial_process_of_data:
    """
    Initial processing of data reads in data of specified category, creates dictionary of data, removes impossible values, merges data to give a df. 
    """
    def __init__(self, data_type:str):
        self.data_type = data_type
        self.check_input_flag = False
        self.processed_data_dir = False
        self.uids_added = False
    
    def check_input(self) -> None:
        assert (self.data_type=='test') or (self.data_type=='train'), f'You gave self.data_type as {self.data_type}. Please define data_type as "test" or "train."'
        self.check_input_flag = True

    def readin_data(self):
        """ This function reads in test or train data, which must be in folders 'testing_data' and 'training_data' in the same directory.

        Parameters
        ----------
        'self.data_type' : str
            This must be 'test' or 'train'.
        """
        # assert (self.data_type=='test') or (self.data_type=='train'), f'You gave self.data_type as {self.data_type}. Please define self.data_type as "test" or "train."'
        if self.data_type == 'test':
            inner_directory = './testing_data/'
            data_list = os.listdir('./testing_data')
        else:
            inner_directory = './training_data/'
            data_list = os.listdir('./training_data')
        self.data_dict = {}
        for file_name in data_list:
            self.data_dict[file_name.split('.')[0]] = pd.read_csv(inner_directory+file_name).drop_duplicates()
        return self.data_dict # TODO is this still necessary to return?

    def processed_data_directory(self) -> None:
        """ Makes 'processed_data' directory if one is not found.

        Parameters
        ----------
        None
        """
        os.makedirs('./processed_data', exist_ok=True)
        self.processed_data_dir = True
        return None

    def add_uids(self) -> None:
        """ This function adds a UID to each row to establish unique instances between person_id & measurement datetimes for the various tables.
        
        This is not done for the demographics file since the information in it is not sensitive to the hour.

        Parameters
        ----------
        'data_dictionary' : dict
            A dictionary of pandas DataFrames.
        
        Details
        -------
        * UID is a concatenation of datetime and person_id, in that order.
        * UID is later used as a key for table joins.
        """
        print("Adding UIDs.")

        for table_ind in list(self.data_dict.keys()):
            if not table_ind.startswith("person_demographics"):
                table = self.data_dict[table_ind]
                datetime_index = np.argmax([i.find('datetime') for i in table.columns])
                date_column = table.columns[datetime_index]
                personid_index = np.argmax([i.find('person_id') for i in table.columns])
                personid_column = table.columns[personid_index]
                table['uid'] = table[date_column].astype(str) + table[personid_column].astype(str)
                table.drop(columns=[date_column,personid_column],inplace=True)
                self.data_dict[table_ind] = table
                # print(f'file {table_ind} with len {len(table)}')
        self.uids_added = True
        
        print("UIDs added")
        return None
    
    def birthday_management(self) -> None:
        """
        This function processes the 'person_demographics' table of given data, which is inputed as a dictionary. The data in dictionary is replaced by index, hence function returns nothing.

        Parameters
        ----------
        'data_dictionary' : dict
            A dictionary of pandas DataFrames.
        
        Details
        -------
        * adds new birthday and visit start date columns with dates formated using datetime package.
        * joins new columns to old table using left join to match the unprocessed to the processed date
        """
        demographics_ind_no = np.argmax([table.startswith("person_demographics") for table in self.data_dict.keys()])
        demographics_index = list(self.data_dict.keys())[demographics_ind_no]
        demographics = self.data_dict[demographics_index]
        print(f"Beginning processing for {demographics_index}.")
        

        new_birthday_col = pd.DataFrame(columns=['birthday_formatted', 'person_id'])
        new_visit_col = pd.DataFrame(columns=['visit_start_date','new_visit_startdate'])
        
        for person in np.unique(demographics['person_id']):
            birthday = demographics[demographics['person_id']==person]['birth_datetime'].to_list()[0]
            birthday_formatted = datetime.strptime(birthday,'%Y-%m-%d')
            new_birthday_col.loc[len(new_birthday_col)] = [birthday_formatted, person]

        for date in np.unique(demographics['visit_start_date']):
            visit_start = demographics[demographics['visit_start_date']==date]['visit_start_date'].to_list()[0]
            new_visit_startdate = datetime.strptime(visit_start,'%Y-%m-%d')
            # print(f'new {new_visit_startdate} old {date}')
            new_visit_col.loc[len(new_visit_col)] = [date, new_visit_startdate]


        demographics = pd.merge(left=demographics,right=new_birthday_col,how='left',on='person_id')
        demographics = pd.merge(left=demographics,right=new_visit_col,how='left',on='visit_start_date')
        demographics.drop(columns=['visit_start_date','birth_datetime'],inplace=True)
        demographics.to_csv(f'./processed_data/processed_{demographics_index}.csv')
        self.data_dict[demographics_index] = demographics
        print(f"Finished processing of {demographics_index}.")
        return None

    def measurement_meds_processing(self) -> None:
        """ This function processes the 'measurement_meds' table of given data, which is inputed in a dictionary. The data in dictionary is replaced by index, hence function returns nothing.

        Parameters
        ----------
        'data_dictionary' : dict
            A dictionary of pandas DataFrames.

        Details
        -------
        * removes body temperature measurements > 46 C
        * removes heart rate < 10 and > 250
        * removes Systolic blood pressure > 250
        * removes Diastolic blood pressure > 200
        * removes Respiratory rate > 200
        * removes Measurement of oxygen saturation at periphery > 150
        * removes Oxygen/Gas total [Pure volume fraction] Inhaled gas > 80
        """
        body_measurements_ind = np.argmax([table.startswith("measurement_meds") for table in self.data_dict.keys()])
        body_measurements_index = list(self.data_dict.keys())[body_measurements_ind]
        measurements = self.data_dict[body_measurements_index]
        print(f"Beginning processing for {body_measurements_index}.")

        
        measurements = measurements.dropna(subset=measurements.select_dtypes(float).columns, how='all')
        # measurements.drop(index=[i for i in measurements[measurements['Body temperature']>45].index], axis=1,inplace=True)
        measurements['Body temperature'] = measurements['Body temperature'].apply(lambda x: np.nan if x > 46 else x)
        measurements['Heart rate'] = measurements['Heart rate'].apply(lambda x: np.nan if ((x < 10) | (x > 250)) else x)
        measurements['Systolic blood pressure'] = measurements['Systolic blood pressure'].apply(lambda x: np.nan if x > 250 else x)
        measurements['Diastolic blood pressure'] = measurements['Diastolic blood pressure'].apply(lambda x: np.nan if x > 200 else x)
        measurements['Respiratory rate'] = measurements['Respiratory rate'].apply(lambda x: np.nan if x > 200 else x)
        measurements['Measurement of oxygen saturation at periphery'] = measurements['Measurement of oxygen saturation at periphery'].apply(lambda x: np.nan if x > 150 else x)
        measurements['Oxygen/Gas total [Pure volume fraction] Inhaled gas'] = measurements['Oxygen/Gas total [Pure volume fraction] Inhaled gas'].apply(lambda x: np.nan if x > 80 else x)
        measurements.to_csv(f'./processed_data/processed_{body_measurements_index}.csv')
        self.data_dict[body_measurements_index] = measurements
        print(f"Finished processing of {body_measurements_index}.")
        return None
    
    def drugs_exposure_processing(self) -> None:
        """ This function processes the 'drugsexposure' table of given data, which is inputed in a dictionary. The data in dictionary is replaced by index, hence function returns nothing.

        Parameters
        ----------
        'data_dictionary' : dict
            A dictionary of pandas DataFrames.

        Requirements
        ------------
        You need to run add_uids first.

        Details
        -------
        * combines rows of the same datetime with different drugs to be one row per datetime with all drugs listed in new 'drugs' column and all drug routes listed in new 'routes' column
        * converts from list to string for new columns to allow categorical encoding later on
        """
        assert self.add_uids, "Please run 'add_uids' before get_details."

        drugs_exposure_ind = np.argmax([table.startswith("drugsexposure") for table in self.data_dict.keys()])
        drugs_exposure_index = list(self.data_dict.keys())[drugs_exposure_ind]
        drugs_exposure = self.data_dict[drugs_exposure_index]
        drugs_exposure.reset_index(inplace = True, drop = True)
        print(f"Beginning processing for {drugs_exposure_index}.")

        drugs_exposure = pd.get_dummies(drugs_exposure, columns = ['drug_concept_id', 'route_concept_id'], drop_first = True, prefix="ohe", prefix_sep="_", dtype=float)

        drugs_count = pd.DataFrame([list(i) for i in Counter(drugs_exposure['uid']).items()],columns=['uid','count'])
        drugs_count['count'].astype(int)
        drugs_rows = []
        drugs_extras = drugs_count[drugs_count['count']>1]
        for j in tqdm([i for i in drugs_extras['uid']]):
            new_row = drugs_exposure[drugs_exposure['uid']==j].max().to_frame().T.values.tolist()
            drugs_rows.extend(new_row)

        drugs_exposure.drop_duplicates(subset='uid', keep = False, inplace=True)

        drugs_rows_df = pd.DataFrame(drugs_rows, columns = drugs_exposure.columns)
        drugs_exposure = pd.concat([drugs_exposure,drugs_rows_df]).reset_index(drop=True)
        self.data_dict[drugs_exposure_index] = drugs_exposure
        drugs_exposure.to_csv(f'./processed_data/processed_{drugs_exposure_index}.csv')
        print(f"Finished processing of {drugs_exposure_index}.")
        return None

    def measurement_lab_processing(self) -> None:
        """ This function processes the 'measurement_lab' table of given data, which is inputed in a dictionary. The data in dictionary is replaced by index, hence function returns nothing.

        Parameters
        ----------
        'data_dictionary' : dict
            A dictionary of pandas DataFrames.

        Requirements
        ------------
        You need to run add_uids first.

        Details
        -------
        * removes rows which are all NA
        * combines rows which have the same datetime but different columns filled (different columns have non-NA values)
        * converts columns to float to resolve typing issue
        """
        assert self.add_uids, "Please run 'add_uids' before get_details."

        measurement_lab_ind = np.argmax([table.startswith("measurement_lab") for table in self.data_dict.keys()])
        measurement_lab_index = list(self.data_dict.keys())[measurement_lab_ind]
        measurement_lab = self.data_dict[measurement_lab_index]
        print(f"Beginning processing for {measurement_lab_index}.")

        measurement_lab['Blood arterial pH'] = measurement_lab['Blood arterial pH'].apply(lambda x: np.nan if x > 30 else x)
        measurement_lab['Base excess in Venous blood by calculation'] = measurement_lab['Base excess in Venous blood by calculation'].apply(lambda x: np.nan if x > 30 else x)
        measurement_lab['Base excess in Arterial blood by calculation'] = measurement_lab['Base excess in Arterial blood by calculation'].apply(lambda x: np.nan if x > 50 else x)
        measurement_lab['Phosphate [Moles/volume] in Serum or Plasma'] = measurement_lab['Phosphate [Moles/volume] in Serum or Plasma'].apply(lambda x: np.nan if x > 50 else x)
        measurement_lab['Potassium [Moles/volume] in Blood'] = measurement_lab['Potassium [Moles/volume] in Blood'].apply(lambda x: np.nan if x > 50 else x)
        measurement_lab['Bilirubin.total [Moles/volume] in Serum or Plasma'] = measurement_lab['Bilirubin.total [Moles/volume] in Serum or Plasma'].apply(lambda x: np.nan if x > 700 else x)
        # Leaving 'Neutrophil Ab [Units/volume] in Serum' as is
        # Leaving 'Bicarbonate [Moles/volume] in Arterial blood' as is
        measurement_lab['Hematocrit [Volume Fraction] of Blood'] = measurement_lab['Hematocrit [Volume Fraction] of Blood'].apply(lambda x: np.nan if x > 150 else x)
        measurement_lab['Glucose [Moles/volume] in Serum or Plasma'] = measurement_lab['Glucose [Moles/volume] in Serum or Plasma'].apply(lambda x: np.nan if x > 800 else x)
        measurement_lab['Calcium [Moles/volume] in Serum or Plasma'] = measurement_lab['Calcium [Moles/volume] in Serum or Plasma'].apply(lambda x: np.nan if x > 20 else x)
        # Leaving 'Chloride [Moles/volume] in Blood' as is
        # Leaving 'Sodium [Moles/volume] in Serum or Plasma' as is
        measurement_lab['C reactive protein [Mass/volume] in Serum or Plasma'] = measurement_lab['C reactive protein [Mass/volume] in Serum or Plasma'].apply(lambda x: np.nan if x > 500 else x)
        # Leaving 'Carbon dioxide [Partial pressure] in Venous blood' as is
        measurement_lab['Oxygen [Partial pressure] in Venous blood'] = measurement_lab['Oxygen [Partial pressure] in Venous blood'].apply(lambda x: np.nan if x > 400 else x)
        # Leaving 'Albumin [Mass/volume] in Serum or Plasma' as is
        measurement_lab['Bicarbonate [Moles/volume] in Venous blood'] = measurement_lab['Bicarbonate [Moles/volume] in Venous blood'].apply(lambda x: np.nan if x > 80 else x)
        # Cuts off a couple on right side: (some of the above do too)
        measurement_lab['Oxygen [Partial pressure] in Arterial blood'] = measurement_lab['Oxygen [Partial pressure] in Arterial blood'].apply(lambda x: np.nan if x > 400 else x)
        # Leaving 'Carbon dioxide [Partial pressure] in Arterial blood' as is
        measurement_lab['Interleukin 6 [Mass/volume] in Body fluid'] = measurement_lab['Interleukin 6 [Mass/volume] in Body fluid'].apply(lambda x: np.nan if x > 800 else x)
        measurement_lab['Magnesium [Moles/volume] in Blood'] = measurement_lab['Magnesium [Moles/volume] in Blood'].apply(lambda x: np.nan if x > 10 else x)
        measurement_lab['Prothrombin time (PT)'] = measurement_lab['Prothrombin time (PT)'].apply(lambda x: np.nan if x > 60 else x)
        measurement_lab['Procalcitonin [Mass/volume] in Serum or Plasma'] = measurement_lab['Procalcitonin [Mass/volume] in Serum or Plasma'].apply(lambda x: np.nan if x > 500 else x)
        measurement_lab['Lactate [Moles/volume] in Blood'] = measurement_lab['Lactate [Moles/volume] in Blood'].apply(lambda x: np.nan if x > 60 else x)
        measurement_lab['Creatinine [Mass/volume] in Blood'] = measurement_lab['Creatinine [Mass/volume] in Blood'].apply(lambda x: np.nan if x > 500 else x)
        measurement_lab['Fibrinogen measurement'] = measurement_lab['Fibrinogen measurement'].apply(lambda x: np.nan if x > 50 else x)
        measurement_lab['Bilirubin measurement'] = measurement_lab['Bilirubin measurement'].apply(lambda x: np.nan if x > 500 else x)
        # Leaving 'Partial thromboplastin time' as is
        measurement_lab[' activated'] = measurement_lab[' activated'].apply(lambda x: np.nan if x > 100 else x)
        # Leaving 'Total white blood count' as is
        # Leaving 'Platelet count' as is
        # Is this correct for WBC vs Total WBC (column referenced above)
        measurement_lab['White blood cell count'] = measurement_lab['White blood cell count'].apply(lambda x: np.nan if x > 100 else x)
        measurement_lab['Blood venous pH'] = measurement_lab['Blood venous pH'].apply(lambda x: np.nan if x > 14 else x)
        measurement_lab['D-dimer level'] = measurement_lab['D-dimer level'].apply(lambda x: np.nan if x > 20 else x)
        measurement_lab['Blood arterial pH'] = measurement_lab['Blood arterial pH'].apply(lambda x: np.nan if x > 14 else x)
        measurement_lab['Hemoglobin [Moles/volume] in Blood'] = measurement_lab['Hemoglobin [Moles/volume] in Blood'].apply(lambda x: np.nan if x > 20 else x)

        nan_col_inds = list(measurement_lab.isna().all())
        nan_col_indices = list(measurement_lab.loc[:,nan_col_inds].columns)
        measurement_lab.drop(columns=nan_col_indices, inplace=True)

        measurement_lab = measurement_lab.dropna(subset=list(measurement_lab.select_dtypes(float).columns), how='all')
        measurement_lab_count = pd.DataFrame([list(i) for i in Counter(measurement_lab['uid']).items()],columns=['uid','count'])
        measurement_lab_count['count'].astype(int)
        measurement_lab_rows = pd.DataFrame()
        measurement_lab_extras = measurement_lab_count[measurement_lab_count['count']>1]
        for j in [i for i in measurement_lab_extras['uid']]:
            measurement_lab_rows = pd.concat([measurement_lab_rows,measurement_lab[measurement_lab['uid']==j].max().to_frame().T]).reset_index(drop=True)
        # measurement_lab_extras_ind = measurement_lab_extras.index
        # measurement_lab = measurement_lab.drop(index=measurement_lab_extras_ind, axis=1,inplace=False)
        measurement_lab = measurement_lab.drop_duplicates(subset='uid', keep = False, inplace=False)

        measurement_lab = pd.concat([measurement_lab,measurement_lab_rows]).reset_index(drop=True)

        col_inds = [not((i.endswith('_id')) or (i=='uid')) for i in list(measurement_lab.columns)]
        col_names = measurement_lab.columns.values[col_inds]
        for column in col_names:
            measurement_lab[column] = measurement_lab[column].astype(float)

        measurement_lab.to_csv(f'./processed_data/processed_{measurement_lab_index}.csv')
        self.data_dict[measurement_lab_index] = measurement_lab
        print(f"Finished processing of {measurement_lab_index}.")
        return None
    
    def measurement_observation_processing(self) -> None:
        """ This function processes the 'measurement_observation' table of given data, which is inputed in a dictionary. The data in dictionary is replaced by index, hence function returns nothing.

        Parameters
        ----------
        'data_dictionary' : dict
            A dictionary of pandas DataFrames.
        
        Details
        -------
        * creates csv to verify this table has been processed.
        """

        measurement_obs_ind = np.argmax([table.startswith("measurement_observation") for table in self.data_dict.keys()])
        measurement_obs_index = list(self.data_dict.keys())[measurement_obs_ind]
        measurement_obs = self.data_dict[measurement_obs_index]
        print(f"Beginning processing for {measurement_obs_index}.")

        # measurement_obs = measurement_obs.dropna(subset=measurement_obs.select_dtypes(float).columns, how='all')
        measurement_obs.to_csv(f'./processed_data/processed_{measurement_obs_index}.csv')
        self.data_dict[measurement_obs_index] = measurement_obs
        print(f"Finished processing of {measurement_obs_index}.")
        return None
    
    def observation_processing(self) -> None:
        """ This function processes the 'observation' table of given data, which is inputed in a dictionary. The data in dictionary is replaced by index, hence function returns nothing.

        Parameters
        ----------
        'data_dictionary' : dict
            A dictionary of pandas DataFrames.
        
        Details
        -------
        * creates csv to verify this table has been processed.
        * removes columns 'observation_concept_id' and 'observation_concept_name' which each have only one value filled to all rows.
        """

        observation_ind = np.argmax([table.startswith("observation") for table in self.data_dict.keys()])
        observation_index = list(self.data_dict.keys())[observation_ind]
        observation = self.data_dict[observation_index]
        print(f"Beginning processing for {observation_index}.")
        observation.drop(columns=['observation_concept_id','observation_concept_name'], inplace = True)


        observation = observation.dropna(subset=observation.select_dtypes(object).columns, how='all')

        observation.to_csv(f'./processed_data/processed_{observation_index}.csv')

        self.data_dict[observation_index] = observation 
        print(f"Finished processing of {observation_index}.")
        return None

    def procedures_processing(self) -> None:
        """ This function processes the 'observation' table of given data, which is inputed in a dictionary. The data in dictionary is replaced by index, hence function returns nothing.

        Parameters
        ----------
        'data_dictionary' : dict
            A dictionary of pandas DataFrames.

        Requirements
        ------------
        You need to run add_uids first.
        
        Details
        -------
        * drops visit_occurrence column
        * drops duplicate entries
        """
        # assert uids_added == True, 'You need to run add_uids before this function.'

        procedures_ind = np.argmax([table.startswith("proceduresoccurrences") for table in self.data_dict.keys()])
        procedures_index = list(self.data_dict.keys())[procedures_ind]
        procedures = self.data_dict[procedures_index]
        print(f"Beginning processing for {procedures_index}.")

        procedures = procedures.dropna(subset=procedures.select_dtypes(object).columns, how='all')


        visit_id_index = np.argmax([i.find('visit_occurrence') for i in procedures.columns])
        visit_column = procedures.columns[visit_id_index]
        procedures.drop(columns=visit_column,inplace=True)
        procedures.drop_duplicates(inplace=True)

        procedures.to_csv(f'./processed_data/processed_{procedures_index}.csv')


        self.data_dict[procedures_index] = procedures 
        print(f"Finished processing of {procedures_index}.")
        return None

    def devices_processing(self) -> None:
        """ This function processes the 'devices' table of given data, which is inputed in a dictionary. The data in dictionary is replaced by index, hence function returns nothing.

        Parameters
        ----------
        'data_dictionary' : dict
            A dictionary of pandas DataFrames.
        
        Details
        -------
        * creates csv to verify this table has been processed.
        """

        devices_ind = np.argmax([table.startswith("devices") for table in self.data_dict.keys()])
        devices_index = list(self.data_dict.keys())[devices_ind]
        devices = self.data_dict[devices_index]
        print(f"Beginning processing for {devices_index}.")

        devices = devices.dropna(subset=devices.select_dtypes(object).columns, how='all')
        devices.drop_duplicates(inplace=True)

        visit_id_index = np.argmax([i.find('visit_occurrence') for i in devices.columns])
        visit_column = devices.columns[visit_id_index]
        devices.drop(columns=visit_column,inplace=True)

        devices = pd.get_dummies(devices, columns = ['device'], drop_first = True, prefix="ohe", prefix_sep="_", dtype=float)

        devices_count = pd.DataFrame([list(i) for i in Counter(devices['uid']).items()],columns=['uid','count'])
        devices_count['count'].astype(int)
        devices_rows = []
        devices_extras = devices_count[devices_count['count']>1]
        for j in tqdm([i for i in devices_extras['uid']]):
            new_row = devices[devices['uid']==j].max().to_frame().T.values.tolist()
            devices_rows.extend(new_row)

        devices.drop_duplicates(subset='uid', keep = False, inplace=True)
        devices_rows_df = pd.DataFrame(devices_rows, columns = devices.columns)
        if devices_rows_df.empty == False:
            devices = pd.concat([devices,devices_rows_df]).reset_index(drop=True)

        devices.to_csv(f'./processed_data/processed_{devices_index}.csv')

        self.data_dict[devices_index] = devices
        print(f"Finished processing of {devices_index}.")
        return None

    def sepsis_processing(self) -> None:
        """ This function processes the 'sepsis' table of given data, which is inputed in a dictionary. The data in dictionary is replaced by index, hence function returns nothing.

        Parameters
        ----------
        'data_dictionary' : dict
            A dictionary of pandas DataFrames.
        
        Details
        -------
        * drops values which have no datetime
        """
        sepsis_ind = np.argmax([table.startswith("SepsisLabel") for table in self.data_dict.keys()])
        sepsis_index = list(self.data_dict.keys())[sepsis_ind]
        sepsis = self.data_dict[sepsis_index]
        print(f"Beginning processing for {sepsis_index}.")

        #Taking out values that have no datetime:
        no_time_rows = list(sepsis.loc[sepsis['uid'].str.startswith('nan', na=False)].index)
        sepsis = sepsis.drop(index=no_time_rows, axis = 1, inplace = False)
        
        self.data_dict[sepsis_index] = sepsis
        sepsis.to_csv(f'./processed_data/processed_{sepsis_index}.csv')
        
        print(f"Finished processing of {sepsis_index}.")
        return None
    
    def process_data(self, load_tables: str):
        """ This function reads in test or train data and goes through functions to preprocess it. For further details see specific functions.

        Processed tables will be saved into the /processed folder.

        Parameters
        ----------
        'self.data_type' : str
            This must be 'test' or 'train'. Determines which data to process.
        'load_tables' : str
            This must be 'yes' or 'no'. 'yes' means load csvs from processed_data folder, of the type given in 'self.data_type' input. 'no' means process data from training or testing folder, depending on 'self.data_type' input given.
        
        Returns
        -------
        'processed_data' : dict
            This is a dictionary of datatables that have been processed
        'factors' : DataFrame
            This is a DataFrame of the data from 'processed_data' joined together.
        """
        # assert (data_type=='test') or (data_type=='train'), f'You gave data_type as {data_type}. Please define data_type as "test" or "train."'
        assert (load_tables=='yes') or (load_tables=='no'), f'You gave load_tables as {load_tables}. Please define load_tables as "test" or "train."'

        self.processed_data_directory()

        if self.data_type == 'train':
            if load_tables == 'no':
                training_data = self.readin_data('train')
                self.add_uids(training_data)
                self.birthday_management(training_data)
                self.measurement_meds_processing(training_data)
                self.drugs_exposure_processing(training_data)
                self.measurement_lab_processing(training_data)
                self.procedures_processing(training_data)
                self.observation_processing(training_data)
                self.measurement_observation_processing(training_data)
                self.devices_processing(training_data)
                self.sepsis_processing(training_data)
            else:
                training_data={}
                inner_directory = './processed_data/'
                data_list = os.listdir('./processed_data')
                separator = '_'
                for file_name in data_list:
                    if file_name.split('.')[0].split('_')[-1]=='train':
                        training_data[separator.join(file_name.split('.')[0].split('_')[1:])] = pd.read_csv(inner_directory+file_name,index_col=0).drop_duplicates()

            factors = pd.merge(left=training_data['measurement_meds_train'], right=training_data['measurement_lab_train'],how='outer',on='uid')
            factors = pd.merge(left=factors, right=training_data['drugsexposure_train'],how='outer',on='uid')
            factors = pd.merge(left=factors, right=training_data['proceduresoccurrences_train'],how='outer',on='uid')
            factors = pd.merge(left=factors, right=training_data['devices_train'],how='outer',on='uid')
            # factors = pd.merge(left=factors, right=training_data['person_demographics_episode_train'], how='outer',on='visit_occurrence_id')
            factors = pd.merge(left=training_data['SepsisLabel_train'],right=factors,how='left',on='uid')
            factors.to_csv(f'./processed_data/factors_train.csv')
            processed_data = training_data

        else:
            if load_tables == 'no':
                testing_data = self.readin_data('test')
                self.add_uids(testing_data)
                self.birthday_management(testing_data)
                self.measurement_meds_processing(testing_data)
                self.drugs_exposure_processing(testing_data)
                self.measurement_lab_processing(testing_data)
                self.procedures_processing(testing_data)
                self.observation_processing(testing_data)
                self.measurement_observation_processing(testing_data)
                self.devices_processing(testing_data)
                self.sepsis_processing(testing_data)
            else:
                training_data={}
                inner_directory = './processed_data/'
                data_list = os.listdir('./processed_data')
                separator = '_'
                for file_name in data_list:
                    if file_name.split('.')[0].split('_')[-1]=='test':
                        training_data[separator.join(file_name.split('.')[0].split('_')[1:])] = pd.read_csv(inner_directory+file_name,index_col=0).drop_duplicates()
                
            factors = pd.merge(left=testing_data['measurement_meds_test'], right=testing_data['measurement_lab_test'],how='outer',on='uid')
            factors = pd.merge(left=factors, right=testing_data['drugsexposure_test'],how='outer',on='uid')
            factors = pd.merge(left=factors, right=testing_data['proceduresoccurrences_test'],how='outer',on='uid')
            factors = pd.merge(left=factors, right=testing_data['devices_test'],how='outer',on='uid')
            # factors = pd.merge(left=factors, right=testing_data['person_demographics_episode_test'], how='outer',on='visit_occurrence_id')
            print(f"factors length {len(factors)} and sepsis has {len(testing_data['SepsisLabel_test'])}")
            factors = pd.merge(left=testing_data['SepsisLabel_test'],right=factors,how='left',on='uid')
            print(f'post merge factors {len(factors)}')
            factors.to_csv(f'./processed_data/factors_test.csv')
            processed_data = testing_data

        factors.drop(columns=['visit_occurrence_id_x','visit_occurrence_id_y'],inplace=True)

        return processed_data, factors