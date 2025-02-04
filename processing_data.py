import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


class initial_intake_process_of_data:
    """
    Initial processing of data reads in data of specified category, creates dictionary of data, removes impossible values, merges data to give a df. 
    """
    def __init__(self, data_type:str, load_tables: str):
        self.data_type = data_type
        self.check_input_flag = False
        self.processed_data_dir = False
        self.uids_added = False
        self.load_tables = load_tables
    
    def check_input(self) -> None:
        assert (self.data_type=='test') or (self.data_type=='train'), f'You gave self.data_type as {self.data_type}. Please define data_type as "test" or "train."'
        assert (self.load_tables=='yes') or (self.load_tables=='no'), f'You gave self.data_type as {self.load_tables}. Please define data_type as "yes" or "no."'
        self.check_input_flag = True
        

    def readin_data(self):
        """ This function reads in test or train data, which must be in folders 'testing_data' and 'training_data' in the same directory.

        Parameters
        ----------
        'self.data_type' : str
            This must be 'test' or 'train'.
        """

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
        assert self.add_uids, "Please run 'add_uids' before get_details."

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
        assert self.add_uids, "Please run 'add_uids' before get_details."

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
        assert self.add_uids, "Please run 'add_uids' before get_details."

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
    
    def process_data(self):
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

        self.processed_data_directory()

        if self.data_type == 'train':
            if self.load_tables == 'no':
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
            self.data_dict = training_data

        else:
            if self.load_tables == 'no':
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
            self.data_dict = testing_data

        factors.drop(columns=['visit_occurrence_id_x','visit_occurrence_id_y'],inplace=True)

        return self.data_dict, factors
    


class post_merge_process:
    """
    Initial processing of data reads in data of specified category, creates dictionary of data, removes impossible values, merges data to give a df. 
    """
    def __init__(self, data_dict: dict, factors: pd.DataFrame):
        self.data_dict = data_dict
        self.factors = factors
        self.new_person_id_flag = False
        self.processed_data_dir = False
        self.uids_added = False

    def new_person_ids(self):
        """
        This function reads person_id column for all rows based on UID and adds a new column with this information.

        Parameters
        ----------
        'df': pandas DataFrame
            This should be a pandas DataFrame which contains a column 'uid' with universal ids, with strings with format s.t. the person_id starts at the 18th index. e.g.: '2021-07-21 12:00:00623183219'
        """
        # assert that uid column exists
        print('Beginning adding new_person_id column based on uids.')
        new_person_ids = pd.DataFrame(columns=['new_person_id','uid'])
        new_id = df['uid'].copy().apply(lambda x: x[19:])
        # new_id.apply(lambda x: int(x))
        new_id = new_id.astype(int)
        new_person_ids['new_person_id'] = new_id
        new_person_ids['uid'] = df['uid'].copy()
        # df.drop(columns=['person_id'],inplace=True)
        df['new_person_id']=new_person_ids['new_person_id']
        self.new_person_id = True
        print('Finished adding new_person_id column based on uids.')
        return None
    
    def birthday_ubiquity(self):
        """
        This function adds age in months to all rows in df based on uid time and birthday. It also adds gender to all rows in df.

        Parameters
        ----------
        'df': pandas DataFrame
            This is a pandas DataFrame of patient data that contains a 'new_person_id' column from running new_person_ids function.
        'data_dictionary': dictionary
            This is a dictionary of pandas DataFrames, should have a table of 'person_demographics_episode' data. The 'person_demographics_episode' data will be used to establish birthdays and genders for patients who occur in the DataFrame given.

        Returns
        -------
        'df': pandas DataFrame
            returns df with age in months and gender for participants with entry in 'person_demographics_episode' from provided data_dictionary.

        Details
        -------
        * makes temporary dataframe 'unique_demographics_rows' of gender and birthday from 'person_demographics_episode'
        * temporarily adds birthday by joining birthday from 'unique_demographics_rows' to df with key as new_person_id
        * temporarily creates datetime column derived from uid
        * adds age in months column by time between uid date and birthday to df in new column
        * deletes birthday column from df
        * adds gender column by joining gender from 'unique_demographics_rows' to df with key as new_person_id
        """
        #assert that new_person_id column exists
        assert self.new_person_id_flag, "Please run 'add_uids' before get_details."

        # factors = pd.merge(left=testing_data['SepsisLabel_test'],right=factors,how='left',on='uid')
        print('Beginning birthday ubiquity.')

        demographics_ind_no = np.argmax([table.startswith("person_demographics") for table in data_dictionary.keys()])
        demographics_index = list(data_dictionary.keys())[demographics_ind_no]
        demographics = data_dictionary[demographics_index]

        unique_demographics_rows = pd.DataFrame(columns=['new_person_id','gender','birthday'])
        for patient in np.unique(demographics['person_id']):
            birthday = list(demographics[demographics['person_id']==patient]['birthday_formatted'])[0]
            gender = list(demographics[demographics['person_id']==patient]['gender'])[0]
            unique_demographics_rows.loc[len(unique_demographics_rows)] = [patient, gender, birthday]
        df = pd.merge(left=df, right=unique_demographics_rows, how='left', on='new_person_id')
        datetime_temp = df['uid'].copy().apply(lambda x: x[:19])
        datetime_temp = pd.to_datetime(datetime_temp)
        birthday_col = df['birthday'].copy()
        age = -round((birthday_col-datetime_temp)/np.timedelta64(1,'D')/30)
        df['age'] = age
        df.drop(columns=['birthday'], inplace = True)
        print('Birthday ubiquity established.')
        return df

    def add_visit_reason(self):
        """

        This function adds 'Admission Reason' column to 'df'. This is done based on the most recent admission reason in the 'observation' table in 'data_dictionary' which occurs at or before the timestamp of a row (via uid).
        
        Please makes sure 'df' and 'data_dictionary' correspond to the same data!
        
        Parameters
        ----------
        'df' : pd.DataFrame
            DataFrame of data.
        'data_dictionary' : dict
            dictionary with DataFrames for values.

        Returns
        -------
        none since df is modified inplace.
        """
        # assert uid column in df
        # potentially assert observation column in df.

        observation_ind = np.argmax([table.startswith("observation") for table in data_dictionary.keys()])
        observation_index = list(data_dictionary.keys())[observation_ind]
        observation = data_dictionary[observation_index]

        print("Adding visit reason")

        admission_times = observation['uid'].copy().apply(lambda x: x[:19])
        admission_times = pd.to_datetime(admission_times)
        
        person = observation['uid'].copy().apply(lambda x: x[19:])
        person = person.astype(int)

        observation['admission_time'] = admission_times
        observation['person'] = person

        most_recent_admission_list = []

        for uid in tqdm(np.unique(df['uid'])):
            instance = pd.to_datetime(uid[:19])
            person = int(uid[19:])
            try:
                # personal_admissions = observation[observation['person']==person]
                recent_reason = observation[(observation['person']==person) & (observation['admission_time'] <= instance)].sort_values(by='admission_time', ascending=False).iloc[0]['valuefilled']
            except:
                recent_reason = np.nan
            most_recent_admission_list.append(recent_reason)
        admission_reason = pd.DataFrame(most_recent_admission_list, columns=['Admission Reason'])
        df['Admission Reason'] = admission_reason
        print("Visit reasons added")
        return None

    def time_interpolation(self):
        """
        This function interpolates values for certain columns based on difference in time via uid.
        
        Parameters
        ----------
        'df' : pd.DataFrame
        """
        # function does not work
        # 'White blood cell count'
        assert self.new_person_id_flag, "Please run 'add_uids' before get_details."
        
        datetime_temp = df['uid'].copy().apply(lambda x: x[:19])
        df['datetime_temp'] = pd.to_datetime(datetime_temp)
        for person in tqdm(np.unique(df['new_person_id'])):
            personal_df = df[df['new_person_id']==person].loc[:,['White blood cell count', 'datetime_temp']]
            og_index = personal_df.index
            personal_df.set_index('datetime_temp', inplace=True)
            personal_df.loc[:,['White blood cell count']].interpolate(method='time')
            personal_df.set_index(og_index, inplace=True)
            print(personal_df)
            df[df['new_person_id']==person]['White blood cell count']
            # Write new df? #TODO 
        return None

    def clearing_cols(df: pd.DataFrame, threshold: int, data_type: str, col_list: list = None):
        """
        Clears columns that shouldn't be there for modeling and columns that don't meet threshold of number of values.
        
        Parameters
        ----------
        'df' : pd.DataFrame
            DataFrame of data
        'threshold' : int
            number of values needed in a column. If column doesn't have this number of entries, it will be dropped from df. Irrelevant if 'data_type' is set to 'train'.
        'data_type' : str
            'train' or 'test'.
        'col_list' : list
            This is a list of columns to clear in test data, based on most recent run of this function with training data.

        Returns
        -------
        'df' : pd.DataFrame
        'sufficiently_empty_cols' : list
        """
        # assert data_type
        print('Clearing columns.')

        if data_type == 'train':
            sufficiently_empty_cols = list(df.loc[:,df.count() < threshold].columns)
            df.drop(columns=sufficiently_empty_cols, inplace=True)
            df.drop(columns=['visit_occurrence_id','uid'], inplace=True)
            # The following columns are dropped because the drugs are not present in test data.
            df.drop(columns=['ohe_ceftolozane','ohe_isoproterenol'],inplace=True)
        
        if data_type == 'test':
            sufficiently_empty_cols = col_list
            df.drop(columns=sufficiently_empty_cols, inplace=True)
            df.drop(columns=['visit_occurrence_id','uid'], inplace=True)

        print('Columns cleared.')

        return df, sufficiently_empty_cols
    
    def fill_from_gaussian(column_value, mean: float, std: float):
        """"
        This function imputes NaN values with values sampled from a normal distribution of parameters specified.

        Parameters
        ----------
        'column_value'
            the column this is applied to
        'mean' : float
            mean of normal distribution sampled.
        'std': float
            standard deviation of normal distribution sampled.
        """
        if np.isnan(column_value) == True:
            column_value = np.round(np.random.normal(mean, std, 1)[0], 1)
        else:
            column_value = column_value
        return column_value

    def fill_nans_gaussian(self):
        """
        Imputes NaN values in certain columns of df provided, based on online information of typical values by age.

        Parameters
        ----------
        'df' : pd.DataFrame
            DataFrame of data.
        
        Returns
        -------
        None. df will be updated by column.
        """
        print('filling NaN values using Gaussians.')
        
        df['Body temperature'] = df['Body temperature'].apply(fill_from_gaussian, **{'mean': 36.9, 'std': .15})

        df.loc[df['age'] <= 12, 'Systolic blood pressure'] = df.loc[df['age'] <= 12, 'Systolic blood pressure'].apply(fill_from_gaussian, **{'mean': 90, 'std': 5})
        df.loc[df['age'].between(12, 60,inclusive='right'), 'Systolic blood pressure'] = df.loc[df['age'].between(12, 60,inclusive='right'), 'Systolic blood pressure'].apply(fill_from_gaussian, **{'mean': 105, 'std': 7})
        df.loc[df['age'].between(60, 120,inclusive='right'), 'Systolic blood pressure'] = df.loc[df['age'].between(60, 120,inclusive='right'), 'Systolic blood pressure'].apply(fill_from_gaussian, **{'mean': 114, 'std': 7})
        df.loc[df['age'] >120, 'Systolic blood pressure'] = df.loc[df['age'] >120, 'Systolic blood pressure'].apply(fill_from_gaussian, **{'mean': 120, 'std': 10})

        df.loc[df['age'] <= 12, 'Diastolic blood pressure'] = df.loc[df['age'] <= 12, 'Diastolic blood pressure'].apply(fill_from_gaussian, **{'mean': 49, 'std': 5})
        df.loc[df['age'].between(12, 60,inclusive='right'), 'Diastolic blood pressure'] = df.loc[df['age'].between(12, 60,inclusive='right'), 'Diastolic blood pressure'].apply(fill_from_gaussian, **{'mean': 60, 'std': 5})
        df.loc[df['age'].between(60, 120,inclusive='right'), 'Diastolic blood pressure'] = df.loc[df['age'].between(60, 120,inclusive='right'), 'Diastolic blood pressure'].apply(fill_from_gaussian, **{'mean': 70, 'std': 5})
        df.loc[df['age'] > 120, 'Diastolic blood pressure'] = df.loc[df['age'] > 120, 'Diastolic blood pressure'].apply(fill_from_gaussian, **{'mean': 75, 'std': 7})

        df.loc[df['age'] <= 2, 'Hematocrit [Volume Fraction] of Blood'] = df.loc[df['age'] <= 2, 'Hematocrit [Volume Fraction] of Blood'].apply(fill_from_gaussian, **{'mean': 42, 'std': 4})
        df.loc[df['age'].between(2, 12,inclusive='right'), 'Hematocrit [Volume Fraction] of Blood'] = df.loc[df['age'].between(2, 12,inclusive='right'), 'Hematocrit [Volume Fraction] of Blood'].apply(fill_from_gaussian, **{'mean': 35, 'std': 4})
        df.loc[df['age'].between(12, 60,inclusive='right'), 'Hematocrit [Volume Fraction] of Blood'] = df.loc[df['age'].between(12, 60,inclusive='right'), 'Hematocrit [Volume Fraction] of Blood'].apply(fill_from_gaussian, **{'mean': 37, 'std': 2})
        # > 60 Hematocrit should vary for M vs F but not implemented here
        df.loc[df['age'] > 60, 'Hematocrit [Volume Fraction] of Blood'] = df.loc[df['age'] > 60, 'Hematocrit [Volume Fraction] of Blood'].apply(fill_from_gaussian, **{'mean': 42, 'std': 2})

        # df.loc[df['age'] <= 36, 'Glucose [Moles/volume] in Serum or Plasma'] = df.loc[df['age'] <= 36, 'Glucose [Moles/volume] in Serum or Plasma'].apply(fill_from_gaussian, **{'mean': 120, 'std': 30})
        # df.loc[df['age'] > 36, 'Glucose [Moles/volume] in Serum or Plasma'] = df.loc[df['age'] > 36, 'Glucose [Moles/volume] in Serum or Plasma'].apply(fill_from_gaussian, **{'mean': 125, 'std': 25})

        # df['Bicarbonate [Moles/volume] in Venous blood'] = df['Bicarbonate [Moles/volume] in Venous blood'].apply(fill_from_gaussian, **{'mean': 24, 'std': 2})
        # df['Blood arterial pH'] = df['Blood arterial pH'].apply(fill_from_gaussian, **{'mean': 7.4, 'std': .05})


        # Not a good idea to impute WBC by expected values!
        # df.loc[df['age'] <= 1, 'White blood cell count'] = df.loc[df['age'] <= 1, 'White blood cell count'].apply(fill_from_gaussian, **{'mean': 12.5, 'std': 3})
        # df.loc[df['age'].between(1, 24,inclusive='right'), 'White blood cell count'] = df.loc[df['age'].between(1, 24,inclusive='right'), 'White blood cell count'].apply(fill_from_gaussian, **{'mean': 11.5, 'std': 3})
        # df.loc[df['age'].between(24, 96,inclusive='right'), 'White blood cell count'] = df.loc[df['age'].between(24, 96,inclusive='right'), 'White blood cell count'].apply(fill_from_gaussian, **{'mean': 10, 'std': 2.5})
        # df.loc[df['age'] > 96, 'White blood cell count'] = df.loc[df['age'] > 96, 'White blood cell count'].apply(fill_from_gaussian, **{'mean': 9, 'std': 2})
        print('filled NaN values using Gaussians.')
        return None

    def fill_zeros_imputation(self):
        """
        This function fills zeros for the empty rows of one-hot-encoded columns.

        Parameters
        ----------
        'df' : pd.DataFrame
            DataFrame of data.

        Returns
        -------
        none.
        """
        print("Filling zero values for one hot encoding columns.")
        ohe_col_inds = [i.startswith('ohe') for i in list(df.columns)]
        ohe_cols = df.columns[ohe_col_inds]
        for col in ohe_cols:
            df[col] = df[col].apply(lambda x: 0 if pd.isna(x) else x)
        print("Zeros filled.")
        return None

    def categorical_encoding(self):
        """
        Encodes categorical data, as determined by columns with object datatypes.

        Parameters
        ----------
        'df' : pd.DataFrame
            DataFrame of Data

        Returns
        -------
        None. df is updated in place.
        """
        print('Beginning categorical encoding.')
        categorical_cols = list(df.select_dtypes(object).columns)
        le_dictionary = {}
        for name in tqdm(categorical_cols):
            le = LabelEncoder()
            le.fit(df.loc[:,f'{name}'])
            new_col = le.transform(df.loc[:,f'{name}'])
            le_dictionary[name] = le
            df.drop(columns=f'{name}', inplace=True)
            df[f'encoded_{name}'] = new_col
        print('Finished categorical encoding.')
        return None

    def post_join_processing_train(self, entry_threshold: int):
        """
        This function is for training data only.

        Runs operations on dataframe that occur after the tables of patient data have been joined together into a table of factors. This is pre-split.

        Parameters
        ----------
        'df' : pd.DataFrame
            This is a DataFrame of patient data.
        'data_dictionary' : dict
            This is a dictionary of DataFrames that was presumably used to create the DataFrame df. Data from 'person_demographics_episode' will be used to populate age and gender in df.
        'entry_threshold' : int
            Columns with count of values less than this threshold will be removed. That is, non-NaN values.
        
        Details
        -------
        * runs new_person_ids
        * runs birthday_ubiquity
        * runs clearing_cols
        * runs categorical_encoding
        """
        # add flags
        
        self.new_person_ids(df=df)
        df = self.birthday_ubiquity(df = df, data_dictionary = data_dictionary)
        self.add_visit_reason(df=df,data_dictionary= data_dictionary)
        df, cols_cleared = self.clearing_cols(df=df, threshold = entry_threshold, data_type = 'train')
        self.fill_nans_gaussian(df)
        self.fill_zeros_imputation(df)
        self.categorical_encoding(df)

        return df, cols_cleared

    def post_join_processing_test(self, col_list: list):
        """
        This function is for test data only. That is, data without sepsis labels.

        Runs operations on dataframe that occur after the tables of patient data have been joined together into a table of factors. This is pre-split.

        Parameters
        ----------
        'df' : pd.DataFrame
            This is a DataFrame of patient data.
        'data_dictionary' : dict
            This is a dictionary of DataFrames that was presumably used to create the DataFrame df. Data from 'person_demographics_episode' will be used to populate age and gender in df.
        'entry_threshold' : int
            Columns with count of values less than this threshold will be removed. That is, non-NaN values.
        
        Details
        -------
        * runs new_person_ids
        * runs birthday_ubiquity
        * runs clearing_cols
        * runs categorical_encoding
        """
        # add flags
        
        self.new_person_ids(df=df)
        df = self.birthday_ubiquity(df = df, data_dictionary = data_dictionary)
        self.add_visit_reason(df=df,data_dictionary= data_dictionary)
        # Note that threshold for clearing_cols is irrelevant if data_type is set to 'test'.
        # Columns will be cleared based on col_list to match columns cleared from training data in most recent run of post_join_processing_train.
        # cols variable unused but function outputs tuple..
        df, cols = clearing_cols(df=df, threshold = 10000, data_type = 'test', col_list = col_list)
        self.fill_nans_gaussian(df)
        self.fill_zeros_imputation(df)
        self.categorical_encoding(df)
        # Since test does not go through later processing that train data does.
        df.drop(columns=['new_person_id'],inplace=True)

        return df