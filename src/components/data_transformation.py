import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.components.data_ingestion import Dataingestion

from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging

@dataclass
class datatransfromationconfig:
    pkl_file_path=os.path.join("artifacts","preprocessor.pkl")
    
class datatransformation:
    def __init__(self) -> None:
        self.datatransfromationconfig=datatransfromationconfig()
        
        
    def transformation(self):
        '''
        this function is responsible
        for data trnasformation
        '''
        try:
            numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
            categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
            # print columns
            print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
            print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))
            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="median")),
                 ("scaler",StandardScaler())
            ])
            
            cat_pipreline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehot",OneHotEncoder()),
                ("scaler",StandardScaler())
            ])
            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,numeric_features),
                ("cat_pipelinr",cat_pipreline,categorical_features)
                
            ])
            
            return preprocessor
            
        except Exception as e :
            raise CustomException(e,sys)
        
        def inititate_data_transformation(self,train_path,test_path):
            
        
            

