import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.components.data_ingestion import Dataingestion
from src.utils import save_object


from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging

@dataclass
class datatransfromationconfig:
    pkl_file_path=str(os.path.join("artifacts","preprocessor.pkl"))
    
class datatransformation(datatransfromationconfig):
    # def __init__(self) -> None:
    #     self.datatransfromationconfig=datatransfromationconfig()
        
        
    def transformation(self):
        '''
        this function is responsible
        for data trnasformation
        '''
        try:
            
            numeric_features = ["writing_score", "reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
            ])
           
            cat_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehot",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
            ])
            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,numeric_features),
                ("cat_pipelinr",cat_pipeline,categorical_features)
                
            ])
           
            return preprocessor
            
        except Exception as e :
            raise CustomException(e,sys)
        
    def inititate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
                
            logging.info("Reading the train and test file")
            preprocessor_obj=self.transformation()
                
            logging.info("dividing the train dataset as xtrain and ytrain")
            target_column="math_score"
            X_train=train_df.drop(columns=[target_column])
            Y_train=train_df[target_column]
            
            logging.info("Dividing the test dataset")
            X_test=test_df.drop(columns=[target_column])
            Y_test=test_df[target_column]
            
            logging.info("applying the preprocessor")
            trained_xdata= preprocessor_obj.fit_transform(X_train)
            test_xdata= preprocessor_obj.transform(X_test)
            
            '''
            now since the train and test data has been transformed we hace the new dataset on which we will
            train our model and so predictions 
            now lets concatenate the diveded xtrain ytrain adn xtest ytest
            '''
            new_train=np.c_[trained_xdata,np.array(Y_train)]
            new_test=np.c_[test_xdata,np.array(Y_test)]
            logging.info("Saved preprocessing object")
            
            save_object(

                file_path=super().pkl_file_path,
                obj=preprocessor_obj
            )
            
            return(
                new_test,
                new_train,
                super().pkl_file_path
            )
            
            
                
                
        except Exception as e:
            raise CustomException(e,sys)
                
        
            

