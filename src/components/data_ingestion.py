import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
@dataclass
class Dataingestionconfig:
    train_data_path=str(os.path.join("artifacts","train.csv"))
    test_data_path=str(os.path.join("artifacts","test.csv"))
    raw_data_path: str=os.path.join("artifacts","data.csv")
    
class Dataingestion(Dataingestionconfig):
    # def __init__(self):
    #     self.ingestion_config=Dataingestionconfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df=pd.read_csv("notebook\data\stud.csv")
            logging.info("read the dataset")
            os.makedirs(os.path.dirname(super().train_data_path),exist_ok=True)
            df.to_csv(super().raw_data_path,index=False,header=True)
            
            logging.info("train_test_split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(super().train_data_path,index=False,header=True)
            test_set.to_csv(super().test_data_path,index=False,header=True)
            logging.info("ingestion of the data is completed")
            return(
                super().raw_data_path,
                super().train_data_path,
                super().test_data_path
                
            )
            
            
        except Exception as e:
            logging.info("Custom Exception")
            raise CustomException(e,sys)
           
           
           
           
if __name__=="__main__":
    obj=Dataingestion()
    obj.initiate_data_ingestion()        
        
        