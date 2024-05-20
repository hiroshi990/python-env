from src.logger import logging
import sys
from src.exception import CustomException
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import Dataingestion
from src.components.data_ingestion import Dataingestionconfig
from src.components.data_transformation import datatransfromationconfig,datatransformation
from src.components.model_trainer import Modeltrainingconfig,Modeltrainer

if __name__=="__main__":
    logging.info("The execution has started")

    try:
        #data_ingestion_config=DataIngestionConfig()
        data_ingestion=Dataingestion()
        raw_data_path,train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        #data_transformation_config=DataTransformationConfig()
        data_transformation = datatransformation()
        new_train,new_test,preprocessor_file_path = data_transformation.inititate_data_transformation(train_data_path , test_data_path)

        ## Model Training

        model_trainer=Modeltrainer()
        print(model_trainer.initiate_training(new_train,new_test))
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)

        