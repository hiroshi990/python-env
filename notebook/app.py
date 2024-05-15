from src.logger import logging
import sys
from src.exception import CustomException
if __name__=="__main__":
    logging.info("testing logging.py")

try:
    a=1
    b="hehe"
    c=a+b;
    print(c)
    
except Exception as e:
    logging.info("CUstom exception")
    raise CustomException(e,sys)

        