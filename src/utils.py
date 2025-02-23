import sys,os
import pandas as pd
import numpy as np
import pickle

from src.exception import CustomException
from src.logger import logging

#Saving the pickle file
def save_object(file_path,obj):#file path where we need to save pkl file
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)

    
    except Exception as e:
        raise CustomException(e,sys)