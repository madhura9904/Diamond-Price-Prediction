'''#Not exactly! Both `DataTransformation` and `ModelTraining.ipynb` will run, but they serve different purposes and will **run at different times in the ML workflow**. Here's how it works:  

---

## **1. `DataTransformation.py` Runs First (Once per Dataset)**
### 📌 **Purpose: Prepare and Save Preprocessing Pipeline**
- Reads the raw dataset.  
- Applies preprocessing (handling missing values, encoding, scaling, etc.).  
- **Fits** the transformations **only on the training data**.  
- Saves the fitted preprocessing pipeline (`preprocessor.pkl`).  
- Saves the transformed dataset (`train_arr`, `test_arr`).  

### **When is this run?**  
✅ Once, when preparing the dataset for model training.  
✅ If you get a **new dataset** or update your features, you need to rerun it.  

---

## **2. `ModelTraining.ipynb` Runs Every Time You Train the Model**
### 📌 **Purpose: Load Processed Data and Train Model**
- **Loads the saved preprocessor (`preprocessor.pkl`)** instead of redefining transformations.  
- Loads the processed `train_arr` and `test_arr`.  
- Trains the ML model.  
- Evaluates performance.  
- Saves the trained model.  

### **When is this run?**  
✅ **Every time you want to train the model** (e.g., trying different models, hyperparameters, etc.).  
✅ If you **change the model architecture or parameters**, you rerun `ModelTraining.ipynb`, but **not `DataTransformation.py` unless the dataset changes**.  

---

### **🚀 Key Takeaway**
- **DataTransformation runs only when the dataset is updated** (to avoid repeated preprocessing).  
- **ModelTraining runs every time you retrain the model** (to test different models, parameters, etc.).  

🔹 **If you never change the dataset, you don't need to rerun `DataTransformation.py`.**  
🔹 **If you tweak the model, you only rerun `ModelTraining.ipynb`.**  

Does this make sense? 😃

1. What Counts as a Dataset Change?
There are two main types of dataset changes:

✅ Adding More Rows (New Data Points)
Example: You collect new sales data or get more customer reviews over time.
In this case, the structure of the dataset remains the same (same columns/features), but there are more rows.
You don’t need to refit the preprocessor; you can just load preprocessor.pkl and transform the new data.
✅ Changing the Structure (New Columns or Different Features)
Example: You introduce a new categorical feature like "Region" in a sales dataset or remove an old column.
This requires refitting the preprocessor since the transformation rules (like encoding, scaling) depend on the columns.
Here, you must rerun DataTransformation.py to fit and save a new preprocessor.pkl.
✅ Completely New Dataset
Example: Switching from a diamonds dataset to a house prices dataset.
Here, the data structure and feature types will likely be different.
You must refit the entire pipeline from scratch and retrain your model.
2. How Does the Machine Know the Dataset Has Changed?
By default, the machine does NOT automatically detect dataset changes—we usually need to tell it manually.

How to Handle Dataset Changes?
🔹 If only new rows are added → No need to rerun DataTransformation.py, just load preprocessor.pkl and apply transform().
🔹 If columns change or a new dataset is used → Rerun DataTransformation.py to create a new preprocessor and save it again.
🔹 Use a versioning system → Track dataset versions (dataset_v1.csv, dataset_v2.csv) to know when preprocessing needs updating.

3. Automating Dataset Change Detection (Optional)
If you want to automate dataset change detection, you can:

Store the dataset’s schema (column names, types) in a file and compare it with the new dataset.
Track dataset hashes (checksum) to detect content changes.
Log dataset updates and set a flag to rerun preprocessing when needed.'''
from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


## Data Transformation config

@dataclass
class DataTransformationconfig:#path of the file where this pipeline should be stored
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')



## Data Ingestionconfig class

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def get_data_transformation_object(self):
         
         try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor#Returns preprocessed file

            logging.info('Pipeline Completed')

         except Exception as e:
            
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)



    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            #Obtaining the preprocessed file
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name,'id']

            ## features into independent and dependent features

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]


            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            ## apply the transformation

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]#Concatenation operation:Combine input feature and targt feature
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            logging.info('Processsor pickle in created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)


    




