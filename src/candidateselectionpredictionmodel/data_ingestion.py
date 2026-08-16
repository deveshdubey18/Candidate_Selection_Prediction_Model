import pandas as pd
import numpy as np 
import logging
def data_ingestion():
    df = pd.read_csv(r'C:\Devesh ITV\Machine Learning\CandidateSelectionPredictionModel\data\resume_screening_dataset.csv')
    logging.info('Data Ingestion : Done')
    return df
    