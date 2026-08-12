import pandas as pd

def data_ingestion():
    df = pd.read_csv(r'C:\Devesh ITV\Machine Learning\CandidateSelectionPredictionModel\data\resume_screening_dataset.csv')
    
    return df