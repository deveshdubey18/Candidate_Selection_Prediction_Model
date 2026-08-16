from src.candidateselectionpredictionmodel.data_ingestion import data_ingestion
from src.candidateselectionpredictionmodel.data_preprocessing import processing
from src.candidateselectionpredictionmodel.model_building import model
import logging
logging.basicConfig(level=logging.INFO,
                    filename='model.log',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)


def main():
    logging.info('Machine Learning Model Started!!')
    
    df=data_ingestion()
    print(df.shape)
    
    X_train,X_test,y_train,y_test = processing(df)
    
    result = model(X_train,X_test,y_train,y_test)
    
    print(result)
    logging.info('Machine Learning Model Ended!!')
main()