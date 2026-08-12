from src.candidateselectionpredictionmodel.data_ingestion import data_ingestion
from src.candidateselectionpredictionmodel.data_preprocessing import processing
from src.candidateselectionpredictionmodel.model_building import model


def main():
    
    df=data_ingestion()
    print(df.shape)
    
    X_train,X_test,y_train,y_test = processing(df)
    
    result = model(X_train,X_test,y_train,y_test)
    
    print(result)
    
main()