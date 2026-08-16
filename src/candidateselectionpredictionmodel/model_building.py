from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
import logging

def model(X_train,X_test,y_train,y_test):
    logging.info('Random Forest Model Started!!')
    
    # train model
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    logging.info('Model Training : Done')
    
    y_pred = model.predict(X_test)
    logging.info('Model Testing : Done')
    
    # Evaluation
    accuracy = accuracy_score(y_test,y_pred)
    logging.info(f'Model Accuracy : {accuracy}')
    report = classification_report(y_test,y_pred)
    
    print('Classification report :\n',report)
    
    # save model.pkl file
    
    os.makedirs('models',exist_ok=True)
    
    with open('models/model.pkl','wb') as f:
        pickle.dump(model,f)
    logging.info('Model Pickle File Generated!!')
        
    result=f'{model.__class__.__name__} Accuracy = {round(accuracy*100,2)}%'
    
    return result
 