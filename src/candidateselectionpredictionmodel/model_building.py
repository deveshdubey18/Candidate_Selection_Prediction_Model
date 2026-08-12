from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
import os
import pickle

def model(X_train,X_test,y_train,y_test):
    
    # train model
    model = RandomForestClassifier(random_state=1)
    model.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_test,y_pred)
    
    # print('Accuracy :',accuracy)
    print('Classification report :\n',report)
    
    # save model.pkl file
    
    os.makedirs('models',exist_ok=True)
    
    with open('models/model.pkl','wb') as f:
        pickle.dump(model,f)
        
    result=f'{model.__class__.__name__} Accuracy = {round(accuracy*100,2)}%'
        
   
    return result
 