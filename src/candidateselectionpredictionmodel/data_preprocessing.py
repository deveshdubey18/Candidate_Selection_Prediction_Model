from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import logging

def processing(df):
    
    logging.info('Data PreProcessing Started!!!')

    # dropping duplicated vals
    df.drop_duplicates()
    logging.info('Duplicates removed : Done')
  
    # dropping unwanted columns
    df = df.drop(columns=['candidate_id','location'])
    logging.info('Unwanted Columns Dropped : Done')
    
    # encoding categorical to numerical
    df['selected'] = df['selected'].map({'No':0,'Yes':1})
    logging.info('Target Columns Encoded : Done')    
    
    # Split the Target column and Input Features
    X = df.drop(columns='selected')
    y = df['selected']
    logging.info('Data Split into Input Fratures : Done')
    
    # split data into categorical and numerical
    categorical = X.select_dtypes(include='object').columns
    numerical = X.select_dtypes(exclude='object').columns 
    logging.info('Data Split into Categorical & Numerical : Done')
    
    # Spliting data in to Training and Testing data
    X_train,X_test,y_train,y_test=train_test_split(X,y,
                                                test_size=0.3,
                                                random_state=1)
    logging.info('Data Split into Training and Testing : Done')

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])
    logging.info('Numerical PipeLine Created!!')
    
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ])
    logging.info('Categorical PipeLine Created!!')
    
    transformer = ColumnTransformer([
        ("num", numerical_pipeline, numerical),
        ("cat", categorical_pipeline, categorical)
    ])
    logging.info('Tranformer Created!!')
    
    X_train = transformer.fit_transform(X_train) # type: ignore
    X_test = transformer.transform(X_test)
    logging.info('PipeLine Running successfully : Done')

    # Balancing bias data
    sm=SMOTE()
    X_train,y_train = sm.fit_resample(X_train,y_train) # type: ignore
    logging.info('Target Columns Balanced by SMOTE : Done')
    
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train) # type: ignore
    X_test = pca.transform(X_test)
    logging.info('PCA Applied : Done')

    logging.info('Pre Processing Done Successfully')
    return X_train,X_test,y_train,y_test
