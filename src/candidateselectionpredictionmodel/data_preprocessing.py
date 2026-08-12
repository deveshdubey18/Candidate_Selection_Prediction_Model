from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

def processing(df):

    # dropping duplicated vals
    df.drop_duplicates()
  
    # dropping unwanted columns
    df = df.drop(columns=['candidate_id','location'])
    
    # encoding categorical to numerical
    df['selected'] = df['selected'].map({'No':0,'Yes':1})
        
    # Split the Target column and Input Features
    X = df.drop(columns='selected')
    y = df['selected']
    
    # split data into categorical and numerical
    categorical = X.select_dtypes(include='object').columns
    numerical = X.select_dtypes(exclude='object').columns

    # Spliting data in to Training and Testing data
    X_train,X_test,y_train,y_test=train_test_split(X,y,
                                                test_size=0.3,
                                                random_state=1)

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ])
    
    transformer = ColumnTransformer([
        ("num", numerical_pipeline, numerical),
        ("cat", categorical_pipeline, categorical)
    ])
    
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)


    # Balancing bias data
    sm=SMOTE()
    X_train,y_train = sm.fit_resample(X_train,y_train) # type: ignore

    return X_train,X_test,y_train,y_test
