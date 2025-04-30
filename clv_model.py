import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pyodbc
import os

def get_db_connection():
    try:
        server = os.getenv('DB_SERVER')
        database = os.getenv('DB_NAME')
        username = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        
        if not all([server, database, username, password]):
            print("Database environment variables not properly set")
            return None
            
        driver = '{ODBC Driver 17 for SQL Server}'
        connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'
        connection = pyodbc.connect(connection_string, timeout=30)  # Add timeout parameter
        return connection
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        return None

def train_clv_model():
    connection = get_db_connection()
    
    if connection is None:
        print("Cannot train model: Database connection failed")
        return None, None
    
    try:
        households = pd.read_sql("SELECT * FROM Households", connection)
        transactions = pd.read_sql("SELECT * FROM Transactions", connection)
        products = pd.read_sql("SELECT * FROM Products", connection)
        
        connection.close()
        
        max_date = pd.to_datetime(transactions['Date']).max()
        
        rfm = transactions.groupby('HSHD_NUM').agg({
            'Date': lambda x: (max_date - pd.to_datetime(x).max()).days,  
            'Basket_NUM': 'nunique',  
            'Spend': 'sum'  
        }).rename(columns={'Date': 'Recency', 'Basket_NUM': 'Frequency', 'Spend': 'Monetary'})
        
        features = rfm.merge(households, on='HSHD_NUM')
        
        features['CLV'] = features['Monetary']
        
        X = features.drop(['HSHD_NUM', 'CLV', 'Loyalty_Flag', 'Age_Range', 'Marital_Status', 
                        'Income_Range', 'Homeowner_Desc', 'Hshd_Composition'], axis=1)
        y = features['CLV']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.2f}")
        
        joblib.dump(model, 'clv_model.joblib')
        joblib.dump(scaler, 'clv_scaler.joblib')
        
        return model, scaler
    
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        if connection:
            connection.close()
        return None, None

def predict_clv(hshd_num):
    try:
        try:
            model = joblib.load('clv_model.joblib')
            scaler = joblib.load('clv_scaler.joblib')
        except FileNotFoundError:
            print("Model files not found, training model...")
            model, scaler = train_clv_model()
            
            if model is None or scaler is None:
                print("Could not train model, returning default value")
                return 1000.0  
        
        connection = get_db_connection()
        
        if connection is None:
            print("Database connection failed, returning default value")
            return 1000.0  
        
        try:
            transactions = pd.read_sql(f"SELECT * FROM Transactions WHERE HSHD_NUM = {hshd_num}", connection)
            household = pd.read_sql(f"SELECT * FROM Households WHERE HSHD_NUM = {hshd_num}", connection)
            
            connection.close()
            
            if transactions.empty or household.empty:
                print(f"No data available for household {hshd_num}")
                return 500.0  
            
            max_date = pd.to_datetime(transactions['Date']).max()
            recency = (max_date - pd.to_datetime(transactions['Date']).max()).days
            frequency = transactions['Basket_NUM'].nunique()
            monetary = transactions['Spend'].sum()
            
            hshd_size = household['Hshd_Size'].values[0]
            children = household['Children'].values[0]
            
            features = np.array([[recency, frequency, monetary, hshd_size, children]])
            
            features_scaled = scaler.transform(features)
            
            predicted_clv = model.predict(features_scaled)[0]
            
            return predicted_clv
            
        except Exception as e:
            print(f"Error in getting household data: {str(e)}")
            if connection:
                connection.close()
            return 750.0  
            
    except Exception as e:
        print(f"Error in CLV prediction: {str(e)}")
        return 750.0  

if __name__ == "__main__":
    train_clv_model()