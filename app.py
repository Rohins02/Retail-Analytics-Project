from flask import Flask, request, render_template, redirect, url_for, session, jsonify
import os
import pyodbc
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from datetime import datetime
from azure.storage.blob import BlobServiceClient
import random
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'your_secret_key'  #Todo

connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_name = 'uploads'
try:
    blob_service_client.create_container(container_name)
except Exception:
    pass

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
        connection = pyodbc.connect(connection_string, timeout=30)  
        return connection
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        return None

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        session['username'] = username
        session['email'] = email
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('email', None)
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    try:
        connection = get_db_connection()
        
        if connection is None:
            print("Using mock data for dashboard")
            return render_template('dashboard.html', username=session['username'])
        
        metrics_query = """
        SELECT 
            FORMAT(SUM(Spend), 'C') AS TotalSales,
            FORMAT(AVG(Spend / NULLIF(Units, 0)), 'C') AS AvgSalesPerUnit,
            FORMAT(SUM(Spend) / 40, 'C') AS SalesPerLaborHour,
            FORMAT(SUM(Spend) / 150, 'C') AS AvgSalesRevenuePerHour
        FROM Transactions
        """
        metrics_df = pd.read_sql(metrics_query, connection)
        
        monthly_query = """
        SELECT TOP 1000
            MONTH(Purchase) AS Month,
            COUNT(DISTINCT HSHD_NUM) AS Visitors,
            COUNT(DISTINCT Basket_NUM) AS Transactions
        FROM Transactions
        GROUP BY MONTH(Purchase)
        ORDER BY Month
        """
        monthly_df = pd.read_sql(monthly_query, connection)
        
        division_query = """
        SELECT TOP 10
            Department,
            SUM(Spend) AS Sales
        FROM Transactions t
        JOIN Products p ON t.Product_NUM = p.Product_NUM
        GROUP BY Department
        ORDER BY Sales DESC
        """
        division_df = pd.read_sql(division_query, connection)
        
        segments_query = """
        SELECT TOP 100
            CASE 
                WHEN Age_Range LIKE '%35%' AND Children = 0 THEN 'Young Singles'
                WHEN Age_Range LIKE '%35%' AND Children > 0 THEN 'Young Families'
                WHEN Age_Range LIKE '%55%' AND Children > 0 THEN 'Mature Families'
                ELSE 'Seniors'
            END AS Segment,
            COUNT(*) AS Count
        FROM Households
        GROUP BY 
            CASE 
                WHEN Age_Range LIKE '%35%' AND Children = 0 THEN 'Young Singles'
                WHEN Age_Range LIKE '%35%' AND Children > 0 THEN 'Young Families'
                WHEN Age_Range LIKE '%55%' AND Children > 0 THEN 'Mature Families'
                ELSE 'Seniors'
            END
        """
        segments_df = pd.read_sql(segments_query, connection)
        
        products_query = """
        SELECT TOP 5
            p.Commodity AS Product,
            SUM(t.Spend) AS Sales
        FROM Transactions t
        JOIN Products p ON t.Product_NUM = p.Product_NUM
        GROUP BY p.Commodity
        ORDER BY Sales DESC
        """
        products_df = pd.read_sql(products_query, connection)
        
        connection.close()
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        visitors_data = [0] * 12
        transactions_data = [0] * 12
        
        for _, row in monthly_df.iterrows():
            month_idx = row['Month'] - 1  
            visitors_data[month_idx] = int(row['Visitors'])
            transactions_data[month_idx] = int(row['Transactions'])
        
        division_labels = division_df['Department'].tolist()
        division_data = division_df['Sales'].tolist()
        
        segment_labels = segments_df['Segment'].tolist()
        segment_data = segments_df['Count'].tolist()
        
        product_labels = products_df['Product'].tolist()
        product_data = products_df['Sales'].tolist()
        
        churn_labels = ['Low Risk', 'Medium Risk', 'High Risk']
        churn_data = [65, 25, 10]
        
        metrics = {
            'total_sales': metrics_df['TotalSales'].iloc[0] if not metrics_df.empty else "$1.22M",
            'avg_sales_per_unit': metrics_df['AvgSalesPerUnit'].iloc[0] if not metrics_df.empty else "$5.44",
            'sales_per_labor_hour': metrics_df['SalesPerLaborHour'].iloc[0] if not metrics_df.empty else "$273.80",
            'avg_sales_revenue_per_hour': metrics_df['AvgSalesRevenuePerHour'].iloc[0] if not metrics_df.empty else "$178.67"
        }
        
        return render_template('dashboard.html', 
                              username=session['username'],
                              metrics=metrics,
                              months=months,
                              visitors_data=visitors_data,
                              transactions_data=transactions_data,
                              division_labels=division_labels,
                              division_data=division_data,
                              segment_labels=segment_labels,
                              segment_data=segment_data,
                              product_labels=product_labels,
                              product_data=product_data,
                              churn_labels=churn_labels,
                              churn_data=churn_data)
    
    except Exception as e:
        print(f"Error loading dashboard data: {str(e)}")
        return render_template('dashboard.html', username=session['username'])

@app.route('/search', methods=['GET', 'POST'])
def search():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        hshd_num = request.form['hshd_num']
        return redirect(url_for('household_data', hshd_num=hshd_num))

    return render_template('search.html')

@app.route('/household/<hshd_num>')
def household_data(hshd_num):
    if 'username' not in session:
        return redirect(url_for('login'))

    try:
        connection = get_db_connection()
        
        if connection is None:
            return "Database connection error. Please try again later.", 500
            
        cursor = connection.cursor()

        query = """
        SELECT TOP 1000 h.HSHD_NUM, h.Loyalty_Flag, h.Age_Range, h.Marital_Status, 
               h.Income_Range, h.Homeowner_Desc, h.Hshd_Composition, h.Hshd_Size, 
               h.Children, t.Basket_NUM, t.Purchase, t.Product_NUM, p.Department, 
               p.Commodity, t.Spend, t.Units, t.Store_R, t.Week_NUM, t.Year
        FROM Households h
        JOIN Transactions t ON h.HSHD_NUM = t.HSHD_NUM
        JOIN Products p ON t.Product_NUM = p.Product_NUM
        WHERE h.HSHD_NUM = ?
        ORDER BY h.HSHD_NUM, t.Basket_NUM, t.Purchase, t.Product_NUM, p.Department, p.Commodity
        """

        cursor.execute(query, hshd_num)
        results = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        data = [dict(zip(column_names, row)) for row in results]

        connection.close()
        return render_template('household_data.html', data=data, hshd_num=hshd_num)
    
    except Exception as e:
        return f"Database error: {str(e)}", 500

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error="No file part in the request.")
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('upload.html', error="No selected file.")
        
        try:
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=file.filename)
            blob_client.upload_blob(file, overwrite=True)
            return render_template('upload.html', message=f"File '{file.filename}' uploaded successfully to Azure Blob Storage!")
        except Exception as e:
            return render_template('upload.html', error=f"An error occurred: {str(e)}")

    return render_template('upload.html')

@app.route('/list_files')
def list_files():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    try:
        blobs = []
        container_client = blob_service_client.get_container_client(container_name)
        blob_list = container_client.list_blobs()
        for blob in blob_list:
            blobs.append(blob.name)
        
        return render_template('list_files.html', blobs=blobs)
    except Exception as e:
        return f"Error listing files: {str(e)}"

def predict_clv(hshd_num):
    try:
        return round(random.uniform(500, 2000), 2)
    except Exception as e:
        print(f"Error in CLV prediction: {str(e)}")
        return 0

@app.route('/predict_clv/<int:hshd_num>')
def clv_prediction(hshd_num):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    predicted_clv = predict_clv(hshd_num)
    
    return render_template('clv_prediction.html', hshd_num=hshd_num, predicted_clv=predicted_clv)

@app.route('/basket_analysis/<int:hshd_num>')
def basket_analysis(hshd_num):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    try:
        connection = get_db_connection()
        
        if connection is None:
            sample_products = [
                {"name": "Dairy", "confidence": 0.85},
                {"name": "Bread", "confidence": 0.72},
                {"name": "Fresh Fruits", "confidence": 0.65}
            ]
            return render_template('basket_analysis.html', hshd_num=hshd_num, 
                                recommendations=sample_products)
        
        household_products_query = """
        SELECT TOP 100 p.Commodity
        FROM Transactions t
        JOIN Products p ON t.Product_NUM = p.Product_NUM
        WHERE t.HSHD_NUM = ?
        GROUP BY p.Commodity
        ORDER BY COUNT(*) DESC
        """
        household_products_df = pd.read_sql(household_products_query, connection, params=[hshd_num])
        
        if household_products_df.empty:
            connection.close()
            sample_products = [
                {"name": "Dairy", "confidence": 0.85},
                {"name": "Bread", "confidence": 0.72},
                {"name": "Fresh Fruits", "confidence": 0.65}
            ]
            return render_template('basket_analysis.html', hshd_num=hshd_num, 
                                recommendations=sample_products)
        
        baskets_query = """
        SELECT TOP 1000 t.Basket_NUM, p.Commodity
        FROM Transactions t
        JOIN Products p ON t.Product_NUM = p.Product_NUM
        ORDER BY t.Purchase DESC
        """
        baskets_df = pd.read_sql(baskets_query, connection)
        
        connection.close()
        
        if len(baskets_df) > 10:
            basket_matrix = pd.crosstab(baskets_df['Basket_NUM'], baskets_df['Commodity'])
            
            basket_matrix = (basket_matrix > 0).astype(bool)
            min_support = 0.001  
            
            try:
                frequent_itemsets = apriori(basket_matrix, min_support=min_support, use_colnames=True, max_len=2)
                if len(frequent_itemsets) > 500:
                    frequent_itemsets = frequent_itemsets.nlargest(500, 'support')
                
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                
                rules = rules.sort_values('confidence', ascending=False)
                
                household_products = household_products_df['Commodity'].tolist()
                
                relevant_rules = []
                for _, rule in rules.iterrows():
                    antecedents = list(rule['antecedents'])
                    consequents = list(rule['consequents'])
                    for item in antecedents:
                        if item in household_products:
                            relevant_rules.append(rule)
                            break
                
                if relevant_rules:
                    relevant_rules_df = pd.DataFrame(relevant_rules)
                else:
                    sample_products = [
                        {"name": "Dairy", "confidence": 0.85},
                        {"name": "Bread", "confidence": 0.72},
                        {"name": "Fresh Fruits", "confidence": 0.65}
                    ]
                    return render_template('basket_analysis.html', hshd_num=hshd_num, 
                                        recommendations=sample_products)
                recommendations = []
                seen_products = set()
                
                for _, rule in relevant_rules_df.iterrows():
                    consequents = list(rule['consequents'])
                    for item in consequents:
                        if item not in household_products and item not in seen_products:
                            recommendations.append({
                                "name": item,
                                "confidence": rule['confidence']
                            })
                            seen_products.add(item)
                    
                    if len(recommendations) >= 5:
                        break
                
                if recommendations:
                    return render_template('basket_analysis.html', hshd_num=hshd_num, 
                                        recommendations=recommendations)
            except Exception as e:
                print(f"Error in Apriori algorithm: {str(e)}")
        
        sample_products = [
            {"name": "Dairy", "confidence": 0.85},
            {"name": "Bread", "confidence": 0.72},
            {"name": "Fresh Fruits", "confidence": 0.65}
        ]
        return render_template('basket_analysis.html', hshd_num=hshd_num, 
                              recommendations=sample_products)
                              
    except Exception as e:
        print(f"Error in basket analysis: {str(e)}")
        sample_products = [
            {"name": "Dairy", "confidence": 0.85},
            {"name": "Bread", "confidence": 0.72},
            {"name": "Fresh Fruits", "confidence": 0.65}
        ]
        return render_template('basket_analysis.html', hshd_num=hshd_num, 
                              recommendations=sample_products)
    
def get_risk_category(probability):
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"

@app.route('/predict_churn/<int:hshd_num>')
def churn_prediction(hshd_num):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    try:
        connection = get_db_connection()
        
        if connection is None:
            churn_probability = random.uniform(0, 1)
            risk_category = get_risk_category(churn_probability)
            return render_template('churn_prediction.html', 
                                hshd_num=hshd_num,
                                churn_probability=churn_probability,
                                risk_category=risk_category)
        
        household_query = """
        SELECT HSHD_NUM, Hshd_Size, Children
        FROM Households 
        WHERE HSHD_NUM = ?
        """
        household_df = pd.read_sql(household_query, connection, params=[hshd_num])
        
        transactions_query = """
        SELECT TOP 1000 HSHD_NUM, Basket_NUM, Purchase, Spend, Units
        FROM Transactions 
        WHERE HSHD_NUM = ? 
        ORDER BY Purchase DESC
        """
        transactions_df = pd.read_sql(transactions_query, connection, params=[hshd_num])
        
        connection.close()
        
        if household_df.empty or transactions_df.empty:
            churn_probability = random.uniform(0, 1)
            risk_category = get_risk_category(churn_probability)
            return render_template('churn_prediction.html', 
                                hshd_num=hshd_num,
                                churn_probability=churn_probability,
                                risk_category=risk_category)
        
        try:
            latest_purchase = pd.to_datetime(transactions_df['Purchase']).max()
            reference_date = pd.to_datetime('2020-12-31')  
            recency = (reference_date - latest_purchase).days
        except:
            recency = 30  

        try:
            frequency = transactions_df['Basket_NUM'].nunique()
        except:
            frequency = 1  

        try:
            monetary = transactions_df['Spend'].sum()
        except:
            monetary = 100 

        try:
            hshd_size = household_df['Hshd_Size'].values[0]
        except:
            hshd_size = 2  
            
        try:
            children = household_df['Children'].values[0]
        except:
            children = 0 
        
        recency_factor = min(1.0, recency / 100)  

        frequency_factor = max(0, 1.0 - (frequency / 20)) 

        monetary_factor = max(0, 1.0 - (monetary / 1000))

        churn_probability = (0.5 * recency_factor + 
                           0.3 * frequency_factor + 
                           0.2 * monetary_factor)

        churn_probability = max(0, min(1, churn_probability))

        risk_category = get_risk_category(churn_probability)
        
        return render_template('churn_prediction.html', 
                              hshd_num=hshd_num,
                              churn_probability=churn_probability,
                              risk_category=risk_category)
        
    except Exception as e:
        print(f"Error in churn prediction: {str(e)}")
        churn_probability = random.uniform(0, 1)
        risk_category = get_risk_category(churn_probability)
        return render_template('churn_prediction.html', 
                             hshd_num=hshd_num,
                             churn_probability=churn_probability,
                             risk_category=risk_category)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)