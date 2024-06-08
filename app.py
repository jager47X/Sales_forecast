from flask import Flask, request, render_template, jsonify, send_file, make_response
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time
import io
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Global variables
data = None
bestModel = None
MAE = None
MSE = None
R2 = None
ACC = None
prediction_csv = None


@app.route('/upload', methods=['POST'])
def upload():
    global data, bestModel, MAE, MSE, R2, ACC

    start_time = time.time()

    try:
        # Load the CSV file
        file = request.files['file']
        if not file:
            raise ValueError("No file provided")
        df = pd.read_csv(file)
        load_time = time.time()
        logging.debug(f"Time to load data: {load_time - start_time:.2f} seconds")
        logging.debug(f"Dataframe head:\n{df.head()}")
        logging.debug(f"Dataframe columns: {df.columns}")

        # Ensure necessary columns are present
        required_columns = ['Date', 'Product', 'Number of Sales', 'Weather', 'Holiday', 'Weekend', 'Price', 'Address']
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"Missing required column: {column}")

        # Convert the `Date` column to datetime with the correct format
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

        # Feature engineering
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'] >= 5

        # Adding interactions between features
        df['Year_Month'] = df['Year'] * df['Month']
        df['Month_Day'] = df['Month'] * df['Day']

        feature_engineering_time = time.time()
        logging.debug(f"Time for feature engineering: {feature_engineering_time - load_time:.2f} seconds")

        # Define the features and target
        features = ['Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekend', 'Weather', 'Holiday', 'Price', 'Year_Month',
                    'Month_Day']
        target = 'Number of Sales'

        # Split the data into training and test sets
        X = df[features]
        y = df[target]
        if X.empty or y.empty:
            raise ValueError("Feature set or target is empty after splitting data")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Log the shape of the datasets
        logging.debug(f'Training set shape: X_train={X_train.shape}, y_train={y_train.shape}')
        logging.debug(f'Test set shape: X_test={X_test.shape}, y_test={y_test.shape}')

        # Preprocessing for numerical and categorical features
        numerical_features = ['Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekend', 'Price', 'Year_Month', 'Month_Day']
        categorical_features = ['Weather', 'Holiday']

        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Define the model pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(random_state=42))
        ])

        # Hyperparameter tuning
        param_grid = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'regressor__max_depth': [3, 4, 5, 6]
        }

        logging.debug('Starting hyperparameter tuning...')
        tuning_start_time = time.time()
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        tuning_end_time = time.time()
        logging.debug(f"Time for hyperparameter tuning: {tuning_end_time - tuning_start_time:.2f} seconds")

        # Best model from GridSearchCV
        best_model = grid_search.best_estimator_
        logging.debug(f'Best model parameters: {grid_search.best_params_}')

        # Predict on the test set
        logging.debug('Predicting on the test set...')
        y_pred = best_model.predict(X_test)

        # Calculate accuracy metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        average_sales = df['Number of Sales'].mean()
        accuracy = mae / average_sales * 100

        # Adjust R-squared value to be non-negative
        r2_percentage = max(r2 * 100, 0)

        logging.debug(f'Accuracy metrics: MAE={mae}, MSE={mse}, R²={r2_percentage}, Accuracy={accuracy}%')

        unique_weather = df['Weather'].unique().tolist()
        unique_address = df['Address'].unique().tolist()
        unique_products = df['Product'].unique().tolist()

        # Debugging: Log the unique values being passed to the template
        logging.debug(f'Unique weather values: {unique_weather}')
        logging.debug(f'Unique address values: {unique_address}')
        logging.debug(f'Unique product values: {unique_products}')

        total_time = time.time()
        logging.debug(f"Total time for processing: {total_time - start_time:.2f} seconds")

        # Assign global variables
        data = df
        bestModel = best_model
        MAE = mae
        MSE = mse
        R2 = r2_percentage
        ACC = accuracy

        # Return the unique values as a JSON response
        return jsonify({
            'weather': unique_weather,
            'address': unique_address,
            'products': unique_products,
            'processing_time': total_time - start_time,
            'load_time': load_time - start_time,
            'feature_engineering_time': feature_engineering_time - load_time,
            'tuning_time': tuning_end_time - tuning_start_time
        })
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global data, bestModel, MAE, MSE, R2, ACC, prediction_csv

    try:
        if data is None or bestModel is None:
            raise ValueError("Model and data must be initialized before prediction")

        requestData = request.json
        logging.debug("Received data: %s", requestData)

        # Parse the selected date
        selected_date = pd.to_datetime(requestData['date'], format='%Y-%m-%d')

        # Check if the selected date is in the future
        if selected_date <= data['Date'].max():
            raise ValueError("Selected date must be in the future.")

        # Prepare input data for prediction
        product_sales = []
        for product in requestData['products']:
            product_name = product['name']
            if product_name not in data['Product'].values:
                raise ValueError(f"Product {product_name} not found in the data")
            product_price = data[data['Product'] == product_name]['Price'].values[0]
            input_data = pd.DataFrame({
                'Year': [selected_date.year],
                'Month': [selected_date.month],                'Day': [selected_date.day],
                'DayOfWeek': [selected_date.dayofweek],
                'IsWeekend': [selected_date.dayofweek >= 5],
                'Weather': [requestData['weather']],
                'Holiday': [requestData['holiday']],
                'Price': [product_price],
                'Year_Month': [selected_date.year * selected_date.month],
                'Month_Day': [selected_date.month * selected_date.day]
            })

            logging.debug(f'Input data for product {product_name}: {input_data}')

            # Predict sales
            forecasted_sales = bestModel.predict(input_data)[0]
            revenue = forecasted_sales * product_price
            product_sales.append({
                'product_name': product_name,
                'forecasted_sales': float(forecasted_sales),
                'revenue': float(revenue),
                'price': float(product_price)
            })

        response = {
            'selected_date': str(selected_date.date()),
            'weather': requestData['weather'],
            'address': requestData['address'],
            'product_sales': product_sales,
            'mae': float(MAE),
            'mse': float(MSE),
            'r2': float(R2),
            'accuracy': float(ACC)
        }

        # Save prediction results to CSV
        prediction_results = pd.DataFrame(product_sales)
        csv_buffer = io.StringIO()
        prediction_results.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        prediction_csv = csv_buffer.getvalue()

        logging.info("Prediction response: %s", response)

        return jsonify(response)
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/download_csv')
def download_csv():
    global prediction_csv
    response = make_response(prediction_csv)
    response.headers["Content-Disposition"] = "attachment; filename=prediction_results.csv"
    response.headers["Content-Type"] = "text/csv"
    return response

if __name__ == '__main__':
    app.run(debug=True)


