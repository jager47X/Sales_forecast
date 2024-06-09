# Sales Prediction App

## Overview

The Sales Prediction App is a web-based framework designed to predict future sales for different products based on historical sales data. This app utilizes machine learning techniques to analyze historical data and provide predictions for future dates. Users can upload a CSV file with historical sales data, select parameters like date, weather, holiday, and address, and get predictions for sales and revenue.

## Features

- Upload CSV files containing historical sales data.
- Perform feature engineering and preprocess data.
- Train an XGBoost model with hyperparameter tuning.
- Provide predictions for future sales based on user-selected parameters.
- Display prediction results with the ability to download them as a CSV file.

## Requirements

- Python 3.x
- Flask
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/sales-prediction-app.git
    cd sales-prediction-app
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Start the Flask application:

    ```bash
    python app.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`.

3. Upload your CSV file containing historical sales data. The CSV file should have the following columns:
    - `Date`: The date of the sale (format: MM/DD/YYYY).
    - `Product`: The name of the product.
    - `Number of Sales`: The number of sales.
    - `Weather`: The weather condition on the date of the sale.
    - `Holiday`: Whether the date was a holiday (Yes/No).
    - `Weekend`: Whether the date was a weekend (Yes/No).
    - `Price`: The price of the product.
    - `Address`: The address where the sale took place.

4. After uploading the CSV file, select the parameters for the prediction:
    - Date
    - Weather
    - Holiday
    - Address
    - Products

5. Click the "Predict" button to get the sales predictions.

6. View the prediction results on the right side of the page. The results include the predicted number of sales, price, and revenue for each product.

7. Click the "Save to CSV" button to download the prediction results as a CSV file.

## Extending the Framework

This project is designed as a framework that can be extended by developers to improve accuracy and add more features. You can implement additional input data, such as more detailed weather conditions, economic indicators, or customer demographics, to enhance the model's accuracy. The current implementation is a starting point, and there are many opportunities to customize and expand the app to fit specific use cases.

## File Structure

- `app.py`: The main Flask application script.
- `templates/index.html`: The HTML template for the web interface.
- `static/style.css`: The CSS file for styling the web interface.
- `requirements.txt`: The list of required Python packages.

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push them to your fork.
4. Create a pull request with a description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

This project was developed using the following libraries and frameworks:
- [Flask](https://flask.palletsprojects.com/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [xgboost](https://xgboost.readthedocs.io/)
- [matplotlib](https://matplotlib.org/)
