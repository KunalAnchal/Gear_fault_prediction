from flask import Flask, render_template, request, send_file, jsonify
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns

app = Flask(__name__)

# Load the model
model = pickle.load(open('model(18).pkl', 'rb'))

# Define the local path to the dataset
DATASET_PATH = ("gear_fault_desc.csv")

@app.route('/')
def index():
    return render_template('index.html', graph=None, data_table=None, excel_file=None, error=None)

@app.route('/predict', methods=['POST'])
def predict_and_plot():
    try:
        # Read the CSV data from the local path
        df = pd.read_csv(DATASET_PATH)


        # Ensure that the CSV file has the expected columns
        required_columns = ['sensor1', 'sensor2', 'speedSet', 'load_value', 'year', 'month', 'day', 'hour', 'minute', 'second']
        if not set(required_columns).issubset(df.columns):
            return render_template('index.html', error="The CSV file is missing required columns.", graph=None, data_table=None, excel_file=None)

        # Perform predictions using the model
        df['Prediction'] = model.predict(df[required_columns])

        # Create distribution plots for Load_Type and Prediction
        plt.rcParams['figure.figsize'] = [10, 6]

        # Create a combined distribution plot
        plt.figure()
        sns.distplot(df['gear_fault'], label='gear_fault')
        sns.distplot(df['Prediction'], label='Prediction')
        plt.title('Distribution of failure and Prediction')
        plt.xlabel('Values')
        plt.ylabel('Density')
        plt.legend()

        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        graph = base64.b64encode(img_buffer.read()).decode('utf-8')

        # Generate data tables for the original data and the prediction
        data_table = df.to_html(classes='table table-condensed table-bordered table-striped')

        # Save the predicted data as an Excel file
        excel_file_path = 'predicted_data.xlsx'
        df.to_excel(excel_file_path, index=False)

        return render_template('index.html', graph=graph, data_table=data_table, excel_file=excel_file_path, error=None)
    except Exception as e:
        return render_template('index.html', error="An error occurred: {}".format(str(e)), graph=None, data_table=None, excel_file=None)

@app.route('/download_excel')
def download_excel():
    try:
        excel_file_path = request.args.get('excel_file')
        return send_file(excel_file_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8088, debug=True)
