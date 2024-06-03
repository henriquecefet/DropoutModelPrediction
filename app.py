from flask import Flask, request, jsonify, send_from_directory, render_template
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd
import joblib
import traceback

app = Flask(__name__, static_folder='static')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model4GPA')
def model4GPA():
    return render_template('model4GPA/model4GPA.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/model4GPA/predict', methods=['POST'])
def predict():
    try:
        # Load the model
        pipeline = joblib.load('model4GPA_model_and_encoders/model4GPA.pkl')

        # Carregar os LabelEncoders
        label_encoders = {
            'ingresso_atual': joblib.load('model4GPA_model_and_encoders/label_encoder_ingresso_atual.pkl'),
            'Categoria': joblib.load('model4GPA_model_and_encoders/label_encoder_Categoria.pkl'),
            'IsTheyBusinessperson': joblib.load('model4GPA_model_and_encoders/label_encoder_IsTheyBusinessperson.pkl'),
            'SEXO': joblib.load('model4GPA_model_and_encoders/label_encoder_SEXO.pkl'),
            'employee_student': joblib.load('model4GPA_model_and_encoders/label_encoder_employee_student.pkl')
        }

        # Define as colunas numéricas e categóricas
        numerical_columns = ['GPA1', 'GPA2', 'GPA3', 'GPA4', 'grade_programming1', 'grade_programming2', 'grade_basic_math', 'grade_calculus1', 'grade_linear_algebra']
        categorical_columns = ['ingresso_atual', 'IsTheyBusinessperson', 'Categoria', 'SEXO', 'employee_student']
        # Get the JSON data from the request
        data = request.json
        
        # Create a DataFrame from the JSON data
        df = pd.DataFrame(data, index=[0])
        
        # Apply label encoding to categorical columns
        for name, le in label_encoders.items():
            try:
                df[name] = le.transform(df[name])
            except ValueError as e:
                return jsonify({'error': f"Unknown category in column '{name}': {str(e)}"}), 400
        
        # Convert all non-categorical columns to float
        for column in numerical_columns:
            df[column] = df[column].astype(float)
        
        # Print column types for debugging
        for column in df.columns:
            print(f"Column '{column}' has type: {type(df[column].iloc[0])}")
        
        # Select features for prediction
        X = df[numerical_columns + categorical_columns]
        
        # Print the transformed DataFrame for debugging
        print(X)
        
        y_pred = pipeline.predict(X)
        print("-----------------------------------------------------------")
        print(y_pred.tolist())
        # Return the predictions as JSON
        return jsonify({'prediction': y_pred.tolist()[0]})
    except Exception as e:
        error_message = f"Error: {str(e)}"
        traceback_str = ''.join(traceback.format_tb(e.__traceback__))
        full_error_message = f"{error_message}\nTraceback:\n{traceback_str}"
        print(full_error_message)
        return jsonify(error=full_error_message), 500

if __name__ == '__main__':
    app.run(debug=True)

