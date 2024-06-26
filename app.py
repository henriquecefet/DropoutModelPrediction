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
@app.route('/static/<path:path>')

def send_static(path):
    return send_from_directory('static', path)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
@app.route('/bsi')
def bsi_index():
    return render_template('BSI/bsiindex.html')

@app.route('/bsi/model4GPA')
def bsi_model4GPA():
    return render_template('BSI/model4GPA/model4GPA.html')


@app.route('/bsi/model4GPA/predict', methods=['POST'])
def bsi_model4gpa_predict():
    try:
        # Load the model
        pipeline = joblib.load('model4GPA_model_and_encoders/BSI/model4GPA/model4GPA.pkl')

        # Carregar os LabelEncoders
        label_encoders = {
            'ingresso_atual': joblib.load('model4GPA_model_and_encoders/BSI/model4GPA/label_encoder_ingresso_atual.pkl'),
            'Categoria': joblib.load('model4GPA_model_and_encoders/BSI/model4GPA/label_encoder_Categoria.pkl'),
            'IsTheyBusinessperson': joblib.load('model4GPA_model_and_encoders/BSI/model4GPA/label_encoder_IsTheyBusinessperson.pkl'),
            'SEXO': joblib.load('model4GPA_model_and_encoders/BSI/model4GPA/label_encoder_SEXO.pkl'),
            'employee_student': joblib.load('model4GPA_model_and_encoders/BSI/model4GPA/label_encoder_employee_student.pkl')
        }

        # Define as colunas numéricas e categóricas
        numerical_columns = ['GPA1', 'GPA2', 'GPA3', 'GPA4', 'grade_programming1', 'grade_programming2', 'grade_basic_math', 'grade_calculus1', 'grade_linear_algebra', 'grade_logic', 'CR_ATUAL']
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
    
#-------------------------------------------------------------------------------------------------------------------------------


@app.route('/bsi/model3GPA')
def bsi_model3GPA():
    return render_template('BSI/model3GPA/model3GPA.html')

@app.route('/bsi/model3GPA/predict', methods=['POST'])
def bsi_model3gpa_predict():
    try:
        # Load the model
        pipeline = joblib.load('model4GPA_model_and_encoders/BSI/model3GPA/model3GPA.pkl')

        # Carregar os LabelEncoders
        label_encoders = {
            'ingresso_atual': joblib.load('model4GPA_model_and_encoders/BSI/model3GPA/label_encoder_ingresso_atual.pkl'),
            'Categoria': joblib.load('model4GPA_model_and_encoders/BSI/model3GPA/label_encoder_Categoria.pkl'),
            'IsTheyBusinessperson': joblib.load('model4GPA_model_and_encoders/BSI/model3GPA/label_encoder_IsTheyBusinessperson.pkl'),
            'SEXO': joblib.load('model4GPA_model_and_encoders/BSI/model3GPA/label_encoder_SEXO.pkl'),
            'employee_student': joblib.load('model4GPA_model_and_encoders/BSI/model3GPA/label_encoder_employee_student.pkl')
        }

        # Define as colunas numéricas e categóricas
        categorical_columns = [ 'ingresso_atual', 'Categoria', 'IsTheyBusinessperson', 'SEXO', 'employee_student']
        numerical_columns = ['CR_ATUAL', 'GPA1', 'GPA2', 'GPA3', 'grade_programming1', 'grade_programming2', 'grade_basic_math', 'grade_calculus1', 'grade_linear_algebra', 'grade_logic']
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



#-------------------------------------------------------------------------------------------------------------------------------


@app.route('/bsi/model2GPA')
def bsi_model2GPA():
    return render_template('BSI/model2GPA/model2GPA.html')

@app.route('/bsi/model2GPA/predict', methods=['POST'])
def bsi_model2gpa_predict():
    try:
        # Load the model
        pipeline = joblib.load('model4GPA_model_and_encoders/BSI/model2GPA/model2GPA.pkl')

        # Carregar os LabelEncoders
        label_encoders = {
            'ingresso_atual': joblib.load('model4GPA_model_and_encoders/BSI/model2GPA/label_encoder_ingresso_atual.pkl'),
            'Categoria': joblib.load('model4GPA_model_and_encoders/BSI/model2GPA/label_encoder_Categoria.pkl'),
            'IsTheyBusinessperson': joblib.load('model4GPA_model_and_encoders/BSI/model2GPA/label_encoder_IsTheyBusinessperson.pkl'),
            'SEXO': joblib.load('model4GPA_model_and_encoders/BSI/model2GPA/label_encoder_SEXO.pkl'),
            'employee_student': joblib.load('model4GPA_model_and_encoders/BSI/model2GPA/label_encoder_employee_student.pkl')
        }

        # Define as colunas numéricas e categóricas
        categorical_columns = ['ingresso_atual', 'Categoria', 'IsTheyBusinessperson', 'SEXO', 'employee_student']
        numerical_columns = ['CR_ATUAL', 'GPA1', 'GPA2', 'grade_programming1', 'grade_programming2', 'grade_basic_math', 'grade_calculus1', 'grade_linear_algebra', 'grade_logic']
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





#-------------------------------------------------------------------------------------------------------------------------------


@app.route('/bsi/model1GPA')
def bsi_model1GPA():
    return render_template('BSI/model1GPA/model1GPA.html')

@app.route('/bsi/model1GPA/predict', methods=['POST'])
def bsi_model1gpa_predict():
    try:
        # Load the model
        pipeline = joblib.load('model4GPA_model_and_encoders/BSI/model1GPA/model1GPA.pkl')

        # Carregar os LabelEncoders
        label_encoders = {
            'ingresso_atual': joblib.load('model4GPA_model_and_encoders/BSI/model1GPA/label_encoder_ingresso_atual.pkl'),
            'Categoria': joblib.load('model4GPA_model_and_encoders/BSI/model1GPA/label_encoder_Categoria.pkl'),
            'IsTheyBusinessperson': joblib.load('model4GPA_model_and_encoders/BSI/model1GPA/label_encoder_IsTheyBusinessperson.pkl'),
            'SEXO': joblib.load('model4GPA_model_and_encoders/BSI/model1GPA/label_encoder_SEXO.pkl'),
            'employee_student': joblib.load('model4GPA_model_and_encoders/BSI/model1GPA/label_encoder_employee_student.pkl')
        }

        # Define as colunas numéricas e categóricas
        categorical_columns = ['ingresso_atual', 'Categoria', 'IsTheyBusinessperson', 'SEXO', 'employee_student']
        numerical_columns = ['CR_ATUAL', 'GPA1', 'grade_programming1', 'grade_basic_math']
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

@app.route('/mat')
def mat_index():
    return render_template('mat/matindex.html')

@app.route('/mat/model4GPA')
def mat_model4GPA():
    return render_template('mat/model4GPA/model4GPA.html')


@app.route('/mat/model4GPA/predict', methods=['POST'])
def mat_model4gpa_predict():
    try:
        # Load the model
        pipeline = joblib.load('model4GPA_model_and_encoders/mat/model4GPA/model4GPA.pkl')

        # Carregar os LabelEncoders
        label_encoders = {
            'ingresso_atual': joblib.load('model4GPA_model_and_encoders/mat/model4GPA/label_encoder_ingresso_atual.pkl'),
            'Categoria': joblib.load('model4GPA_model_and_encoders/mat/model4GPA/label_encoder_Categoria.pkl'),
            'IsTheyBusinessperson': joblib.load('model4GPA_model_and_encoders/mat/model4GPA/label_encoder_IsTheyBusinessperson.pkl'),
            'SEXO': joblib.load('model4GPA_model_and_encoders/mat/model4GPA/label_encoder_SEXO.pkl'),
            'employee_student': joblib.load('model4GPA_model_and_encoders/mat/model4GPA/label_encoder_employee_student.pkl')
        }

        # Define as colunas numéricas e categóricas
        categorical_columns = ['ingresso_atual', 'Categoria', 'IsTheyBusinessperson', 'SEXO', 'employee_student']
        numerical_columns = ['CR_ATUAL',  'GPA1', 'GPA2', 'GPA3', 'GPA4', 'grade_programming1', "grade_enviroment", 
                    'grade_math_foundation', 'grade_analytic_geometry']
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

#-------------------------------------------------------------------------------------------------------------------------------


@app.route('/mat/model3GPA')
def mat_model3GPA():
    return render_template('mat/model3GPA/model3GPA.html')

@app.route('/mat/model3GPA/predict', methods=['POST'])
def mat_model3gpa_predict():
    try:
        # Load the model
        pipeline = joblib.load('model4GPA_model_and_encoders/mat/model3GPA/model3GPA.pkl')

        # Carregar os LabelEncoders
        label_encoders = {
            'ingresso_atual': joblib.load('model4GPA_model_and_encoders/mat/model3GPA/label_encoder_ingresso_atual.pkl'),
            'Categoria': joblib.load('model4GPA_model_and_encoders/mat/model3GPA/label_encoder_Categoria.pkl'),
            'IsTheyBusinessperson': joblib.load('model4GPA_model_and_encoders/mat/model3GPA/label_encoder_IsTheyBusinessperson.pkl'),
            'SEXO': joblib.load('model4GPA_model_and_encoders/mat/model3GPA/label_encoder_SEXO.pkl'),
            'employee_student': joblib.load('model4GPA_model_and_encoders/mat/model3GPA/label_encoder_employee_student.pkl')
        }

        # Define as colunas numéricas e categóricas
        categorical_columns = ['ingresso_atual', 'Categoria', 'IsTheyBusinessperson', 'SEXO', 'employee_student']
        numerical_columns = ['CR_ATUAL', 'GPA1', 'GPA2', 'GPA3', 'grade_programming1', "grade_enviroment", 
                    'grade_math_foundation', 'grade_analytic_geometry']
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



#-------------------------------------------------------------------------------------------------------------------------------


@app.route('/mat/model2GPA')
def mat_model2GPA():
    return render_template('mat/model2GPA/model2GPA.html')

@app.route('/mat/model2GPA/predict', methods=['POST'])
def mat_model2gpa_predict():
    try:
        # Load the model
        pipeline = joblib.load('model4GPA_model_and_encoders/mat/model2GPA/model2GPA.pkl')

        # Carregar os LabelEncoders
        label_encoders = {
            'ingresso_atual': joblib.load('model4GPA_model_and_encoders/mat/model2GPA/label_encoder_ingresso_atual.pkl'),
            'Categoria': joblib.load('model4GPA_model_and_encoders/mat/model2GPA/label_encoder_Categoria.pkl'),
            'IsTheyBusinessperson': joblib.load('model4GPA_model_and_encoders/mat/model2GPA/label_encoder_IsTheyBusinessperson.pkl'),
            'SEXO': joblib.load('model4GPA_model_and_encoders/mat/model2GPA/label_encoder_SEXO.pkl'),
            'employee_student': joblib.load('model4GPA_model_and_encoders/mat/model2GPA/label_encoder_employee_student.pkl')
        }

        # Define as colunas numéricas e categóricas
        categorical_columns = ['ingresso_atual', 'Categoria', 'IsTheyBusinessperson', 'SEXO', 'employee_student']
        numerical_columns = ['GPA1', 'GPA2', 'grade_programming1', "grade_enviroment", 
                    'grade_math_foundation', 'grade_analytic_geometry']
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





#-------------------------------------------------------------------------------------------------------------------------------


@app.route('/mat/model1GPA')
def mat_model1GPA():
    return render_template('mat/model1GPA/model1GPA.html')

@app.route('/mat/model1GPA/predict', methods=['POST'])
def mat_model1gpa_predict():
    try:
        # Load the model
        pipeline = joblib.load('model4GPA_model_and_encoders/mat/model1GPA/model1GPA.pkl')

        # Carregar os LabelEncoders
        label_encoders = {
            'ingresso_atual': joblib.load('model4GPA_model_and_encoders/mat/model1GPA/label_encoder_ingresso_atual.pkl'),
            'Categoria': joblib.load('model4GPA_model_and_encoders/mat/model1GPA/label_encoder_Categoria.pkl'),
            'IsTheyBusinessperson': joblib.load('model4GPA_model_and_encoders/mat/model1GPA/label_encoder_IsTheyBusinessperson.pkl'),
            'SEXO': joblib.load('model4GPA_model_and_encoders/mat/model1GPA/label_encoder_SEXO.pkl'),
            'employee_student': joblib.load('model4GPA_model_and_encoders/mat/model1GPA/label_encoder_employee_student.pkl')
        }

        # Define as colunas numéricas e categóricas
        categorical_columns = ['ingresso_atual', 'Categoria', 'IsTheyBusinessperson', 'SEXO', 'employee_student']
        numerical_columns = ['CR_ATUAL', 'GPA1', 'grade_programming1', "grade_enviroment", 
                    'grade_math_foundation', 'grade_analytic_geometry']
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
























    
    
###############################################################################################################################
@app.route('/ccet')
def ccet_index():
    return render_template('CCET/ccet.html')

###############################################################################################################################
@app.route('/ccet/model4gpa')
def ccet_model4gpa():
    return render_template('CCET/model4GPA/generalmodel.html')

@app.route('/ccet/model4GPA/predict', methods=['POST'])
def ccet_model4gpa_predict():
    try:
         # Load the model
        pipeline = joblib.load('model4GPA_model_and_encoders/CCET/model4GPA/model4GPA.pkl')
    
        # Carregar os LabelEncoders
        label_encoders = {
            'ingresso_atual': joblib.load('model4GPA_model_and_encoders/CCET/model4GPA/label_encoder_ingresso_atual.pkl'),
            'Categoria': joblib.load('model4GPA_model_and_encoders/CCET/model4GPA/label_encoder_Categoria.pkl'),
            'IsTheyBusinessperson': joblib.load('model4GPA_model_and_encoders/CCET/model4GPA/label_encoder_IsTheyBusinessperson.pkl'),
            'SEXO': joblib.load('model4GPA_model_and_encoders/CCET/model4GPA/label_encoder_SEXO.pkl'),
            'employee_student': joblib.load('model4GPA_model_and_encoders/CCET/model4GPA/label_encoder_employee_student.pkl'),
            'NOME_CURSO': joblib.load('model4GPA_model_and_encoders/CCET/model4GPA/label_encoder_Curso.pkl')
        }    

        # Define as colunas numéricas e categóricas
        numerical_columns = ['CR_ATUAL','GPA1', 'GPA2', 'GPA3', 'GPA4']
        categorical_columns = ['ingresso_atual', 'IsTheyBusinessperson', 'Categoria', 'SEXO', 'employee_student','NOME_CURSO']
        # Get the JSON data from the request
        data = request.json
    
        # Create a DataFrame from the JSON data
        df = pd.DataFrame(data, index=[0])
         # Apply label encoding to categorical columns
        for name, le in label_encoders.items():
            try:
                df[name] = le.transform(df[name])
            except ValueError as e:
                print("error Unknown category in column "+name)
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
 
            
        y_pred = pipeline.predict(X)
        print(y_pred.tolist())
        # Return the predictions as JSON
        return jsonify({'prediction': y_pred.tolist()[0]})
    except Exception as e:
            error_message = f"Error: {str(e)}"
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            full_error_message = f"{error_message}\nTraceback:\n{traceback_str}"
            print(full_error_message)
            return jsonify(error=full_error_message), 500
###############################################################################################################################
@app.route('/ccet/model3gpa')
def ccet_model3gpa():
    return render_template('CCET/model3GPA/generalmodel.html')

@app.route('/ccet/model3GPA/predict', methods=['POST'])
def ccet_model3gpa_predict():
    try:
         # Load the model
        pipeline = joblib.load('model4GPA_model_and_encoders/CCET/model3GPA/model3GPA.pkl')
    
        # Carregar os LabelEncoders
        label_encoders = {   
            'Categoria': joblib.load('model4GPA_model_and_encoders/CCET/model3GPA/label_encoder_Categoria.pkl'),
            'IsTheyBusinessperson': joblib.load('model4GPA_model_and_encoders/CCET/model3GPA/label_encoder_IsTheyBusinessperson.pkl'),
            'SEXO': joblib.load('model4GPA_model_and_encoders/CCET/model3GPA/label_encoder_SEXO.pkl'),
            'employee_student': joblib.load('model4GPA_model_and_encoders/CCET/model3GPA/label_encoder_employee_student.pkl'),
            'NOME_CURSO': joblib.load('model4GPA_model_and_encoders/CCET/model3GPA/label_encoder_Curso.pkl'),
            'ingresso_atual': joblib.load('model4GPA_model_and_encoders/CCET/model3GPA/label_encoder_ingresso_atual.pkl')
        }    

        # Define as colunas numéricas e categóricas
        numerical_columns = ['CR_ATUAL','GPA1', 'GPA2', 'GPA3']
        categorical_columns = ['ingresso_atual', 'IsTheyBusinessperson', 'Categoria', 'SEXO', 'employee_student','NOME_CURSO']
        # Get the JSON data from the request
        data = request.json
    
        # Create a DataFrame from the JSON data
        df = pd.DataFrame(data, index=[0])
         # Apply label encoding to categorical columns
        for name, le in label_encoders.items():
            try:
                df[name] = le.transform(df[name])
            except ValueError as e:
                print("error Unknown category in column "+name)
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
 
            
        y_pred = pipeline.predict(X)
        print(y_pred.tolist())
        # Return the predictions as JSON
        return jsonify({'prediction': y_pred.tolist()[0]})
    except Exception as e:
            error_message = f"Error: {str(e)}"
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            full_error_message = f"{error_message}\nTraceback:\n{traceback_str}"
            print(full_error_message)
            return jsonify(error=full_error_message), 500
###############################################################################################################################
@app.route('/ccet/model2gpa')
def ccet_model2gpa():
    return render_template('CCET/model2GPA/generalmodel.html')

@app.route('/ccet/model2GPA/predict', methods=['POST'])
def ccet_model2gpa_predict():
    try:
         # Load the model
        pipeline = joblib.load('model4GPA_model_and_encoders/CCET/model2GPA/model2GPA.pkl')
    
        # Carregar os LabelEncoders
        label_encoders = {
            'ingresso_atual': joblib.load('model4GPA_model_and_encoders/CCET/model2GPA/label_encoder_ingresso_atual.pkl'),
            'Categoria': joblib.load('model4GPA_model_and_encoders/CCET/model2GPA/label_encoder_Categoria.pkl'),
            'IsTheyBusinessperson': joblib.load('model4GPA_model_and_encoders/CCET/model2GPA/label_encoder_IsTheyBusinessperson.pkl'),
            'SEXO': joblib.load('model4GPA_model_and_encoders/CCET/model2GPA/label_encoder_SEXO.pkl'),
            'employee_student': joblib.load('model4GPA_model_and_encoders/CCET/model2GPA/label_encoder_employee_student.pkl'),
            'NOME_CURSO': joblib.load('model4GPA_model_and_encoders/CCET/model2GPA/label_encoder_Curso.pkl')
        }    

        # Define as colunas numéricas e categóricas
        numerical_columns = ['CR_ATUAL','GPA1', 'GPA2']
        categorical_columns = ['ingresso_atual', 'IsTheyBusinessperson', 'Categoria', 'SEXO', 'employee_student','NOME_CURSO']
        # Get the JSON data from the request
        data = request.json
    
        # Create a DataFrame from the JSON data
        df = pd.DataFrame(data, index=[0])
         # Apply label encoding to categorical columns
        for name, le in label_encoders.items():
            try:
                df[name] = le.transform(df[name])
            except ValueError as e:
                print("error Unknown category in column "+name)
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
 
            
        y_pred = pipeline.predict(X)
        print(y_pred.tolist())
        # Return the predictions as JSON
        return jsonify({'prediction': y_pred.tolist()[0]})
    except Exception as e:
            error_message = f"Error: {str(e)}"
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            full_error_message = f"{error_message}\nTraceback:\n{traceback_str}"
            print(full_error_message)
            return jsonify(error=full_error_message), 500
###############################################################################################################################
@app.route('/ccet/model1gpa')
def ccet_model1gpa():
    return render_template('CCET/model1GPA/generalmodel.html')

@app.route('/ccet/model1GPA/predict', methods=['POST'])
def ccet_model1gpa_predict():
    try:
         # Load the model
        pipeline = joblib.load('model4GPA_model_and_encoders/CCET/model1GPA/model1GPA.pkl')
    
        # Carregar os LabelEncoders
        label_encoders = {
            'ingresso_atual': joblib.load('model4GPA_model_and_encoders/CCET/model1GPA/label_encoder_ingresso_atual.pkl'),
            'Categoria': joblib.load('model4GPA_model_and_encoders/CCET/model1GPA/label_encoder_Categoria.pkl'),
            'IsTheyBusinessperson': joblib.load('model4GPA_model_and_encoders/CCET/model1GPA/label_encoder_IsTheyBusinessperson.pkl'),
            'SEXO': joblib.load('model4GPA_model_and_encoders/CCET/model1GPA/label_encoder_SEXO.pkl'),
            'employee_student': joblib.load('model4GPA_model_and_encoders/CCET/model1GPA/label_encoder_employee_student.pkl'),
            'NOME_CURSO': joblib.load('model4GPA_model_and_encoders/CCET/model1GPA/label_encoder_Curso.pkl')
        }    

        # Define as colunas numéricas e categóricas
        numerical_columns = ['CR_ATUAL','GPA1']
        categorical_columns = ['ingresso_atual', 'IsTheyBusinessperson', 'Categoria', 'SEXO', 'employee_student','NOME_CURSO']
        # Get the JSON data from the request
        data = request.json
    
        # Create a DataFrame from the JSON data
        df = pd.DataFrame(data, index=[0])
         # Apply label encoding to categorical columns
        for name, le in label_encoders.items():
            try:
                df[name] = le.transform(df[name])
            except ValueError as e:
                print("error Unknown category in column "+name)
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
 
            
        y_pred = pipeline.predict(X)
        print(y_pred.tolist())
        # Return the predictions as JSON
        return jsonify({'prediction': y_pred.tolist()[0]})
    except Exception as e:
            error_message = f"Error: {str(e)}"
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            full_error_message = f"{error_message}\nTraceback:\n{traceback_str}"
            print(full_error_message)
            return jsonify(error=full_error_message), 500


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
@app.route('/eng')
def eng_index():
    return render_template('eng/engindex.html')

@app.route('/eng/model4GPA')
def eng_model4GPA():
    return render_template('eng/model4GPA/model4GPA.html')


@app.route('/eng/model4GPA/predict', methods=['POST'])
def eng_model4gpa_predict():
    try:
        # Load the model
        pipeline = joblib.load('model4GPA_model_and_encoders/eng/model4GPA/model4GPA.pkl')

        # Carregar os LabelEncoders
        label_encoders = {
            'ingresso_atual': joblib.load('model4GPA_model_and_encoders/eng/model4GPA/label_encoder_ingresso_atual.pkl'),
            'Categoria': joblib.load('model4GPA_model_and_encoders/eng/model4GPA/label_encoder_Categoria.pkl'),
            'IsTheyBusinessperson': joblib.load('model4GPA_model_and_encoders/eng/model4GPA/label_encoder_IsTheyBusinessperson.pkl'),
            'SEXO': joblib.load('model4GPA_model_and_encoders/eng/model4GPA/label_encoder_SEXO.pkl'),
            'employee_student': joblib.load('model4GPA_model_and_encoders/eng/model4GPA/label_encoder_employee_student.pkl')
        }

        # Define as colunas numéricas e categóricas
        categorical_columns = ['ingresso_atual', 'Categoria', 'IsTheyBusinessperson', 'SEXO', 'employee_student']
        numerical_columns = ['CR_ATUAL',  'GPA1', 'GPA2', 'GPA3', 'GPA4','grade_programming1', 'grade_calculus0','grade_calculus1', 'grade_linear_algebra',
                   'grade_eng_introduction']
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
    
#-------------------------------------------------------------------------------------------------------------------------------


@app.route('/eng/model3GPA')
def eng_model3GPA():
    return render_template('eng/model3GPA/model3GPA.html')

@app.route('/eng/model3GPA/predict', methods=['POST'])
def eng_model3gpa_predict():
    try:
        # Load the model
        pipeline = joblib.load('model4GPA_model_and_encoders/eng/model3GPA/model3GPA.pkl')

        # Carregar os LabelEncoders
        label_encoders = {
            'ingresso_atual': joblib.load('model4GPA_model_and_encoders/eng/model3GPA/label_encoder_ingresso_atual.pkl'),
            'Categoria': joblib.load('model4GPA_model_and_encoders/eng/model3GPA/label_encoder_Categoria.pkl'),
            'IsTheyBusinessperson': joblib.load('model4GPA_model_and_encoders/eng/model3GPA/label_encoder_IsTheyBusinessperson.pkl'),
            'SEXO': joblib.load('model4GPA_model_and_encoders/eng/model3GPA/label_encoder_SEXO.pkl'),
            'employee_student': joblib.load('model4GPA_model_and_encoders/eng/model3GPA/label_encoder_employee_student.pkl')
        }

        # Define as colunas numéricas e categóricas
        categorical_columns = ['ingresso_atual', 'Categoria', 'IsTheyBusinessperson', 'SEXO', 'employee_student']
        numerical_columns = ['CR_ATUAL',  'GPA1', 'GPA2', 'GPA3','grade_programming1', 'grade_calculus0','grade_calculus1', 'grade_linear_algebra',
                   'grade_eng_introduction']
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



#-------------------------------------------------------------------------------------------------------------------------------


@app.route('/eng/model2GPA')
def eng_model2GPA():
    return render_template('eng/model2GPA/model2GPA.html')

@app.route('/eng/model2GPA/predict', methods=['POST'])
def eng_model2gpa_predict():
    try:
        # Load the model
        pipeline = joblib.load('model4GPA_model_and_encoders/eng/model2GPA/model2GPA.pkl')

        # Carregar os LabelEncoders
        label_encoders = {
            'ingresso_atual': joblib.load('model4GPA_model_and_encoders/eng/model2GPA/label_encoder_ingresso_atual.pkl'),
            'Categoria': joblib.load('model4GPA_model_and_encoders/eng/model2GPA/label_encoder_Categoria.pkl'),
            'IsTheyBusinessperson': joblib.load('model4GPA_model_and_encoders/eng/model2GPA/label_encoder_IsTheyBusinessperson.pkl'),
            'SEXO': joblib.load('model4GPA_model_and_encoders/eng/model2GPA/label_encoder_SEXO.pkl'),
            'employee_student': joblib.load('model4GPA_model_and_encoders/eng/model2GPA/label_encoder_employee_student.pkl')
        }

        # Define as colunas numéricas e categóricas
        categorical_columns = ['ingresso_atual', 'Categoria', 'IsTheyBusinessperson', 'SEXO', 'employee_student']
        numerical_columns = ['CR_ATUAL',  'GPA1', 'GPA2', 'grade_programming1', 'grade_calculus0','grade_calculus1', 'grade_linear_algebra',
                   'grade_eng_introduction']
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





#-------------------------------------------------------------------------------------------------------------------------------


@app.route('/eng/model1GPA')
def eng_model1GPA():
    return render_template('eng/model1GPA/model1GPA.html')

@app.route('/eng/model1GPA/predict', methods=['POST'])
def eng_model1gpa_predict():
    try:
        # Load the model
        pipeline = joblib.load('model4GPA_model_and_encoders/eng/model1GPA/model1GPA.pkl')

        # Carregar os LabelEncoders
        label_encoders = {
            'ingresso_atual': joblib.load('model4GPA_model_and_encoders/eng/model1GPA/label_encoder_ingresso_atual.pkl'),
            'Categoria': joblib.load('model4GPA_model_and_encoders/eng/model1GPA/label_encoder_Categoria.pkl'),
            'IsTheyBusinessperson': joblib.load('model4GPA_model_and_encoders/eng/model1GPA/label_encoder_IsTheyBusinessperson.pkl'),
            'SEXO': joblib.load('model4GPA_model_and_encoders/eng/model1GPA/label_encoder_SEXO.pkl'),
            'employee_student': joblib.load('model4GPA_model_and_encoders/eng/model1GPA/label_encoder_employee_student.pkl')
        }

        # Define as colunas numéricas e categóricas
        categorical_columns = ['ingresso_atual', 'Categoria', 'IsTheyBusinessperson', 'SEXO', 'employee_student']
        numerical_columns = ['CR_ATUAL', 'GPA1', 'grade_programming1', 'grade_calculus0', 'grade_eng_introduction']
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

