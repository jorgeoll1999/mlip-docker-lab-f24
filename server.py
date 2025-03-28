from flask import Flask, request, jsonify
import numpy as np
import joblib
from sklearn import datasets
import os

app = Flask(__name__)

# Ruta al modelo
model_path = os.path.join("model_storage", "model.pkl")
model = joblib.load(model_path)

# Cargar nombres de clases desde Iris dataset
iris = datasets.load_iris()
class_names = iris.target_names

@app.route('/predict', methods=['GET'])
def predict():
    # Obtener input del request
    get_json = request.get_json()
    iris_input = get_json['input']

    # Convertir a numpy array y predecir
    input_array = np.array(iris_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]  # obtenemos solo el número

    # Obtener nombre de clase
    class_name = class_names[prediction]

    # Devolver resultado en formato bonito
    return jsonify({
        "prediction": int(prediction),
        "class_name": class_name
    })

@app.route('/')
def hello():
    return 'Welcome to Docker Lab'

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
