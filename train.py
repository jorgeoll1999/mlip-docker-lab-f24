from sklearn.svm import SVC
from sklearn import datasets
import joblib
import os

# Cargar dataset
iris = datasets.load_iris()

# Crear clasificador SVM
clf = SVC()

# Entrenar con etiquetas numéricas
clf.fit(iris.data, iris.target)

# Crear carpeta si no existe (buena práctica)
os.makedirs("/app/model_storage", exist_ok=True)

# Guardar el modelo
joblib.dump(clf, "/app/model_storage/model.pkl")

print("✅ Modelo entrenado y guardado correctamente en /app/model_storage/model.pkl")
