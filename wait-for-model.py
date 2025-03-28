import time
import os

model_path = "model_storage/model.pkl"

print("⏳ Esperando que el modelo esté disponible...")

# Esperar hasta que el archivo exista
while not os.path.exists(model_path):
    time.sleep(1)

print("✅ Modelo encontrado. Iniciando servidor Flask...")
