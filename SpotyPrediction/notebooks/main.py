from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
import warnings

# 1. Ignorer les warnings
warnings.filterwarnings("ignore")

# 2. Création de l'application FastAPI
mlflow.set_tracking_uri("http://127.0.0.1:5001/")
app = FastAPI(title="API Prédiction - Logistic Regression")

# 3. Chargement du modèle MLflow (version 1)
MODEL_NAME = "Logistic regression"  # respecte bien l’espace
MODEL_VERSION = 1

try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
    print(f"✅ Modèle '{MODEL_NAME}' version {MODEL_VERSION} chargé avec succès !")
except Exception as e:
    print(f"❌ Erreur de chargement du modèle : {e}")
    model = None

# 4. Définition de l'endpoint FastAPI
@app.post("/predict")
async def predict(features: dict):
    if model is None:
        return {"error": "Modèle non disponible"}

    try:
        df = pd.DataFrame([features])  # transforme le dict en DataFrame
        prediction = model.predict(df)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}
