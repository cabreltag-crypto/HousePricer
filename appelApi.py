import requests
import json

# URL de l'API
url = "http://localhost:5000/api/predict"

# Données de la maison
house_data = {
    "grade": 9,
    "waterfront": 1,  # Vue sur l'eau
    "sqft_living": 3200,
    "bathrooms": 3.0,
    "lat": 47.6205,
    "view": 4,  # Excellente vue
    "long": -122.3493,
    "yr_built": 2005,
    "zipcode": 98101,
    "sqft_lot": 9000,
    "sqft_basement": 800,
    "annee_construction": 2005,
    "sqft_lot15": 8500,
    "condition": 4,
    "yr_renovated": 0,  # Jamais rénové
    "model": "random_forest"
}

# Faire la requête
response = requests.post(url, json=house_data)

# Afficher la réponse
if response.status_code == 200:
    result = response.json()
    print(f"Prix prédit: {result['prediction']['formatted_price']}")
    print(f"Modèle utilisé: {result['model_used']}")
else:
    print(f"Erreur: {response.json()}")