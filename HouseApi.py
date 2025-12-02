from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)  # Permet les requêtes cross-origin

# Configuration
ALLOWED_MODELS = ['random_forest', 'xgboost']

# Chargement des modèles au démarrage de l'application
models = {}

def load_models():
    """Charge les modèles ML au démarrage"""
    try:
        models['random_forest'] = joblib.load('random_forest_model.pkl')
        print("✅ Modèle Random Forest chargé")
    except Exception as e:
        print(f"Erreur chargement Random Forest: {e}")
    
    try:
        models['xgboost'] = joblib.load('xgb_house_price_model.pkl')
        print("Modèle XGBoost chargé")
    except Exception as e:
        print(f"Erreur chargement XGBoost: {e}")

# Charger les modèles au démarrage
load_models()


def validate_input(data):
    """Valide les données d'entrée"""
    required_fields = [
        'grade', 
        'waterfront', 
        'sqft_living', 
        'bathrooms', 
        'lat', 
        'view', 
        'long', 
        'yr_built', 
        'zipcode', 
        'sqft_lot', 
        'sqft_basement', 
        'annee_construction', 
        'sqft_lot15', 
        'condition', 
        'yr_renovated'
    ]
    
    # Vérifier que tous les champs requis sont présents
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Champs manquants: {', '.join(missing_fields)}"
    
    # Validation des types et des plages de valeurs
    try:
        # grade: Note de construction (1-13)
        grade = int(data['grade'])
        if grade < 1 or grade > 13:
            return False, "Le grade doit être entre 1 et 13"
        
        # waterfront: Vue sur l'eau (0 ou 1)
        waterfront = int(data['waterfront'])
        if waterfront not in [0, 1]:
            return False, "Waterfront doit être 0 (Non) ou 1 (Oui)"
        
        # sqft_living: Surface habitable (100-20000)
        sqft_living = float(data['sqft_living'])
        if sqft_living < 100 or sqft_living > 20000:
            return False, "La surface habitable (sqft_living) doit être entre 100 et 20000 sqft"
        
        # bathrooms: Nombre de salles de bain (0.5-10)
        bathrooms = float(data['bathrooms'])
        if bathrooms < 0.5 or bathrooms > 10:
            return False, "Le nombre de salles de bain doit être entre 0.5 et 10"
        
        # lat: Latitude (-90 à 90)
        lat = float(data['lat'])
        if lat < -90 or lat > 90:
            return False, "La latitude doit être entre -90 et 90"
        
        # view: Qualité de la vue (0-4)
        view = int(data['view'])
        if view < 0 or view > 4:
            return False, "La qualité de la vue (view) doit être entre 0 et 4"
        
        # long: Longitude (-180 à 180)
        long = float(data['long'])
        if long < -180 or long > 180:
            return False, "La longitude doit être entre -180 et 180"
        
        # yr_built: Année de construction (1800-année actuelle)
        yr_built = int(data['yr_built'])
        current_year = datetime.now().year
        if yr_built < 1800 or yr_built > current_year:
            return False, f"L'année de construction (yr_built) doit être entre 1800 et {current_year}"
        
        # zipcode: Code postal (5 chiffres)
        zipcode = int(data['zipcode'])
        if zipcode < 10000 or zipcode > 99999:
            return False, "Le zipcode doit être un code postal valide (5 chiffres)"
        
        # sqft_lot: Surface du terrain (500-1000000)
        sqft_lot = float(data['sqft_lot'])
        if sqft_lot < 500 or sqft_lot > 1000000:
            return False, "La surface du terrain (sqft_lot) doit être entre 500 et 1000000 sqft"
        
        # sqft_basement: Surface du sous-sol (0-10000)
        sqft_basement = float(data['sqft_basement'])
        if sqft_basement < 0 or sqft_basement > 10000:
            return False, "La surface du sous-sol (sqft_basement) doit être entre 0 et 10000 sqft"
        
        # annee_construction: Année de construction (1800-année actuelle)
        annee_construction = int(data['annee_construction'])
        if annee_construction < 1800 or annee_construction > current_year:
            return False, f"L'année de construction (annee_construction) doit être entre 1800 et {current_year}"
        
        # sqft_lot15: Surface moyenne des 15 terrains voisins (500-1000000)
        sqft_lot15 = float(data['sqft_lot15'])
        if sqft_lot15 < 500 or sqft_lot15 > 1000000:
            return False, "La surface des terrains voisins (sqft_lot15) doit être entre 500 et 1000000 sqft"
        
        # condition: État de la maison (1-5)
        condition = int(data['condition'])
        if condition < 1 or condition > 5:
            return False, "L'état de la maison (condition) doit être entre 1 et 5"
        
        # yr_renovated: Année de rénovation (0 ou 1900-année actuelle)
        yr_renovated = int(data['yr_renovated'])
        if yr_renovated != 0 and (yr_renovated < 1900 or yr_renovated > current_year):
            return False, f"L'année de rénovation (yr_renovated) doit être 0 (non rénové) ou entre 1900 et {current_year}"
        
        return True, "Validation réussie"
    
    except ValueError as e:
        return False, f"Erreur de type de données: {str(e)}"


def prepare_features(data):
    """Prépare les features pour la prédiction dans l'ordre exact attendu par les modèles"""
    features = pd.DataFrame({
        'grade': [int(data['grade'])],
        'waterfront': [int(data['waterfront'])],
        'sqft_living': [float(data['sqft_living'])],
        'bathrooms': [float(data['bathrooms'])],
        'lat': [float(data['lat'])],
        'view': [int(data['view'])],
        'long': [float(data['long'])],
        'yr_built': [int(data['yr_built'])],
        'zipcode': [int(data['zipcode'])],
        'sqft_lot': [float(data['sqft_lot'])],
        'sqft_basement': [float(data['sqft_basement'])],
        'annee_construction': [int(data['annee_construction'])],
        'sqft_lot15': [float(data['sqft_lot15'])],
        'condition': [int(data['condition'])],
        'yr_renovated': [int(data['yr_renovated'])]
    })
    
    return features


@app.route('/')
def home():
    """Route d'accueil"""
    return jsonify({
        'message': 'API HomePricer - Prédiction de prix immobiliers',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'predict': '/api/predict',
            'models': '/api/models',
            'info': '/api/info'
        }
    }), 200


@app.route('/health')
def health():
    """Endpoint de santé pour vérifier que l'API fonctionne"""
    models_status = {
        model_name: 'loaded' if model_name in models else 'not loaded'
        for model_name in ALLOWED_MODELS
    }
    
    return jsonify({
        'status': 'healthy',
        'models': models_status,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/models', methods=['GET'])
def get_models():
    """Liste les modèles disponibles"""
    available_models = []
    for model_name in ALLOWED_MODELS:
        if model_name in models:
            available_models.append({
                'name': model_name,
                'display_name': model_name.replace('_', ' ').title(),
                'status': 'available'
            })
    
    return jsonify({
        'models': available_models,
        'total': len(available_models)
    }), 200


@app.route('/api/info', methods=['GET'])
def get_info():
    """Informations sur les paramètres attendus"""
    return jsonify({
        'required_parameters': {
            'grade': {
                'type': 'integer',
                'range': '1-13',
                'description': 'Note de construction (qualité globale)'
            },
            'waterfront': {
                'type': 'integer',
                'values': [0, 1],
                'description': 'Vue sur l\'eau (0=Non, 1=Oui)'
            },
            'sqft_living': {
                'type': 'float',
                'range': '100-20000',
                'unit': 'sqft',
                'description': 'Surface habitable'
            },
            'bathrooms': {
                'type': 'float',
                'range': '0.5-10',
                'description': 'Nombre de salles de bain (demi-salles acceptées)'
            },
            'lat': {
                'type': 'float',
                'range': '-90 à 90',
                'description': 'Latitude de la propriété'
            },
            'view': {
                'type': 'integer',
                'range': '0-4',
                'description': 'Qualité de la vue (0=Aucune, 4=Excellente)'
            },
            'long': {
                'type': 'float',
                'range': '-180 à 180',
                'description': 'Longitude de la propriété'
            },
            'yr_built': {
                'type': 'integer',
                'range': '1800-2025',
                'description': 'Année de construction originale'
            },
            'zipcode': {
                'type': 'integer',
                'range': '10000-99999',
                'description': 'Code postal (5 chiffres)'
            },
            'sqft_lot': {
                'type': 'float',
                'range': '500-1000000',
                'unit': 'sqft',
                'description': 'Surface totale du terrain'
            },
            'sqft_basement': {
                'type': 'float',
                'range': '0-10000',
                'unit': 'sqft',
                'description': 'Surface du sous-sol'
            },
            'annee_construction': {
                'type': 'integer',
                'range': '1800-2025',
                'description': 'Année de construction'
            },
            'sqft_lot15': {
                'type': 'float',
                'range': '500-1000000',
                'unit': 'sqft',
                'description': 'Surface moyenne des 15 terrains voisins les plus proches'
            },
            'condition': {
                'type': 'integer',
                'range': '1-5',
                'description': 'État de la maison (1=Très mauvais, 5=Excellent)'
            },
            'yr_renovated': {
                'type': 'integer',
                'range': '0 ou 1900-2025',
                'description': 'Année de rénovation (0 si jamais rénové)'
            }
        },
        'optional_parameters': {
            'model': {
                'type': 'string',
                'values': ALLOWED_MODELS,
                'default': 'xgboost',
                'description': 'Modèle à utiliser pour la prédiction'
            }
        }
    }), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint principal de prédiction"""
    try:
        # Récupérer les données JSON
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Aucune donnée fournie',
                'message': 'Le corps de la requête doit contenir des données JSON'
            }), 400
        
        # Valider les données
        is_valid, validation_message = validate_input(data)
        if not is_valid:
            return jsonify({
                'error': 'Validation échouée',
                'message': validation_message
            }), 400
        
        # Sélectionner le modèle
        model_name = data.get('model', 'xgboost').lower()
        if model_name not in ALLOWED_MODELS:
            return jsonify({
                'error': 'Modèle invalide',
                'message': f'Modèle doit être l\'un de: {", ".join(ALLOWED_MODELS)}'
            }), 400
        
        if model_name not in models:
            return jsonify({
                'error': 'Modèle non disponible',
                'message': f'Le modèle {model_name} n\'est pas chargé'
            }), 503
        
        # Préparer les features
        features = prepare_features(data)
        
        # Faire la prédiction
        model = models[model_name]
        prediction = model.predict(features)[0]
        
        # Préparer la réponse
        response = {
            'success': True,
            'prediction': {
                'price': float(prediction),
                'currency': 'USD',
                'formatted_price': f'${prediction:,.2f}'
            },
            'model_used': model_name,
            'input_data': {
                'grade': int(data['grade']),
                'waterfront': bool(int(data['waterfront'])),
                'sqft_living': float(data['sqft_living']),
                'bathrooms': float(data['bathrooms']),
                'lat': float(data['lat']),
                'view': int(data['view']),
                'long': float(data['long']),
                'yr_built': int(data['yr_built']),
                'zipcode': int(data['zipcode']),
                'sqft_lot': float(data['sqft_lot']),
                'sqft_basement': float(data['sqft_basement']),
                'annee_construction': int(data['annee_construction']),
                'sqft_lot15': float(data['sqft_lot15']),
                'condition': int(data['condition']),
                'yr_renovated': int(data['yr_renovated'])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'error': 'Erreur de prédiction',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Gestion des erreurs 404"""
    return jsonify({
        'error': 'Route non trouvée',
        'message': 'L\'endpoint demandé n\'existe pas'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Gestion des erreurs 500"""
    return jsonify({
        'error': 'Erreur interne du serveur',
        'message': 'Une erreur s\'est produite lors du traitement de votre requête'
    }), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Démarrage de l'API HomePricer")
    print("="*50)
    print(f"Modèles disponibles: {', '.join([m for m in ALLOWED_MODELS if m in models])}")
    print("="*50 + "\n")
    
    # Démarrer l'application Flask
    app.run(debug=True, host='0.0.0.0', port=5000)