import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import xgboost as xgb
import pickle
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="HomePricer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Chargement du CSS externe
def load_css():
    css_file = Path("style4.css")
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("Fichier style.css non trouvé")

load_css()

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('xgb_house_price_model.pkl', 'rb'))
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# Fonction pour obtenir le code postal à partir des coordonnées GPS
@st.cache_data
def get_zipcode_from_coordinates(lat, lon):
    """
    Récupère le code postal basé sur les coordonnées GPS en utilisant l'API Nominatim (OpenStreetMap)
    """
    try:
        import requests
        import time
        
        # API Nominatim pour le reverse geocoding
        url = f"https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json',
            'addressdetails': 1
        }
        headers = {
            'User-Agent': 'XGBoost-Property-Prediction-App/1.0'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extraire le code postal
            address = data.get('address', {})
            zipcode = address.get('postcode', None)
            
            if zipcode:
                # Nettoyer le code postal (enlever les espaces, garder que les chiffres)
                zipcode_clean = ''.join(filter(str.isdigit, zipcode))
                
                # Vérifier si c'est un code postal US valide (5 chiffres)
                if len(zipcode_clean) >= 5:
                    return int(zipcode_clean[:5])
            
            # Si pas de code postal trouvé, chercher dans les coordonnées communes de King County, WA
            # (zone de Seattle où se trouve le dataset de maisons)
            return estimate_zipcode_by_proximity(lat, lon)
            
    except Exception as e:
        print(f"Erreur lors de la récupération du code postal : {e}")
        return estimate_zipcode_by_proximity(lat, lon)
    
    return None

def estimate_zipcode_by_proximity(lat, lon):
    """
    Estime le code postal basé sur la proximité avec des codes postaux connus de King County, WA
    """
    # Dictionnaire des codes postaux de King County (Seattle area) avec leurs coordonnées approximatives
    king_county_zipcodes = {
        98001: (47.3073, -122.2290),  # Auburn
        98002: (47.2879, -122.2351),  # Auburn
        98003: (47.3087, -122.3426),  # Federal Way
        98004: (47.6229, -122.2043),  # Bellevue
        98005: (47.6168, -122.1460),  # Bellevue
        98006: (47.5632, -122.1493),  # Bellevue
        98007: (47.6068, -122.1315),  # Bellevue
        98008: (47.6168, -122.1196),  # Bellevue
        98010: (47.2929, -121.9726),  # Black Diamond
        98011: (47.7523, -122.2054),  # Bothell
        98014: (47.6473, -121.7868),  # Carnation
        98019: (47.7301, -122.2054),  # Duvall
        98022: (47.4262, -121.7826),  # Enumclaw
        98023: (47.3101, -122.3493),  # Federal Way
        98024: (47.5373, -121.8232),  # Fall City
        98027: (47.5262, -122.0326),  # Issaquah
        98028: (47.7540, -122.2290),  # Kenmore
        98029: (47.5262, -122.0326),  # Issaquah
        98030: (47.3837, -122.2176),  # Kent
        98031: (47.3887, -122.2343),  # Kent
        98032: (47.3698, -122.2562),  # Kent
        98033: (47.6779, -122.1910),  # Kirkland
        98034: (47.7176, -122.1910),  # Kirkland
        98038: (47.3632, -122.0690),  # Maple Valley
        98039: (47.6351, -122.2290),  # Medina
        98040: (47.5718, -122.2176),  # Mercer Island
        98042: (47.3651, -122.1193),  # Kent
        98045: (47.4826, -121.7493),  # North Bend
        98047: (47.2729, -122.3493),  # Pacific
        98052: (47.6779, -122.1212),  # Redmond
        98053: (47.6707, -122.0426),  # Redmond
        98055: (47.4512, -122.2093),  # Renton
        98056: (47.4873, -122.1910),  # Renton
        98057: (47.4762, -122.2176),  # Renton
        98058: (47.4401, -122.1426),  # Renton
        98059: (47.4873, -122.1193),  # Renton
        98065: (47.5651, -121.9893),  # Snoqualmie
        98070: (47.3837, -122.3176),  # Vashon
        98074: (47.6262, -122.0326),  # Sammamish
        98075: (47.5762, -122.0326),  # Sammamish
        98077: (47.7540, -122.0643),  # Woodinville
        98092: (47.2929, -122.2176),  # Auburn
        98101: (47.6101, -122.3426),  # Seattle
        98102: (47.6301, -122.3243),  # Seattle
        98103: (47.6779, -122.3426),  # Seattle
        98104: (47.6034, -122.3293),  # Seattle
        98105: (47.6629, -122.3043),  # Seattle
        98106: (47.5318, -122.3543),  # Seattle
        98107: (47.6668, -122.3793),  # Seattle
        98108: (47.5429, -122.3143),  # Seattle
        98109: (47.6379, -122.3476),  # Seattle
        98112: (47.6318, -122.2993),  # Seattle
        98115: (47.6818, -122.3043),  # Seattle
        98116: (47.5718, -122.3943),  # Seattle
        98117: (47.6868, -122.3793),  # Seattle
        98118: (47.5429, -122.2793),  # Seattle
        98119: (47.6379, -122.3743),  # Seattle
        98122: (47.6101, -122.3026),  # Seattle
        98125: (47.7176, -122.3043),  # Seattle
        98126: (47.5429, -122.3743),  # Seattle
        98133: (47.7351, -122.3426),  # Seattle
        98134: (47.5762, -122.3326),  # Seattle
        98136: (47.5429, -122.3943),  # Seattle
        98144: (47.5818, -122.3043),  # Seattle
        98146: (47.5040, -122.3543),  # Seattle
        98148: (47.4401, -122.3326),  # Burien
        98155: (47.7540, -122.3043),  # Seattle
        98166: (47.4540, -122.3543),  # Burien
        98168: (47.4929, -122.3043),  # Burien
        98177: (47.7540, -122.3743),  # Seattle
        98178: (47.4929, -122.2626),  # Seattle
        98188: (47.4540, -122.2926),  # SeaTac
        98198: (47.4079, -122.3326),  # Des Moines
        98199: (47.6379, -122.3993),  # Seattle
    }
    
    # Calculer la distance avec chaque code postal et retourner le plus proche
    min_distance = float('inf')
    closest_zipcode = 98001
    
    for zipcode, (zip_lat, zip_lon) in king_county_zipcodes.items():
        # Distance euclidienne simple
        distance = ((lat - zip_lat) ** 2 + (lon - zip_lon) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_zipcode = zipcode
    
    return closest_zipcode

# Header
st.markdown("""
    <div class="header">
        <div class="logo-container">
            <div class="logo-icon">📈</div>
            <div class="logo-text">
                <div class="logo-title">HomePricer</div>
                <div class="logo-subtitle">prediction du Prix des maisons avec IA</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Section d'introduction
intro_html = """
<div class="intro-section">
<h1 class="main-title">Estimation Immobilière par Intelligence Artificielle</h1>
<p class="intro-text">
        Notre plateforme utilise un modèle XGBoost avancé pour prédire avec précision la valeur de biens
        immobiliers en fonction de leur localisation géographique et de leurs caractéristiques. Sélectionnez une
        position sur la carte et renseignez les paramètres pour obtenir une estimation instantanée.
</p>
    
<div class="features-grid">
<div class="feature-card">
<div class="feature-icon"></div>
<h3 class="feature-title">Précision Géographique</h3>
<p class="feature-desc">Sélectionnez n'importe quel point sur la carte pour une analyse localisée</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon"></div>
            <h3 class="feature-title">Résultats Instantanés</h3>
            <p class="feature-desc">Obtenez une estimation en temps réel grâce à notre modèle optimisé</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon"></div>
            <h3 class="feature-title">Analyse Multi-critères</h3>
            <p class="feature-desc">Prise en compte de multiples variables pour une prédiction fiable</p>
        </div>
    </div>
</div>
"""
st.markdown(intro_html, unsafe_allow_html=True)
st.write("")  # force Streamlit à clôturer les blocs markdown

# Initialisation des variables de session
if 'latitude' not in st.session_state:
    st.session_state.latitude = 47.6062
if 'longitude' not in st.session_state:
    st.session_state.longitude = -122.3321
if 'zipcode' not in st.session_state:
    st.session_state.zipcode = 98001
if 'auto_update_zipcode' not in st.session_state:
    st.session_state.auto_update_zipcode = True

# Section Carte - En haut et pleine largeur
st.markdown('<div class="card-header">Cliquez pour sélectionner une position sur la carte</div>', unsafe_allow_html=True)

# Création de la carte
m = folium.Map(
    location=[st.session_state.latitude, st.session_state.longitude],
    zoom_start=10,
    tiles='OpenStreetMap'
)

# Ajout du marqueur
folium.Marker(
    [st.session_state.latitude, st.session_state.longitude],
    popup="Position sélectionnée",
    tooltip="Position sélectionnée",
    icon=folium.Icon(color='red', icon='home')
).add_to(m)

# Ajout d'un cercle de rayon
folium.Circle(
    location=[st.session_state.latitude, st.session_state.longitude],
    radius=2500,
    color='#7C3AED',
    fill=True,
    fillColor='#7C3AED',
    fillOpacity=0.1,
    weight=2
).add_to(m)

# Affichage de la carte
map_data = st_folium(m, width=None, height=600, key="map")

# Mise à jour des coordonnées si clic sur la carte
if map_data and map_data.get('last_clicked'):
    new_lat = map_data['last_clicked']['lat']
    new_lon = map_data['last_clicked']['lng']
    
    # Vérifier si les coordonnées ont changé
    if (new_lat != st.session_state.latitude or new_lon != st.session_state.longitude):
        st.session_state.latitude = new_lat
        st.session_state.longitude = new_lon
        
        # Mettre à jour automatiquement le code postal
        if st.session_state.auto_update_zipcode:
            with st.spinner("🔍 Recherche du code postal..."):
                new_zipcode = get_zipcode_from_coordinates(new_lat, new_lon)
                if new_zipcode:
                    st.session_state.zipcode = new_zipcode
                    st.success(f"✅ Code postal mis à jour: {new_zipcode}")
        
        st.rerun()

# Affichage des coordonnées
st.markdown(f"""
    <div class="coordinates-display">
        <div class="coord-item">
            <span class="coord-label">Latitude:</span>
            <span class="coord-value">{st.session_state.latitude:.6f}°</span>
        </div>
        <div class="coord-item">
            <span class="coord-label">Longitude:</span>
            <span class="coord-value">{st.session_state.longitude:.6f}°</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Séparateur
st.markdown("<br>", unsafe_allow_html=True)

# Section Formulaire 
st.markdown("""
    <div class="params-header">
        <div class="params-icon">⚙️</div>
        <div>
            <div class="params-title">Caracteristiques de la maison</div>
            <div class="params-subtitle">Remplissez les informations pour obtenir une estimation</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Section GPS
st.markdown('<div class="section-divider"><span class="section-icon"></span> Localisation GPS</div>', unsafe_allow_html=True)

# Option pour activer/désactiver la mise à jour automatique du code postal
col_toggle = st.columns([3, 1])[1]
with col_toggle:
    st.session_state.auto_update_zipcode = st.checkbox(
        "Auto ZIP",
        value=st.session_state.auto_update_zipcode,
        help="Mettre à jour automatiquement le code postal basé sur les coordonnées GPS"
    )

gps_col1, gps_col2, gps_col3 = st.columns([1, 1, 1])
with gps_col1:
    lat = st.number_input(
        "Latitude",
        value=st.session_state.latitude,
        format="%.6f",
        key="lat_input",
        help="Coordonnée de latitude"
    )
with gps_col2:
    long = st.number_input(
        "Longitude",
        value=st.session_state.longitude,
        format="%.6f",
        key="lon_input",
        help="Coordonnée de longitude"
    )
with gps_col3:
    zipcode = st.number_input(
        "Code Postal (Zipcode)",
        min_value=10000,
        max_value=99999,
        value=st.session_state.zipcode,
        step=1,
        help="Code postal de la propriété",
        key="zipcode_input"
    )


# Section Surfaces
st.markdown('<div class="section-divider"><span class="section-icon"></span> Surfaces et dimensions</div>', unsafe_allow_html=True)

surf_col1, surf_col2, surf_col3, surf_col4 = st.columns([1, 1, 1, 1])
with surf_col1:
    sqft_living = st.number_input(
        "Surface habitable (sqft)",
        min_value=300,
        max_value=13000,
        value=2000,
        step=50,
        help="Surface habitable en pieds carrés"
    )
with surf_col2:
    sqft_lot = st.number_input(
        "Surface terrain (sqft)",
        min_value=500,
        max_value=1500000,
        value=5000,
        step=100,
        help="Surface totale du terrain"
    )
with surf_col3:
    sqft_basement = st.number_input(
        "Surface sous-sol (sqft)",
        min_value=0,
        max_value=5000,
        value=0,
        step=50,
        help="Surface du sous-sol (0 si aucun)"
    )
with surf_col4:
    sqft_lot15 = st.number_input(
        "Surface moyenne terrain voisins (sqft)",
        min_value=500,
        max_value=1500000,
        value=5000,
        step=100,
        help="Moyenne des surfaces des 15 plus proches voisins"
    )

# Section Pièces
st.markdown('<div class="section-divider"><span class="section-icon"></span> Pièces et aménagements</div>', unsafe_allow_html=True)

pieces_col1, pieces_col2, pieces_col3, pieces_col4 = st.columns([1, 1, 1, 1])
with pieces_col1:
    bathrooms = st.number_input(
        "Nombre salles de bain",
        min_value=0.5,
        max_value=8.0,
        value=2.0,
        step=0.25,
        help="Nombre de salles de bain (0.5 = toilettes)"
    )
with pieces_col2:
    waterfront = st.selectbox(
        "Vue sur l'eau",
        options=[0, 1],
        format_func=lambda x: "Oui" if x == 1 else "Non",
        help="Propriété avec vue sur l'eau"
    )
with pieces_col3:
    view = st.selectbox(
        "Qualité de la vue",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: ["Aucune", "Moyenne", "Bonne", "Excellente", "Exceptionnelle"][x],
        help="Qualité de la vue (0-4)"
    )
with pieces_col4:
    condition = st.selectbox(
        "État du bien",
        options=[1, 2, 3, 4, 5],
        index=2,
        format_func=lambda x: ["Très mauvais", "Mauvais", "Moyen", "Bon", "Très bon"][x-1],
        help="État général de la propriété (1-5)"
    )

# Section Qualité
st.markdown('<div class="section-divider"><span class="section-icon"></span> Qualité et construction</div>', unsafe_allow_html=True)

qual_col1, qual_col2, qual_col3 = st.columns([1, 1, 1])
with qual_col1:
    grade = st.number_input(
        "Grade de construction",
        min_value=1,
        max_value=14,
        value=7,
        step=1,
        help="Qualité de construction et design (1-14)"
    )
    # Afficher la catégorie basée sur le grade
    if grade <= 4:
        grade_category = "Basique"
    elif grade <= 7:
        grade_category = "Standard"
    elif grade <= 10:
        grade_category = "Haut gamme"
    else:
        grade_category = "Luxe"
    st.caption(f"Catégorie: {grade_category}")
    
with qual_col2:
    from datetime import datetime
    current_year = datetime.now().year
    
    yr_built = st.number_input(
        "Année construction",
        min_value=1000,
        max_value=current_year,
        value=1990,
        step=1,
        help="Année de construction initiale"
    )
with qual_col3:
    # L'année de rénovation doit être >= année de construction et <= année en cours
    min_yr_renovated = yr_built if yr_built > 0 else 0
    
    yr_renovated = st.number_input(
        "Année rénovation",
        min_value=0,
        max_value=current_year,
        value=0,
        step=1,
        help=f"Année de dernière rénovation (0 si jamais rénové, min: {min_yr_renovated if min_yr_renovated > 0 else 'N/A'})"
    )
    
    # Validation : vérifier que l'année de rénovation n'est pas inférieure à l'année de construction
    if yr_renovated > 0 and yr_renovated < yr_built:
        st.warning(f" L'année de rénovation ({yr_renovated}) ne peut pas être inférieure à l'année de construction ({yr_built})")

# Note : annee_construction sera égale à yr_built
annee_construction = yr_built

# Bouton de prédiction
st.markdown("<br>", unsafe_allow_html=True)

if st.button("calculer le prix", use_container_width=True, type="primary"):
    model = load_model()
    
    if model is None:
        st.error(" Impossible de charger le modèle. Veuillez vérifier que le fichier 'xgb_house_price_model.pkl' existe.")
    else:
        with st.spinner("🔍 Analyse en cours..."):
            try:
                # Préparation des données pour la prédiction
                features_dict = {
                    'grade': grade,
                    'waterfront': waterfront,
                    'sqft_living': sqft_living,
                    'bathrooms': bathrooms,
                    'lat': lat,
                    'view': view,
                    'long': long,
                    'yr_built': yr_built,
                    'zipcode': zipcode,
                    'sqft_lot': sqft_lot,
                    'sqft_basement': sqft_basement,
                    'annee_construction': annee_construction,
                    'sqft_lot15': sqft_lot15,
                    'condition': condition,
                    'yr_renovated': yr_renovated
                }
                
                # Création du DataFrame avec l'ordre exact des features
                features_df = pd.DataFrame([features_dict])
                
                # # Affichage des features pour debug (optionnel)
                # with st.expander(" Voir les données envoyées au modèle"):
                #     st.dataframe(features_df, use_container_width=True)
                
                # Prédiction
                prediction = model.predict(features_df)[0]
                
                # Affichage du résultat
                st.markdown(f"""
                    <div class="prediction-result">
                        <div class="result-icon">💰</div>
                        <div>
                            <div class="result-label">Estimation du prix</div>
                            <div class="result-value">${prediction:,.2f}</div>
                            <div class="result-confidence">Modèle: XGBoost</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.success("✅ Prédiction effectuée avec succès!")
                
                # Calcul du prix au sqft
                price_per_sqft = prediction / sqft_living
                confidence_lower = prediction * 0.92
                confidence_upper = prediction * 1.08
                
                # Informations complémentaires
                st.markdown(f"""
                    <div class="info-box">
                        <strong>ℹ Informations complémentaires:</strong><br>
                        • Prix au pied carré: ${price_per_sqft:,.2f}/sqft<br>
                        • Fourchette estimée (±8%): ${confidence_lower:,.2f} - ${confidence_upper:,.2f}<br>
                        • Surface habitable: {sqft_living:,} sqft ({sqft_living * 0.092903:.1f} m²)<br>
                        • Grade de qualité: {grade}/14 ({grade_category})<br>
                        • Année de construction: {yr_built} {f'(rénové en {yr_renovated})' if yr_renovated > 0 else ''}
                    </div>
                """, unsafe_allow_html=True)
                
                # Graphique comparatif (optionnel)
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.metric(
                        label="Prix estimé",
                        value=f"${prediction:,.0f}",
                        delta=f"${prediction - (sqft_living * 200):,.0f} vs moyenne"
                    )
                with col_info2:
                    st.metric(
                        label="Prix/sqft",
                        value=f"${price_per_sqft:.2f}",
                        delta=f"Grade {grade}"
                    )
            
            except Exception as e:
                st.error(f" Erreur lors de la prédiction : {str(e)}")
                st.info(" Vérifiez que votre modèle attend bien les features dans cet ordre.")

# Footer
st.markdown("""
    <div class="footer">
        © 2025 HomePricer. Tous droits réservés.
    </div>
""", unsafe_allow_html=True)