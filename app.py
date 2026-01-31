import streamlit as st
from PyPDF2 import PdfReader
from supabase import create_client
from groq import Groq
from sentence_transformers import SentenceTransformer
from datetime import datetime
import pandas as pd

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Al-Moussaid", page_icon="🇰🇮", layout="centered")

# --- INITIALISATION DES CLIENTS ---
@st.cache_resource
def init_connection():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

@st.cache_resource
def load_models():
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return model, client

supabase = init_connection()
model_embed, client_groq = load_models()

# --- FONCTIONS CŒUR ---
def extraire_competences(cv_text):
    prompt = f"Extrais uniquement les compétences techniques de ce profil sous forme de liste : {cv_text[:1500]}"
    completion = client_groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

# --- FONCTION D'EXTRACTION DE TEXTE ---
def extraire_texte_fichier(uploaded_file):
    texte = ""
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            texte += page.extract_text()
    elif uploaded_file.type in ["image/jpeg", "image/png"]:
        # Note: Pour les images, l'extraction nécessite un OCR plus complexe.
        # Pour commencer simple, on peut limiter aux PDF ou utiliser une consigne :
        st.warning("L'analyse directe d'images (JPG/PNG) arrive bientôt. Utilisez le PDF pour l'instant.")
    return texte

# --- INTERFACE PRINCIPALE ---
st.title("🇰🇮 Al-Moussaid")

# Option de saisie : Fichier ou Texte
mode_saisie = st.radio("Comment voulez-vous soumettre votre CV ?", ("📤 Importer un fichier", "⌨️ Copier-coller le texte"))

cv_texte_final = ""

if mode_saisie == "📤 Importer un fichier":
    uploaded_file = st.file_uploader("Choisissez votre CV (PDF, JPG, PNG)", type=["pdf", "jpg", "png"])
    if uploaded_file is not None:
        # Limite de taille (ex: 2Mo)
        if uploaded_file.size > 2 * 1024 * 1024:
            st.error("Le fichier est trop lourd. Limite : 2 Mo.")
        else:
            with st.spinner("Lecture du fichier..."):
                cv_texte_final = extraire_texte_fichier(uploaded_file)
                if cv_texte_final:
                    st.success("CV chargé avec succès !")
                    with st.expander("Aperçu du texte extrait"):
                        st.write(cv_texte_final[:500] + "...")
else:
    cv_texte_final = st.text_area("Collez votre profil ici :", height=150)

# --- BOUTON DE RECHERCHE ---
if st.button("🔍 Rechercher mon match"):
    if cv_texte_final:
        # On utilise cv_texte_final pour la suite de ton code (IA + Supabase)
# --- GESTION DE LA SESSION ---
if 'resultats' not in st.session_state:
    st.session_state.resultats = None
if 'competences_detectees' not in st.session_state:
    st.session_state.competences_detectees = ""

# --- BARRE LATÉRALE (FILTRES) ---
st.sidebar.header("📍 Localisation")
# On récupère les villes uniques de ta base si possible, sinon on liste les principales
villes_disponibles = ["Toutes", "N'Djamena", "Moundou", "Abéché", "Sarh", "Koumra", "Pala"]
ville_choisie = st.sidebar.selectbox("Filtrer par ville :", villes_disponibles)

# --- SECTION ALERTES DANS LA SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.subheader("📩 Alerte Emploi")
email_user = st.sidebar.text_input("Ton email pour les alertes :", placeholder="exemple@email.com")

if st.sidebar.button("M'avertir des nouveaux jobs"):
    if email_user and st.session_state.competences_detectees:
        # Vectorisation du profil de l'utilisateur
        vecteur_user = model_embed.encode(st.session_state.competences_detectees).tolist()
        
        # Enregistrement dans la table des alertes
        supabase.table("alertes_emails").insert({
            "email": email_user,
            "competences_detectees": st.session_state.competences_detectees,
            "embedding": vecteur_user,
            "seuil_match": 0.9
        }).execute()
        
        st.sidebar.success("Super ! Tu recevras un mail dès qu'un job à +90% de match est publié.")
    else:
        st.sidebar.warning("Fais d'abord une recherche pour que l'IA connaisse ton profil !")


# --- SECTION STATISTIQUES ---
st.sidebar.markdown("---")
if st.sidebar.checkbox("📊 Voir les tendances du marché"):
    st.markdown("## 📈 Statistiques du recrutement au Tchad")
    
    # 1. Récupération des données
    stats_res = supabase.table("jobs").select("location").execute()
    df = pd.DataFrame(stats_res.data)
    
    if not df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top des villes qui recrutent :**")
            # Graphique simple des villes
            st.bar_chart(df['location'].value_counts())
        
        with col2:
            st.write("**Volume total d'offres :**")
            st.metric("Offres actives", len(df))
            
        # 2. Analyse des mots-clés (Compétences demandées)
        st.write("**Compétences les plus recherchées :**")
        # On simule ici, mais on pourrait extraire les mots du titre
        titres = " ".join(supabase.table("jobs").select("title").execute().data[0].values())
        # (Optionnel : ajouter un nuage de mots ici)
    else:
        st.info("Pas assez de données pour les statistiques.")

# --- INTERFACE PRINCIPALE ---
st.title("🇰🇮 Al-Moussaid")
st.markdown("### Trouvez un emploi au Tchad grâce à l'IA")

cv_input = st.text_area("Collez votre CV ou décrivez votre profil :", height=150)

if st.button("🔍 Rechercher mon match"):
    if cv_input:
        with st.spinner('Analyse sémantique en cours...'):
            st.session_state.competences_detectees = extraire_competences(cv_input)
            vecteur = model_embed.encode(st.session_state.competences_detectees).tolist()
            
            # Appel RPC
            res = supabase.rpc("match_jobs", {
                "query_embedding": vecteur,
                "match_threshold": 0.35,
                "match_count": 20 # On en prend plus pour pouvoir filtrer par ville après
            }).execute()
            
            st.session_state.resultats = res.data
    else:
        st.error("Veuillez entrer un profil.")

# --- AFFICHAGE FILTRÉ ---
if st.session_state.resultats:
    # Logique de filtrage par ville
    if ville_choisie != "Toutes":
        resultats_a_afficher = [j for j in st.session_state.resultats if ville_choisie.lower() in j.get('location', '').lower()]
    else:
        resultats_a_afficher = st.session_state.resultats

    if resultats_a_afficher:
        st.success(f"🎯 {len(resultats_a_afficher)} offres correspondent à votre profil à {ville_choisie if ville_choisie != 'Toutes' else 'au Tchad'}")
        
        for job in resultats_a_afficher:
            with st.expander(f"💼 {job.get('title')} - {job.get('company')} (Match: {int(job.get('similarity',0)*100)}%)"):
                st.write(f"📍 **Ville :** {job.get('location', 'Non précisé')}")
                st.write(f"📝 **Description :** {job.get('description', 'Pas de description.')}")
                
                # Bouton Lettre
                if st.button(f"📄 Générer ma lettre pour {job.get('title')}", key=f"btn_{job.get('id')}"):
                    with st.spinner('Rédaction...'):
                        prompt_lettre = f"Rédige une lettre de motivation pour le poste {job.get('title')} chez {job.get('company')}. Compétences : {st.session_state.competences_detectees}"
                        lettre = client_groq.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt_lettre}])
                        st.text_area("Lettre :", value=lettre.choices[0].message.content, height=200, key=f"txt_{job.get('id')}")
                st.divider()
    else:
        st.warning(f"Aucune offre correspondante trouvée à {ville_choisie} pour le moment.")

st.sidebar.markdown("---")
st.sidebar.caption("Projet Al-Moussaid v1.2")
