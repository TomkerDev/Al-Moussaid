import streamlit as st
from PyPDF2 import PdfReader
from supabase import create_client
from groq import Groq
from sentence_transformers import SentenceTransformer
from datetime import datetime
import pandas as pd

# --- 1. CONFIGURATION ET SESSION (Doit être au début) ---
st.set_page_config(page_title="Al-Moussaid", page_icon="🇰🇮", layout="centered")

if 'resultats' not in st.session_state:
    st.session_state.resultats = None
if 'competences_detectees' not in st.session_state:
    st.session_state.competences_detectees = ""

# --- 2. INITIALISATION DES CLIENTS ---
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

# --- 3. FONCTIONS CŒUR ---
def extraire_competences(cv_text):
    prompt = f"Extrais uniquement les compétences techniques de ce profil sous forme de liste : {cv_text[:1500]}"
    completion = client_groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

def extraire_texte_fichier(uploaded_file):
    texte = ""
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            texte += page.extract_text()
    elif uploaded_file.type in ["image/jpeg", "image/png"]:
        st.warning("L'analyse directe d'images arrive bientôt. Utilisez le PDF pour l'instant.")
    return texte

# --- 4. BARRE LATÉRALE (FILTRES ET STATS) ---
st.sidebar.header("📍 Localisation")
villes_disponibles = ["Toutes", "N'Djamena", "Moundou", "Abéché", "Sarh", "Koumra", "Pala"]
ville_choisie = st.sidebar.selectbox("Filtrer par ville :", villes_disponibles)

# Section Alertes
st.sidebar.markdown("---")
st.sidebar.subheader("📩 Alerte Emploi")
email_user = st.sidebar.text_input("Ton email pour les alertes :", placeholder="exemple@email.com")

if st.sidebar.button("M'avertir des nouveaux jobs"):
    if email_user and st.session_state.competences_detectees:
        vecteur_user = model_embed.encode(st.session_state.competences_detectees).tolist()
        supabase.table("alertes_emails").insert({
            "email": email_user,
            "competences_detectees": st.session_state.competences_detectees,
            "embedding": vecteur_user,
            "seuil_match": 0.9
        }).execute()
        st.sidebar.success("Alerte activée !")
    else:
        st.sidebar.warning("Fais d'abord une recherche !")

# Section Statistiques
st.sidebar.markdown("---")
if st.sidebar.checkbox("📊 Voir les tendances"):
    stats_res = supabase.table("jobs").select("location").execute()
    df = pd.DataFrame(stats_res.data)
    if not df.empty:
        st.sidebar.bar_chart(df['location'].value_counts())
        st.sidebar.metric("Offres actives", len(df))

# --- 5. INTERFACE PRINCIPALE ---
st.title("🇰🇮 Al-Moussaid")
st.markdown("### Votre assistant IA pour l'emploi au Tchad")

mode_saisie = st.radio("Soumettre votre CV :", ("📤 Importer un fichier", "⌨️ Copier-coller le texte"))

cv_texte_final = ""

if mode_saisie == "📤 Importer un fichier":
    uploaded_file = st.file_uploader("Choisissez votre CV (PDF)", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Lecture du PDF..."):
            cv_texte_final = extraire_texte_fichier(uploaded_file)
            if cv_texte_final:
                st.success("Texte extrait avec succès.")
else:
    cv_texte_final = st.text_area("Collez votre profil ici :", height=150)

# --- 6. BOUTON DE RECHERCHE ---
if st.button("🔍 Rechercher mon match"):
    if cv_texte_final:
        with st.spinner('Analyse sémantique...'):
            st.session_state.competences_detectees = extraire_competences(cv_texte_final)
            vecteur = model_embed.encode(st.session_state.competences_detectees).tolist()
            
            res = supabase.rpc("match_jobs", {
                "query_embedding": vecteur,
                "match_threshold": 0.35,
                "match_count": 20 
            }).execute()
            
            st.session_state.resultats = res.data
    else:
        st.error("Contenu vide !")

# --- 7. AFFICHAGE DES RÉSULTATS ---
if st.session_state.resultats:
    # Filtrage par ville
    if ville_choisie != "Toutes":
        resultats_a_afficher = [j for j in st.session_state.resultats if ville_choisie.lower() in j.get('location', '').lower()]
    else:
        resultats_a_afficher = st.session_state.resultats

    if resultats_a_afficher:
        st.success(f"🎯 {len(resultats_a_afficher)} offres trouvées.")
        for job in resultats_a_afficher:
            with st.expander(f"💼 {job.get('title')} - {job.get('company')} ({int(job.get('similarity',0)*100)}%)"):
                st.write(f"📍 **Lieu :** {job.get('location')}")
                st.write(f"📝 {job.get('description')}")
                
                if st.button(f"📄 Lettre de motivation", key=f"btn_{job.get('id')}"):
                    prompt = f"Rédige une lettre pour {job.get('title')} chez {job.get('company')}. Compétences : {st.session_state.competences_detectees}"
                    lettre = client_groq.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}])
                    st.text_area("Votre lettre :", value=lettre.choices[0].message.content, height=200, key=f"txt_{job.get('id')}")
    else:
        st.warning("Aucune offre pour cette ville.")

with st.expander("ℹ️ Comment fonctionne Al-Moussaid ?"):
    st.markdown("""
    **Al-Moussaid** (L'Assistant) est la première plateforme de recrutement au Tchad propulsée par l'Intelligence Artificielle.
    
    1. **Analyse Intelligente** : Grâce aux modèles de langage (LLM), nous extrayons vos compétences réelles de votre CV, même s'il est au format PDF.
    2. **Matching Sémantique** : Au lieu de chercher des mots-clés exacts, notre IA comprend le sens de votre profil. Si vous êtes "Expert en Réseaux", elle vous proposera des postes de "Technicien Cisco" ou "Administrateur Système".
    3. **Aide à la Postulation** : L'IA rédige pour vous une ébauche de lettre de motivation personnalisée pour chaque offre trouvée, adaptée au contexte tchadien.
    
    *L'objectif est de réduire le chômage en connectant plus rapidement les talents aux opportunités locales.*
    """)
st.caption("Al-Moussaid v1.2 - N'Djamena, Tchad")
