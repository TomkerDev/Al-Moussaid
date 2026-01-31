import streamlit as st
from supabase import create_client
from groq import Groq
from sentence_transformers import SentenceTransformer
from datetime import datetime

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
