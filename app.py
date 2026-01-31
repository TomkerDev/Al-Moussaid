import streamlit as st
from supabase import create_client
from groq import Groq
from sentence_transformers import SentenceTransformer
import streamlit as st
url = st.secrets["SUPABASE_URL"]

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Al-Moussaid", page_icon="🇰🇮", layout="centered")

# --- INITIALISATION DES CLIENTS (Via Streamlit Secrets) ---
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

@st.cache_resource
def load_models():
    # Modèle d'embedding (Open-source)
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    # Client Groq
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return model, client

supabase = init_connection()
model_embed, client_groq = load_models()

# --- FONCTIONS CŒUR ---
def extraire_competences(cv_text):
    prompt = f"Extrais uniquement les compétences techniques et outils de ce profil sous forme de liste : {cv_text[:1500]}"
    completion = client_groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

# --- INTERFACE UTILISATEUR ---
st.title("🇰🇮 Al-Moussaid")
st.markdown("### L'Assistant intelligent pour l'emploi au Tchad")
st.info("Collez votre CV ou décrivez vos compétences ci-dessous pour trouver les meilleures offres.")

# Zone de saisie
cv_input = st.text_area("Votre profil (CV ou résumé) :", height=200, placeholder="Ex: Informaticien spécialisé en maintenance et réseaux Cisco...")

if st.button("🔍 Rechercher mon match"):
    if cv_input:
        with st.spinner('Analyse de votre profil et recherche en cours...'):
            # 1. Extraction des compétences
            competences = extraire_competences(cv_input)
            
            # 2. Vectorisation (384 dimensions)
            vecteur = model_embed.encode(competences).tolist()
            
            # 3. Recherche dans Supabase
            res = supabase.rpc("match_jobs", {
                "query_embedding": vecteur,
                "match_threshold": 0.35,
                "match_count": 5
            }).execute()
            
            # 4. Affichage des résultats
            # --- AJOUT DANS LA BOUCLE DES RÉSULTATS ---
if res.data:
    st.balloons()
    st.success(f"Nous avons trouvé {len(res.data)} offres pour vous !")
    
    for job in res.data:
        with st.expander(f"🎯 {job['title']} - {job['company']} (Match: {int(job['similarity']*100)}%)"):
            st.write(f"**Lieu :** {job['location']}")
            st.write(f"**Description :** {job['description']}")
            
            # Nouveau bouton pour la lettre de motivation
            if st.button(f"📄 Générer ma lettre pour {job['title']}", key=job['id']):
                with st.spinner('Rédaction de votre lettre personnalisée...'):
                    prompt_lettre = f"""
                    Rédige une lettre de motivation professionnelle et convaincante pour un étudiant tchadien.
                    Poste : {job['title']} chez {job['company']}.
                    Compétences du candidat : {competences}
                    Contexte : Le ton doit être respectueux et adapté au marché du travail au Tchad.
                    """
                    
                    lettre = client_groq.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt_lettre}]
                    )
                    
                    st.text_area("Votre lettre de motivation :", value=lettre.choices[0].message.content, height=300)
                    st.download_button("📥 Télécharger la lettre", lettre.choices[0].message.content, file_name=f"Lettre_{job['title']}.txt")
            st.divider()

# --- FOOTER ---
st.markdown("---")
st.caption("Projet Al-Moussaid - Propulsé par l'IA Open-source et Supabase.")
