import streamlit as st
from supabase import create_client
from groq import Groq
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Al-Moussaid", page_icon="🇰🇮", layout="centered")

# --- INITIALISATION DES CLIENTS ---
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

@st.cache_resource
def load_models():
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
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

# --- GESTION DE L'ÉTAT (SESSION STATE) ---
if 'resultats' not in st.session_state:
    st.session_state.resultats = None
if 'competences_detectees' not in st.session_state:
    st.session_state.competences_detectees = ""

# --- INTERFACE UTILISATEUR ---
st.title("🇰🇮 Al-Moussaid")
st.markdown("### L'Assistant intelligent pour l'emploi au Tchad")

cv_input = st.text_area("Votre profil (CV ou résumé) :", height=150, placeholder="Ex: Informaticien spécialisé en maintenance...")

if st.button("🔍 Rechercher mon match"):
    if cv_input:
        with st.spinner('Analyse et recherche en cours...'):
            # Sauvegarde des compétences pour la lettre plus tard
            st.session_state.competences_detectees = extraire_competences(cv_input)
            
            # Vectorisation
            vecteur = model_embed.encode(st.session_state.competences_detectees).tolist()
            
            # Recherche
            res = supabase.rpc("match_jobs", {
                "query_embedding": vecteur,
                "match_threshold": 0.35,
                "match_count": 5
            }).execute()
            
            # Stockage des résultats dans la session
            st.session_state.resultats = res.data
    else:
        st.error("Veuillez entrer votre profil.")

# --- AFFICHAGE DES RÉSULTATS ---
if st.session_state.resultats:
    st.success(f"Nous avons trouvé {len(st.session_state.resultats)} offres !")
    
    for job in st.session_state.resultats:
        with st.expander(f"🎯 {job['title']} - {job['company']} (Match: {int(job['similarity']*100)}%)"):
            st.write(f"**Lieu :** {job['location']}")
            st.write(f"**Description :** {job['description']}")
            
            # Bouton de génération
            if st.button(f"📄 Générer ma lettre pour {job['title']}", key=f"btn_{job['id']}"):
                with st.spinner('Rédaction...'):
                    prompt_lettre = f"""
                    Rédige une lettre de motivation courte pour le poste de {job['title']} chez {job['company']}.
                    Compétences du candidat : {st.session_state.competences_detectees}
                    Ton : Professionnel, respectueux, adapté au Tchad.
                    """
                    
                    lettre = client_groq.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt_lettre}]
                    )
                    
                    st.text_area("Lettre générée :", value=lettre.choices[0].message.content, height=200, key=f"txt_{job['id']}")
            st.divider()

st.markdown("---")
st.caption("Projet Al-Moussaid - Propulsé par l'IA au Tchad.")
