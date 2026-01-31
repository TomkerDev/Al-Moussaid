import requests
from bs4 import BeautifulSoup
import time
from supabase import create_client
from groq import Groq
from sentence_transformers import SentenceTransformer
import os

# --- INITIALISATION ---
url_supabase = os.environ.get('SUPABASE_URL')
key_supabase = os.environ.get('SUPABASE_KEY')
supabase = create_client(url_supabase, key_supabase)
client_groq = Groq(api_key=os.environ.get('GROQ_API_KEY'))
model_embed = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def get_vector(text):
    return model_embed.encode(text).tolist()

def scraper_tchad_offres():
    # URL exemple (à adapter selon le site cible)
    target_url = "https://www.tchadcarriere.com/offres-emploi" 
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(target_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. On cherche toutes les cartes d'offres
        # Note : Les classes 'job-item' ou 'card' varient selon le site
        offres = soup.find_all('div', class_='job-item') 

        for offre in offres:
            titre = offre.find('h2').text.strip()
            
            # Vérifier si l'offre existe déjà pour ne pas gaspiller de tokens Groq
            check = supabase.table("jobs").select("id").eq("title", titre).execute()
            if len(check.data) > 0:
                print(f"⏩ Déjà en base : {titre}")
                continue

            # 2. IA : On demande à Groq de nettoyer le texte brut récupéré
            print(f"🧠 Structuration de : {titre}")
            texte_brut = offre.text.strip()
            
            prompt = f"Récupère uniquement ces infos en JSON (title, company, description, location) de ce texte : {texte_brut}"
            
            chat = client_groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            
            import json
            data = json.loads(chat.choices[0].message.content)
            
            # 3. Vectorisation et Insertion
            vecteur = get_vector(data.get('description', titre))
            
            supabase.table("jobs").insert({
                "title": data.get('title', titre),
                "company": data.get('company', 'Non précisé'),
                "description": data.get('description', ''),
                "location": data.get('location', 'Tchad'),
                "embedding": vecteur
            }).execute()
            
            print(f"✅ Ajouté : {titre}")
            time.sleep(1) # Pause éthique

    except Exception as e:
        print(f"❌ Erreur lors du scraping : {e}")

if __name__ == "__main__":
    scraper_tchad_offres()
