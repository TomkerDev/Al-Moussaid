import os
import requests
from bs4 import BeautifulSoup
from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq

# 1. Connexion via les secrets GitHub
url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')
groq_key = os.environ.get('GROQ_API_KEY')

supabase = create_client(url, key)
client_groq = Groq(api_key=groq_key)
model_embed = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. Ta fonction de scraping (ajoute ici celle qu'on a écrite)
def scraper():
    # ... ton code de scraping ...
    print("Scraping en cours...")

if __name__ == "__main__":
    scraper()
