import pandas as pd
import requests
import io

# Les URLs
URLS = {
    "CATALOGUE (RPA)": "https://data.montreal.ca/dataset/c5bf5e9c-528b-4c28-b52f-218215992e35/resource/0795f422-b53b-41ca-89be-abc1069a88c9/download/signalisation-codification-rpa.json",
    "PANNEAUX (SIG_STA)": "https://data.montreal.ca/dataset/8ac6dd33-b0d3-4eab-a334-5a6283eb7940/resource/7f1d4ae9-1a12-46d7-953e-6b9c18c78680/download/signalisation_stationnement.csv",
    "PARKINGS (PLACES)": "https://www.agencemobilitedurable.ca/images/data/Places.csv"
}

# Ceci est pour éviter l'erreur 403 sinon je risque d'avoir un forbidden
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def inspect_url(name, url):
    print(f"\n--- INSPECTION : {name} ---")
    print(f"URL: {url}")
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        if url.endswith('.csv'):
            content = response.content
            
            try:
                decoded_content = content.decode('utf-8')
            except UnicodeDecodeError:
                print(" Encodage UTF-8 échoué, tentative en Latin-1 (Windows)...")
                decoded_content = content.decode('latin-1')

            df = pd.read_csv(io.StringIO(decoded_content), nrows=5)
            print("\nCOLONNES TROUVÉES :")
            print(df.columns.tolist())
            
            print("\nExemple de données (1ère ligne) :")
            print(df.iloc[0].to_dict())

        elif url.endswith('.json'):
            try:
                df = pd.read_json(io.StringIO(response.text))
                print("\nCOLONNES TROUVÉES :")
                print(df.columns.tolist())
                print("\nExemple de données (1ère ligne) :")
                print(df.iloc[0].to_dict())
            except ValueError:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    print("\nCLÉS DU PREMIER OBJET :")
                    print(list(data[0].keys()))

    except Exception as e:
        print(f"Erreur lors de l'analyse : {e}")

if __name__ == "__main__":
    print("🔍 DÉBUT DE L'ANALYSE DES DONNÉES DISTANTES...")
    for name, url in URLS.items():
        inspect_url(name, url)