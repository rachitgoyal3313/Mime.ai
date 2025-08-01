import json
import os
from django.conf import settings
import requests
from nltk.corpus import stopwords
import nltk
c = 0
if(c == 0):
    nltk.download('stopwords')
    c = 1

# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


# Load animation vocab
with open('Main/vocab/animation_words.txt', 'r') as f:
    animation_words = [line.strip().lower() for line in f if line.strip()]

# Path to the JSON cache file
CACHE_FILE = 'Main/vocab/word_synonym_map.json'

# Load synonym cache from file
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as f:
        synonym_cache = json.load(f)
else:
    synonym_cache = {}

# Save mapping to cache file
def save_cache():
    with open(CACHE_FILE, 'w') as f:
        json.dump(synonym_cache, f, indent=2)

def get_synonym_in_vocab_spacy(word):
    word = word.lower()

    # 1. Check if already in cache
    if word in synonym_cache:
        return synonym_cache[word]

    # 2. If it's already a valid animation word
    if word in animation_words:
        synonym_cache[word] = word
        save_cache()
        return word

    # 3. Otherwise call OpenRouter AI
    prompt = f"""
    You are an AI that helps translate text into sign language. You are given:

    - An input word: "{word}"
    - A list of known words that can be animated: {animation_words}

    Your task:
    - If the input word is in the list, return it.
    - If a **close synonym** is in the list (e.g., "father" → "dad"), return it.
    - If **no close synonym** exists, return "-1"

    ⚠️ Do not guess. If you are unsure or the match is too weak, return "-1".
    Only return the chosen word or "-1". No explanation.
    Do not return any additional text like Word not available or something, just return the closest word available in our list, otherwise return -1.
    
    EXAMPLES:
    INPUT: buddy,
    OUTPUT: friend
    """

    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Mime AI Glossifier",
        "X-User": "MimeAIUser"
    }

    payload = {
        "model": "x-ai/grok-3-mini",
        # "model": "deepseek/deepseek-r1-distill-llama-70b",
        "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 6000
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        synonym = data['choices'][0]['message']['content'].strip()

        # Cache result
        if synonym == "-1":
            synonym_cache[word] = word  # Cache the original word if no synonym found
        else:
            synonym_cache[word] = synonym

        save_cache()
        
        return synonym

    except Exception as e:
        print(f"Error getting synonym for '{word}':", e)
        return word  # fallback


def normalize_and_glossify(text):
    text = text.lower()
    gloss = []

    for token in text.split():

        if token in animation_words:
            gloss.append(token)

        elif token in stop_words:
            continue

        else:
            s = get_synonym_in_vocab_spacy(token)
            if s != "-1":
                gloss.append(s)
            else:
                gloss.append(token)

    return gloss
