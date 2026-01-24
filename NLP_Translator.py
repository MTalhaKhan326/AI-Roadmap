from transformers import pipeline
from langdetect import detect

# Load both directions
en_to_de = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
de_to_en = pipeline("translation_de_to_en", model="Helsinki-NLP/opus-mt-de-en")

def universal_translator(text):
    # STEP 1: Detect the language automatically
    lang = detect(text) 
    
    # STEP 2: Use the correct Sequence-to-Sequence path
    if lang == 'en':
        print(f"Detected: English -> Translating to German")
        return en_to_de(text)[0]['translation_text']
    elif lang == 'de':
        print(f"Detected: German -> Translating to English")
        return de_to_en(text)[0]['translation_text']
    else:
        return "Language not supported yet!"

# Test it again
print(universal_translator("Hi ich heise Talha. Ich bin student im Detuchland"))