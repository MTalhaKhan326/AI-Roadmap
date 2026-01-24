# NER stands for Name Entity Recognition

import spacy

# Load the model
nlp = spacy.load("en_core_web_md")

text = "this guy is working i Audi. He is AI engineer. His name is Talha and he was very passionate his monthly salary is 20,000 euro"

doc = nlp(text)

print(f"{'Text':<20} | {'Label':<10} | {'Description'}")
print("-" * 50)

# doc.ents contains the 'Named Entities' found by the AI
for ent in doc.ents:
    print(f"{ent.text:<20} | {ent.label_:<10} | {spacy.explain(ent.label_)}")