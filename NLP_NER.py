# NER stands for Name Entity Recognition

import spacy

# Load the model
nlp = spacy.load("en_core_web_md")

text = "Apple is looking at buying a startup in the United Kingdom for $1 billion."

doc = nlp(text)

print(f"{'Text':<20} | {'Label':<10} | {'Description'}")
print("-" * 50)

# doc.ents contains the 'Named Entities' found by the AI
for ent in doc.ents:
    print(f"{ent.text:<20} | {ent.label_:<10} | {spacy.explain(ent.label_)}")