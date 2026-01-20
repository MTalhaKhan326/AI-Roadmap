# import spacy

# # 1. Load the English language model
# nlp = spacy.load("en_core_web_sm")

# # 2. Raw medical text
# text = "The Patient's Blood Tests were conducted yesterday the Blood test result is ok now now see what doctore says! Results: NORMAL."

# # 3. Apply the NLP Pipeline (Tokenization happens here)
# doc = nlp(text)

# print(f"{'Word':<15} | {'Is Stopword?':<12} | {'Lemma (Root)':<12}")
# print("-" * 45)

# # 4. Filter and Clean
# for token in doc:
#     # Check if it's punctuation or a stop word
#     if not token.is_stop and not token.is_punct:
#         print(f"{token.text} Vector: {token.vector[:5]}...")
#         # print(f"{token.text:<15} | {str(token.is_stop):<12} | {token.lemma_:<12}")

import spacy
import numpy as np

# 1. Load the model
nlp = spacy.load("en_core_web_md")

# 2. Sample text
text = "The doctor says the blood test result is normal."

# 3. Process the text
doc = nlp(text)

print(f"{'Word':<15} | {'Vector Shape':<15} | {'First 3 Values of Vector'}")
print("-" * 60)

for token in doc:
    # We only look at meaningful words (no stop words/punctuation)
    if not token.is_stop and not token.is_punct:
        
        # token.vector is the actual embedding
        vector_data = token.vector
        
        # We show the first 3 numbers just to see the 'math'
        print(f"{token.text:<15} | {str(vector_data.shape):<15} | {vector_data[:3]}")

# 4. Purpose Demo: Mathematical Similarity
# Let's compare two words manually
word1 = nlp("blood")
word2 = nlp("plasma")
word3 = nlp("apple")

print(f"\n--- Similarity Scores ---")
print(f"Blood vs Plasma: {word1.similarity(word2):.4f}") # Should be higher
print(f"Blood vs Apple: {word1.similarity(word3):.4f}")  # Should be lower