import spacy
import numpy as np

# 1. Load the "Big Brain" model
nlp = spacy.load("en_core_web_md")

# 2. Two sentences with different meanings but similar words
sentences = [
    "Results are normal",
    "Results are NOT normal"
]

# 3. Create a "Sequence" for the model
sequence_data = []

for text in sentences:
    doc = nlp(text)
    
    # Extract vectors for EVERY word in the exact order
    # This creates a "Time Series" of 300-dimension vectors
    sentence_vectors = [token.vector for token in doc]
    
    sequence_data.append(sentence_vectors)

# Convert to a NumPy array for math operations
# Note: In real deep learning, we "pad" these to be the same length
sequence_array = np.array(sequence_data, dtype=object)

print(f"Number of Sentences: {len(sequence_array)}")
print(f"Words in Sentence 1: {len(sequence_array[0])}")
print(f"Words in Sentence 2: {len(sequence_array[1])}")

# 4. The "RNN" View
print("\n--- RNN Processing View ---")
print(f"The model first sees 'Results' vector: {sequence_array[1][0][:3]}...")
print(f"Then it sees 'are' vector: {sequence_array[1][1][:3]}...")
print(f"Then it sees 'NOT' vector: {sequence_array[1][2][:3]}...")