import pandas as pd
import numpy as np
import pickle

# Load Dataset
dataset_path = "updated_merged_dataset.csv"
data = pd.read_csv(dataset_path)

# Use the correct column names
word_column = "Word"
pos_column = "Pos_meaning"  # Updated to use 'Pos_meaning' column

# Extract unique POS tags and words
unique_words = set(data[word_column].values)
unique_tags = set(data[pos_column].values)

# Count occurrences for transition and emission probabilities
transition_counts = {}
emission_counts = {}
tag_counts = {}

for i in range(len(data) - 1):
    current_tag = data.iloc[i][pos_column]
    next_tag = data.iloc[i + 1][pos_column]
    word = data.iloc[i][word_column]
    
    # Count transitions
    if current_tag not in transition_counts:
        transition_counts[current_tag] = {}
    transition_counts[current_tag][next_tag] = transition_counts[current_tag].get(next_tag, 0) + 1
    
    # Count emissions
    if current_tag not in emission_counts:
        emission_counts[current_tag] = {}
    emission_counts[current_tag][word] = emission_counts[current_tag].get(word, 0) + 1
    
    # Count tag occurrences
    tag_counts[current_tag] = tag_counts.get(current_tag, 0) + 1

# Convert counts to probabilities
transition_probs = {tag: {next_tag: count / sum(transition_counts[tag].values()) for next_tag, count in transition_counts[tag].items()} for tag in transition_counts}

emission_probs = {tag: {word: count / tag_counts[tag] for word, count in emission_counts[tag].items()} for tag in emission_counts}

# Save the trained model to a pickle file
with open("marathi_pos_model.pkl", "wb") as f:
    pickle.dump((transition_probs, emission_probs, unique_tags), f)

print("Model saved as marathi_pos_model.pkl")
