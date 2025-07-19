from flask import Flask, request, jsonify, render_template
import pickle
import re

app = Flask(__name__)

# Load the trained model
with open("marathi_pos_model.pkl", "rb") as f:
    transition_probs, emission_probs, unique_tags = pickle.load(f)

# Function to get the POS meaning of a given word
def get_pos_meaning(word):
    possible_tags = {tag: emission_probs[tag].get(word, 0) for tag in unique_tags}
    best_tag = max(possible_tags, key=possible_tags.get) if possible_tags else "Unknown"
    return best_tag

# Function to tokenize Marathi text
def tokenize_marathi(text):
    # Remove punctuation and split by whitespace
    # Keep punctuation as separate tokens
    tokens = []
    # First, separate punctuation with spaces
    text = re.sub(r'([।,.!?;:\'\"()])', r' \1 ', text)
    # Split by whitespace
    for token in text.split():
        if token.strip():
            tokens.append(token.strip())
    return tokens

# Function to tag a sentence
def tag_sentence(sentence):
    tokens = tokenize_marathi(sentence)
    tagged_words = []
    
    for token in tokens:
        pos_tag = get_pos_meaning(token)
        tagged_words.append({"word": token, "pos_tag": pos_tag})
    
    return tagged_words

# Function to generate graph data for word relationships
def generate_graph_data(tagged_words):
    nodes = []
    links = []
    
    # Create nodes for each word
    for i, item in enumerate(tagged_words):
        nodes.append({
            "id": i,
            "word": item["word"],
            "pos_tag": item["pos_tag"],
            "group": get_pos_group(item["pos_tag"])
        })
    
    # Create links between consecutive words
    for i in range(len(tagged_words) - 1):
        links.append({
            "source": i,
            "target": i + 1,
            "value": 1
        })
    
    return {"nodes": nodes, "links": links}

# Function to assign group numbers based on POS tags
def get_pos_group(pos_tag):
    pos_groups = {
        "NN": 1,  # Noun
        "NNP": 1, # Proper Noun
        "PRP": 2, # Pronoun
        "JJ": 3,  # Adjective
        "RB": 4,  # Adverb
        "VB": 5,  # Verb
        "VM": 5,  # Main Verb
        "VAUX": 5, # Auxiliary Verb
        "CC": 6,  # Conjunction
        "PSP": 7, # Postposition
        "QF": 8,  # Quantifier
        "QC": 8,  # Cardinal
        "QO": 8,  # Ordinal
        "SYM": 9, # Symbol
        "RDP": 10, # Reduplication
        "ECH": 11, # Echo
        "UNK": 12, # Unknown
    }
    
    # Default to group 0 if POS tag not found
    return pos_groups.get(pos_tag, 0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/pos-meaning', methods=['POST', 'GET'])
def pos_meaning():
    if request.method == 'POST':
        data = request.json
        text = data.get("word", "").strip()
    else:
        text = request.args.get("word", "").strip()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Tag the entire sentence
    tagged_sentence = tag_sentence(text)
    
    return jsonify({
        "original_text": text,
        "tagged_words": tagged_sentence
    })

@app.route('/transition-probabilities', methods=['GET', 'POST'])
def get_transition_probabilities():
    return jsonify({"transition_probabilities": transition_probs})

@app.route('/word-relationships', methods=['GET'])
def word_relationships():
    text = request.args.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Tag the sentence
    tagged_sentence = tag_sentence(text)
    
    # Generate graph data
    graph_data = generate_graph_data(tagged_sentence)
    
    return jsonify({
        "original_text": text,
        "graph_data": graph_data
    })

if __name__ == '__main__':
    app.run(debug=True, port=5051)