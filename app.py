from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import random
import re

app = Flask(__name__)

# Load the trained model
model = load_model('generator/poem_model.h5')

# Load the tokenizer
tokenizer_file_path = 'generator/tokenizer.pkl'
try:
    with open(tokenizer_file_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded successfully.")
except FileNotFoundError:
    print(f"Error: Tokenizer file not found at {tokenizer_file_path}")
    tokenizer = None
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer = None

# Temperature sampling function for creativity
def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

# Function to generate text using the loaded model
def generate_poem(topic, num_words, model, tokenizer):
    if not tokenizer:
        return "Error: Tokenizer not loaded."

    max_sequence_length = 100  # adjust this to your model's expected input length

    # Create a prompt seed phrase to guide generation
    seed_text = f"poem about {topic}"

    # Convert seed_text to sequence of tokens
    start_sequence = tokenizer.texts_to_sequences([seed_text])[0]

    # Pad the sequence
    padded_sequence = pad_sequences([start_sequence], maxlen=max_sequence_length, padding='pre')

    generated_sequence = list(start_sequence)
    output_word_count = len(seed_text.split())

    while output_word_count < num_words:
        predicted_probs = model.predict(padded_sequence, verbose=0)[0]
        next_token = np.argmax(predicted_probs)
        next_word = tokenizer.index_word.get(next_token, '')

        if next_word and next_word != '<unk>':
            generated_sequence.append(next_token)
            output_word_count += 1
            current_sequence = generated_sequence[-max_sequence_length:]
            padded_sequence = pad_sequences([current_sequence], maxlen=max_sequence_length, padding='pre')
        else:
            break

    generated_text = tokenizer.sequences_to_texts([generated_sequence])[0]

    # Replace special newline tokens if your tokenizer/model uses any
    generated_text = generated_text.replace('<newline>', '\n')

    # Here, split into stanzas and lines as you currently do
    potential_lines = generated_text.split('\n')
    structured_stanzas = []
    current_stanza_lines = []
    lines_per_stanza = 4
    line_length_target = 10

    for line in potential_lines:
        words = line.split()
        current_line_words = []
        for word in words:
            current_line_words.append(word)
            if len(current_line_words) >= line_length_target:
                current_stanza_lines.append(" ".join(current_line_words))
                current_line_words = []
        if current_line_words:
            current_stanza_lines.append(" ".join(current_line_words))
        if len(current_stanza_lines) >= lines_per_stanza:
            structured_stanzas.append(current_stanza_lines)
            current_stanza_lines = []

    if current_stanza_lines:
        structured_stanzas.append(current_stanza_lines)

    return structured_stanzas

@app.route('/', methods=['GET', 'POST'])
def index():
    generated_poems = []
    poem_topic = ""
    poem_length = "short"
    poem_type = "default"
    language = "english"
    selected_poem_index = 0

    if request.method == 'POST':
        poem_topic = request.form.get('poem_topic', '')
        poem_length = request.form.get('poem_length', 'short')
        poem_type = request.form.get('poem_type', 'default')
        language = request.form.get('language', 'english')

        if poem_length == "short":
            num_words = 30
        elif poem_length == "medium":
            num_words = 60
        else:
            num_words = 100

        if poem_topic:
            num_versions = 2
            for _ in range(num_versions):
                generated_poem = generate_poem(poem_topic, num_words, model, tokenizer)
                generated_poems.append(generated_poem)

    return render_template('generator_form.html', generated_poems=generated_poems, poem_topic=poem_topic, poem_length=poem_length, poem_type=poem_type, language=language, selected_poem_index=selected_poem_index)

if __name__ == '__main__':
    app.run(debug=True)
