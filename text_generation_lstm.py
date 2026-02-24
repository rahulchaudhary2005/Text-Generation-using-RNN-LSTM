import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load Text
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

# Create Input Sequences
input_sequences = []

for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)

# Padding
max_len = max(len(seq) for seq in input_sequences)

input_sequences = pad_sequences(
    input_sequences,
    maxlen=max_len,
    padding='pre'
)

# Split X and Y
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot encode output
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Build Model
model = Sequential()

model.add(Embedding(total_words, 128, input_length=max_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

# Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Train
model.fit(X, y, epochs=100, batch_size=32)

# Save
model.save("text_generator.h5")

# Text Generation Function
def generate_text(seed_text, next_words=20):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_len-1,
            padding='pre'
        )

        predicted = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted)

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break

    return seed_text


# Test Generator
print(generate_text("machine learning", 15))