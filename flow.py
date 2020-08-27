from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Embedding
 
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
	for _ in range(n_words):
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		yhat = model.predict_classes(encoded, verbose=0)
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		in_text += ' ' + out_word
	return in_text

def generate_word(model, tokenizer, max_length, seed_text):
    in_text = seed_text
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
    yhat = model.predict_classes(encoded, verbose=0)
    out_word = ''
    for word, index in tokenizer.word_index.items():
        if index == yhat:
            out_word = word
            break
    return out_word
 
 
def load_data(file_name, size):
    data = ''
    with open(file_name) as f:
        head = [next(f) for x in range(size)]
    data = ''.join(head)
    return data

def model_tol(max_length, vocab_size, X, y, load=True):
    if(load):
        
        model = load_model('./obj/model_LSTM.h5')
        print("Loaded model from disk.")
        return model
    else:
        model = Sequential()
        model.add(Embedding(vocab_size, 10, input_length=max_length-1))
        model.add(LSTM(100))
        model.add(Dense(vocab_size, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, epochs=50) # Epochs: 500

        model.save('./obj/model_LSTM.h5')
        return model

def load_tokenizer():
    tokenizer = Tokenizer()
    data = load_data('./data/annotated.txt', 1500) # Max: 10000
    tokenizer.fit_on_texts([data])
    encoded = tokenizer.texts_to_sequences([data])[0]

    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)

    sequences = list()
    for i in range(2, len(encoded)):
        sequence = encoded[i-2:i+1]
        sequences.append(sequence)
    print('Total Sequences: %d' % len(sequences))

    max_length = max([len(seq) for seq in sequences])
    sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
    print('Max Sequence Length: %d' % max_length)

    sequences = array(sequences)
    X, y = sequences[:,:-1],sequences[:,-1]
    y = to_categorical(y, num_classes=vocab_size)

    return tokenizer, max_length, vocab_size, X, y
string = input("Input Sentence: ")
tokenizer, max_length, vocab_size, X, y = load_tokenizer()
model = model_tol(max_length, vocab_size, X, y, load=True)
print(generate_seq(model, tokenizer, max_length-1, string, 1))