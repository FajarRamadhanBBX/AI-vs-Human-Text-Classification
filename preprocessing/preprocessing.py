import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

with open('/tokenizer/tokenizer.pkl', 'rb') as f:
    loaded_tokenizer = pickle.load(f)
    
special_char = re.compile(r'[^a-zA-Z0-9\s]')

def preprocess_text(text):
  text = remove_whitespace(text)
  text = lower_remove_punctuation(text)
  text = stopword(text)
  text = loaded_tokenizer.texts_to_sequences([text])
  text = pad_sequences(text)
  return text

def remove_whitespace(text):
  text = str(text)
  text = re.sub(r"\s+", " ", text)
  return text.strip()

def lower_remove_punctuation(text):
  text = text.lower()
  text = special_char.sub(' ', text)
  return text

def stopword(text):
  stop_words = set(stopwords.words('english'))
  words = text.split()
  filtered_words = [word for word in words if word not in stop_words]
  return " ".join(filtered_words)