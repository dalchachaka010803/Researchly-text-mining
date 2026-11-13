"""Text cleaning used by both the training script and the web app.

The app must clean an incoming abstract in exactly the same way as the
training data (Question 1.1), otherwise the model would see words in a
different form from what it learned.
"""
import re
import string
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()

# English stop words + academic words that appear in every abstract regardless of topic
stop_words = set(stopwords.words('english')).union({
    'paper', 'study', 'article', 'research', 'result', 'results', 'method', 'methods',
    'conclusion', 'conclusions', 'objective', 'background', 'finding', 'findings',
    'propose', 'proposed', 'show', 'shows', 'also', 'however', 'using', 'use', 'used',
    'based', 'approach', 'author', 'authors', 'within', 'among', 'may', 'well', 'one', 'two'
})


def denoise_text(text):
    """Remove publisher and formatting noise with regular expressions."""
    text = re.sub(r'(copyright\s*)?©.*$', ' ', text, flags=re.IGNORECASE)        # copyright notice
    text = re.sub(r'http\S+|www\.\S+', ' ', text)                                 # URLs
    text = re.sub(r'\[[^\]]*\]', ' ', text)                                        # [bracketed text]
    text = re.sub(r'^\s*(abstract|background)\s*[.:]?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'[‐-―−]', ' ', text)                              # unicode dashes
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\d+', ' ', text)                                                # numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)                                        # punctuation / symbols
    return re.sub(r'\s+', ' ', text).strip()


def preprocess_text(text):
    """Lowercase, tokenize, remove stop words and lemmatize."""
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens
              if word not in string.punctuation and word not in stop_words and len(word) > 2]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def clean(text):
    """Full pipeline: denoise, then normalise."""
    return preprocess_text(denoise_text(text))
