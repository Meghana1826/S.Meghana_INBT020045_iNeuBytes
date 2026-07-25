import re
import string

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
}

def clean_text(text, remove_stopwords=True):
    """
    Cleans raw review text:
    - Lowercase
    - Remove punctuation & HTML
    - Remove stopwords
    """
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()

    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS]

    return " ".join(words)

class CustomTokenizer:
    """Sequence Tokenizer & Padder for Neural Network."""
    def __init__(self, vocab_size=2000, max_len=100):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def fit_on_texts(self, texts):
        word_counts = {}
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1

        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[: self.vocab_size - 2]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def text_to_sequence(self, text):
        seq = [self.word2idx.get(w, 1) for w in text.split()]
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        else:
            seq = seq + [0] * (self.max_len - len(seq))
        return seq
