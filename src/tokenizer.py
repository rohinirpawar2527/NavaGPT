
class Tokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}

    def fit(self, text):
        words = text.split()
        vocabulary = sorted(set(words))

        self.word_to_id = {
            word: index for index, word in enumerate(vocabulary)
        }

        self.id_to_word = {
            index: word for word, index in self.word_to_id.items()
        }

    def encode(self, text):
        words = text.split()
        return [self.word_to_id[word] for word in words]

    def decode(self, token_ids):
        return " ".join(
            self.id_to_word[token_id] for token_id in token_ids
        )
