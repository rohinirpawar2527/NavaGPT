
class BPETokenizer:
    def __init__(self):
        self.merges = []
        self.vocab = {}
        self.inverse_vocab = {}

    def fit(self, text, num_merges=100):
        tokens = list(text)
        vocabulary = set(tokens)

        for _ in range(num_merges):
            pairs = {}

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)

            if pairs[best_pair] < 2:
                break

            merged_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            vocabulary.add(merged_token)

            new_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and (tokens[i], tokens[i + 1]) == best_pair
                ):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        vocabulary.add("<UNK>")

        self.vocab = {
            token: index
            for index, token in enumerate(sorted(vocabulary))
        }

        self.inverse_vocab = {
            index: token
            for token, index in self.vocab.items()
        }

    def encode(self, text):
        tokens = list(text)

        for pair in self.merges:
            merged_token = pair[0] + pair[1]
            new_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and (tokens[i], tokens[i + 1]) == pair
                ):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return [
            self.vocab.get(token, self.vocab["<UNK>"])
            for token in tokens
        ]

    def decode(self, token_ids):
        return "".join(
            self.inverse_vocab[token_id]
            for token_id in token_ids
        )
