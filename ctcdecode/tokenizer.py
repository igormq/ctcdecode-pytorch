import torch


class BaseTokenizer:
    def __init__(self,
                 vocab_file,
                 ignored_tokens=[],
                 allow_unknown=False,
                 reserved_tokens={'blank': '<blank>'}):
        self.ignored_tokens = ignored_tokens
        self.reserved_tokens = reserved_tokens
        self._token2idx = {}

        if allow_unknown:
            self.reserved_tokens += {'unk': '<unk>'}

        with open(vocab_file, 'r', encoding='utf8') as f:
            for idx, line in enumerate(f):
                c = line.strip().split(' ')[0]

                if c in self.ignored_tokens:
                    print(f'Ignoring {c}')
                    continue

                self._token2idx[c] = idx

        for k, v in self.reserved_tokens.items():
            if v not in self._token2idx:
                print(f'Token {k} not found. Defining it with index {len(self._token2idx)}')
                self._token2idx[v] = len(self._token2idx)

        self._idx2token = {v: k for k, v in self._token2idx.items()}
        self._len = len(self._token2idx)

    def __len__(self):
        return self._len

    def token2idx(self, tokens):
        if self.reserved_tokens['unk'] in self._token2idx:
            return [
                self._token2idx.get(t, self._token2idx[self.reserved_tokens['unk']]) for t in tokens
            ]

        return [self._token2idx[t] for t in tokens]

    def idx2token(self, token_ids, as_list=False, join=''):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        token_ids = list(map(lambda c: self._idx2token[c], token_ids))

        if as_list:
            return token_ids

        return join.join(token_ids)

    @property
    def space_token(self):
        return self.reserved_tokens['space']

    @property
    def blank_token(self):
        return self.reserved_tokens['blank']

    @property
    def blank_idx(self):
        return self._token2idx[self.blank_token]

    @property
    def space_idx(self):
        return self._token2idx.get(self.space_token, None)


class CharTokenizer(BaseTokenizer):
    def __init__(self,
                 vocab_file,
                 ignored_tokens=[],
                 allow_unknown=False,
                 reserved_tokens={
                     'blank': '<blank>',
                     'space': '<space>'
                 }):
        super().__init__(vocab_file, ignored_tokens, allow_unknown, reserved_tokens)

    def token2idx(self, text):
        chars = map(lambda x: x.replace(' ', self.reserved_tokens['space']), list(text))
        return super().token2idx(chars)

    def idx2token(self, token_ids, as_list=False):
        out = super().idx2token(token_ids, as_list=as_list)

        if as_list:
            return out

        return out.replace(self.reserved_tokens['space'], ' ')


class WordTokenizer(BaseTokenizer):
    def __init__(self, vocab_file, ignored_tokens=[], allow_unknown=False):
        super().__init__(vocab_file, ignored_tokens, allow_unknown)

    def token2idx(self, text):
        words = text.split(' ')
        return super().token2idx(words)

    def idx2token(self, token_ids, as_list=False):
        return super().idx2token(token_ids, as_list=as_list, join=' ')
