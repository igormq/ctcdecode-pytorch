import torch
from ctcdecode.csrc import _C


Tokenizer = _C.Tokenizer
# class CharTokenizer(_C.BaseTokenizer):

#     reserved_tokens = ['<blank>', '<space>']

#     def __init__(self, vocab_file, ignored_tokens=[], allow_unknown=False):
#         super().__init__()

#         self.ignored_tokens = ignored_tokens
#         self._entry2idx = {}

#         if allow_unknown:
#             self.reserved_tokens += ['<unk>']

#         with open(vocab_file, 'r', encoding='utf8') as f:
#             for idx, line in enumerate(f):
#                 c = line.strip().split(' ')[0]

#                 if c in self.ignored_tokens:
#                     print(f'Ignoring {c}')
#                     continue

#                 self._entry2idx[c] = idx

#         for t in self.reserved_tokens:
#             if t not in self._entry2idx:
#                 print(f'Token {t} not found. Defining it with index {len(self._entry2idx)}')
#                 self._entry2idx[t] = len(self._entry2idx)

#         self._idx2entry = {v: k for k, v in self._entry2idx.items()}

#     def __len__(self):
#         return len(self._entry2idx)

#     def entry2idx(self, entry):
#         if '<unk>' in self._entry2idx:
#             return [self._entry2idx.get(t, self._entry2idx['<unk>']) for t in entry]

#         return [self._entry2idx[t] for t in entry]

#     def idx2entry(self, idxs):
#         if isinstance(idxs, torch.Tensor):
#             idxs = idxs.tolist()

#         idxs = list(map(lambda c: self._idx2entry[c], idxs))

#         return ''.join(idxs)

#     @property
#     def blank_idx(self):
#         return self._entry2idx['<blank>']

#     @property
#     def space_idx(self):
#         return self._entry2idx.get('<space>', None)

#     def get_space_idx(self):
#         return self.space_idx

#     def get_blank_idx(self):
#         return self.blank_idx

#     def __del__(self):
#         print('CharTokenizer foi embora')


#     def idx2token(self, token_ids, as_list=False, join=''):
#         if isinstance(token_ids, torch.Tensor):
#             token_ids = token_ids.tolist()

#         token_ids = list(map(lambda c: self._idx2entry[c], token_ids))

#         if as_list:
#             return token_ids

#         return join.join(token_ids)

#     @property
#     def blank_idx(self):
#         return self._entry2idx['<blank>']

#     @property
#     def space_idx(self):
#         return self._entry2idx.get('<space>', None)

# class CharTokenizer(BaseTokenizer):

#     reserved_tokens = ['<blank>', '<space>']

#     def __init__(self, vocab_file, ignored_tokens=[], allow_unknown=False):
#         super().__init__(vocab_file, ignored_tokens, allow_unknown)

#     def token2idx(self, text):
#         chars = map(lambda x: x.replace(' ', '<space>'), list(text))
#         return super().token2idx(chars)

#     def idx2token(self, token_ids, as_list=False):
#         out = super().idx2token(token_ids, as_list=as_list)

#         if as_list:
#             return out

#         return out.replace('<space>', ' ')

# class WordTokenizer(BaseTokenizer):
#     def __init__(self, vocab_file, ignored_tokens=[], allow_unknown=False):
#         super().__init__(vocab_file, ignored_tokens, allow_unknown)

#     def token2idx(self, text):
#         words = text.split(' ')
#         return super().token2idx(words)

#     def idx2token(self, token_ids, as_list=False):
#         return super().idx2token(token_ids, as_list=as_list, join=' ')
