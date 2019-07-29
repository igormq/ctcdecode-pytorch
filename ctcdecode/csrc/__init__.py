import os

from torch.utils.cpp_extension import load

print(2)

base_dir = os.path.dirname(os.path.realpath(__file__))
print(base_dir)

binding = load(name='binding',
               sources=[
                   os.path.join(base_dir, 'decoder_utils.cpp'),
                   os.path.join(base_dir, 'path_trie.cpp'),
                   os.path.join(base_dir, 'binding.cpp')
               ],
               verbose=True)
