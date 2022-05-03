"""CoreNLP-related utilities."""
def rejoin(tokens, sep=None):
  """Rejoin tokens into the original sentence.
  
  Args:
    tokens: a list of dicts containing 'originalText' and 'before' fields.
        All other fields will be ignored.
    sep: if provided, use the given character as a separator instead of
        the 'before' field (e.g. if you want to preserve where tokens are).
  Returns: the original sentence that generated this CoreNLP token list.
  """
  if sep is None:
    return ''.join('%s%s' % (t['before'], t['originalText']) for t in tokens)
  else:
    # Use the given separator instead
    return sep.join(t['originalText'] for t in tokens)

class ConstituencyParse(object):
  """A CoreNLP constituency parse (or a node in a parse tree).
  
  Word-level constituents have |word| and |index| set and no children.
  Phrase-level constituents have no |word| or |index| and have at least one child.
  """
  def __init__(self, tag, children=None, word=None, index=None):
    self.tag = tag
    if children:
      self.children = children
    else:
      self.children = None
    self.word = word
    self.index = index

  @classmethod
  def _recursive_parse_corenlp(cls, tokens, i, j):
    orig_i = i
    if tokens[i] == '(':
      tag = tokens[i + 1]
      children = []
      i = i + 2
      while True:
        child, i, j = cls._recursive_parse_corenlp(tokens, i, j)
        if isinstance(child, cls):
          children.append(child)
          if tokens[i] == ')': 
            return cls(tag, children), i + 1, j
        else:
          if tokens[i] != ')':
            raise ValueError('Expected ")" following leaf')
          return cls(tag, word=child, index=j), i + 1, j + 1
    else:
      # Only other possibility is it's a word
      return tokens[i], i + 1, j

  @classmethod
  def from_corenlp(cls, s):
    """Parses the "parse" attribute returned by CoreNLP parse annotator."""
    # "parse": "(ROOT\n  (SBARQ\n    (WHNP (WDT What)\n      (NP (NN portion)\n        (PP (IN                       of)\n          (NP\n            (NP (NNS households))\n            (PP (IN in)\n              (NP (NNP             Jacksonville)))))))\n    (SQ\n      (VP (VBP have)\n        (NP (RB only) (CD one) (NN person))))\n    (. ?        )))",
    s_spaced = s.replace('\n', ' ').replace('(', ' ( ').replace(')', ' ) ')
    tokens = [t for t in s_spaced.split(' ') if t]
    tree, index, num_words = cls._recursive_parse_corenlp(tokens, 0, 0)
    if index != len(tokens):
      raise ValueError('Only parsed %d of %d tokens' % (index, len(tokens)))
    return tree

  def is_singleton(self):
    if self.word: return True
    if len(self.children) > 1: return False
    return self.children[0].is_singleton()
    
  def print_tree(self, indent=0):
    spaces = '  ' * indent
    if self.word:
      print('%s%s: %s (%d)' % (spaces, self.tag, self.word, self.index))
    else:
      print('%s%s:' % (spaces, self.tag))
      for c in self.children:
        c.print_tree(indent=indent + 1)

  def get_phrase(self):
    if self.word: return self.word
    toks = []
    for i, c in enumerate(self.children):
      p = c.get_phrase()
      if i == 0 or p.startswith("'"):
        toks.append(p)
      else:
        toks.append(' ' + p)
    return ''.join(toks)

  def get_start_index(self):
    if self.index is not None: return self.index
    return self.children[0].get_start_index()

  def get_end_index(self):
    if self.index is not None: return self.index + 1
    return self.children[-1].get_end_index()

  @classmethod
  def _recursive_replace_words(cls, tree, new_words, i):
    if tree.word:
      new_word = new_words[i]
      return (cls(tree.tag, word=new_word, index=tree.index), i + 1)
    new_children = []
    for c in tree.children:
      new_child, i = cls._recursive_replace_words(c, new_words, i)
      new_children.append(new_child)
    return cls(tree.tag, children=new_children), i

  @classmethod
  def replace_words(cls, tree, new_words):
    """Return a new tree, with new words replacing old ones."""
    new_tree, i = cls._recursive_replace_words(tree, new_words, 0)
    if i != len(new_words):
      raise ValueError('len(new_words) == %d != i == %d' % (len(new_words), i))
    return new_tree
