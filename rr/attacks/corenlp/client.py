"""A client for a CoreNLP Server."""
import json
import os
import requests

from rr.attacks.corenlp.server import CoreNLPServer

class CoreNLPClient(object):
  """A client that interacts with the CoreNLPServer."""
  def __init__(self, hostname='http://localhost', port=7000,
               start_server=False, server_flags=None, server_log=None,
               cache_file=None,):
    """Create the client.

    Args:
      hostname: hostname of server.
      port: port of server.
      start_server: start the server on first cache miss.
      server_flags: passed to CoreNLPServer.__init__()
      server_log: passed to CoreNLPServer.__init__()
      cache_file: load and save cache to this file.
    """
    self.hostname = hostname
    self.port = port
    self.start_server = start_server
    self.server_flags = server_flags
    self.server_log = server_log
    self.server = None
    self.cache_file = cache_file
    self.has_cache_misses = False
    if cache_file:
      if os.path.exists(cache_file):
        with open(cache_file) as f:
          self.cache = json.load(f)
      else:
        self.cache = {}
    else:
      self.cache = None

  def save_cache(self):
    if self.cache_file and self.has_cache_misses:
      with open(self.cache_file, 'w') as f:
        json.dump(self.cache, f)
    self.has_cache_misses = False

  def query(self, sents, properties):
    """Most general way to query the server.
    
    Args:
      sents: Either a string or a list of strings.
      properties: CoreNLP properties to send as part of the request.
    """
    url = '%s:%d' % (self.hostname, self.port)
    params = {'properties': str(properties)}
    if isinstance(sents, list):
      data = '\n'.join(sents)
    else:
      data = sents
    key = '%s\t%s' % (data, str(properties))
    if self.cache and key in self.cache:
      return self.cache[key]
    self.has_cache_misses = True
    if self.start_server and not self.server:
      self.server = CoreNLPServer(port=self.port, flags=self.server_flags,
                                  logfile=self.server_log)
      self.server.start()
    r = requests.post(url, params=params, data=data.encode('utf-8'))
    r.encoding = 'utf-8'
    json_response = json.loads(r.text, strict=False)
    if self.cache is not None:
      self.cache[key] = json_response
    return json_response

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    if self.server:
      self.server.stop()
    if self.cache_file:
      self.save_cache()

  def query_pos(self, sents):
    """Standard query for getting POS tags."""
    properties = {
        'ssplit.newlineIsSentenceBreak': 'always',
        'annotators': 'tokenize,ssplit,pos',
        'outputFormat':'json'
    }
    return self.query(sents, properties)

  def query_ner(self, paragraphs):
    """Standard query for getting NERs on raw paragraphs."""
    annotators = 'tokenize,ssplit,pos,ner,entitymentions'
    properties = {
        'ssplit.newlineIsSentenceBreak': 'always',
        'annotators': annotators,
        'outputFormat':'json'
    }
    return self.query(paragraphs, properties)

  def query_depparse_ptb(self, sents, use_sd=False):
    """Standard query for getting dependency parses on PTB-tokenized input."""
    annotators = 'tokenize,ssplit,pos,depparse'
    properties = {
        'tokenize.whitespace': True,
        'ssplit.eolonly': True,
        'ssplit.newlineIsSentenceBreak': 'always',
        'annotators': annotators,
        'outputFormat':'json'
    }
    if use_sd:
      # Use Stanford Dependencies trained on PTB
      # Default is Universal Dependencies
      properties['depparse.model'] = 'edu/stanford/nlp/models/parser/nndep/english_SD.gz'
    return self.query(sents, properties)

  def query_depparse(self, sents, use_sd=False, add_ner=False):
    """Standard query for getting dependency parses on raw sentences."""
    annotators = 'tokenize,ssplit,pos,depparse'
    if add_ner:
      annotators += ',ner'
    properties = {
        'ssplit.eolonly': True,
        'ssplit.newlineIsSentenceBreak': 'always',
        'annotators': annotators,
        'outputFormat':'json'
    }
    if use_sd:
      # Use Stanford Dependencies trained on PTB
      # Default is Universal Dependencies
      properties['depparse.model'] = 'edu/stanford/nlp/models/parser/nndep/english_SD.gz'
    return self.query(sents, properties)

  def query_const_parse(self, sents, add_ner=False):
    """Standard query for getting constituency parses on raw sentences."""
    annotators = 'tokenize,ssplit,pos,parse'
    if add_ner:
      annotators += ',ner'
    properties = {
        'ssplit.eolonly': True,
        'ssplit.newlineIsSentenceBreak': 'always',
        'annotators': annotators,
        'outputFormat':'json'
    }
    return self.query(sents, properties)
