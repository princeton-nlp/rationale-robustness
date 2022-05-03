"""Run a CoreNLP Server."""
import atexit
import errno
import os
import socket
import subprocess
import sys
import time

LIB_PATH = '/n/fs/nlp-jh70/stanford-corenlp-4.2.0/*'
DEVNULL = open(os.devnull, 'wb')

class CoreNLPServer(object):
  """An object that runs the CoreNLP server."""
  def __init__(self, port=7000, lib_path=LIB_PATH, flags=None, logfile=None):
    """Create the CoreNLPServer object.

    Args:
      port: Port on which to serve requests.
      flags: If provided, pass this list of additional flags to the java server.
      logfile: If provided, log stderr to this file.
      lib_path: The path to the CoreNLP *.jar files.
    """
    self.port = port
    self.lib_path = LIB_PATH
    self.process = None
    self.p_stderr = None
    if flags:
      self.flags = flags
    else:
      self.flags = []
    if logfile:
      self.logfd = open(logfile, 'wb')
    else:
      self.logfd = DEVNULL

  def start(self, flags=None):
    """Start up the server on a separate process."""
    print('Using lib directory %s' % self.lib_path)
    if not flags:
      flags = self.flags
    p = subprocess.Popen(
        ['java', '-mx4g', '-cp', self.lib_path,
         'edu.stanford.nlp.pipeline.StanfordCoreNLPServer',
         '--port', str(self.port)] + flags,
        stderr=self.logfd, stdout=self.logfd)
    self.process = p
    atexit.register(self.stop)  

    # Keep trying to connect until the server is up
    s = socket.socket()
    while True:
      time.sleep(1)
      try:
        s.connect(('127.0.0.1', self.port))
      except socket.error as e:
        if e.errno != errno.ECONNREFUSED:
          # Something other than Connection refused means server is running
          break
    s.close()


  def stop(self):
    """Stop running the server on a separate process."""
    if self.process:
      self.process.terminate()
    if self.logfd != DEVNULL:
      self.logfd.close()

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, type, value, traceback):
    self.stop()
