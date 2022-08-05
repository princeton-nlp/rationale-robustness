import json

from corenlp.client import CoreNLPClient
from corenlp.server import CoreNLPServer

def main():
  """Start a REPL to interact with the server."""
  # TODO: actuall make it a REPL
  c = CoreNLPClient()
  with CoreNLPServer() as s:
    print json.dumps(c.query_depparse(['Bills on ports and immigration were submitted by Senator Brownback , Republican of Kansas']), indent=2)

if __name__ == '__main__':
  main()
