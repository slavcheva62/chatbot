import sys
import logging

logging.basicConfig(stream=sys.stderr)
sys.path.insert(0, '/home/tretogdc/chatbot')

from chatbot_app import app as application
