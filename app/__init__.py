
from dotenv import load_dotenv, find_dotenv
import logging

from rainbow import RainbowLogger as Logger

logger = Logger('Twitter Sentiment Analysis')
# logging.getLogger('botocore').setLevel(logging.ERROR)

labels = [
    'negative', 
    'neutral', 
    'positive'
]

try:
    load_dotenv(find_dotenv())
except:
    ...