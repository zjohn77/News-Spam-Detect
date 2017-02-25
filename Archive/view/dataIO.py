'''Controls the ingestion, language processing, and creation of the doc-term matrix based on the text input.'''
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from os import chdir
import re
import sys
import csv
import pandas as pd
import numpy as np
#from lemma import LemmaTokenizer
chdir('C:/Users/jjung/Documents/GitHub/')
#sys.path.insert(0,'/bkmark_organizer/test_parser_stemmer/prototype/view/')

maxInt = sys.maxsize
decrement = True
while decrement:
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

# 1. READ TEXT FILE LINE BY LINE INTO LIST
#with open('data.txt') as f:
#	snippets = f.read().splitlines()

# 1. Read a 415k row CSV line by line into np.array with each element being a list.
with open('newsCorpora.csv'
          , 'r'
          , encoding='utf8') as f:
    newsList = list(csv.reader(f))
    
# 2. Sample 500 observations. 
newsSample = np.random.choice(newsList, 500)

# 3. Tokenize each line of tab-delimited string into a list of strings.  
def splitLine(lineOfStr, delimiter="\t"):
    values = lineOfStr.split(delimiter)
    return values
ndArr = [splitLine(x[0]) for x in newsSample]

# 4. Convert this 2d list to a pd.DataFrame for easier columnar access.        
cols = ['ID','TITLE','URL','PUBLISHER','CATEGORY','STORY','HOSTNAME','TIMESTAMP']
df = pd.DataFrame(ndArr, columns=cols)

# 5. Backup sample to csv.
df.to_csv('./bkmark_organizer/test_parser_stemmer/prototype/view/newsSample.csv', index=False) 