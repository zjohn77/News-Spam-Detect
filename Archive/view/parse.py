import pandas as pd
import re
from os import chdir
chdir('C:/Users/john/Documents/GitHub/')

#
newsDf = pd.read_csv('/bkmark_organizer/test_parser_stemmer/prototype/view/newsSample.csv')

##snippets =  
##
### 2. Strip away numbers, punctuations, unicodes, and hexadecimals from string, before lemmatizing/tokenizing
##alphanumSnippets = [re.sub('[^A-Za-z\s]+', ' ', snippet) for snippet in snippets] 
##
### 3. Produce term-document matrix after 1-gram tokenizing, lemmatizing(stemming), and filtering English stop-words.
##Vectorizer = TfidfVectorizer(tokenizer = LemmaTokenizer(), stop_words = 'english', strip_accents = 'unicode')   
##doc_term_matrix = Vectorizer.fit_transform(alphanumSnippets) # term_doc_matrix is a Sparse Matrix object    
##terms = Vectorizer.get_feature_names()
##pprint(terms)
##
##
##dtMatr_arr = doc_term_matrix.toarray()
##print(dtMatr_arr.shape)
