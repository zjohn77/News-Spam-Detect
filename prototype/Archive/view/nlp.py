'''defines the callback function to provide to TfidfVectorizer for custom lemmatization of tokens'''
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from re import sub
from numpy.random import choice
from sklearn.feature_extraction.text import TfidfVectorizer

class LemmaTokenizer(object):
    '''Define class for pre-processing raw string into 1-gram lemmas exclusive of stop-words.'''
    def __init__(self):
        self.wnl = WordNetLemmatizer() # unpack WordNetLemmatizer into a property in this class
    def __call__(self, doc):
        lemmas = [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        return filter(lambda token: len(token) != 1, lemmas)

def letterSnippets(snippets):
    '''Strip away numbers, punctuations, unicodes, and hexadecimals from string, before tokenizing.'''
    return [sub('[^A-Za-z\s]+', ' ', snippet) for snippet in snippets] 

def doc_term_matrix(snippets, param_dict, tokenizerObj):
    '''Compute document-term matrix based on corpus.'''
    corpus = choice(snippets, param_dict['samp_size'])  # take sample    
    
    docTerm_X = TfidfVectorizer(tokenizer = tokenizerObj, 
                                   stop_words = 'english', 
                                   strip_accents = 'unicode',
                                   sublinear_tf = param_dict['tf_dampen']).fit_transform(corpus)


    
    ###  Rotate and then project the data onto eigenspace
    if param_dict.dim_redu: 
        return TruncatedSVD(n_components = docTerm_X.shape[1]/5).fit_transform(docTerm_X)
    
    return docTerm_X   