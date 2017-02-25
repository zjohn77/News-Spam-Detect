# In[6]:
import sys
sys.path.append('C:/Users/jjung/Documents/GitHub//bkmark_organizer/test_parser_stemmer/prototype/view/')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabaz_score
import numpy as np
from numpy.linalg import norm
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import re
from lemma import LemmaTokenizer
#import jqmcvi.basec as jqmcvi
import jqmcvi.base as measure

### 1. Read csv to Pandas DataFrame
newsDf = pd.read_csv('C:/Users/jjung/Documents/GitHub/bkmark_organizer/test_parser_stemmer/prototype/view/newsSample.csv'
                    , encoding = 'latin1')
# In[9]:
### 2. Strip away numbers, punctuations, unicodes, and hexadecimals from string, before lemmatizing/tokenizing
snippets =  newsDf.TITLE
alphanumSnippets = [re.sub('[^A-Za-z\s]+', ' ', snippet) for snippet in snippets] 
# In[13]:


### 3A. Sample from 500 observations to create 3 different datasets. 
corpus1 = np.random.choice(alphanumSnippets, 25)
corpus2 = np.random.choice(alphanumSnippets, 75)
corpus3 = np.random.choice(alphanumSnippets, 125)

### 3B. Produce term-document matrix after 1-gram tokenizing, lemmatizing(stemming), and filtering English stop-words.
##  RUN 1
docTerm_X1 = TfidfVectorizer(tokenizer = LemmaTokenizer(), 
                                   stop_words = 'english', 
                                   strip_accents = 'unicode',
                                   sublinear_tf = True).fit_transform(corpus1) # term_doc_matrix1 is a Sparse Matrix object    

#  RUN 2
docTerm_X2 = TfidfVectorizer(tokenizer = LemmaTokenizer(), 
                                   stop_words = 'english', 
                                   strip_accents = 'unicode',
                                   max_df = 0.7,
                                   min_df = 0.05).fit_transform(corpus2) 
#  RUN 3
vectorizer = TfidfVectorizer(tokenizer = LemmaTokenizer(), 
                                   stop_words = 'english', 
                                   strip_accents = 'unicode',
                                   sublinear_tf = True)
docTerm_X3 = vectorizer.fit_transform(corpus3) 
#terms = vectorizer.get_feature_names() # 525 unique terms (i.e. dimensions) for 125 docs

### 4A. See that the 1st 100 Principal Components capture roughly 97% of the 
###    variance of the document-term matrix.
svdObj = TruncatedSVD(n_components=100, n_iter=7)
svdObj.fit(docTerm_X3) 
print(svdObj.explained_variance_ratio_.sum())  

### 4B. Rotate and then project the data onto eigenspace
docTerm_E3 = svdObj.transform(docTerm_X3) 

### 5. Check that TfidfVectorizer, by default, normalized each doc vector 
###    so that it's inscribed in the unit hypersphere.
docTerm_X1_arr = docTerm_X1.toarray()  # (25, 118) matrix
norm(docTerm_X1_arr, axis=1)

### 6. Cluster the data.
kmeansFit1 = KMeans(n_clusters=2).fit(docTerm_X1_arr)
kmeansFit1.labels_
kmeansFit1.cluster_centers_.shape


# Calc and store to list the average silhouette score for KMeans on this number of clusters.
avg_silhouette = silhouette_score(docTerm_X1_arr, labels = kmeansFit1.labels_)     
avg_calinski = calinski_harabaz_score(docTerm_X1_arr, labels = kmeansFit1.labels_)                            
avg_dunn = measure.dunn_fast(docTerm_X1_arr, labels = kmeansFit1.labels_) 

def reached_elbow(metric_cur, metric_prev, k_cur, k_prev):
    return (metric_cur - metric_prev) / (k_cur - k_prev) < metric_cur / k_cur

def _2toSeq(n):
    return 2 ** np.arange(n)    
        
np.floor(np.log2(len(X)))


def best_model(clusterFcn, clusterQual, stopCondition):
    
    
                           


#3. For each dataset, compute the Elbow by evaluating with 4 different internal metrics to find the best K.

