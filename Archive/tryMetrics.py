
# coding: utf-8

# In[52]:

from pandas import read_csv, DataFrame
import os
# from collections import defaultdict
import sys
sys.path.insert(0, 'C:/Users/jjung/Documents/GitHub//bkmark_organizer/test_parser_stemmer/prototype/TxtClus/')
from nlp.termWeighting import doc_term_matrix
from EstimateK.seqFit import sensitiv


# In[53]:

def text_groupings(param_dict):
    news_df = read_csv(param_dict['file_loc'], encoding = 'latin1')
    X = doc_term_matrix(news_df.TITLE, param_dict).toarray()
    sensitiv(X)


# In[54]:

if __name__ == "__main__":
    news_file = sys.path[0] + 'Input/newsSample.csv'
    text_groupings({'run': 1,
                    'file_loc': news_file,
                    'samp_size': 25,
                    'tf_dampen': True,
                    'common_word_pct': 1,
                    'rare_word_pct': 1,
                    'dim_redu': False})  

    KMeans(k).fit(X).labels_     


