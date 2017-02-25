from pandas import read_csv
import os
# from collections import defaultdict
import sys
sys.path.append('C:/Users/john/Documents/GitHub//bkmark_organizer/test_parser_stemmer/prototype/TxtClus/')
from nlp.termWeighting import doc_term_matrix
from EstimateK.seqFit import sensitiv

### 1. Input a csv and a dictionary that contains the settings for Run1
### create the document-term matrix for Run1, and call it X1.

# news_df = read_csv('C:/Users/John/Documents/GitHub/bkmark_organizer/test_parser_stemmer/prototype/Input/newsSample.csv'
#                     , encoding = 'latin1')

def text_groupings(param_dict):	
	news_df = read_csv(param_dict.file_location, encoding = 'latin1')
	X = doc_term_matrix(news_df.TITLE, param_dict).toarray()
	sensitiv(X)

if __name__ == "__main__":
    news_file = sys.path[0] + 'Input/newsSample.csv'
	text_groupings({'run': 1,
				  'file_location': news_file,
	              'samp_size': 25,
	              'tf_dampen': True,
	              'common_word_pct': 1,
	              'rare_word_pct': 1,
	              'dim_redu': False})
	# text_groupings({'run': 1,
	# 			  'file_location': 'C:/Users/John/Documents/GitHub/bkmark_organizer/test_parser_stemmer/prototype/Input/newsSample.csv',
	#               'samp_size': 25,
	#               'tf_dampen': True,
	#               'common_word_pct': 1,
	#               'rare_word_pct': 1,
	#               'dim_redu': False})