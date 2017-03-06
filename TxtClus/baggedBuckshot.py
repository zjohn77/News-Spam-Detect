from pandas import read_csv, DataFrame
from nlp.termWeighting import doc_term_matrix
from EstimateK.seqFit import sensitiv

class Clusterings(object):
    '''Define a class that encapsulates textual processing tools.'''
    def __init__(self, param_dict):
        self.__param_dict = param_dict        
    
    def get_file(self):
        '''read csv input into a pandas data frame'''
        return read_csv(self.__param_dict['file_loc'], encoding = 'latin1')

    def __term_weight_matr(self, snippetsArr):
        '''compute a document-term matrix based on a collection of text documents'''
        return doc_term_matrix(snippetsArr, self.__param_dict)
    
    def __resample(self, df, NUM_BOOTSTRAPS = 3):
        '''take 3 bootstrap sub-samples for faster, bagged kmeans fits'''        
        bootstraps = [None] * NUM_BOOTSTRAPS
        for bootI in range(NUM_BOOTSTRAPS):  
            bootstraps[bootI] = df.sample(frac=1/NUM_BOOTSTRAPS, replace=True)
        return bootstraps
    
    def buckshot(self, df):
        '''Public controller method that computes bagged kmeans fits across metrics & K.'''
        bstraps = self.__resample(df)
        # Calc the TF-IDF matrix based on headlines; then fits kmeans across different K:    
        return [sensitiv(self.__term_weight_matr(bstrap.TITLE)) for bstrap in bstraps]      