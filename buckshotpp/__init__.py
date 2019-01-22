from .nlp.termWeighting import doc_term_matrix
from .EstimateK.seqFit import sensitiv
from functools import reduce
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame

def plot_mult_samples(listOf_df, row_name):     
    '''For each bootstrap iteration, plot metrics across K''' 
    for i in range(len(listOf_df)):
        plt.plot(list(listOf_df[i].columns), list(listOf_df[i].loc[row_name,:]), c="green")  
    
    # Use the reduce function to do elementwise average for several data frames:
    avg_metrics = reduce(lambda df1, df2: df1.add(df2), listOf_df).div(len(listOf_df))        
    plt.plot(list(avg_metrics.columns), list(avg_metrics.loc[row_name,:]), c="red")    
    plt.ylabel(row_name + ' score')   
    plt.xlabel('no. of clusters')        
    plt.show()      

class Clusterings(object):
    '''Define a class that encapsulates textual processing tools.'''
    def __init__(self, param_dict):
        self.__param_dict = param_dict        
    
    def get_file(self):
        '''read csv input into a pandas data frame'''
        return read_csv(self.__param_dict['file_loc'], encoding = 'latin1')

    def term_weight_matr(self, snippetsArr):
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
        return [sensitiv(self.term_weight_matr(bstrap.TITLE)) for bstrap in bstraps]      