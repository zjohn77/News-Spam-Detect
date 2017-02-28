
# coding: utf-8

# In[113]:

'''Sample 20 stories from UCI-News dataset; for each story, sample 10% of articles on that story.''' 
import numpy as np
import pandas as pd
from os import chdir


# In[114]:

# Use pandas to input csv file into news_df:
chdir('/home/jz/proj/News-Spam-Detect/TxtClus/Input')
news_df = pd.read_csv('uci-news-aggregator.csv')


# In[115]:

def _story_sampler(storyArr, NUM_STORIES = 20):
    '''Sample 20 out of about 1000 stories, first.'''
    return np.random.choice(storyArr.unique(), NUM_STORIES, replace=False)

def _stagewise_sample(df, story_sampler, PCT = .1):
    #Isolate articles belonging to those 20 selected stories:
    articles = df[df.STORY.isin(story_sampler(df.STORY))] 
    return articles.sample(frac = PCT)    


# In[116]:

if __name__ == "__main__":
    news_sample = _stagewise_sample(news_df, _story_sampler)
    news_sample.to_csv('newsSample.csv')

