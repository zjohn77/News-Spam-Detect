
### Buckshot++: A Scalable, Noise-Tolerant Bagging Approach to  Clustering.  
Despite the k-means algorithm being one of the most important unsupervised learning methods today, the Achilles' heel of this algorithm is the need to determine beforehand the number of clusters. A common way to find K is like this: take some metric, evaluate at multiple values of K, and let some greedy search determine when to stop (typically elbow heuristic). Here, I show a better way. I introduce 3 improvements:   
1. I show that some metrics are powerful, while others are too inconsistent to be useful
2. A big problem with k-means is that every single point is clustered--but what we really want is the pattern and not the noise. I show a beautiful way to solve this problem--and much simpler than k-medoids and k-medians to boot. 
3. Running k-means multiple times to find the best K is expensive in computation time, and just doesn't scale. I show a surprisingly simple alternative that *does* scale. 


```python
'''A Scalable, Noise-Tolerant Bagging approach to the Buckshot Algorithm.
Reference to the original buckshot algorithm (https://pdfs.semanticscholar.org/1134/3448f8a817fa391e3a7897a95f975ad2873a.pdf)
Author: John Jung'''
```


```python
from numpy.random import choice
from pandas import read_csv, DataFrame
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
import matplotlib.pyplot as plt
from functools import reduce
import os, sys
sys.path.insert(0, '/home/jz/proj/News-Spam-Detect/TxtClus')
from baggedBuckshot import Clusterings
```


```python
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
```

#### Background
The novel algorithm that I will explain here, which hereinafter I will call Buckshot++, is motivated by the [Pedersen et al.](https://pdfs.semanticscholar.org/1134/3448f8a817fa391e3a7897a95f975ad2873a.pdf) paper. The essence of the Buckshot algorithm outline in that paper is to find cluster centers by performing hierarchical clustering on a sample and then perform k-means by taking those cluster centers as input. Table 1 below explains why the Buckshot algorithm takes this approach:


```python
print('Table 1')
tbl = DataFrame({'k-means': ['O(N * k * d * i)', 'random initial means; local minimum; outlier'],
                 'hierarchical': ['O(N^2 * logN)', 'outlier']}
                , index=['Computational Complexity', 'Sources of Instability'])
tbl
```

    Table 1
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hierarchical</th>
      <th>k-means</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Computational Complexity</th>
      <td>O(N^2 * logN)</td>
      <td>O(N * k * d * i)</td>
    </tr>
    <tr>
      <th>Sources of Instability</th>
      <td>outlier</td>
      <td>random initial means; local minimum; outlier</td>
    </tr>
  </tbody>
</table>
</div>



The shockingly high time complexity of hierarchical clustering is why the Buckshot algorithm performs hierarchical only on a sample. Let's look at what Buckshot++ is and how it simultaneously tackles scalability and stability issues.  

#### The Buckshot++ Algorithm  
1. Take B bootstrap samples where each sample is of size 1/B.  
2. For each K from 2 to F do:  
        2a. Pass parameter K into kmeans computation.  
        2b. Compute the metric M.
3. Compute the centroid of all metrics vectors.  
4. Output: argmax of the centroid vector.        

#### Implementation & Empirical Discoveries
Having given the background and the concept of Buckshot++, let's see how I implemented the algorithm. In the cell below, I import the [Kaggle News Aggregator Dataset](https://www.kaggle.com/uciml/news-aggregator-dataset) that I will use to illustrate my algorithm. In this data, there's a field that contains news headlines and another field that holds an ID indicating which story a headline belongs to. 


```python
if __name__ == "__main__":
    # Pass in settings to instantiate a Clusterings object called vecSpaceMod1:
    vecSpaceMod = Clusterings({'file_loc': sys.path[0] + '/Input/newsSample.csv',
                               'tf_dampen': True,
                               'common_word_pct': 1,
                               'rare_word_pct': 1,
                               'dim_redu': False})
    news_df = vecSpaceMod.get_file() # Read CSV input to data frame.
```


```python
    # Repeatedly run kmeans on resamples, compute the silhouette and calinski metrics for each K that I plug in.  
    metrics_byK = vecSpaceMod.buckshot(news_df)        
```

#### Bagging *does* make sense here!
Each green curve is generated from a bootstrap sample, and the red curve is their average. Remember the sources of instability for k-means listed in Table 1? One instability is outlier. The concept of outlier has a somewhat different meaning in the context of clustering. In supervised learning, an outlier is a rare observation that happens to be far from other observations. In clustering, a far away observation is its own well-separated cluster. So my interpretation is that "rare" is the operative word here and that outliers are singleton clusters that exert undue influence on the formation of other clusters. Look at how [bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating) imparted a more stable estimate of the relationship between the number of clusters and the silhouette score in the graph below.


```python
    plot_mult_samples(metrics_byK, 'silhouette')
```


![png](output_12_0.png)


#### Useful and not-so-useful internal clustering metrics currently out there.
The two internal clustering metrics implemented in scikit-learn are: the Silhouette Coefficient and the Calinski-Harabasz criterion. Comparing the Silhouette plotted above with the Calinski plotted below, it's clear that Silhouette is clearly better.


```python
    plot_mult_samples(metrics_byK, 'calinski')
```


![png](output_14_0.png)



```python
    X = vecSpaceMod.term_weight_matr(news_df.TITLE)
    kmeans_fit = KMeans(20).fit(X)
```

#### Are the quality of the clusters good?
Taking a look at the documents and their corresponding "predictedCluster", the results certainly do impress!


```python
    cluster_results = DataFrame({'predictedCluster': kmeans_fit.labels_,
                                 'document': news_df.TITLE})
    cluster_results.sort_values(by='predictedCluster', inplace=True)
    cluster_results
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>document</th>
      <th>predictedCluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>Boulder's Wealth May Be A Factor For Lowest Ob...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Virginia's Governor Challenges Abortion Clinic...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>82</th>
      <td>The Least Obese City in the Country</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Fine Tuning: Good Wife just gets better</td>
      <td>0</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Zebra Technologies to Acquire Enterprise Busin...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Second Twitter exec resigns with goodbye tweet...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Ali Rowghani, Twitter's COO, resigns after mon...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Twitter COO Ali Rowghani Just Announced Via Tw...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Twitter COO Ali Rowghani Exits</td>
      <td>1</td>
    </tr>
    <tr>
      <th>57</th>
      <td>'Goodbye Twitter' COO Ali Rowghani, says bye t...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Weiner reflects on the beginning of the end of...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>37</th>
      <td>TV Review: Mad Men Season 7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>'Mad Men' Preview: Buckle Up For 7 'Dense' Epi...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Fargo (FX) Season Finale 2014 âMorton's Forkâ</td>
      <td>2</td>
    </tr>
    <tr>
      <th>65</th>
      <td>'Fargo' Season 1 Spoilers: Episode 10 Synopsis...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>70</th>
      <td>10 Things You Never Knew About 'Mad Men'!</td>
      <td>2</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Rich Sommer from AMC's 'Mad Men' Season Premiere</td>
      <td>2</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Why Fargo Works So Well as a TV Show</td>
      <td>2</td>
    </tr>
    <tr>
      <th>84</th>
      <td>'Mad Men' takes off on its final flight</td>
      <td>2</td>
    </tr>
    <tr>
      <th>53</th>
      <td>'Mad Men' Season 7 Spoilers: Everything We Kno...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Before 'Fargo's' season finale, a sequel (or p...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>46</th>
      <td>'Mad Men': Season 7 Premiere Guide (Video)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Mad Men - the (Blaxploitation) Movie</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>'Mad Men' mixology</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>'Mad Men' end in sight for Weiner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>'Mad Men': 7 things to know for Season 7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Rising carbon dioxide levels reduce nutrients ...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Nutrition in Crops Are Cut down Drastically by...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Rising CO2 Levels Will Lower Nutritional Value...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>68</th>
      <td>With carbon dioxide levels up, nutrients in cr...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>64</th>
      <td>Inflation back up: Modest rise to 1.8% in Apri...</td>
      <td>12</td>
    </tr>
    <tr>
      <th>55</th>
      <td>UK Inflation Rise To 1.8% Delays Real Wage Ris...</td>
      <td>12</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Inflation rise stalls wage hopes in the UK</td>
      <td>12</td>
    </tr>
    <tr>
      <th>44</th>
      <td>UK CPI rises to 1.8% in April, core CPI hits 2%</td>
      <td>12</td>
    </tr>
    <tr>
      <th>51</th>
      <td>BREAKING NEWS: Transport costs lead to hike in...</td>
      <td>12</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Twitter's head of media Chloe Sladden steps do...</td>
      <td>13</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Twitter's revolving door: media head Chloe Sla...</td>
      <td>13</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Twitter Exec Exodus Continues with Media Chief...</td>
      <td>13</td>
    </tr>
    <tr>
      <th>90</th>
      <td>Sony Xperia Z4 Concept Emerges as Fan Imagines...</td>
      <td>14</td>
    </tr>
    <tr>
      <th>78</th>
      <td>If you hate the word 'selfie' look away now, t...</td>
      <td>14</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Leaked: Images Of Sony's Xperia C3 'Selfie Phone'</td>
      <td>14</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Sony Xperia C3 arrives with 5MP selfie camera,...</td>
      <td>14</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Sony Xperia Z2 Encased In A Block Of Ice, Cont...</td>
      <td>14</td>
    </tr>
    <tr>
      <th>40</th>
      <td>GM June Sales Up 9 Percent, Best June Since 2007</td>
      <td>15</td>
    </tr>
    <tr>
      <th>79</th>
      <td>Despite Safety Issues, GM's Sales Still Increa...</td>
      <td>15</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Chrysler Group LLC reports June 2014 US sales ...</td>
      <td>15</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Net-a-Porter Embraces Google Glass</td>
      <td>16</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Google Glass Still Trying To Look Cool</td>
      <td>16</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Google Glass headsets get new designs in colla...</td>
      <td>16</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Google's first fashionable Glass frames are de...</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Here's How Climate Change Will Make Food Less ...</td>
      <td>17</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Why almost everything you've been told about u...</td>
      <td>17</td>
    </tr>
    <tr>
      <th>81</th>
      <td>Rising CO2 levels also make our food less nutr...</td>
      <td>17</td>
    </tr>
    <tr>
      <th>24</th>
      <td>UPDATE 5-JPMorgan profit weaker than expected ...</td>
      <td>18</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Marks and Spencer's profits fall for third year</td>
      <td>18</td>
    </tr>
    <tr>
      <th>29</th>
      <td>JPMorgan profit weaker than expected</td>
      <td>18</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Technology stocks falling for 2nd day in a row</td>
      <td>18</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Microsoft wants Windows XP dead and has announ...</td>
      <td>19</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Gov. McAuliffe Makes Health Announcements</td>
      <td>19</td>
    </tr>
    <tr>
      <th>74</th>
      <td>McAuliffe puts focus on women's health</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
<p>93 rows × 2 columns</p>
</div>



#### Internal or External Clustering Metrics?
With the ground truth labels, we can evaluate Mutual Information, which is the most commonly used external metric. Looking at 0.67 Mutual Information compared with about 0.35 Silhouette, the relationship between the metrics is fairly consistent with other studies.


```python
    mutual_info = adjusted_mutual_info_score(labels_true=news_df.STORY, labels_pred=kmeans_fit.labels_) 
    print(mutual_info)
```

    0.668634947446
    

#### Buckshot++ Scales Better Than Buckshot.
The fact is that hierarchical clustering has a much higher order of time complexity than k-means. This means that, for sizable inputs, running k-means multiple times is still faster than running hierarchical clustering just once. The Buckshot algorithm runs hierarchical just once on a small sample in order to initialize cluster centers for k-means. Since O(N^2 * logN) grows really fast, the sample must be really small to make it work computationally. But a key critique of Buckshot is [failure to find the right structure with a small sample](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.136.7906&rep=rep1&type=pdf).  

#### The Selling Points of Buckshot++
* **Extremely accurate** method of estimating the number of clusters (a clearly best Silhouette emerged every time, while hill climbing searches can hit or miss).
* **Highly scalable** (faster search for K achieved by using k-means rather than hierarchical; running k-means on subsample rather than everything).
* **Robust to noise** when used in conjunction with k-means++ (sampling with replacement lessens the chance of selecting an outlier in the bootstrap sample).
