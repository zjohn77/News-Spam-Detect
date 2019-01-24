
## Buckshot++: An Outlier-Resistant and Scalable Clustering Algorithm. (Inspired by the [Buckshot Algorithm](https://pdfs.semanticscholar.org/1134/3448f8a817fa391e3a7897a95f975ad2873a.pdf).)

Here, we introduce a new algorithm, which we name Buckshot++. Buckshot++ improves upon the k-means by dealing with the main shortcoming thereof, namely, the need to predetermine the number of clusters, K. Typically, K is found in the following manner: 
1. settle on some metric,
2. evaluate that metric at multiple values of K, 
3. use a greedy stopping rule to determine when to stop (typically the bend in an elbow curve).

There must be a better way. We detail the following 3 improvements that the Buckshot++ algorithm makes to k-means.   
1. Not all metrics are create equal. And since **K-means doesn't prescribe which metric to use for finding K**, we analyzed that some of the commonly implemented metrics are too inconsistent from one iteration to the next. Buckshot++ prescribes the silhouette score for finding K.
2. **In k-means, every single point is clustered -- even the noise and outliers**. But what we really care about is the pattern and not the noise. We show here an elegant way to overcome this problem -- even simpler than [k-medoids](https://en.wikipedia.org/wiki/K-medoids) or [k-medians](https://en.wikipedia.org/wiki/K-medians_clustering). 
3. Finally, the [computational complexity](https://en.wikipedia.org/wiki/Computational_complexity) of running k-means multiple times on the whole dataset to find the best K can be prohibitive. We show below a surprisingly simple alternative with better asymptotics.

### Details of the Buckshot++ algorithm

**ALGORITHM**: Buckshot++ <br>
**INPUTS**: population of *N* vectors <br>
*B* := number of bootstrap samples <br>
*F* := max number of clusters to try <br>
*M* := cluster quality metric <br>
**OUTPUT**: the optimal *K* for kmeans

Take *B* bootstrap samples where each sample is of size 1/*B*.  
**for each** counter *k* from 2 to *F* **do** <br>
&emsp;&emsp;Compute kmeans with *k* centers. <br>
&emsp;&emsp;Compute the metric *M* on the clusters. <br>
Compute the centroid of all metrics vectors.  
Get argmax of the centroid vector.


### Explanation of Buckshot++
The Buckshot++ algorithm was motivated by the [Buckshot algorithm](https://pdfs.semanticscholar.org/1134/3448f8a817fa391e3a7897a95f975ad2873a.pdf), which essentially finds cluster centers by performing [hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) on a sample and then performing k-means by taking those cluster centers as inputs. [Hierarchical has relatively high time complexity](https://nlp.stanford.edu/IR-book/html/htmledition/time-complexity-of-hac-1.html), which is why Buckshot performs hierarchical only on a sample. The key difference between hierarchical and kmeans is that the former is more deterministic/stable but less scalable than the latter, as the next table elucidates.


```python
%matplotlib inline
import pandas as pd
pd.set_option('display.max_rows', 500)
tbl = pd.DataFrame({'k-means': ['O(N * k * d * i)', 'random initial means; local minimum; outlier'],
                    'hierarchical': ['O(N^2 * logN)', 'outlier']}
                   , index=['Computational Complexity', 'Sources of Instability'])
tbl
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k-means</th>
      <th>hierarchical</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Computational Complexity</th>
      <td>O(N * k * d * i)</td>
      <td>O(N^2 * logN)</td>
    </tr>
    <tr>
      <th>Sources of Instability</th>
      <td>random initial means; local minimum; outlier</td>
      <td>outlier</td>
    </tr>
  </tbody>
</table>
</div>



Hierarchical's higher time complexity means that, for large inputs, running k-means multiple times is still faster than running hierarchical just once. The Buckshot algorithm runs hierarchical just once on a small sample in order to initialize cluster centers for k-means. Since O(N^2 * logN) grows really fast, the sample must be really small to make it work computationally. But a key critique of Buckshot is [failure to find the right structure with a small sample](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.136.7906&rep=rep1&type=pdf).  

Buckshot++'s key innovation lies in the step "Take B [bootstrap samples](http://www.stat.rutgers.edu/home/mxie/rcpapers/bootstrap.pdf) where each sample is of size 1/B." While Buckshot is doing hierarchical on a sample, Buckshot++ is doing multiple kmeans on *bootstrap* samples. Doing kmeans many times can still finish sooner than doing hierarchical just once, as the time complexities above show. An added bonus is that bootstrapping is a great way to smooth out noise and improve stability. In fact, that is exactly why **Bagging** (a.k.a. **B**ootstrap **Agg**regat**ing**) and [Random Forests](https://en.wikipedia.org/wiki/Random_forest#From_bagging_to_random_forests) work so well.

### Python implementation of Buckshot++
The core algorithm implementation is in the [buckshotpp module](https://github.com/zjohn77/buckshotpp/tree/master/buckshotpp). We use it below to cluster a news headlines dataset.


```python
from buckshotpp import Clusterings, plot_mult_samples
from numpy.random import choice
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
import nltk; nltk.download('punkt', quiet=True)
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 120
import warnings; warnings.filterwarnings('ignore')

vecSpaceMod = Clusterings({'file_loc': 'data/news_headlines.csv',
                           'tf_dampen': True,
                           'common_word_pct': 1,
                           'rare_word_pct': 1,
                           'dim_redu': False}
                         )  # Instantiate a Clusterings object using parameters.
news_df = vecSpaceMod.get_file() # Read news_headlines.csv into a df.
metrics_byK = vecSpaceMod.buckshot(news_df)
plot_mult_samples(metrics_byK, 'silhouette')
```


![png](output_10_0.png)


### An insight from this chart
Each green curve is generated from a bootstrap sample, and the red curve is their average. Remember the sources of instability for k-means listed in the table above? Outlier is one. The concept of outlier has somewhat different meaning in the context of clustering. In supervised learning, an outlier is a rare observation that's far from other observations distance-wise. In clustering, a far away observation is its own well-separated cluster. Here, our interpretation is that "rare" is the operative word here and that outliers are singleton clusters that exert undue influence on the formation of other clusters. Look at how **[bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating) led to a more stable estimate of the optimal number of clusters** in the graph above.

### Not all metrics are create equal
The two internal clustering metrics implemented in scikit-learn are: the Silhouette Coefficient and the [Calinski-Harabasz criterion](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabaz_score.html). Comparing the Silhouette plotted above with the Calinski plotted below, it's clear that Calinski is far more extreme, perhaps implausibly extreme.


```python
plot_mult_samples(metrics_byK, 'calinski')
```


![png](output_13_0.png)


### Internal or External Clustering Metrics?
This data contains a field named "STORY" that indicates which story a headline belongs to. With this field as the ground truth, we compute Mutual Information (the most common external metric) using the code below. Mutual Information's possible range is 0-1. Using the K resulting from Buckshot++, we obtained a Mutual Information of about 0.6, an indicator that the model performance is reasonable.


```python
X = vecSpaceMod.term_weight_matr(news_df.TITLE)
kmeans_fit = KMeans(20).fit(X)  # the argument comes from inflectin point of silhouette plot
mutual_info = adjusted_mutual_info_score(labels_true=news_df.STORY, labels_pred=kmeans_fit.labels_) 
mutual_info
```




    0.6435601965984835



### Practically, does Buckshot++ produce well-separated clusters?
Taking a look at the documents and their corresponding "predictedCluster", the results certainly do seem reasonable.


```python
cluster_results = pd.DataFrame({'predictedCluster': kmeans_fit.labels_,
                                'document': news_df.TITLE})
cluster_results.sort_values(by='predictedCluster', inplace=True)

cluster_results
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predictedCluster</th>
      <th>document</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>SAC Capital Starts Anew as Point72</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0</td>
      <td>Zebra Technologies to Acquire Enterprise Busin...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>Fine Tuning: Good Wife just gets better</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>Boulder's Wealth May Be A Factor For Lowest Ob...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>Power restored to nuclear plant in Waterford, ...</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0</td>
      <td>Electricity out as Millstone shifts to diesel</td>
    </tr>
    <tr>
      <th>59</th>
      <td>1</td>
      <td>Twitter's head of media Chloe Sladden steps do...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>Twitter's revolving door: media head Chloe Sla...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>Twitter Exec Exodus Continues with Media Chief...</td>
    </tr>
    <tr>
      <th>67</th>
      <td>2</td>
      <td>Sony Xperia C3 arrives with 5MP selfie camera,...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2</td>
      <td>Leaked: Images Of Sony's Xperia C3 'Selfie Phone'</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2</td>
      <td>Sony Xperia Z2 Encased In A Block Of Ice, Cont...</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2</td>
      <td>Sony Xperia Z4 Concept Emerges as Fan Imagines...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2</td>
      <td>If you hate the word 'selfie' look away now, t...</td>
    </tr>
    <tr>
      <th>71</th>
      <td>3</td>
      <td>Twitter Executive Quits Amid Stalling Growth</td>
    </tr>
    <tr>
      <th>47</th>
      <td>3</td>
      <td>Twitter COO quits, signalling management shake-up</td>
    </tr>
    <tr>
      <th>52</th>
      <td>3</td>
      <td>Twitter Loses a Powerful Executive</td>
    </tr>
    <tr>
      <th>31</th>
      <td>3</td>
      <td>Second Twitter executive quits hours after Row...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>3</td>
      <td>Twitter COO resigns as growth lags</td>
    </tr>
    <tr>
      <th>61</th>
      <td>3</td>
      <td>Twitter COO Rowghani resigns amid lacklustre g...</td>
    </tr>
    <tr>
      <th>57</th>
      <td>4</td>
      <td>'Goodbye Twitter' COO Ali Rowghani, says bye t...</td>
    </tr>
    <tr>
      <th>69</th>
      <td>4</td>
      <td>Twitter chief operating officer resigns as use...</td>
    </tr>
    <tr>
      <th>66</th>
      <td>4</td>
      <td>UPDATE 3-Twitter chief operating officer resig...</td>
    </tr>
    <tr>
      <th>86</th>
      <td>4</td>
      <td>Twitter chief operating officer Ali Rowghani h...</td>
    </tr>
    <tr>
      <th>76</th>
      <td>4</td>
      <td>Ali Rowghani, Twitter's COO, resigns after mon...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>4</td>
      <td>Twitter COO Ali Rowghani Just Announced Via Tw...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4</td>
      <td>Twitter COO Ali Rowghani Exits</td>
    </tr>
    <tr>
      <th>35</th>
      <td>4</td>
      <td>Second Twitter exec resigns with goodbye tweet...</td>
    </tr>
    <tr>
      <th>39</th>
      <td>5</td>
      <td>Why almost everything you've been told about u...</td>
    </tr>
    <tr>
      <th>77</th>
      <td>5</td>
      <td>Why Fargo Works So Well as a TV Show</td>
    </tr>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>'Mad Men' Preview: Buckle Up For 7 'Dense' Epi...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>'Mad Men' end in sight for Weiner</td>
    </tr>
    <tr>
      <th>36</th>
      <td>6</td>
      <td>Weiner reflects on the beginning of the end of...</td>
    </tr>
    <tr>
      <th>42</th>
      <td>7</td>
      <td>Giant mystery crater in Siberia has scientists...</td>
    </tr>
    <tr>
      <th>85</th>
      <td>7</td>
      <td>Mysterious giant crater in the earth discovere...</td>
    </tr>
    <tr>
      <th>60</th>
      <td>7</td>
      <td>Massive Crater Discovered in Siberia</td>
    </tr>
    <tr>
      <th>92</th>
      <td>7</td>
      <td>Massive mystery crater at 'end of the world'</td>
    </tr>
    <tr>
      <th>16</th>
      <td>7</td>
      <td>Mysterious crater in Siberia spawns wild Inter...</td>
    </tr>
    <tr>
      <th>43</th>
      <td>8</td>
      <td>Inflation rise stalls wage hopes in the UK</td>
    </tr>
    <tr>
      <th>82</th>
      <td>8</td>
      <td>The Least Obese City in the Country</td>
    </tr>
    <tr>
      <th>19</th>
      <td>8</td>
      <td>Real wages could resume fall as "Easter effect...</td>
    </tr>
    <tr>
      <th>55</th>
      <td>8</td>
      <td>UK Inflation Rise To 1.8% Delays Real Wage Ris...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>8</td>
      <td>Virginia's Governor Challenges Abortion Clinic...</td>
    </tr>
    <tr>
      <th>51</th>
      <td>8</td>
      <td>BREAKING NEWS: Transport costs lead to hike in...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>Cable prices climb 4 times faster than inflati...</td>
    </tr>
    <tr>
      <th>79</th>
      <td>9</td>
      <td>Despite Safety Issues, GM's Sales Still Increa...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>9</td>
      <td>Chrysler Group LLC reports June 2014 US sales ...</td>
    </tr>
    <tr>
      <th>40</th>
      <td>9</td>
      <td>GM June Sales Up 9 Percent, Best June Since 2007</td>
    </tr>
    <tr>
      <th>87</th>
      <td>9</td>
      <td>Ford sales fall, GM barely even; Jeep powers C...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>10</td>
      <td>Gov. McAuliffe Makes Health Announcements</td>
    </tr>
    <tr>
      <th>48</th>
      <td>10</td>
      <td>Microsoft wants Windows XP dead and has announ...</td>
    </tr>
    <tr>
      <th>74</th>
      <td>10</td>
      <td>McAuliffe puts focus on women's health</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>Sony makes duckfacing official with Xperia C3,...</td>
    </tr>
    <tr>
      <th>54</th>
      <td>11</td>
      <td>Sony to announce 'Selfie' phone on July 8th wi...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>11</td>
      <td>Sony prepares to launch a smartphone that has ...</td>
    </tr>
    <tr>
      <th>91</th>
      <td>11</td>
      <td>Sony Xperia C3 launches as "world's best selfi...</td>
    </tr>
    <tr>
      <th>88</th>
      <td>11</td>
      <td>Sony unveils Xperia C3 smartphone with LED fla...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>Sony Xperia C3 Boasts 5MP "PROselfie" Front-fa...</td>
    </tr>
    <tr>
      <th>44</th>
      <td>12</td>
      <td>UK CPI rises to 1.8% in April, core CPI hits 2%</td>
    </tr>
    <tr>
      <th>75</th>
      <td>12</td>
      <td>Rising CO2 Levels Will Lower Nutritional Value...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12</td>
      <td>Here's How Climate Change Will Make Food Less ...</td>
    </tr>
    <tr>
      <th>81</th>
      <td>12</td>
      <td>Rising CO2 levels also make our food less nutr...</td>
    </tr>
    <tr>
      <th>80</th>
      <td>13</td>
      <td>Nutrition in Crops Are Cut down Drastically by...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
      <td>Rising carbon dioxide levels reduce nutrients ...</td>
    </tr>
    <tr>
      <th>68</th>
      <td>13</td>
      <td>With carbon dioxide levels up, nutrients in cr...</td>
    </tr>
    <tr>
      <th>64</th>
      <td>14</td>
      <td>Inflation back up: Modest rise to 1.8% in Apri...</td>
    </tr>
    <tr>
      <th>83</th>
      <td>14</td>
      <td>US plants prepare for long-term nuclear waste ...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>14</td>
      <td>Nuclear Plant Operators Deal With Radioactive ...</td>
    </tr>
    <tr>
      <th>32</th>
      <td>14</td>
      <td>US plants prepare long-term nuclear waste stor...</td>
    </tr>
    <tr>
      <th>84</th>
      <td>15</td>
      <td>'Mad Men' takes off on its final flight</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>'Mad Men' mixology</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15</td>
      <td>'Mad Men': 7 things to know for Season 7</td>
    </tr>
    <tr>
      <th>9</th>
      <td>15</td>
      <td>Mad Men - the (Blaxploitation) Movie</td>
    </tr>
    <tr>
      <th>37</th>
      <td>15</td>
      <td>TV Review: Mad Men Season 7</td>
    </tr>
    <tr>
      <th>46</th>
      <td>15</td>
      <td>'Mad Men': Season 7 Premiere Guide (Video)</td>
    </tr>
    <tr>
      <th>70</th>
      <td>15</td>
      <td>10 Things You Never Knew About 'Mad Men'!</td>
    </tr>
    <tr>
      <th>53</th>
      <td>15</td>
      <td>'Mad Men' Season 7 Spoilers: Everything We Kno...</td>
    </tr>
    <tr>
      <th>72</th>
      <td>15</td>
      <td>Rich Sommer from AMC's 'Mad Men' Season Premiere</td>
    </tr>
    <tr>
      <th>63</th>
      <td>16</td>
      <td>Fargo (FX) Season Finale 2014 âMorton's Forkâ</td>
    </tr>
    <tr>
      <th>56</th>
      <td>16</td>
      <td>Before 'Fargo's' season finale, a sequel (or p...</td>
    </tr>
    <tr>
      <th>65</th>
      <td>16</td>
      <td>'Fargo' Season 1 Spoilers: Episode 10 Synopsis...</td>
    </tr>
    <tr>
      <th>62</th>
      <td>17</td>
      <td>Google Glass headsets get new designs in colla...</td>
    </tr>
    <tr>
      <th>41</th>
      <td>17</td>
      <td>Google's first fashionable Glass frames are de...</td>
    </tr>
    <tr>
      <th>89</th>
      <td>17</td>
      <td>Google Glass Still Trying To Look Cool</td>
    </tr>
    <tr>
      <th>34</th>
      <td>17</td>
      <td>Net-a-Porter Embraces Google Glass</td>
    </tr>
    <tr>
      <th>15</th>
      <td>18</td>
      <td>Routine pelvic exams not recommended under new...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>18</td>
      <td>Doctors group nixes routine pelvic exams</td>
    </tr>
    <tr>
      <th>38</th>
      <td>18</td>
      <td>Metro Detroit doctors wary of recommendation a...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>18</td>
      <td>Doctors against having frequent pelvic exams</td>
    </tr>
    <tr>
      <th>58</th>
      <td>19</td>
      <td>Technology stocks falling for 2nd day in a row</td>
    </tr>
    <tr>
      <th>24</th>
      <td>19</td>
      <td>UPDATE 5-JPMorgan profit weaker than expected ...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>19</td>
      <td>JPMorgan profit weaker than expected</td>
    </tr>
    <tr>
      <th>33</th>
      <td>19</td>
      <td>Marks and Spencer's profits fall for third year</td>
    </tr>
  </tbody>
</table>
</div>



### Summary of the key advantages of Buckshot++
* **Accurate** method of estimating the number of clusters (a clearly best Silhouette emerged every time, while typical elbow heuristic searches can hit or miss).
* **Scalable** (faster search for K achieved by using k-means rather than hierarchical; running k-means on subsample rather than everything).
* **Noise resistant** when used in conjunction with k-means++ (sampling with replacement lessens the chance of selecting an outlier in the bootstrap sample).
