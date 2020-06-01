# Unsupervised Learning 

Say you have a collection of customers with a variety of characteristics such as age, location, and financial history, and you wish to discover patterns and sort them into clusters. Or perhaps you have a set of texts, such as wikipedia pages, and you wish to segment them into categories based on their content. This is the world of unsupervised learning, called as such because you are not guiding, or supervising, the pattern discovery by some prediction task, but instead uncovering hidden structure from unlabeled data. Unsupervised learning encompasses a variety of techniques in machine learning, from clustering to dimension reduction to matrix factorization. In this course, you'll learn the fundamentals of unsupervised learning and implement the essential algorithms using scikit-learn and scipy. You will learn how to cluster, transform, visualize, and extract insights from unlabeled datasets, and end the course by building a recommender system to recommend popular musical artists.

Unsupervised Learning is a class of machine learning techniques for discovering patterns in data. For instance, finding the natural clusters of customers based on their purchases histories or searching for patterns and correlations among these purchases, and using these patterns to express the data in a compressed form. These are examples of unsupervised learning techniques called clustering and dimension reduction. Unsupervised learning is learning without labels. It is pure pattern discovery, unguided by a prediction task. 

Rows are dimensions. We can’t visualize more than two dimensions directly, but using unsupervised learning techniques we can still gain insights. 

K-means finds a specified number of clusters in the samples. 

In iris dataset, 3 clusters are defined since there are 3 species of flowers. 

Centroids are the mean of the samples in each cluster. New samples are assigned to the cluster whose centroid is closest. 



```python
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
```


```python
file_path = ('/Users/MuhammadBilal/Desktop/Data Camp/Unsupervised learning in Python/Data/Grains/seeds.csv')
```


```python
df = pd.read_csv(file_path, header=None)
df.head(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>15.26</td>
      <td>14.84</td>
      <td>0.8710</td>
      <td>5.763</td>
      <td>3.312</td>
      <td>2.221</td>
      <td>5.220</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>14.88</td>
      <td>14.57</td>
      <td>0.8811</td>
      <td>5.554</td>
      <td>3.333</td>
      <td>1.018</td>
      <td>4.956</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>14.29</td>
      <td>14.09</td>
      <td>0.9050</td>
      <td>5.291</td>
      <td>3.337</td>
      <td>2.699</td>
      <td>4.825</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>13.84</td>
      <td>13.94</td>
      <td>0.8955</td>
      <td>5.324</td>
      <td>3.379</td>
      <td>2.259</td>
      <td>4.805</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = df.iloc[:,[0,1,2,3,4,5,6]].values
```

###### Evaluating your clusters: 
    
A direct approach is to compare the clusters with the data. We can measure the quality of a clustering in a way that doesn’t require our samples to come pre-grouped into species. This measure of quality can then be used to make an informed choice about the number of clusters to look for. 
Cross tabulations help us to determine the quality of clusters. The table shows number of values of each variable in a cluster. Same variable can be present in different clusters. Cross tabulations provide insights into which sort of samples are in which cluster. 
A good clustering has tight clusters, meaning that the samples in each cluster are bunched together, not spread out. How spread out the samples within each cluster are can be measured by the inertia. Inertia measures how far samples are from their centroids. Lower inertia is better. K-means clustering creates the clusters in a way that minimized the inertia. 
A good clustering has tight clusters (meaning low inertia) but it also doesn’t have too many clusters. A good method is to choose elbow in the inertia plot. That is a point where the interia begins to decrease more slowly. 



```python
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fitting model to samples
    model.fit(data)
    
    # Appending the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plotting ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

```


![output_8_0](https://user-images.githubusercontent.com/49030506/83438758-faf44a80-a40f-11ea-9aae-7063623660b5.png)


The inertia decreases very slowly from 3 clusters to 4, so it looks like 3 clusters would be a good choice for this data.

###### Evaluating the grain clustering

It is observed above from the inertia plot that 3 is a good number of clusters for the grain data. In fact, the grain samples come from a mix of 3 different grain varieties: "Kama", "Rosa" and "Canadian". Now, I will cluster the grain samples into three clusters, and compare the clusters to the grain varieties using a cross-tabulation.


```python
# Creating a list of wheat varieties
```


```python
varieties = ['Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat']
```


```python
# Creating a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Using fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(data)

# Creating a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Creating crosstab: ct
ct = ct = pd.crosstab(df['labels'], df['varieties'])

# Displaying ct
print(ct)

```

    varieties  Canadian wheat  Kama wheat  Rosa wheat
    labels                                           
    0                      68           9           0
    1                       0           1          60
    2                       2          60          10


The cross-tabulation shows that the 3 varieties of grain separate really well into 3 clusters. But depending on the type of data we are working with, the clustering may not always be this good. Is there anything we can do in such situations to improve the clustering. Let's find out. 


###### Transforming variables for better clustering

Sometimes KMeans clustering doesn’t work well on the data and doesn’t create distinct clusters. Its because some features in the data have very different variance. The variance of a feature measures the spread of its values. In KMeans clustering, the variance of a feature corresponds to its influence on the clustering algorithm. To give every feature a chance, the data needs to be transformed so that features have equal variance. This can be achieved with the StandardScaler from scikit-learn. It transforms every feature to have mean 0 and variance 1. The resulting standardized features can be very informative. StandardScaler is a preprocessing step, there are others available as well like MaxAbsScaler and Normalizer. 



```python
Scaling fish data for clustering

Here I will work with the dataset on samples giving measurements of fish. Each row represents an individual fish. The measurements, such as weight in grams, length in centimeters, and the percentage ratio of height to length, have very different scales. In order to cluster this data effectively, I'll need to standardize these features first. I'll build a pipeline to standardize and cluster the data.
```


```python
# Importing the data set
df1 = pd.read_csv('fish.csv',header = None)
```


```python
df1.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Bream</td>
      <td>242.0</td>
      <td>23.2</td>
      <td>25.4</td>
      <td>30.0</td>
      <td>38.4</td>
      <td>13.4</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Bream</td>
      <td>290.0</td>
      <td>24.0</td>
      <td>26.3</td>
      <td>31.2</td>
      <td>40.0</td>
      <td>13.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Selecting the features and converting them into array for ML modeling
data1 = df1.iloc[:,[1,2,3,4,5,6]].values
```


```python
# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)

```

###### Clustering the fish data

I'll now use your standardization and clustering pipeline from above to cluster the fish by their measurements, and then create a cross-tabulation to compare the cluster labels with the fish species.

As before, data1 is the 2D array of fish measurements. Pipeline is available as pipeline, and the species of every fish sample is given by the list species.


```python
species = ['Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Bream',
 'Roach',
 'Roach',
 'Roach',
 'Roach',
 'Roach',
 'Roach',
 'Roach',
 'Roach',
 'Roach',
 'Roach',
 'Roach',
 'Roach',
 'Roach',
 'Roach',
 'Roach',
 'Roach',
 'Roach',
 'Roach',
 'Roach',
 'Roach',
 'Smelt',
 'Smelt',
 'Smelt',
 'Smelt',
 'Smelt',
 'Smelt',
 'Smelt',
 'Smelt',
 'Smelt',
 'Smelt',
 'Smelt',
 'Smelt',
 'Smelt',
 'Smelt',
 'Pike',
 'Pike',
 'Pike',
 'Pike',
 'Pike',
 'Pike',
 'Pike',
 'Pike',
 'Pike',
 'Pike',
 'Pike',
 'Pike',
 'Pike',
 'Pike',
 'Pike',
 'Pike',
 'Pike']
```


```python
# Importing pandas
import pandas as pd

# Fitting the pipeline to samples
pipeline.fit(data1)

# Calculating the cluster labels: labels
labels = pipeline.predict(data1)

# Creating a DataFrame with labels and species as columns: df
df1 = pd.DataFrame({'labels':labels, 'species':species})

# Creating crosstab: ct
ct = pd.crosstab(df1['labels'], df1['species'])

# Displaying ct
print(ct)

```

    species  Bream  Pike  Roach  Smelt
    labels                            
    0           33     0      1      0
    1            1     0     19      1
    2            0     0      0     13
    3            0    17      0      0


It looks like the fish data separates really well into 4 clusters

###### Clustering stocks using KMeans

Here I'll cluster companies using their daily stock price movements (i.e. the dollar difference between the closing and opening prices for each trading day). I have given a NumPy array movements of daily price movements from 2010 to 2015 (obtained from Yahoo! Finance), where each row corresponds to a company, and each column corresponds to a trading day.

Some stocks are more expensive than others. To account for this, I am including a Normalizer at the beginning of the pipeline. The Normalizer will separately transform each company's stock price to a relative scale before the clustering begins.

Note that Normalizer() is different to StandardScaler(), which I used above. While StandardScaler() standardizes features (such as the features of the fish data from the previous exercise) by removing the mean and scaling to unit variance, Normalizer() rescales each sample - here, each company's stock price works independently of the other.


```python
df2 = pd.read_csv('company-stock-movements-2010-2015-incl.csv')
df2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 60 entries, 0 to 59
    Columns: 964 entries, Unnamed: 0 to 2013-10-29
    dtypes: float64(963), object(1)
    memory usage: 452.0+ KB



```python
# Selecting all the columns 
data = df2.iloc[:,np.r_[:,1:963]].values
```

df2 had colnames but when it is converted into numpy 'data' through iloc it disregards the colnames and 'data' can directly be used for ML models. 


```python
data.shape
```




    (60, 962)




```python
# Importing Normalizer
from sklearn.preprocessing import Normalizer

# Creating a normalizer: normalizer
normalizer = Normalizer()

# Creating a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Making a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fitting pipeline to the daily price movements
pipeline.fit(data)

```




    Pipeline(memory=None,
             steps=[('normalizer', Normalizer(copy=True, norm='l2')),
                    ('kmeans',
                     KMeans(algorithm='auto', copy_x=True, init='k-means++',
                            max_iter=300, n_clusters=10, n_init=10, n_jobs=None,
                            precompute_distances='auto', random_state=None,
                            tol=0.0001, verbose=0))],
             verbose=False)



###### Which stocks move together?

Above I clustered companies by their daily stock price movements. So which company have stock prices that tend to change in the same way? I'll now inspect the cluster labels from the clustering to find out.

The solution to the above code has already been run. I constructed a pipeline containing a KMeans model and fit it to the NumPy array movements of daily stock movements. In addition, I created a list companies of the company names.


```python
companies = ['Apple',
 'AIG',
 'Amazon',
 'American express',
 'Boeing',
 'Bank of America',
 'British American Tobacco',
 'Canon',
 'Caterpillar',
 'Colgate-Palmolive',
 'ConocoPhillips',
 'Cisco',
 'Chevron',
 'DuPont de Nemours',
 'Dell',
 'Ford',
 'General Electrics',
 'Google/Alphabet',
 'Goldman Sachs',
 'GlaxoSmithKline',
 'Home Depot',
 'Honda',
 'HP',
 'IBM',
 'Intel',
 'Johnson & Johnson',
 'JPMorgan Chase',
 'Kimberly-Clark',
 'Coca Cola',
 'Lookheed Martin',
 'MasterCard',
 'McDonalds',
 '3M',
 'Microsoft',
 'Mitsubishi',
 'Navistar',
 'Northrop Grumman',
 'Novartis',
 'Pepsi',
 'Pfizer',
 'Procter Gamble',
 'Philip Morris',
 'Royal Dutch Shell',
 'SAP',
 'Schlumberger',
 'Sony',
 'Sanofi-Aventis',
 'Symantec',
 'Toyota',
 'Total',
 'Taiwan Semiconductor Manufacturing',
 'Texas instruments',
 'Unilever',
 'Valero Energy',
 'Walgreen',
 'Wells Fargo',
 'Wal-Mart',
 'Exxon',
 'Xerox',
 'Yahoo']
```


```python
# Importing pandas
import pandas as pd

# Predicting the cluster labels: labels
labels = pipeline.predict(data)

# Creating a DataFrame aligning labels and companies: df2
df2 = pd.DataFrame({'labels': labels, 'companies': companies})

# Displaying df2 sorted by cluster label
print(df2.sort_values('labels'))

```

        labels                           companies
    0        0                               Apple
    2        0                              Amazon
    17       0                     Google/Alphabet
    35       1                            Navistar
    32       1                                  3M
    30       1                          MasterCard
    13       1                   DuPont de Nemours
    8        1                         Caterpillar
    59       1                               Yahoo
    56       2                            Wal-Mart
    31       2                           McDonalds
    41       2                       Philip Morris
    28       2                           Coca Cola
    38       2                               Pepsi
    55       3                         Wells Fargo
    26       3                      JPMorgan Chase
    3        3                    American express
    16       3                   General Electrics
    5        3                     Bank of America
    18       3                       Goldman Sachs
    1        3                                 AIG
    34       4                          Mitsubishi
    7        4                               Canon
    58       4                               Xerox
    21       4                               Honda
    15       4                                Ford
    45       4                                Sony
    48       4                              Toyota
    22       4                                  HP
    40       5                      Procter Gamble
    9        5                   Colgate-Palmolive
    27       5                      Kimberly-Clark
    33       6                           Microsoft
    24       6                               Intel
    11       6                               Cisco
    51       6                   Texas instruments
    50       6  Taiwan Semiconductor Manufacturing
    57       7                               Exxon
    10       7                      ConocoPhillips
    53       7                       Valero Energy
    52       7                            Unilever
    49       7                               Total
    12       7                             Chevron
    46       7                      Sanofi-Aventis
    6        7            British American Tobacco
    43       7                                 SAP
    42       7                   Royal Dutch Shell
    19       7                     GlaxoSmithKline
    39       7                              Pfizer
    20       7                          Home Depot
    37       7                            Novartis
    23       7                                 IBM
    25       7                   Johnson & Johnson
    44       7                        Schlumberger
    4        8                              Boeing
    54       8                            Walgreen
    36       8                    Northrop Grumman
    29       8                     Lookheed Martin
    47       9                            Symantec
    14       9                                Dell


###### Visualizing Hierarchies: 
    
There are two unsupervised visualization techniques t-SNE and hierarchical clustering. T-SNE creates a 2d map of any dataset and conveys useful information about the proximity of the samples to one another. 

The example of hierarchical clustering can be organized into small narrow groups, like humans, apes, snakes and lizards, or into larger broader groups like mammals and reptiles, or even broader groups like animals and plants. These groups are contained in one another and form a hierarchy. Analogously, hierarchical clustering arranges samples into a hierarchy of clusters. Hierarchical clustering can organize any sort of data into a hierarchy, not just samples of plants and animals. 

Lets consider a type of dataset describing how countries scored performances at the Eurovision 2016 song contest. The data is arranged in a rectangular array, where the rows of the array show how many points a country gave to each song. The samples in this case are the countries. The result of applying hierarchical clustering to the Eurovision scores can be visualized as a tree like diagram called a dendrogram. The single picture reveals a great deal of information about the voting behavior of countries at the Eurovision.   The dendrogram groups the countries into larger and larger clusters, any many of these clusters are immediately recognizable as containing countries that are close to one another geographically, or that have close cultural or political ties, or that belong to single language group. Hierarchical clustering proceeds in steps. In the beginning every country is its own cluster, so there are as many clusters as there are countries. At each step, the two closest clusters are merged. This decreases the number of clusters and eventually there is only one cluster left, and it contains all the countries. This process is actually a particular type of hierarchical clustering called agglomerative clustering. There is also divisive clustering, which works the other way around.  The entire process of the Hierarchical clustering is encoded in the dendrogram. At the bottom each country is in a cluster of its own. The clustering then proceeds from the bottom up. Clusters are represented as vertical lines and a joining of vertical lines indicates a merging of clusters. This process continues until there is only one cluster left and it contains all the countries. 



```python
# Importing data for hierarchical clustering 
file_path1 = ('/Users/MuhammadBilal/Desktop/Data Camp/Unsupervised learning in Python/Data/Grains/seeds_samples.csv')
```


```python
df2 = pd.read_csv(file_path1)
```


```python
samples = df2.iloc[0:43,np.r_[:,0:7]].values
```


```python
samples.shape
```




    (42, 7)




```python
varieties = ['Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Kama wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Rosa wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat',
 'Canadian wheat']
```

###### Hierarchical clustering of the grain data

SciPy linkage() function performs hierarchical clustering on an array of samples. I will use the linkage() function to obtain a hierarchical clustering of the grain samples, and use dendrogram() to visualize the result. I have created a sample of the grain measurements in the array samples, while the variety of each grain sample is given by the list varieties.


```python
# Performing the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Calculating the linkage: mergings
mergings = linkage(samples, method='complete')

# Plotting the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()

```


![output_41_0](https://user-images.githubusercontent.com/49030506/83438761-faf44a80-a40f-11ea-8df5-15b5b22b8ba2.png)


Dendrograms are a great way to illustrate the arrangement of the clusters produced by hierarchical clustering.

If the hierarchical clustering were stopped at height 6 on the dendrogram, there would be 3 clusters. 

As intermediate clustering of the grain samples at height 6 has 3 clusters. Now, using the fcluster() function to extract the cluster labels for this intermediate clustering, and comparing the labels with the grain varieties using a cross-tabulation.



```python
# Performing the necessary imports
import pandas as pd
from scipy.cluster.hierarchy import fcluster

# Using fcluster to extract labels: labels
labels = fcluster(mergings, 6, criterion='distance')

# Creating a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Creating crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Displaying ct
print(ct)

```

    varieties  Canadian wheat  Kama wheat  Rosa wheat
    labels                                           
    1                      14           3           0
    2                       0           0          14
    3                       0          11           0


###### Hierarchies of stocks

Above I used k-means clustering to cluster companies according to their stock price movements. Now, I'll perform hierarchical clustering of the companies. I have created a NumPy array of price movements, where the rows correspond to companies, and a list of the company names companies is already created above. SciPy hierarchical clustering doesn't fit into a sklearn pipeline, so I'll need to use the normalize() function from sklearn.preprocessing instead of Normalizer.

linkage and dendrogram have already been imported from scipy.cluster.hierarchy, and PyPlot has been imported as plt.


```python
df2 = pd.read_csv('company-stock-movements-2010-2015-incl.csv')
df2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 60 entries, 0 to 59
    Columns: 964 entries, Unnamed: 0 to 2013-10-29
    dtypes: float64(963), object(1)
    memory usage: 452.0+ KB



```python
# Selecting all the columns of the dataset df2
movements = df2.iloc[:,np.r_[:,1:964]].values
```


```python
movements.shape
```




    (60, 963)




```python
# Importing normalize
from sklearn.preprocessing import normalize

# Normalizing the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculating the linkage: mergings
mergings = linkage(normalized_movements, 'complete')

# Plotting the dendrogram
dendrogram(mergings, 
          labels=companies, 
          leaf_rotation=90,
          leaf_font_size=6)
plt.show()

```


![output_50_0](https://user-images.githubusercontent.com/49030506/83438762-faf44a80-a40f-11ea-88d6-88f7e495c3fe.png)


We can produce great visualizations such as this with hierarchical clustering, but it can be used for more than just visualizations.

###### Cluster labels in hierarchical clustering

Hierarchical clustering creates a visualization of the voting behavior at the Eurovision. We can also extract the clusters from intermediate stages of a hierarchical clustering. The cluster labels for these intermediate clustering can then be used in further computations, such as cross tabulations, just like the cluster labels from k-means. An intermediate stage in the hierarchical clustering is specified by choosing a height on the dendrogram. The y-axis of the dendrogram encodes the distance between merging clusters. The height that specifies an intermediate clustering corresponds to a distance. The distance between two clusters is measured using a linkage method. In the complete linkage the distance between the two clusters is the maximum distance between their samples. Different linkage methods give different hierarchical clustering. 

The cluster labels for any intermediate stage of the hierarchical clustering can be extracted using the fcluster function. 



```python
file_path2 = ('/Users/MuhammadBilal/Desktop/Data Camp/Unsupervised learning in Python/Data/ESC2017_GF.csv')
```


```python
samples = pd.read_csv(file_path2)
```


```python
Samples = samples.iloc[0:26:,np.r_[:,3:45]].values
```


```python
Samples.shape
```




    (26, 42)




```python
Samples
```




    array([[14, 20, 17, ..., 19, 10, 14],
           [20, 17, 24, ..., 14, 10, 13],
           [ 0,  3, 10, ...,  8, 20, 22],
           ...,
           [ 0,  0,  1, ...,  0,  7,  0],
           [ 0,  0,  0, ...,  0,  0,  0],
           [ 0,  0,  0, ...,  0,  0,  0]])



Each row in 'Samples' corresponds to a voting country, and each column corresponds to a performance that was voted for.


```python
country_names = ['Albania',
 'Armenia',
 'Australia',
 'Austria',
 'Azerbaijan',
 'Belarus',
 'Belgium',
 'Bosnia & Herzegovina',
 'Bulgaria',
 'Croatia',
 'Cyprus',
 'Czech Republic',
 'Denmark',
 'Estonia',
 'F.Y.R. Macedonia',
 'Finland',
 'France',
 'Georgia',
 'Germany',
 'Greece',
 'Hungary',
 'Iceland',
 'Ireland',
 'Israel',
 'Italy',
 'Latvia',
 'Lithuania',
 'Malta',
 'Moldova',
 'Montenegro',
 'Norway',
 'Poland',
 'Russia',
 'San Marino',
 'Serbia',
 'Slovenia',
 'Spain',
 'Sweden',
 'Switzerland',
 'The Netherlands',
 'Ukraine',
 'United Kingdom']
```


```python
The list countries gives the name of each voting country. This dataset was obtained from Eurovision (https://eurovision.tv/history/full-split-results).
```


```python
countries = ['Portugal',
'Bulgaria',
'Moldova',
'Belgium',
'Sweden',
'Italy',
'Romania',
'Hungary',
'Australia',
'Norway',
'Netherlands',
'France',
'Croatia',
'Azerbaijan',
'United Kingdom',
'Austria',
'Belarus',
'Armenia',
'Greece',
'Denmark',
'Cyprus',
'Poland',
'Israel',
'Ukraine',
'Germany',
'Spain']
```


```python
# Performing the necessary imports
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Calculating the linkage: mergings
mergings = linkage(Samples, method='single')

# Plotting the dendrogram
dendrogram(mergings,
           labels=countries,
           leaf_rotation=90,
           leaf_font_size=6)
plt.show()

```


![output_62_0](https://user-images.githubusercontent.com/49030506/83438763-faf44a80-a40f-11ea-93d2-ac819a9757ba.png)



```python
from scipy.cluster.hierarchy import linkage
mergings = linkage(Samples, method='complete')
from scipy.cluster.hierarchy import fcluster 
labels = fcluster(mergings, 15, criterion='distance')
print(labels)

# To inspect the cluster labels, using a DataFrame to align the lables with the country names.
pairs = pd.DataFrame({'labels': labels, 'countries': countries})
print(pairs.sort_values('labels'))

```

    [ 1  2  6  5  3 24  4 23 13 15 14  9 22  7 16 21 10  8 11 17 12 20 19 18
     18 18]
        labels       countries
    0        1        Portugal
    1        2        Bulgaria
    4        3          Sweden
    6        4         Romania
    3        5         Belgium
    2        6         Moldova
    13       7      Azerbaijan
    17       8         Armenia
    11       9          France
    16      10         Belarus
    18      11          Greece
    20      12          Cyprus
    8       13       Australia
    10      14     Netherlands
    9       15          Norway
    14      16  United Kingdom
    19      17         Denmark
    23      18         Ukraine
    25      18           Spain
    24      18         Germany
    22      19          Israel
    21      20          Poland
    15      21         Austria
    12      22         Croatia
    7       23         Hungary
    5       24           Italy


The linkage method defines how the distance between clusters is measured. In complete linkage, the distance between clusters is the distance between the furthest points of the clusters. In single linkage, the distance between clusters is the distance between the closest points of the clusters.

Below, hierarchical clustering of the voting countries at the Eurovision song contest is performed using 'complete' linkage.


```python
# Performing the necessary imports
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Calculating the linkage: mergings
mergings = linkage(Samples, method='complete')

# Plotting the dendrogram
dendrogram(mergings,
           labels=countries,
           leaf_rotation=90,
           leaf_font_size=6)
plt.show()

```


![output_66_0](https://user-images.githubusercontent.com/49030506/83438764-fb8ce100-a40f-11ea-902d-88105c679004.png)



```python
from scipy.cluster.hierarchy import linkage
mergings = linkage(Samples, method='complete')
from scipy.cluster.hierarchy import fcluster 
labels = fcluster(mergings, 15, criterion='distance')
print(labels)

# To inspect the cluster labels, using a DataFrame to align the lables with the country names.
pairs = pd.DataFrame({'labels': labels, 'countries': countries})
print(pairs.sort_values('labels'))

```

    [ 1  2  6  5  3 24  4 23 13 15 14  9 22  7 16 21 10  8 11 17 12 20 19 18
     18 18]
        labels       countries
    0        1        Portugal
    1        2        Bulgaria
    4        3          Sweden
    6        4         Romania
    3        5         Belgium
    2        6         Moldova
    13       7      Azerbaijan
    17       8         Armenia
    11       9          France
    16      10         Belarus
    18      11          Greece
    20      12          Cyprus
    8       13       Australia
    10      14     Netherlands
    9       15          Norway
    14      16  United Kingdom
    19      17         Denmark
    23      18         Ukraine
    25      18           Spain
    24      18         Germany
    22      19          Israel
    21      20          Poland
    15      21         Austria
    12      22         Croatia
    7       23         Hungary
    5       24           Italy


Above I worked on the fundamentals of k-Means and agglomerative hierarchical clustering. Now I will work on t-SNE, which is a powerful tool for visualizing high dimensional data.

###### t-SNE for 2-dimensional maps: 
    
t-SNE is an unsupervised learning method for visualization called t-SNE. t-SNE stands for t-distributed stochastic neighbor embedding. It maps samples from their high dimensional space into a 2 or 3 dimensional space so they can be visualized. t-SNE does a great job of approximately representing the distances between the samples. For this reason, t-SNE s an invaluable visual aid for understanding a dataset. 

The iris samples are in four dimensional space, where each dimension where each dimension corresponds to one of the four iris measurements, such as petal length and petal width. If we apply t-SNE on the iris dataset and give it only the measurements of the iris samples and not given any information about the three species of iris. If we color the species differently on the scatter plot, we will see that t-SNE will keep the species separate. But we will see that the samples of versicolor and virginica are closer in space. So it could happen that the iris dataset appears to have two clusters instead of three. This is compatible with the previous example using k-means where we saw  that clustering with 2 clusters also had relatively low inertia, meaning tight clusters.

t-SNE has fit transform method and the learning rate. However, t-SNE only has a fit_transform method. Hence, t-SNE simultaneously fits the model and transforms the data. t-SNE does not have separate fit and transform methods. This means that we can’t extend a t-SNE map to include new samples. Instead we have to start over each time. The second thing to notice is the learning rate. The learning rate makes t-SNE more complicated than some other techniques. We may need to try different learning rates for different datasets. It gets clear when we make a bad choice because all the samples appear to be bunched together in the scatter plot. Normally it’s enough to try a few values between 50 and 200. A final thing to be aware of is that the axes of a t-SNE plot do not have any interpretable meaning. In fact, they are different every time t-SNE is applied, even of the same dataset. However,  if the orientation of the plot is different each time, the clusters represented get the same position relative to one another. 


###### t-SNE visualization of grain dataset

I'll apply t-SNE to the grain samples data and inspect the resulting t-SNE features using a scatter plot. I have created an array samples of grain samples and a list variety_numbers giving the variety number of each grain sample.



```python
samples = df.iloc[:,[0,1,2,3,4,5,6]].values
```


```python
samples.shape
```




    (210, 7)




```python
variety_numbers = [1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 2,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3,
 3]

```


```python
# Importing TSNE
from sklearn.manifold import TSNE

# Creating a TSNE instance: model
model = TSNE(learning_rate=200)

# Applying fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples)

# Selecting the 0th feature: xs
xs = tsne_features[:,0]

# Selecting the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys, c=variety_numbers)
plt.show()

```


![output_74_0](https://user-images.githubusercontent.com/49030506/83438768-fb8ce100-a40f-11ea-8cca-0a790e06f0b3.png)


As we can see, the t-SNE visualization manages to separate the 3 varieties of grain samples. But how will it perform on the stock data? We'll find out next. 


###### A t-SNE map of the stock market

t-SNE provides great visualizations when the individual samples can be labeled. Here I'll apply t-SNE to the company stock price data. A scatter plot of the resulting t-SNE features, labeled by the company names, gives a map of the stock market! The stock price movements for each company are available as the array normalized_movements (as it was already created above). The list companies gives the name of each company. 



```python
# Importing TSNE
from sklearn.manifold import TSNE

# Creating a TSNE instance: model
model = TSNE(learning_rate=50)

# Applying fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Selecting the 0th feature: xs
xs = tsne_features[:,0]

# Selecting the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs, ys, alpha=0.5)

# Annotating the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()

```


![output_77_0](https://user-images.githubusercontent.com/49030506/83438770-fb8ce100-a40f-11ea-837d-009d6d0ba665.png)


It's visualizations such as this that make t-SNE such a powerful tool for extracting quick insights from high dimensional data.

###### Decorrelating your data and dimension reduction

Dimension reduction summarizes a dataset using its common occuring patterns. Here I will use fundamental of dimension reduction techniques, "Principal Component Analysis" ("PCA"). PCA is often used before supervised learning to improve model performance and generalization. It can also be useful for unsupervised learning. For example, I'll employ a variant of PCA that will allow to cluster Wikipedia articles by their content.



###### Correlated data in nature

I will work on the grains dataset giving the width and length of samples of grain. To confirm any correlation between length and width, I will make a scatter plot of width vs length and measure their Pearson correlation.


```python
fp3 = '/Users/MuhammadBilal/Desktop/Data Camp/Unsupervised learning in Python/Data/Grains/seeds-width-vs-length.csv'
```


```python
grains = pd.read_csv(fp3, header=None)
```


```python
grains.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.312</td>
      <td>5.763</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.333</td>
      <td>5.554</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.337</td>
      <td>5.291</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3.379</td>
      <td>5.324</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3.562</td>
      <td>5.658</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Performing the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Assigning the 0th column of grains: width
width = grains.iloc[:,0]

# Assigning the 1st column of grains: length
length = grains.iloc[:,1]

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculating the Pearson correlation
correlation, pvalue = pearsonr(width, length)

# Displaying the correlation
print(correlation)
```


![output_84_0](https://user-images.githubusercontent.com/49030506/83438771-fb8ce100-a40f-11ea-91cd-e7edef48be4d.png)


    0.8604149377143466


The width and length of the grain samples are highly correlated.


```python
# Importing PCA
from sklearn.decomposition import PCA

# Creating PCA instance: model
model = PCA()

# Applying the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assigning 0th column of pca_features: xs
xs = pca_features[:,0]

# Assigning 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculating the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Displaying the correlation
print(correlation)
```


![output_86_0](https://user-images.githubusercontent.com/49030506/83438772-fc257780-a40f-11ea-9172-5bc3b279d729.png)


    1.0408340855860843e-16


I've successfully decorrelated the grain measurements with PCA.

###### Intrinsic dimension: 

The intrinsic dimension of a dataset is the number of features required to approximate a dataset. The intrinsic dimension informs dimension reduction, because it tells us how much a dataset can be compressed. Through scatter plot we can identify closely related features and eliminate some features without losing information. Scatter plot works only on max 3 dimensional data. intrinsic dimensions can be identified even if there are more than 3 features. If the data has more than 3 dimensions this is where PCA comes into play. The intrinsic dimension can be identified by counting the PCA features that have high variance. 

The intrinsic dimension is the number of features that have significant variance. 


###### The first principal component

The first principal component of the data is the direction in which the data varies the most. Here I will use PCA to find the first principal component of the length and width measurements of the grain samples, and represent it as an arrow on the scatter plot.

The array grains gives the length and width of the grain samples. 


```python
# Making a scatter plot of the untransformed points
plt.scatter(grains.iloc[:,0], grains.iloc[:,1])

# Creating a PCA instance: model
model = PCA()

# Fitting model to points
model.fit(grains)

# Getting the mean of the grain samples: mean
mean = model.mean_

# Getting the first principal component: first_pc
first_pc = model.components_[0,:]

# Plotting first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keeping axes on same scale
plt.axis('equal')
plt.show()
```


![output_90_0](https://user-images.githubusercontent.com/49030506/83438773-fc257780-a40f-11ea-876e-e4ddf3da11e1.png)


Red line shows the direction in which the grain data varies the most.

###### Variance of the PCA features

The fish dataset is 6-dimensional. But what is its intrinsic dimension? I will make a plot of the variances of the PCA features to find out. data1 is a 2D array, where each row represents a fish. I'll need to standardize the features first.


```python
# Performing the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Creating scaler: scaler
scaler = StandardScaler()

# Creating a PCA instance: pca
pca = PCA()

# Creating pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Fitting the pipeline to 'samples'
pipeline.fit(data1)

# Plotting the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

```


![output_93_0](https://user-images.githubusercontent.com/49030506/83438774-fc257780-a40f-11ea-8582-c106b28a2f23.png)


Since PCA features 0 and 1 have significant variance, the intrinsic dimension of this dataset appears to be 2.

###### Dimension reduction with PCA

Dimension reduction represents the same data using less features and is vital for building machine learning pipelines using real-world data. We have seen that PCA features are in decreasing order of variance. PCA performs dimension reduction by discarding the PCA features with lower variance which it assumes to be noise, and retaining the higher variance PCA feature which it assumes to be informative. To use PCA for dimension reduction, one needs to specify how many PCA features to keep. For example, specifying n_components=2 when creating a PCA model tells it to keep only the first tow PCA features. A good choice is the intrinsic dimension of the dataset. PCA takes features with highest variance. In some cases an alternative implementation of PCA needs to be used. Word frequency arrays are a great example. In a word frequency array, each row corresponds to a document, and each column corresponds to a word from a fixed vocabulary.  The entries of the word-frequency array measure how often each word appears in each document. Only some of the words from the vocabulary appear in any one document, so most entries of the word frequency array are zero. Arrays like this are said to be sparse and are often represented using a special type of array called a csr_matrix. Csr_matrix save space by remembering only the non-zero entries of the array. Scikit-learn’s PCA doesn’t support csr_matrics and we will need to use TruncatedSVD instead. 


###### Dimension reduction of the fish measurements

Now I will use PCA for dimensionality reduction of the fish measurements (df1), retaining only the 2 most important components.

Firstly I will scale the fish measurements(df1).


```python
# Importing scale
from sklearn.preprocessing import scale
data1 = scale(data1)
data1.shape
```




    (85, 6)




```python
# Importing PCA
from sklearn.decomposition import PCA

# Creating a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fitting the PCA instance to the scaled samples
pca.fit(data1)

# Transforming the scaled samples: pca_features
pca_features = pca.transform(data1)

# Printing the shape of pca_features
print(pca_features.shape)

```

    (85, 2)



```python
I have reduced the dimenstions from 6 to 2. 
```

###### A tf-idf word-frequency array

Now I'll create a tf-idf word frequency array for a toy collection of documents. For this, I will use the TfidfVectorizer from sklearn. It transforms a list of documents into a word frequency array, which it outputs as a csr_matrix. It has fit() and transform() methods like other sklearn objects.

I will create a list documents of toy documents about pets. 


```python
documents = ['cats say meow', 'dogs say woof', 'dogs chase cats']
```


```python
# Importing TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Creating a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer()

# Applying fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Printing result of toarray() method
print(csr_mat.toarray())

# Getting the words: words
words = tfidf.get_feature_names()

# Printing words
print(words)

```

    [[0.51785612 0.         0.         0.68091856 0.51785612 0.        ]
     [0.         0.         0.51785612 0.         0.51785612 0.68091856]
     [0.51785612 0.68091856 0.51785612 0.         0.         0.        ]]
    ['cats', 'chase', 'dogs', 'meow', 'say', 'woof']


###### I'll now move to clustering Wikipedia articles!


```python
fp2 = '/Users/MuhammadBilal/Desktop/Data Camp/Unsupervised learning in Python/Data/Wikipedia articles/wikipedia-vectors.csv'
```


```python
df2 = pd.read_csv(fp2, index_col=0)
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HTTP 404</th>
      <th>Alexa Internet</th>
      <th>Internet Explorer</th>
      <th>HTTP cookie</th>
      <th>Google Search</th>
      <th>Tumblr</th>
      <th>Hypertext Transfer Protocol</th>
      <th>Social search</th>
      <th>Firefox</th>
      <th>LinkedIn</th>
      <th>...</th>
      <th>Chad Kroeger</th>
      <th>Nate Ruess</th>
      <th>The Wanted</th>
      <th>Stevie Nicks</th>
      <th>Arctic Monkeys</th>
      <th>Black Sabbath</th>
      <th>Skrillex</th>
      <th>Red Hot Chili Peppers</th>
      <th>Sepsis</th>
      <th>Adam Levine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.008878</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.049502</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00611</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.0</td>
      <td>0.029607</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.005646</td>
      <td>0.00000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 60 columns</p>
</div>




```python
Each row represents frequency of words in articles. Entries in the columns measure the frequency of the words in each article using tf-idf. tf is the frequency of the word. If 10 percent of the words in a document are science, then the tf of science will be 0.1. idf is the weighing scheme that reduces the influence of frequent words like 'the'. Below I am grouping the articles together based on the similarity of words in them. 
```


```python
from scipy.sparse import csr_matrix
articles = csr_matrix(df2.transpose())
```


```python
articles.shape
```




    (60, 13125)



The reason for taking this transpose is that without it, there would be 13,000 columns (corresponding to the 13,000 words in the file), which is a lot of columns for a CSV to have.


```python
articles
```




    <60x13125 sparse matrix of type '<class 'numpy.float64'>'
    	with 42091 stored elements in Compressed Sparse Row format>




```python
titles = ['HTTP 404',
 'Alexa Internet',
 'Internet Explorer',
 'HTTP cookie',
 'Google Search',
 'Tumblr',
 'Hypertext Transfer Protocol',
 'Social search',
 'Firefox',
 'LinkedIn',
 'Global warming',
 'Nationally Appropriate Mitigation Action',
 'Nigel Lawson',
 'Connie Hedegaard',
 'Climate change',
 'Kyoto Protocol',
 '350.org',
 'Greenhouse gas emissions by the United States',
 '2010 United Nations Climate Change Conference',
 '2007 United Nations Climate Change Conference',
 'Angelina Jolie',
 'Michael Fassbender',
 'Denzel Washington',
 'Catherine Zeta-Jones',
 'Jessica Biel',
 'Russell Crowe',
 'Mila Kunis',
 'Dakota Fanning',
 'Anne Hathaway',
 'Jennifer Aniston',
 'France national football team',
 'Cristiano Ronaldo',
 'Arsenal F.C.',
 'Radamel Falcao',
 'Zlatan Ibrahimović',
 'Colombia national football team',
 '2014 FIFA World Cup qualification',
 'Football',
 'Neymar',
 'Franck Ribéry',
 'Tonsillitis',
 'Hepatitis B',
 'Doxycycline',
 'Leukemia',
 'Gout',
 'Hepatitis C',
 'Prednisone',
 'Fever',
 'Gabapentin',
 'Lymphoma',
 'Chad Kroeger',
 'Nate Ruess',
 'The Wanted',
 'Stevie Nicks',
 'Arctic Monkeys',
 'Black Sabbath',
 'Skrillex',
 'Red Hot Chili Peppers',
 'Sepsis',
 'Adam Levine']
```

###### Clustering Wikipedia part I

TruncatedSVD is able to perform PCA on sparse arrays in csr_matrix format, such as word-frequency arrays. Here I will combine the knowledge of TruncatedSVD and k-means to cluster some popular pages from Wikipedia. I will build a pipeline and apply it to the word-frequency array of some Wikipedia articles.

The Wikipedia dataset I will be working with was obtained from here: https://blog.lateral.io/2015/06/the-unknown-perils-of-mining-wikipedia/



```python
# Performing the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Creating a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Creating a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Creating a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)

```

###### Clustering Wikipedia part II

Here I will put pipeline to work! The array articles of tf-idf word-frequencies of some popular Wikipedia articles is available, and a list titles of their titles. I will use the pipeline to cluster the Wikipedia articles.



```python
# Importing pandas
import pandas as pd

# Fitting the pipeline to articles
pipeline.fit(articles)

# Calculating the cluster labels: labels
labels = pipeline.predict(articles)

# Creating a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Displaying df sorted by cluster label
print(df.sort_values(['label']))

```

        label                                        article
    0       0                                       HTTP 404
    8       0                                        Firefox
    7       0                                  Social search
    6       0                    Hypertext Transfer Protocol
    5       0                                         Tumblr
    9       0                                       LinkedIn
    3       0                                    HTTP cookie
    2       0                              Internet Explorer
    1       0                                 Alexa Internet
    4       0                                  Google Search
    21      1                             Michael Fassbender
    28      1                                  Anne Hathaway
    27      1                                 Dakota Fanning
    26      1                                     Mila Kunis
    25      1                                  Russell Crowe
    24      1                                   Jessica Biel
    23      1                           Catherine Zeta-Jones
    22      1                              Denzel Washington
    20      1                                 Angelina Jolie
    29      1                               Jennifer Aniston
    59      2                                    Adam Levine
    50      2                                   Chad Kroeger
    51      2                                     Nate Ruess
    52      2                                     The Wanted
    53      2                                   Stevie Nicks
    58      2                                         Sepsis
    55      2                                  Black Sabbath
    56      2                                       Skrillex
    57      2                          Red Hot Chili Peppers
    54      2                                 Arctic Monkeys
    41      3                                    Hepatitis B
    49      3                                       Lymphoma
    48      3                                     Gabapentin
    47      3                                          Fever
    46      3                                     Prednisone
    45      3                                    Hepatitis C
    44      3                                           Gout
    43      3                                       Leukemia
    42      3                                    Doxycycline
    40      3                                    Tonsillitis
    18      4  2010 United Nations Climate Change Conference
    10      4                                 Global warming
    17      4  Greenhouse gas emissions by the United States
    16      4                                        350.org
    15      4                                 Kyoto Protocol
    14      4                                 Climate change
    13      4                               Connie Hedegaard
    12      4                                   Nigel Lawson
    11      4       Nationally Appropriate Mitigation Action
    19      4  2007 United Nations Climate Change Conference
    38      5                                         Neymar
    31      5                              Cristiano Ronaldo
    32      5                                   Arsenal F.C.
    33      5                                 Radamel Falcao
    34      5                             Zlatan Ibrahimović
    35      5                Colombia national football team
    36      5              2014 FIFA World Cup qualification
    37      5                                       Football
    30      5                  France national football team
    39      5                                  Franck Ribéry


Looking at the cluster labels patterns can be identified. 

###### Discovering interpretable features

Here I will work on a dimension reduction technique called "Non-negative matrix factorization" ("NMF") that expresses samples as combinations of interpretable parts. For example, it expresses documents as combinations of topics, and images in terms of commonly occurring visual patterns. I will also use NMF to build recommender systems that can find similar articles to read, or musical artists that match ones listening history!

###### Non-negative matrix factorization (NMF) 

NMF is a dimension reduction technique. In contrast to PCA, NMF models are interpretable. Which means that NMF models are easier to interpret and much easier to explain to others. NMF cannot be applied to every dataset. It is required that the sample features be non-negative, so greater than or equal to 0. NMF achieves its interpretability by decomposing samples as sums of their parts. For example, NMF decomposes documents as combinations of common themes and images as combinations of common themes. NMF is available in scikit learn, and follows the same fit transform pattern as PCA. However, unlike PCA, the desired number of components must always be specified. NMF works with numpy arrays and sparse arrays in the csr_matrix format. 
tf is the frequency of the word in the document. So if 10% of the words in the document are “datacamp”, then the tf of datacamp for that document is 0.1. idf is a weighting scheme that reduces the influence  of frequent words like “the”. The entries of the NMF components are always non-negative. The NMF feature values are non-negative as well. 


###### NMF applied to Wikipedia articles

I will apply NMF using the tf-idf word-frequency array of Wikipedia articles, given as a csr matrix articles. Here, I will fit the model and transform the articles. Next I'll explore the result.


```python
# Importing NMF
from sklearn.decomposition import NMF

# Creating an NMF instance: model
model = NMF(n_components=6 )

# Fitting the model to articles
model.fit(articles)

# Transforming the articles: nmf_features
nmf_features = model.transform(articles)

# Printing the NMF features
print(nmf_features)

```

These NMF features don't make much sense at this point, but I will explore them next. 


```python
# Importing pandas
import pandas as pd

# Creating a pandas DataFrame: df
df = pd.DataFrame(nmf_features, index=titles)

# Printing the row for 'Anne Hathaway'
print(df.loc['Anne Hathaway'])

# Printing the row for 'Denzel Washington'
print(df.loc['Denzel Washington'])

```

    0    0.003845
    1    0.000000
    2    0.000000
    3    0.575675
    4    0.000000
    5    0.000000
    Name: Anne Hathaway, dtype: float64
    0    0.000000
    1    0.005601
    2    0.000000
    3    0.422354
    4    0.000000
    5    0.000000
    Name: Denzel Washington, dtype: float64


Notice that for both actors, the NMF feature 3 has by far the highest value. This means that both articles are reconstructed using mainly the 3rd NMF component. Now I'll see why: NMF components represent topics (for instance, acting!).


###### NMF learns interpretable parts

Components of NMF represent patterns that frequently occur in the samples. If we consider a concrete example, where scientific articles are represented by their word frequencies. If there are 20000 articles and 800 words, so they array will have 800 columns. If we fit an NMF model with 10 components to the articles, the 10 components are stored as the 10 rows of a 2-dimensional numpy array. The rows, or components live in an 800-dimensional space – there is one dimension for each of the words. Aligning the words of our vocabulary with the columns of the NMF components allows them to be interpreted. Choosing a component, and looking at which words have the highest values, we can find themes. If any map is applied to documents and the components correspond to topics and NMF features reconstruct the documents from topics. If NMF is applied to a document of images, then NMF represents patterns of frequently occurring images. NMF decomposes images from a LCD display into the individual cells of the display. An image in which shades range from black to white is called a gray scale image. If the image is only gray, the image can be encoded by the brightness of every pixel. Representing the brightness with number between 0 and 1 where 0 is totally black and 1 is totally white. The image can be represented as a 2 dimensional array of numbers. These arrays can be flattened by enumerating the numbers. 


###### NMF learns topics of documents

When NMF is applied to documents, the components correspond to topics of documents, and the NMF features reconstruct the documents from the topics. This can be verified for the NMF model that I built earlier using the Wikipedia articles. Previously, we saw that the 3rd NMF feature value was high for the articles about actors Anne Hathaway and Denzel Washington. Below I will identify the topic of the corresponding NMF component.

Below I am building the model again, while words is a list of the words that label the columns of the word-frequency array.

After I am done, I will try to recognise the topic that the articles about Anne Hathaway and Denzel Washington have in common!


```python
# Importing NMF
from sklearn.decomposition import NMF

# Creating an NMF instance: model
model = NMF(n_components=6 )

# Fitting the model to articles
model.fit(articles)

# Transforming the articles: nmf_features
nmf_features = model.transform(articles)

# Printing the NMF features
print(nmf_features)

```

    [[0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 4.40464614e-01]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 5.66603937e-01]
     [3.82087997e-03 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 3.98645855e-01]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 3.81739161e-01]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 4.85516363e-01]
     [1.29300706e-02 1.37891511e-02 7.76335647e-03 3.34488358e-02
      0.00000000e+00 3.34521484e-01]
     [0.00000000e+00 0.00000000e+00 2.06745469e-02 0.00000000e+00
      6.04493804e-03 3.59060444e-01]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 4.90975955e-01]
     [1.54286342e-02 1.42819636e-02 3.76640537e-03 2.37112856e-02
      2.62622969e-02 4.80773635e-01]
     [1.11747165e-02 3.13682252e-02 3.09490283e-02 6.57003000e-02
      1.96679416e-02 3.38288317e-01]
     [0.00000000e+00 0.00000000e+00 5.30727493e-01 0.00000000e+00
      2.83682730e-02 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 3.56514712e-01 0.00000000e+00
      0.00000000e+00 0.00000000e+00]
     [1.20136780e-02 6.50045129e-03 3.12249989e-01 6.09773781e-02
      1.13862648e-02 1.92602109e-02]
     [3.93516838e-03 6.24443344e-03 3.42378449e-01 1.10769510e-02
      0.00000000e+00 0.00000000e+00]
     [4.63858354e-03 0.00000000e+00 4.34921659e-01 0.00000000e+00
      3.84279512e-02 3.08133662e-03]
     [0.00000000e+00 0.00000000e+00 4.83296431e-01 0.00000000e+00
      0.00000000e+00 0.00000000e+00]
     [5.65061385e-03 1.83535705e-02 3.76538707e-01 3.25462500e-02
      0.00000000e+00 1.13334525e-02]
     [0.00000000e+00 0.00000000e+00 4.80921065e-01 0.00000000e+00
      0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 9.01865430e-03 5.51016283e-01 0.00000000e+00
      0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 4.65976689e-01 0.00000000e+00
      0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 1.14080913e-02 2.08658828e-02 5.17769851e-01
      5.81415432e-02 1.37853905e-02]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 5.10477767e-01
      0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 5.60104437e-03 0.00000000e+00 4.22381917e-01
      0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 4.36753384e-01
      0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 4.98094468e-01
      0.00000000e+00 0.00000000e+00]
     [9.88472471e-02 8.60044623e-02 3.91041687e-03 3.81019401e-01
      4.39244975e-04 5.22152295e-03]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 5.72172678e-01
      0.00000000e+00 7.13542958e-03]
     [1.31479095e-02 1.04853408e-02 0.00000000e+00 4.68908343e-01
      0.00000000e+00 1.16309898e-02]
     [3.84578616e-03 0.00000000e+00 0.00000000e+00 5.75713369e-01
      0.00000000e+00 0.00000000e+00]
     [2.25261385e-03 1.38736113e-03 0.00000000e+00 5.27948352e-01
      1.20266094e-02 1.49483982e-02]
     [0.00000000e+00 4.07548251e-01 1.85717515e-03 0.00000000e+00
      2.96614240e-03 4.52346665e-04]
     [1.53433488e-03 6.08173145e-01 5.22286672e-04 6.24856753e-03
      1.18446159e-03 4.40080970e-04]
     [5.38862085e-03 2.65017097e-01 5.38519380e-04 1.86926903e-02
      6.38658838e-03 2.90104919e-03]
     [0.00000000e+00 6.44916019e-01 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 6.08907083e-01 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 3.43685314e-01 0.00000000e+00 0.00000000e+00
      3.97799929e-03 0.00000000e+00]
     [6.10557443e-03 3.15312875e-01 1.54882357e-02 0.00000000e+00
      5.06250680e-03 4.74335574e-03]
     [6.47424341e-03 2.13328588e-01 9.49510270e-03 4.56983531e-02
      1.71916565e-02 9.52062709e-03]
     [7.99210735e-03 4.67595250e-01 0.00000000e+00 2.43426645e-02
      0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 6.42820227e-01 0.00000000e+00 2.35857028e-03
      0.00000000e+00 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      4.77085915e-01 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      4.94259231e-01 0.00000000e+00]
     [0.00000000e+00 2.99069154e-04 2.14492177e-03 0.00000000e+00
      3.81780945e-01 5.83781074e-03]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 5.64696198e-03
      5.42244757e-01 0.00000000e+00]
     [1.78073870e-03 7.84414030e-04 1.41630240e-02 4.59819589e-04
      4.24304998e-01 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      5.11395046e-01 0.00000000e+00]
     [0.00000000e+00 0.00000000e+00 3.28392156e-03 0.00000000e+00
      3.72889069e-01 0.00000000e+00]
     [0.00000000e+00 2.62084425e-04 3.61110247e-02 2.32338903e-04
      2.30512121e-01 0.00000000e+00]
     [1.12526634e-02 2.12327511e-03 1.60975118e-02 1.02485358e-02
      3.25463535e-01 3.75880324e-02]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      4.18960795e-01 3.57704264e-04]
     [3.08397968e-01 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00]
     [3.68210837e-01 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00]
     [3.97984829e-01 2.81703156e-02 3.67017829e-03 1.70067660e-02
      1.95968931e-03 2.11644302e-02]
     [3.75832351e-01 2.07520838e-03 0.00000000e+00 3.72156230e-02
      0.00000000e+00 5.85927438e-03]
     [4.38072206e-01 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00]
     [4.57927015e-01 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00]
     [2.75504894e-01 4.46956834e-03 0.00000000e+00 5.29658126e-02
      0.00000000e+00 1.90997480e-02]
     [4.45238648e-01 0.00000000e+00 0.00000000e+00 0.00000000e+00
      5.48702561e-03 0.00000000e+00]
     [2.92769798e-01 1.33664850e-02 1.14265266e-02 1.05200573e-02
      1.87697646e-01 9.23964474e-03]
     [3.78304483e-01 1.43970301e-02 0.00000000e+00 9.85244376e-02
      1.35901317e-02 0.00000000e+00]]



```python
words = ['aaron', 'abandon', 'abandoned', 'abandoning', 'abandonment', 'abbas', 'abbey', 'abbreviated', 'abbreviation', 'abc', 'abdomen', 'abdominal', 'abdul', 'abel', 'abilities', 'ability', 'able', 'abnormal', 'abnormalities', 'abnormally', 'aboard', 'abolish', 'abolished', 'abolishing', 'abolition', 'aboriginal', 'abortion', 'abraham', 'abroad', 'abrupt', 'abruptly', 'absence', 'absent', 'absolute', 'absolutely', 'absorb', 'absorbed', 'absorbing', 'absorbs', 'absorption', 'abstract', 'abstraction', 'absurd', 'abu', 'abundance', 'abundant', 'abuse', 'abused', 'abuses', 'abusive', 'academia', 'academic', 'academics', 'academies', 'academy', 'accelerate', 'accelerated', 'accelerating', 'acceleration', 'accent', 'accents', 'accept', 'acceptable', 'acceptance', 'accepted', 'accepting', 'accepts', 'access', 'accessed', 'accessible', 'accessing', 'accession', 'accessories', 'accessory', 'accident', 'accidental', 'accidentally', 'accidents', 'acclaim', 'acclaimed', 'accolades', 'accommodate', 'accompanied', 'accompaniment', 'accompany', 'accompanying', 'accomplish', 'accomplished', 'accomplishment', 'accomplishments', 'accord', 'accordance', 'according', 'accordingly', 'account', 'accountability', 'accountable', 'accounted', 'accounting', 'accounts', 'accredited', 'accumulate', 'accumulated', 'accumulation', 'accuracy', 'accurate', 'accurately', 'accusation', 'accusations', 'accused', 'accusing', 'ace', 'achieve', 'achieved', 'achievement', 'achievements', 'achieves', 'achieving', 'acid', 'acidic', 'acids', 'acknowledge', 'acknowledged', 'acknowledges', 'acknowledging', 'acoustic', 'acquaintance', 'acquainted', 'acquire', 'acquired', 'acquiring', 'acquisition', 'acquisitions', 'acquitted', 'acre', 'acres', 'acronym', 'act', 'acted', 'acting', 'action', 'actions', 'activate', 'activated', 'activates', 'activation', 'active', 'actively', 'activism', 'activist', 'activists', 'activities', 'activity', 'actor', 'actors', 'actress', 'actresses', 'acts', 'actual', 'actually', 'acute', 'adam', 'adams', 'adapt', 'adaptation', 'adaptations', 'adapted', 'adapting', 'adaption', 'adaptive', 'add', 'added', 'addicted', 'addiction', 'adding', 'addition', 'additional', 'additionally', 'additions', 'additive', 'address', 'addressed', 'addresses', 'addressing', 'adds', 'adept', 'adequate', 'adequately', 'adhere', 'adhered', 'adherence', 'adherents', 'adjacent', 'adjective', 'adjoining', 'adjust', 'adjusted', 'adjusting', 'adjustment', 'adjustments', 'administer', 'administered', 'administering', 'administration', 'administrations', 'administrative', 'administrator', 'administrators', 'admiral', 'admiration', 'admired', 'admission', 'admissions', 'admit', 'admits', 'admitted', 'admitting', 'adolescence', 'adolescent', 'adolescents', 'adolf', 'adopt', 'adopted', 'adopting', 'adoption', 'adrian', 'ads', 'adult', 'adultery', 'adulthood', 'adults', 'advance', 'advanced', 'advancement', 'advances', 'advancing', 'advantage', 'advantageous', 'advantages', 'advent', 'adventure', 'adventures', 'adversary', 'adverse', 'adversely', 'advertise', 'advertised', 'advertisement', 'advertisements', 'advertising', 'advice', 'advise', 'advised', 'adviser', 'advisers', 'advises', 'advising', 'advisor', 'advisors', 'advisory', 'advocacy', 'advocate', 'advocated', 'advocates', 'advocating', 'aerial', 'aerospace', 'aesthetic', 'aesthetics', 'affair', 'affairs', 'affect', 'affected', 'affecting', 'affection', 'affects', 'affiliate', 'affiliated', 'affiliates', 'affiliation', 'affinity', 'affirmative', 'affirmed', 'afflicted', 'affluent', 'afford', 'affordable', 'afforded', 'afghan', 'afghanistan', 'aforementioned', 'afraid', 'africa', 'african', 'africans', 'afro', 'afterlife', 'aftermath', 'afternoon', 'afterward', 'age', 'aged', 'agencies', 'agency', 'agenda', 'agent', 'agents', 'ages', 'aggravated', 'aggregate', 'aggregator', 'aggression', 'aggressive', 'aggressively', 'aging', 'agitation', 'agnostic', 'ago', 'agrarian', 'agree', 'agreed', 'agreeing', 'agreement', 'agreements', 'agrees', 'agricultural', 'agriculture', 'ahead', 'ahmad', 'ahmed', 'aid', 'aide', 'aided', 'aides', 'aiding', 'aids', 'ailments', 'aim', 'aimed', 'aiming', 'aims', 'ain', 'air', 'airborne', 'aircraft', 'aired', 'airing', 'airline', 'airlines', 'airplane', 'airplanes', 'airplay', 'airport', 'airports', 'airs', 'airways', 'ajax', 'aka', 'akin', 'alabama', 'alan', 'alarm', 'alaska', 'albania', 'albeit', 'albert', 'alberto', 'album', 'albums', 'alcohol', 'alcoholic', 'alcoholism', 'alec', 'alert', 'alex', 'alexander', 'alexandra', 'alexandre', 'alexandria', 'alexis', 'alfred', 'algae', 'algeria', 'algorithm', 'algorithms', 'ali', 'alias', 'alice', 'alicia', 'alien', 'alienated', 'alienation', 'aliens', 'align', 'aligned', 'alignment', 'alike', 'alive', 'allan', 'allegation', 'allegations', 'alleged', 'allegedly', 'allegiance', 'alleging', 'allegory', 'allen', 'allergic', 'alleviate', 'alley', 'alliance', 'alliances', 'allied', 'allies', 'allmusic', 'allocated', 'allocation', 'allow', 'allowance', 'allowed', 'allowing', 'allows', 'alloy', 'allusions', 'ally', 'alma', 'alongside', 'alpha', 'alphabet', 'alpine', 'alps', 'alt', 'altar', 'alter', 'alteration', 'alterations', 'altercation', 'altered', 'altering', 'alternate', 'alternately', 'alternating', 'alternative', 'alternatively', 'alternatives', 'altitude', 'altitudes', 'alto', 'altogether', 'aluminium', 'aluminum', 'alumni', 'amanda', 'amassed', 'amateur', 'amazing', 'amazon', 'ambassador', 'ambassadors', 'amber', 'ambient', 'ambiguity', 'ambiguous', 'ambition', 'ambitions', 'ambitious', 'ambulance', 'ambush', 'ambushed', 'amended', 'amendment', 'amendments', 'america', 'american', 'americans', 'americas', 'amid', 'amidst', 'amino', 'ammonia', 'ammunition', 'amnesty', 'amounted', 'amounting', 'amounts', 'amphibians', 'ample', 'amplification', 'amsterdam', 'amusement', 'amy', 'ana', 'anal', 'analog', 'analogous', 'analogue', 'analogy', 'analyses', 'analysis', 'analyst', 'analysts', 'analytical', 'analyze', 'analyzed', 'analyzing', 'anarchy', 'anatolia', 'anatomical', 'anatomy', 'ancestor', 'ancestors', 'ancestral', 'ancestry', 'anchor', 'anchored', 'ancient', 'anderson', 'andr', 'andrea', 'andrew', 'andrews', 'android', 'andy', 'anemia', 'angel', 'angela', 'angeles', 'angelo', 'angels', 'anger', 'angered', 'angle', 'angles', 'anglican', 'anglo', 'angola', 'angrily', 'angry', 'angular', 'animal', 'animals', 'animated', 'animation', 'anime', 'animosity', 'ankle', 'ann', 'anna', 'anne', 'annex', 'annexation', 'annexed', 'annie', 'anniversary', 'announce', 'announced', 'announcement', 'announcements', 'announcing', 'annual', 'annually', 'anonymous', 'anonymously', 'answer', 'answered', 'answering', 'answers', 'ant', 'antagonist', 'antagonists', 'antarctic', 'antarctica', 'anterior', 'anthem', 'anthology', 'anthony', 'anthropogenic', 'anthropologist', 'anthropology', 'anti', 'antibiotics', 'antibodies', 'anticipated', 'anticipation', 'antics', 'antiquity', 'antoine', 'anton', 'antonio', 'anus', 'anxiety', 'anxious', 'anybody', 'anymore', 'apache', 'apart', 'apartheid', 'apartment', 'apartments', 'apex', 'api', 'apocalypse', 'apocalyptic', 'apollo', 'apologize', 'apologized', 'apology', 'app', 'appalled', 'apparatus', 'apparel', 'apparent', 'apparently', 'appeal', 'appealed', 'appealing', 'appeals', 'appear', 'appearance', 'appearances', 'appeared', 'appearing', 'appears', 'appetite', 'applauded', 'applause', 'apple', 'apples', 'applicable', 'applicants', 'application', 'applications', 'applied', 'applies', 'apply', 'applying', 'appoint', 'appointed', 'appointing', 'appointment', 'appointments', 'appoints', 'appreciate', 'appreciated', 'appreciation', 'apprentice', 'approach', 'approached', 'approaches', 'approaching', 'appropriate', 'appropriately', 'approval', 'approve', 'approved', 'approving', 'approximate', 'approximately', 'approximation', 'apps', 'april', 'aquatic', 'arab', 'arabia', 'arabian', 'arabic', 'arable', 'arabs', 'arbitrarily', 'arbitrary', 'arbitration', 'arc', 'arcade', 'arch', 'archaeological', 'archaeologists', 'archaeology', 'archaic', 'archbishop', 'archer', 'archipelago', 'architect', 'architects', 'architectural', 'architecture', 'archive', 'archives', 'arctic', 'area', 'areas', 'aren', 'arena', 'arenas', 'argentina', 'argentine', 'arguably', 'argue', 'argued', 'argues', 'arguing', 'argument', 'arguments', 'arid', 'arise', 'arisen', 'arises', 'arising', 'aristocracy', 'aristocratic', 'aristocrats', 'aristotle', 'arithmetic', 'arizona', 'arkansas', 'arm', 'armed', 'armenia', 'armenian', 'armies', 'armistice', 'armor', 'armored', 'arms', 'armstrong', 'army', 'arnold', 'arose', 'arousal', 'aroused', 'arrange', 'arranged', 'arrangement', 'arrangements', 'arranging', 'array', 'arrest', 'arrested', 'arresting', 'arrests', 'arrival', 'arrivals', 'arrive', 'arrived', 'arrives', 'arriving', 'arrow', 'arrows', 'arsenal', 'art', 'arteries', 'artery', 'arthur', 'article', 'articles', 'articulated', 'artifacts', 'artificial', 'artificially', 'artillery', 'artisans', 'artist', 'artistic', 'artistry', 'artists', 'arts', 'artwork', 'aryan', 'ascended', 'ascertain', 'ascribed', 'ash', 'ashes', 'ashley', 'ashton', 'asia', 'asian', 'asians', 'asiatic', 'aside', 'ask', 'asked', 'asking', 'asks', 'asleep', 'aspect', 'aspects', 'aspirations', 'aspiring', 'ass', 'assassin', 'assassinate', 'assassinated', 'assassination', 'assassinations', 'assassins', 'assault', 'assaulted', 'assaults', 'assemble', 'assembled', 'assemblies', 'assembly', 'assert', 'asserted', 'asserting', 'assertion', 'assertions', 'asserts', 'assess', 'assessed', 'assessing', 'assessment', 'assessments', 'asset', 'assets', 'assign', 'assigned', 'assignment', 'assigns', 'assimilated', 'assimilation', 'assist', 'assistance', 'assistant', 'assistants', 'assisted', 'assisting', 'assists', 'associate', 'associated', 'associates', 'association', 'associations', 'assortment', 'assume', 'assumed', 'assumes', 'assuming', 'assumption', 'assumptions', 'assurance', 'assure', 'assured', 'asteroid', 'asthma', 'astonishing', 'astronaut', 'astronauts', 'astronomer', 'astronomers', 'astronomical', 'astronomy', 'asylum', 'ate', 'atheist', 'atheists', 'athens', 'athlete', 'athletes', 'athletic', 'athletics', 'atlanta', 'atlantic', 'atlas', 'atm', 'atmosphere', 'atmospheric', 'atom', 'atomic', 'atoms', 'atop', 'atp', 'atrocities', 'attach', 'attached', 'attachment', 'attack', 'attacked', 'attackers', 'attacking', 'attacks', 'attain', 'attained', 'attaining', 'attempt', 'attempted', 'attempting', 'attempts', 'attend', 'attendance', 'attendant', 'attended', 'attending', 'attends', 'attention', 'attested', 'attire', 'attitude', 'attitudes', 'attorney', 'attorneys', 'attract', 'attracted', 'attracting', 'attraction', 'attractions', 'attractive', 'attracts', 'attributable', 'attribute', 'attributed', 'attributes', 'attributing', 'atypical', 'auction', 'auctioned', 'audience', 'audiences', 'audio', 'audition', 'auditioned', 'auditioning', 'auditions', 'auditorium', 'augmented', 'august', 'augustine', 'augustus', 'aunt', 'auspices', 'austin', 'australia', 'australian', 'australians', 'austria', 'austrian', 'authentic', 'authentication', 'authenticity', 'author', 'authored', 'authorised', 'authoritarian', 'authorities', 'authority', 'authorization', 'authorized', 'authors', 'authorship', 'autism', 'auto', 'autobiographical', 'autobiography', 'automated', 'automatic', 'automatically', 'automobile', 'automobiles', 'automotive', 'autonomous', 'autonomy', 'autopsy', 'autumn', 'auxiliary', 'availability', 'available', 'avant', 'avatar', 'avenue', 'avenues', 'average', 'averaged', 'averages', 'averaging', 'aviation', 'avid', 'avoid', 'avoidance', 'avoided', 'avoiding', 'avoids', 'awaiting', 'awakening', 'award', 'awarded', 'awards', 'aware', 'awareness', 'away', 'awful', 'awkward', 'axe', 'axis', 'azerbaijan', 'babies', 'baby', 'babylonian', 'bachelor', 'backbone', 'backdrop', 'backed', 'background', 'backgrounds', 'backing', 'backlash', 'backs', 'backstage', 'backup', 'backward', 'backwards', 'bacon', 'bacteria', 'bacterial', 'bacterium', 'bad', 'badge', 'badly', 'bafta', 'bag', 'baghdad', 'bags', 'bah', 'bahamas', 'bail', 'baked', 'baker', 'baking', 'balance', 'balanced', 'balances', 'balancing', 'bald', 'baldwin', 'bali', 'balkans', 'ball', 'ballad', 'ballads', 'ballet', 'ballistic', 'ballot', 'balls', 'baltic', 'baltimore', 'bamboo', 'ban', 'banana', 'band', 'bands', 'bandwidth', 'bang', 'bangalore', 'bangladesh', 'bank', 'banker', 'bankers', 'banking', 'bankrupt', 'bankruptcy', 'banks', 'banned', 'banner', 'banning', 'bans', 'baptised', 'baptism', 'baptist', 'baptized', 'bar', 'barack', 'barbara', 'barcelona', 'bare', 'barely', 'bargaining', 'barley', 'barnes', 'baron', 'baroque', 'barred', 'barrel', 'barrels', 'barrier', 'barriers', 'barry', 'bars', 'basal', 'base', 'baseball', 'based', 'baseline', 'basement', 'bases', 'basic', 'basically', 'basilica', 'basin', 'basins', 'basis', 'basket', 'basketball', 'bass', 'bassist', 'bat', 'batch', 'bath', 'bathing', 'bathroom', 'baths', 'batman', 'battalion', 'batteries', 'battery', 'battle', 'battlefield', 'battles', 'battling', 'bavaria', 'bay', 'bbc', 'bce', 'beach', 'beaches', 'beam', 'bean', 'beans', 'bear', 'beard', 'bearer', 'bearing', 'bears', 'beast', 'beat', 'beaten', 'beating', 'beatles', 'beats', 'beautiful', 'beauty', 'beck', 'bed', 'bedroom', 'beds', 'bee', 'beef', 'beer', 'befriended', 'began', 'begin', 'beginning', 'beginnings', 'begins', 'begun', 'behalf', 'behave', 'behaved', 'behavior', 'behavioral', 'behaviors', 'behaviour', 'behest', 'beijing', 'beings', 'bel', 'belarus', 'belgian', 'belgium', 'belief', 'beliefs', 'believe', 'believed', 'believer', 'believers', 'believes', 'believing', 'bell', 'belle', 'bells', 'belly', 'belong', 'belonged', 'belonging', 'belongs', 'beloved', 'belt', 'belts', 'ben', 'bench', 'bend', 'beneath', 'benedict', 'beneficial', 'benefit', 'benefited', 'benefiting', 'benefits', 'bengal', 'bengali', 'benign', 'benjamin', 'bennett', 'benny', 'bent', 'berkeley', 'berlin', 'bernard', 'berry', 'bertrand', 'besieged', 'best', 'bestowed', 'bestseller', 'bet', 'beta', 'betrayal', 'betrayed', 'better', 'betty', 'beverage', 'beverages', 'beverly', 'beyonc', 'bias', 'biased', 'bible', 'biblical', 'bibliography', 'bicameral', 'bicycle', 'bicycles', 'bid', 'bidding', 'big', 'bigger', 'biggest', 'bike', 'bilateral', 'bilingual', 'billboard', 'billed', 'billing', 'billion', 'billionaire', 'billions', 'bills', 'billy', 'bin', 'binary', 'bind', 'binding', 'binds', 'bing', 'bio', 'biochemical', 'biodiversity', 'biographer', 'biographers', 'biographical', 'biographies', 'biography', 'biological', 'biologically', 'biologist', 'biology', 'biomass', 'biomedical', 'biopic', 'biosphere', 'biotechnology', 'bipolar', 'birch', 'bird', 'birds', 'birmingham', 'birth', 'birthday', 'birthplace', 'births', 'bisexual', 'bishop', 'bishops', 'bit', 'bitch', 'bite', 'bites', 'biting', 'bits', 'bitter', 'bizarre', 'black', 'blacks', 'bladder', 'blade', 'blades', 'blair', 'blake', 'blame', 'blamed', 'blaming', 'blank', 'blast', 'bleeding', 'blend', 'blended', 'blending', 'blessed', 'blessing', 'blind', 'blindness', 'bloc', 'block', 'blockade', 'blockbuster', 'blocked', 'blocking', 'blocks', 'blog', 'blogs', 'blonde', 'blood', 'blooded', 'bloodstream', 'bloody', 'bloom', 'bloomberg', 'blow', 'blowing', 'blown', 'blows', 'blu', 'blue', 'blues', 'blunt', 'blurred', 'board', 'boarded', 'boarding', 'boards', 'boasted', 'boasts', 'boat', 'boats', 'bob', 'bobby', 'bodied', 'bodies', 'bodily', 'body', 'bodyguard', 'boeing', 'bohemian', 'boiled', 'boiling', 'bold', 'bolivia', 'bollywood', 'bolt', 'bomb', 'bombardment', 'bombed', 'bombers', 'bombing', 'bombings', 'bombs', 'bond', 'bonded', 'bonding', 'bonds', 'bone', 'bones', 'bonnie', 'bonus', 'book', 'booked', 'books', 'boom', 'boost', 'boosted', 'boot', 'booth', 'boots', 'border', 'bordered', 'bordering', 'borderline', 'borders', 'bore', 'bored', 'boris', 'born', 'borne', 'borough', 'borrow', 'borrowed', 'borrowing', 'bosnia', 'boss', 'boston', 'botanical', 'bottle', 'bottles', 'bought', 'boulevard', 'bounce', 'bound', 'boundaries', 'boundary', 'bounded', 'bounds', 'bounty', 'bourgeois', 'bout', 'bouts', 'bow', 'bowel', 'bowie', 'bowl', 'bowling', 'box', 'boxer', 'boxes', 'boxing', 'boy', 'boycott', 'boycotted', 'boyfriend', 'boyle', 'boys', 'brad', 'bradley', 'brain', 'brains', 'branch', 'branched', 'branches', 'brand', 'branded', 'branding', 'brandon', 'brands', 'brass', 'brave', 'bravery', 'brawl', 'brazil', 'brazilian', 'breach', 'bread', 'break', 'breakdown', 'breakfast', 'breaking', 'breakout', 'breaks', 'breakthrough', 'breakup', 'breast', 'breasts', 'breath', 'breathe', 'breathing', 'bred', 'breed', 'breeding', 'brian', 'brick', 'bricks', 'bride', 'bridge', 'bridges', 'brief', 'briefly', 'brien', 'brigade', 'bright', 'brighter', 'brilliant', 'bring', 'bringing', 'brings', 'bristol', 'brit', 'britain', 'british', 'britney', 'britons', 'broad', 'broadband', 'broadcast', 'broadcaster', 'broadcasters', 'broadcasting', 'broadcasts', 'broader', 'broadly', 'broadway', 'broke', 'broken', 'bronx', 'bronze', 'brooklyn', 'brooks', 'bros', 'brother', 'brotherhood', 'brothers', 'brought', 'brown', 'browser', 'browsers', 'browsing', 'bruce', 'bruno', 'brush', 'brussels', 'brutal', 'brutality', 'bryan', 'bryant', 'bubble', 'bubbles', 'buchanan', 'bud', 'buddha', 'buddhism', 'buddhist', 'buddhists', 'buddy', 'budget', 'budgets', 'buffalo', 'buffer', 'bug', 'bugs', 'build', 'builder', 'builders', 'building', 'buildings', 'builds', 'built', 'bulb', 'bulgaria', 'bulgarian', 'bulk', 'bull', 'bullet', 'bullets', 'bulls', 'bullying', 'bunch', 'bundled', 'bunny', 'burden', 'bureau', 'bureaucracy', 'burgeoning', 'burial', 'buried', 'burke', 'burma', 'burn', 'burned', 'burning', 'burns', 'burnt', 'burst', 'bursts', 'burton', 'bury', 'bus', 'buses', 'bush', 'busiest', 'business', 'businesses', 'businessman', 'businessmen', 'bust', 'busy', 'butler', 'butter', 'butterfly', 'button', 'buttons', 'buy', 'buyer', 'buyers', 'buying', 'buys', 'buzz', 'bypass', 'byzantine', 'cab', 'cabin', 'cabinet', 'cable', 'cables', 'caesar', 'caf', 'cafe', 'cage', 'cairo', 'cake', 'cakes', 'calcium', 'calculate', 'calculated', 'calculating', 'calculation', 'calculations', 'calendar', 'calf', 'caliber', 'california', 'called', 'calling', 'calls', 'calm', 'calories', 'calvin', 'cam', 'cambodia', 'cambridge', 'came', 'cameo', 'camera', 'cameras', 'cameron', 'camp', 'campaign', 'campaigned', 'campaigning', 'campaigns', 'campbell', 'camps', 'campus', 'campuses', 'canada', 'canadian', 'canadians', 'canal', 'canals', 'cancel', 'canceled', 'cancellation', 'cancelled', 'cancer', 'cancers', 'candidacy', 'candidate', 'candidates', 'candy', 'cane', 'cannabis', 'canned', 'cannes', 'cannon', 'canon', 'canonical', 'cans', 'canterbury', 'canyon', 'cap', 'capabilities', 'capability', 'capable', 'capacities', 'capacity', 'cape', 'capita', 'capital', 'capitalism', 'capitalist', 'capitals', 'capitol', 'capped', 'caps', 'captain', 'captive', 'captivity', 'capture', 'captured', 'captures', 'capturing', 'car', 'carbohydrate', 'carbohydrates', 'carbon', 'card', 'cardiac', 'cardinal', 'cardiovascular', 'cards', 'care', 'cared', 'career', 'careers', 'careful', 'carefully', 'carey', 'cargo', 'caribbean', 'caring', 'carl', 'carlo', 'carlos', 'carnegie', 'carnival', 'carol', 'carolina', 'caroline', 'carpenter', 'carpet', 'carr', 'carriage', 'carried', 'carrier', 'carriers', 'carries', 'carroll', 'carry', 'carrying', 'cars', 'carson', 'carter', 'cartoon', 'cartoonist', 'cartoons', 'carved', 'case', 'cases', 'casey', 'cash', 'casino', 'casket', 'caspian', 'cast', 'caste', 'casting', 'castle', 'castro', 'casts', 'casual', 'casualties', 'casualty', 'cat', 'catalog', 'catalogue', 'catalyst', 'catastrophe', 'catastrophic', 'catch', 'catches', 'catching', 'categories', 'categorized', 'category', 'cater', 'cathedral', 'catherine', 'catholic', 'catholicism', 'catholics', 'cats', 'cattle', 'caucasus', 'caught', 'causal', 'cause', 'caused', 'causes', 'causing', 'caution', 'cautious', 'cavalry', 'cave', 'caves', 'cavity', 'cbs', 'cds', 'cease', 'ceased', 'ceasefire', 'cecil', 'ceded', 'ceiling', 'celebrate', 'celebrated', 'celebrates', 'celebrating', 'celebration', 'celebrations', 'celebrities', 'celebrity', 'celestial', 'cell', 'cells', 'cellular', 'celtic', 'cement', 'cemented', 'cemetery', 'censored', 'censorship', 'census', 'cent', 'centennial', 'center', 'centered', 'centers', 'central', 'centralised', 'centralized', 'centrally', 'centre', 'centred', 'centres', 'centric', 'cents', 'centuries', 'century', 'ceo', 'ceramics', 'cereal', 'cereals', 'cerebral', 'ceremonial', 'ceremonies', 'ceremony', 'certain', 'certainly', 'certainty', 'certificate', 'certificates', 'certification', 'certified', 'cervical', 'cessation', 'cgi', 'chad', 'chain', 'chains', 'chair', 'chaired', 'chairman', 'chairs', 'challenge', 'challenged', 'challenger', 'challenges', 'challenging', 'chamber', 'chamberlain', 'chambers', 'champion', 'championed', 'champions', 'championship', 'championships', 'chan', 'chance', 'chancellor', 'chances', 'chandler', 'change', 'changed', 'changes', 'changing', 'channel', 'channels', 'chaos', 'chaotic', 'chapel', 'chapter', 'chapters', 'character', 'characterised', 'characteristic', 'characteristics', 'characterization', 'characterize', 'characterized', 'characters', 'charcoal', 'charge', 'charged', 'charges', 'charging', 'charisma', 'charismatic', 'charitable', 'charities', 'charity', 'charlemagne', 'charles', 'charleston', 'charlie', 'charlotte', 'charm', 'charming', 'chart', 'charted', 'charter', 'chartered', 'charting', 'charts', 'chase', 'chased', 'chasing', 'chat', 'cheap', 'cheaper', 'cheating', 'check', 'checked', 'checking', 'checks', 'cheek', 'cheese', 'chef', 'chelsea', 'chemical', 'chemically', 'chemicals', 'chemist', 'chemistry', 'chemists', 'chennai', 'cherry', 'chess', 'chest', 'chi', 'chicago', 'chicken', 'chickens', 'chief', 'chiefly', 'chiefs', 'child', 'childbirth', 'childhood', 'children', 'chile', 'chilean', 'chili', 'chin', 'china', 'chinese', 'chip', 'chips', 'chloride', 'chocolate', 'choice', 'choices', 'choir', 'cholera', 'cholesterol', 'choose', 'chooses', 'choosing', 'chopra', 'choreographer', 'chorus', 'chose', 'chosen', 'chris', 'christ', 'christian', 'christianity', 'christians', 'christie', 'christina', 'christine', 'christmas', 'christopher', 'chrome', 'chromosome', 'chromosomes', 'chronic', 'chronicle', 'chronicles', 'chronological', 'chuck', 'church', 'churches', 'churchill', 'cia', 'cigarette', 'cigarettes', 'cincinnati', 'cinema', 'cinemas', 'cinematic', 'circa', 'circle', 'circles', 'circuit', 'circuits', 'circular', 'circulated', 'circulating', 'circulation', 'circumstances', 'circus', 'citation', 'cite', 'cited', 'cites', 'cities', 'citing', 'citizen', 'citizens', 'citizenship', 'city', 'civic', 'civil', 'civilian', 'civilians', 'civilization', 'civilizations', 'clad', 'claim', 'claimed', 'claiming', 'claims', 'claire', 'clan', 'clandestine', 'clans', 'clara', 'clarence', 'clarified', 'clarify', 'clarity', 'clark', 'clarke', 'clarkson', 'clash', 'clashed', 'clashes', 'class', 'classed', 'classes', 'classic', 'classical', 'classically', 'classics', 'classification', 'classifications', 'classified', 'classify', 'classmate', 'classmates', 'classroom', 'claude', 'claudia', 'clause', 'clay', 'clean', 'cleaning', 'clear', 'clearance', 'cleared', 'clearing', 'clearly', 'clement', 'clergy', 'clerical', 'clerk', 'cleveland', 'clever', 'clich', 'click', 'client', 'clients', 'cliff', 'cliffs', 'climate', 'climates', 'climatic', 'climax', 'climb', 'climbed', 'climbing', 'clinic', 'clinical', 'clinically', 'clinics', 'clint', 'clinton', 'clip', 'clips', 'clock', 'close', 'closed', 'closely', 'closer', 'closest', 'closet', 'closing', 'closure', 'cloth', 'clothes', 'clothing', 'cloud', 'clouds', 'club', 'clubs', 'clue', 'clues', 'cluster', 'clusters', 'cnn', 'coach', 'coaches', 'coaching', 'coal', 'coalition', 'coast', 'coastal', 'coastline', 'coasts', 'coat', 'coated', 'coats', 'coca', 'cocaine', 'cocktail', 'cocoa', 'coconut', 'code', 'coded', 'codes', 'codice', 'codified', 'coding', 'coefficient', 'coffee', 'coffin', 'cognate', 'cognition', 'cognitive', 'cohen', 'coherent', 'coin', 'coincide', 'coincided', 'coincidence', 'coincidentally', 'coinciding', 'coined', 'coins', 'coke', 'col', 'cola', 'cold', 'colder', 'coldest', 'cole', 'coleman', 'colin', 'collaborate', 'collaborated', 'collaborating', 'collaboration', 'collaborations', 'collaborative', 'collaborator', 'collaborators', 'collapse', 'collapsed', 'collapsing', 'collar', 'colleague', 'colleagues', 'collect', 'collected', 'collecting', 'collection', 'collections', 'collective', 'collectively', 'collector', 'collectors', 'collects', 'college', 'colleges', 'collegiate', 'collins', 'collision', 'collisions', 'colloquial', 'colloquially', 'cologne', 'colombia', 'colombian', 'colon', 'colonel', 'colonial', 'colonialism', 'colonies', 'colonists', 'colonization', 'colony', 'color', 'colorado', 'colored', 'colorful', 'coloring', 'colors', 'colour', 'coloured', 'colours', 'columbia', 'columbian', 'columbus', 'column', 'columnist', 'columns', 'com', 'coma', 'combat', 'combatants', 'combination', 'combinations', 'combine', 'combined', 'combines', 'combining', 'combustion', 'come', 'comeback', 'comedian', 'comedic', 'comedies', 'comedy', 'comes', 'comfort', 'comfortable', 'comic', 'comics', 'coming', 'command', 'commanded', 'commander', 'commanders', 'commanding', 'commands', 'commemorate', 'commemorated', 'commemorating', 'commemoration', 'commemorative', 'commence', 'commenced', 'commencement', 'comment', 'commentaries', 'commentary', 'commentator', 'commentators', 'commented', 'commenting', 'comments', 'commerce', 'commercial', 'commercially', 'commercials', 'commission', 'commissioned', 'commissioner', 'commissions', 'commit', 'commitment', 'commitments', 'committed', 'committee', 'committees', 'committing', 'commodities', 'commodity', 'common', 'commonly', 'commonplace', 'commons', 'commonwealth', 'communal', 'communicate', 'communicated', 'communicating', 'communication', 'communications', 'communion', 'communism', 'communist', 'communists', 'communities', 'community', 'commuter', 'compact', 'companies', 'companion', 'companions', 'company', 'comparable', 'comparative', 'comparatively', 'compare', 'compared', 'compares', 'comparing', 'comparison', 'comparisons', 'compass', 'compassion', 'compatibility', 'compatible', 'compelled', 'compelling', 'compensate', 'compensated', 'compensation', 'compete', 'competed', 'competence', 'competent', 'competes', 'competing', 'competition', 'competitions', 'competitive', 'competitiveness', 'competitor', 'competitors', 'compilation', 'compiled', 'complained', 'complaint', 'complaints', 'complement', 'complementary', 'complemented', 'complete', 'completed', 'completely', 'completing', 'completion', 'complex', 'complexes', 'complexity', 'compliance', 'compliant', 'complicated', 'complication', 'complications', 'complied', 'comply', 'component', 'components', 'compose', 'composed', 'composer', 'composers', 'composing', 'composite', 'composition', 'compositions', 'compound', 'compounded', 'compounds', 'comprehensive', 'compressed', 'compression', 'comprise', 'comprised', 'comprises', 'comprising', 'compromise', 'compromised', 'compulsive', 'compulsory', 'computation', 'computational', 'compute', 'computer', 'computers', 'computing', 'conan', 'conceal', 'concealed', 'conceded', 'conceived', 'concentrate', 'concentrated', 'concentrating', 'concentration', 'concentrations', 'concept', 'conception', 'conceptions', 'concepts', 'conceptual', 'concern', 'concerned', 'concerning', 'concerns', 'concert', 'concerts', 'concession', 'concessions', 'conclude', 'concluded', 'concludes', 'concluding', 'conclusion', 'conclusions', 'conclusive', 'concrete', 'concurrent', 'concurrently', 'condemn', 'condemnation', 'condemned', 'condensation', 'condensed', 'condition', 'conditional', 'conditioning', 'conditions', 'conduct', 'conducted', 'conducting', 'conductor', 'conducts', 'cone', 'confederacy', 'confederate', 'confederation', 'conference', 'conferences', 'conferred', 'confessed', 'confession', 'confessions', 'confidence', 'confident', 'confidential', 'configuration', 'configurations', 'configured', 'confined', 'confinement', 'confirm', 'confirmation', 'confirmed', 'confirming', 'confirms', 'confiscated', 'conflict', 'conflicted', 'conflicting', 'conflicts', 'conform', 'confront', 'confrontation', 'confrontations', 'confronted', 'confronting', 'confronts', 'confused', 'confusing', 'confusion', 'congenital', 'congestion', 'congo', 'congregation', 'congregations', 'congress', 'congressional', 'congressman', 'conjecture', 'conjunction', 'connect', 'connected', 'connecticut', 'connecting', 'connection', 'connections', 'connective', 'connectivity', 'connects', 'connor', 'connotations', 'conquer', 'conquered', 'conquering', 'conquest', 'conquests', 'conrad', 'conscience', 'conscious', 'consciously', 'consciousness', 'conscription', 'consecutive', 'consensus', 'consent', 'consequence', 'consequences', 'consequent', 'consequently', 'conservation', 'conservative', 'conservatives', 'conserve', 'consider', 'considerable', 'considerably', 'consideration', 'considerations', 'considered', 'considering', 'considers', 'consist', 'consisted', 'consistency', 'consistent', 'consistently', 'consisting', 'consists', 'console', 'consoles', 'consolidate', 'consolidated', 'consolidation', 'consortium', 'conspiracy', 'constant', 'constantine', 'constantinople', 'constantly', 'constellation', 'constituencies', 'constituent', 'constituents', 'constitute', 'constituted', 'constitutes', 'constituting', 'constitution', 'constitutional', 'constitutionally', 'constitutions', 'constrained', 'constraints', 'construct', 'constructed', 'constructing', 'construction', 'constructions', 'constructs', 'consult', 'consultant', 'consultation', 'consulted', 'consulting', 'consume', 'consumed', 'consumer', 'consumers', 'consumes', 'consuming', 'consumption', 'contact', 'contacted', 'contacts', 'contain', 'contained', 'container', 'containers', 'containing', 'containment', 'contains', 'contaminated', 'contamination', 'contemporaries', 'contemporary', 'contempt', 'contend', 'contended', 'contender', 'contenders', 'contends', 'content', 'contention', 'contentious', 'contents', 'contest', 'contestant', 'contestants', 'contested', 'contests', 'context', 'contexts', 'contiguous', 'continent', 'continental', 'continents', 'contingent', 'continual', 'continually', 'continuation', 'continue', 'continued', 'continues', 'continuing', 'continuity', 'continuous', 'continuously', 'continuum', 'contract', 'contracted', 'contracting', 'contraction', 'contractor', 'contractors', 'contracts', 'contractual', 'contradict', 'contradicted', 'contradiction', 'contradictions', 'contradictory', 'contrary', 'contrast', 'contrasted', 'contrasting', 'contrasts', 'contribute', 'contributed', 'contributes', 'contributing', 'contribution', 'contributions', 'contributor', 'contributors', 'control', 'controlled', 'controller', 'controllers', 'controlling', 'controls', 'controversial', 'controversially', 'controversies', 'controversy', 'convened', 'convenience', 'convenient', 'convention', 'conventional', 'conventionally', 'conventions', 'converge', 'convergence', 'conversation', 'conversations', 'conversely', 'conversion', 'convert', 'converted', 'converting', 'converts', 'convey', 'conveyed', 'convicted', 'conviction', 'convictions', 'convince', 'convinced', 'convinces', 'convincing', 'cook', 'cooked', 'cookies', 'cooking', 'cool', 'cooled', 'cooler', 'cooling', 'cooper', 'cooperate', 'cooperation', 'cooperative', 'coordinate', 'coordinated', 'coordinates', 'coordinating', 'coordination', 'cop', 'cope', 'copenhagen', 'copied', 'copies', 'copper', 'copy', 'copying', 'copyright', 'coral', 'cord', 'core', 'cork', 'corn', 'cornell', 'corner', 'corners', 'coronary', 'coronation', 'corporate', 'corporation', 'corporations', 'corps', 'corpse', 'corpses', 'corpus', 'correct', 'corrected', 'correction', 'correctly', 'correlate', 'correlated', 'correlates', 'correlation', 'correspond', 'corresponded', 'correspondence', 'correspondent', 'corresponding', 'corresponds', 'corridor', 'corrupt', 'corrupted', 'corruption', 'cortex', 'cosmetic', 'cosmetics', 'cosmic', 'cosmopolitan', 'cost', 'costa', 'costing', 'costly', 'costs', 'costume', 'costumes', 'cottage', 'cotton', 'couldn', 'council', 'councils', 'counsel', 'counseling', 'counselor', 'count', 'countdown', 'counted', 'counter', 'counteract', 'countered', 'counterpart', 'counterparts', 'counties', 'counting', 'countless', 'countries', 'country', 'countryside', 'counts', 'county', 'coup', 'couple', 'coupled', 'couples', 'courage', 'course', 'courses', 'court', 'courts', 'courtship', 'cousin', 'cousins', 'cover', 'coverage', 'covered', 'covering', 'covers', 'covert', 'coveted', 'cow', 'cowboy', 'cows', 'cox', 'cpu', 'crack', 'cracked', 'craft', 'crafted', 'crafts', 'craig', 'crane', 'crash', 'crashed', 'crashing', 'crater', 'crawford', 'crazy', 'cream', 'create', 'created', 'creates', 'creating', 'creation', 'creations', 'creative', 'creativity', 'creator', 'creators', 'creature', 'creatures', 'credentials', 'credibility', 'credible', 'credit', 'credited', 'credits', 'creed', 'creek', 'cremated', 'creole', 'crescent', 'crest', 'crete', 'crew', 'crews', 'cricket', 'cried', 'crime', 'crimes', 'criminal', 'criminals', 'crippled', 'crises', 'crisis', 'criteria', 'criterion', 'critic', 'critical', 'critically', 'criticised', 'criticism', 'criticisms', 'criticize', 'criticized', 'criticizing', 'critics', 'critique', 'croatia', 'croatian', 'crop', 'crops', 'cross', 'crossed', 'crosses', 'crossing', 'crossover', 'crossroads', 'crow', 'crowd', 'crowded', 'crowds', 'crown', 'crowned', 'crucial', 'crude', 'cruel', 'cruelty', 'cruise', 'crusade', 'crush', 'crushed', 'crushing', 'crust', 'cruz', 'crying', 'crystal', 'crystalline', 'crystals', 'cuba', 'cuban', 'cube', 'cubic', 'cues', 'cuisine', 'cuisines', 'culinary', 'culminated', 'culminating', 'cult', 'cultivate', 'cultivated', 'cultivation', 'cults', 'cultural', 'culturally', 'culture', 'cultures', 'cumulative', 'cup', 'cups', 'curb', 'cure', 'cured', 'curiosity', 'curious', 'currency', 'current', 'currently', 'currents', 'curriculum', 'curry', 'curse', 'curtailed', 'curtain', 'curtis', 'curve', 'curved', 'custody', 'custom', 'customary', 'customer', 'customers', 'customs', 'cut', 'cuts', 'cutting', 'cyber', 'cycle', 'cycles', 'cycling', 'cyclones', 'cynical', 'cyprus', 'cyrus', 'czech', 'czechoslovakia', 'dad', 'daddy', 'daily', 'dairy', 'dakota', 'dale', 'dallas', 'daly', 'dam', 'damage', 'damaged', 'damages', 'damaging', 'damascus', 'dame', 'damn', 'damon', 'dams', 'dan', 'dana', 'dance', 'danced', 'dancer', 'dancers', 'dances', 'dancing', 'danger', 'dangerous', 'dangers', 'daniel', 'daniels', 'danish', 'danny', 'dante', 'dare', 'daring', 'dark', 'darker', 'darkness', 'darren', 'darwin', 'das', 'dash', 'data', 'database', 'databases', 'date', 'dated', 'dates', 'dating', 'daughter', 'daughters', 'dave', 'david', 'davies', 'davis', 'dawn', 'day', 'daylight', 'days', 'daytime', 'dead', 'deadliest', 'deadline', 'deadly', 'deaf', 'deal', 'dealer', 'dealers', 'dealing', 'deals', 'dealt', 'dean', 'dear', 'death', 'deaths', 'debate', 'debated', 'debates', 'debris', 'debt', 'debts', 'debut', 'debuted', 'debuting', 'dec', 'decade', 'decades', 'decay', 'deceased', 'december', 'decent', 'decentralized', 'deception', 'decide', 'decided', 'decides', 'deciding', 'decimal', 'decision', 'decisions', 'decisive', 'decisively', 'deck', 'declaration', 'declarations', 'declare', 'declared', 'declares', 'declaring', 'decline', 'declined', 'declines', 'declining', 'decomposition', 'decorated', 'decoration', 'decorations', 'decorative', 'decrease', 'decreased', 'decreases', 'decreasing', 'decree', 'decreed', 'dedicated', 'dedication', 'deeds', 'deemed', 'deep', 'deeper', 'deepest', 'deeply', 'deer', 'def', 'defamation', 'default', 'defeat', 'defeated', 'defeating', 'defeats', 'defect', 'defects', 'defence', 'defend', 'defendants', 'defended', 'defender', 'defenders', 'defending', 'defense', 'defenses', 'defensive', 'defiance', 'deficiencies', 'deficiency', 'deficient', 'deficit', 'deficits', 'defied', 'define', 'defined', 'defines', 'defining', 'definite', 'definitely', 'definition', 'definitions', 'definitive', 'definitively', 'deforestation', 'defunct', 'degradation', 'degraded', 'degree', 'degrees', 'dehydration', 'deities', 'deity', 'del', 'delaware', 'delay', 'delayed', 'delaying', 'delays', 'delegate', 'delegated', 'delegates', 'delegation', 'deleted', 'delhi', 'deliberate', 'deliberately', 'delicate', 'delight', 'delighted', 'deliver', 'delivered', 'deliveries', 'delivering', 'delivers', 'delivery', 'dell', 'delta', 'deluxe', 'demand', 'demanded', 'demanding', 'demands', 'demise', 'demo', 'democracy', 'democrat', 'democratic', 'democratically', 'democrats', 'demographic', 'demographics', 'demolished', 'demon', 'demons', 'demonstrate', 'demonstrated', 'demonstrates', 'demonstrating', 'demonstration', 'demonstrations', 'demos', 'den', 'denial', 'denied', 'denies', 'denis', 'denmark', 'dennis', 'denomination', 'denominations', 'denote', 'denoted', 'denotes', 'denoting', 'denounced', 'dense', 'densely', 'denser', 'densities', 'density', 'dental', 'denver', 'deny', 'denying', 'depart', 'departed', 'departing', 'department', 'departments', 'departure', 'depend', 'depended', 'dependence', 'dependency', 'dependent', 'depending', 'depends', 'depict', 'depicted', 'depicting', 'depiction', 'depictions', 'depicts', 'depleted', 'depletion', 'deploy', 'deployed', 'deploying', 'deployment', 'deportation', 'deported', 'deposed', 'deposit', 'deposited', 'deposition', 'deposits', 'depp', 'depressed', 'depression', 'depressive', 'deprived', 'depth', 'depths', 'deputies', 'deputy', 'der', 'derby', 'derek', 'derivation', 'derivative', 'derivatives', 'derive', 'derived', 'derives', 'deriving', 'derogatory', 'des', 'descend', 'descendant', 'descendants', 'descended', 'descending', 'descends', 'descent', 'described', 'describes', 'describing', 'description', 'descriptions', 'descriptive', 'desert', 'deserted', 'deserts', 'deserve', 'deserved', 'deserves', 'design', 'designate', 'designated', 'designation', 'designations', 'designed', 'designer', 'designers', 'designing', 'designs', 'desirable', 'desire', 'desired', 'desires', 'desk', 'desktop', 'desmond', 'despair', 'desperate', 'desperately', 'despite', 'destination', 'destinations', 'destined', 'destiny', 'destroy', 'destroyed', 'destroying', 'destroys', 'destruction', 'destructive', 'detached', 'detachment', 'detailed', 'detailing', 'details', 'detained', 'detect', 'detectable', 'detected', 'detecting', 'detection', 'detective', 'detention', 'deteriorate', 'deteriorated', 'deteriorating', 'deterioration', 'determination', 'determine', 'determined', 'determines', 'determining', 'detrimental', 'detroit', 'deutsche', 'devastated', 'devastating', 'devastation', 'develop', 'developed', 'developer', 'developers', 'developing', 'development', 'developmental', 'developments', 'develops', 'deviation', 'deviations', 'device', 'devices', 'devil', 'devised', 'devoid', 'devote', 'devoted', 'devotion', 'devout', 'dharma', 'dia', 'diabetes', 'diagnose', 'diagnosed', 'diagnosis', 'diagnostic', 'diagram', 'dialect', 'dialects', 'dialogue', 'diameter', 'diamond', 'diamonds', 'diana', 'diane', 'diaries', 'diarrhea', 'diary', 'diaspora', 'dicaprio', 'dick', 'dickens', 'dictate', 'dictated', 'dictator', 'dictatorship', 'dictionary', 'did', 'didn', 'die', 'died', 'diego', 'dies', 'diesel', 'diet', 'dietary', 'diets', 'differ', 'differed', 'difference', 'differences', 'different', 'differential', 'differentiate', 'differentiated', 'differentiation', 'differently', 'differing', 'differs', 'difficult', 'difficulties', 'difficulty', 'diffuse', 'diffusion', 'dig', 'digest', 'digestion', 'digestive', 'digging', 'digit', 'digital', 'digitally', 'digits', 'dignity', 'dimension', 'dimensional', 'dimensions', 'diminish', 'diminished', 'din', 'dining', 'dinner', 'dioxide', 'diploma', 'diplomacy', 'diplomat', 'diplomatic', 'diplomats', 'dire', 'direct', 'directed', 'directing', 'direction', 'directional', 'directions', 'directive', 'directly', 'director', 'directorial', 'directors', 'directory', 'directs', 'dirt', 'dirty', 'disabilities', 'disability', 'disable', 'disabled', 'disadvantage', 'disadvantaged', 'disadvantages', 'disagree', 'disagreed', 'disagreement', 'disagreements', 'disappear', 'disappearance', 'disappeared', 'disappears', 'disappointed', 'disappointing', 'disappointment', 'disapproval', 'disapproved', 'disarmament', 'disaster', 'disasters', 'disastrous', 'disbanded', 'disc', 'discarded', 'discharge', 'discharged', 'disciples', 'discipline', 'disciplined', 'disciplines', 'disclose', 'disclosed', 'disclosure', 'disco', 'discography', 'discomfort', 'discontent', 'discontinued', 'discourage', 'discouraged', 'discourse', 'discover', 'discovered', 'discoveries', 'discovering', 'discovers', 'discovery', 'discrete', 'discretion', 'discrimination', 'discs', 'discuss', 'discussed', 'discusses', 'discussing', 'discussion', 'discussions', 'disease', 'diseases', 'disguise', 'disguised', 'dish', 'dishes', 'disk', 'disks', 'dislike', 'disliked', 'dismantled', 'dismay', 'dismiss', 'dismissal', 'dismissed', 'disney', 'disorder', 'disorders', 'disparate', 'disparity', 'dispatch', 'dispatched', 'dispersal', 'dispersed', 'displaced', 'displacement', 'display', 'displayed', 'displaying', 'displays', 'disposal', 'disposed', 'disposition', 'dispute', 'disputed', 'disputes', 'disregard', 'disrupt', 'disrupted', 'disrupting', 'disruption', 'disruptive', 'dissatisfaction', 'dissatisfied', 'disseminated', 'dissemination', 'dissent', 'dissolution', 'dissolve', 'dissolved', 'distance', 'distanced', 'distances', 'distant', 'distinct', 'distinction', 'distinctions', 'distinctive', 'distinctly', 'distinguish', 'distinguished', 'distinguishes', 'distinguishing', 'distorted', 'distortion', 'distracted', 'distress', 'distribute', 'distributed', 'distributing', 'distribution', 'distributions', 'distributor', 'distributors', 'district', 'districts', 'distrust', 'disturbance', 'disturbances', 'disturbed', 'disturbing', 'diverged', 'divergence', 'diverse', 'diversified', 'diversion', 'diversity', 'diverted', 'divide', 'divided', 'divides', 'dividing', 'divine', 'diving', 'divinity', 'division', 'divisions', 'divorce', 'divorced', 'dna', 'dock', 'doctor', 'doctoral', 'doctorate', 'doctors', 'doctrine', 'doctrines', 'document', 'documentaries', 'documentary', 'documentation', 'documented', 'documenting', 'documents', 'does', 'doesn', 'dog', 'dogg', 'dogs', 'doing', 'doll', 'dollar', 'dollars', 'dolls', 'dolphins', 'dom', 'domain', 'domains', 'dome', 'domestic', 'domestically', 'domesticated', 'domestication', 'dominance', 'dominant', 'dominate', 'dominated', 'dominates', 'dominating', 'domination', 'dominic', 'dominican', 'dominion', 'don', 'donald', 'donate', 'donated', 'donating', 'donation', 'donations', 'donna', 'donor', 'donors', 'doom', 'doomed', 'door', 'doors', 'dopamine', 'dormant', 'dorothy', 'dorsal', 'dos', 'dose', 'doses', 'dot', 'double', 'doubled', 'doubles', 'doubling', 'doubt', 'doubted', 'doubtful', 'doubts', 'doug', 'douglas', 'downfall', 'download', 'downloadable', 'downloaded', 'downloads', 'downtown', 'downturn', 'downward', 'doyle', 'dozen', 'dozens', 'draft', 'drafted', 'drafting', 'drafts', 'drag', 'dragged', 'dragon', 'dragons', 'drain', 'drainage', 'drained', 'drake', 'drama', 'dramas', 'dramatic', 'dramatically', 'drank', 'drastic', 'drastically', 'draw', 'drawing', 'drawings', 'drawn', 'draws', 'dre', 'dream', 'dreams', 'dress', 'dressed', 'dresses', 'dressing', 'drew', 'dried', 'drier', 'drift', 'drill', 'drilling', 'drink', 'drinking', 'drinks', 'drive', 'driven', 'driver', 'drivers', 'drives', 'driving', 'drop', 'dropped', 'dropping', 'drops', 'drought', 'drove', 'drowned', 'drowning', 'drug', 'drugs', 'drum', 'drummer', 'drums', 'drunk', 'dry', 'drying', 'dual', 'dub', 'dubai', 'dubbed', 'dubious', 'dublin', 'duchy', 'duck', 'duet', 'duets', 'duke', 'dull', 'dumb', 'dumped', 'duncan', 'duo', 'duplicate', 'durable', 'duration', 'dust', 'dutch', 'duties', 'duty', 'dvd', 'dvds', 'dwarf', 'dwelling', 'dwight', 'dying', 'dylan', 'dynamic', 'dynamically', 'dynamics', 'dynastic', 'dynasties', 'dynasty', 'dysfunction', 'eager', 'eagle', 'eagles', 'ear', 'earl', 'earlier', 'earliest', 'early', 'earn', 'earned', 'earnest', 'earning', 'earnings', 'ears', 'earth', 'earthquake', 'earthquakes', 'ease', 'easier', 'easily', 'east', 'easter', 'eastern', 'eastward', 'easy', 'eat', 'eaten', 'eating', 'ebay', 'ebert', 'eccentric', 'ecclesiastical', 'echo', 'echoed', 'echoes', 'eclectic', 'eclipse', 'eclipsed', 'eco', 'ecological', 'ecology', 'economic', 'economical', 'economically', 'economics', 'economies', 'economist', 'economists', 'economy', 'ecosystem', 'ecosystems', 'ecuador', 'eddie', 'eden', 'edgar', 'edge', 'edges', 'edible', 'edinburgh', 'edit', 'edited', 'edith', 'editing', 'edition', 'editions', 'editor', 'editorial', 'editors', 'edmund', 'educate', 'educated', 'educating', 'education', 'educational', 'edward', 'edwards', 'edwin', 'effect', 'effective', 'effectively', 'effectiveness', 'effects', 'efficacy', 'efficiency', 'efficient', 'efficiently', 'effort', 'efforts', 'egg', 'eggs', 'ego', 'egypt', 'egyptian', 'egyptians', 'eighteen', 'eighteenth', 'eighth', 'eighty', 'einstein', 'eisenhower', 'ejected', 'elaborate', 'elaborated', 'elastic', 'elbow', 'elder', 'elderly', 'elders', 'eldest', 'eleanor', 'elect', 'elected', 'election', 'elections', 'elective', 'electoral', 'electorate', 'electric', 'electrical', 'electrically', 'electricity', 'electrified', 'electro', 'electromagnetic', 'electron', 'electronic', 'electronically', 'electronics', 'electrons', 'elegant', 'element', 'elementary', 'elements', 'elephant', 'elephants', 'elevated', 'elevation', 'elevations', 'elevator', 'eleventh', 'eli', 'eligibility', 'eligible', 'eliminate', 'eliminated', 'eliminates', 'eliminating', 'elimination', 'eliot', 'elisabeth', 'elite', 'elites', 'elizabeth', 'elle', 'ellen', 'elliot', 'elliott', 'ellis', 'elongated', 'elton', 'elusive', 'elvis', 'email', 'emails', 'emancipation', 'embargo', 'embarked', 'embarrassed', 'embarrassing', 'embarrassment', 'embassies', 'embassy', 'embedded', 'emblem', 'embodied', 'embodiment', 'embrace', 'embraced', 'embracing', 'embroiled', 'embryo', 'embryonic', 'emerge', 'emerged', 'emergence', 'emergencies', 'emergency', 'emerges', 'emerging', 'emi', 'emigrated', 'emigration', 'emily', 'eminem', 'eminent', 'emirates', 'emission', 'emissions', 'emit', 'emitted', 'emma', 'emmy', 'emotion', 'emotional', 'emotionally', 'emotions', 'empathy', 'emperor', 'emperors', 'emphasis', 'emphasised', 'emphasize', 'emphasized', 'emphasizes', 'emphasizing', 'empire', 'empires', 'empirical', 'employ', 'employed', 'employee', 'employees', 'employer', 'employers', 'employing', 'employment', 'employs', 'empowerment', 'empress', 'emulate', 'enable', 'enabled', 'enables', 'enabling', 'enact', 'enacted', 'enactment', 'enclosed', 'encoded', 'encoding', 'encompass', 'encompassed', 'encompasses', 'encompassing', 'encounter', 'encountered', 'encounters', 'encourage', 'encouraged', 'encouragement', 'encourages', 'encouraging', 'encrypted', 'encryption', 'encyclopedia', 'end', 'endangered', 'endeavor', 'endeavors', 'ended', 'endemic', 'ending', 'endings', 'endless', 'endorse', 'endorsed', 'endorsement', 'endorsements', 'endorsing', 'endowed', 'ends', 'endurance', 'endure', 'endured', 'enduring', 'enemies', 'enemy', 'energetic', 'energies', 'energy', 'enforce', 'enforced', 'enforcement', 'enforcing', 'engage', 'engaged', 'engagement', 'engagements', 'engages', 'engaging', 'engine', 'engineer', 'engineered', 'engineering', 'engineers', 'engines', 'england', 'english', 'enhance', 'enhanced', 'enhancement', 'enhancements', 'enhances', 'enhancing', 'enjoy', 'enjoyable', 'enjoyed', 'enjoying', 'enjoyment', 'enjoys', 'enlarged', 'enlargement', 'enlightenment', 'enlisted', 'enormous', 'enormously', 'enraged', 'enriched', 'enrolled', 'enrollment', 'ensemble', 'enslaved', 'ensued', 'ensuing', 'ensure', 'ensured', 'ensures', 'ensuring', 'entails', 'enter', 'entered', 'entering', 'enterprise', 'enterprises', 'enters', 'entertain', 'entertained', 'entertainer', 'entertainers', 'entertaining', 'entertainment', 'enthusiasm', 'enthusiastic', 'enthusiasts', 'entire', 'entirely', 'entirety', 'entities', 'entitled', 'entity', 'entourage', 'entrance', 'entrenched', 'entrepreneur', 'entrepreneurs', 'entries', 'entrusted', 'entry', 'envelope', 'environment', 'environmental', 'environmentally', 'environments', 'envisioned', 'envoy', 'enzyme', 'enzymes', 'epa', 'epic', 'epidemic', 'epidemics', 'epidemiology', 'epilepsy', 'episcopal', 'episode', 'episodes', 'epithet', 'epoch', 'eponymous', 'equal', 'equality', 'equally', 'equals', 'equation', 'equations', 'equator', 'equatorial', 'equilibrium', 'equipment', 'equipped', 'equity', 'equivalent', 'equivalents', 'era', 'eras', 'erased', 'erect', 'erected', 'eric', 'erik', 'ernest', 'ernst', 'eroded', 'erosion', 'erotic', 'erratic', 'erroneous', 'error', 'errors', 'ers', 'erupted', 'eruption', 'eruptions', 'escalated', 'escape', 'escaped', 'escapes', 'escaping', 'escort', 'especially', 'espionage', 'espn', 'espoused', 'esquire', 'essay', 'essays', 'essence', 'essential', 'essentially', 'essex', 'est', 'establish', 'established', 'establishes', 'establishing', 'establishment', 'establishments', 'estate', 'estates', 'esteem', 'estimate', 'estimated', 'estimates', 'estimating', 'estimation', 'estonia', 'estranged', 'eternal', 'eternity', 'ethan', 'ethanol', 'ethic', 'ethical', 'ethics', 'ethiopia', 'ethiopian', 'ethnic', 'ethnically', 'ethnicities', 'ethnicity', 'etymology', 'eugene', 'eurasia', 'eurasian', 'euro', 'europe', 'european', 'europeans', 'eva', 'evacuated', 'evacuation', 'evaluate', 'evaluated', 'evaluating', 'evaluation', 'evan', 'evangelical', 'evans', 'evaporation', 'eve', 'evening', 'evenings', 'evenly', 'event', 'events', 'eventual', 'eventually', 'everett', 'evergreen', 'everybody', 'everyday', 'evidence', 'evidenced', 'evident', 'evil', 'evolution', 'evolutionary', 'evolve', 'evolved', 'evolving', 'exacerbated', 'exact', 'exactly', 'exaggerated', 'exam', 'examination', 'examinations', 'examine', 'examined', 'examiner', 'examines', 'examining', 'example', 'examples', 'exams', 'excavated', 'excavations', 'exceed', 'exceeded', 'exceeding', 'exceeds', 'excel', 'excelled', 'excellence', 'excellent', 'exception', 'exceptional', 'exceptionally', 'exceptions', 'excess', 'excesses', 'excessive', 'excessively', 'exchange', 'exchanged', 'exchanges', 'exchanging', 'excited', 'excitement', 'exciting', 'exclude', 'excluded', 'excludes', 'excluding', 'exclusion', 'exclusive', 'exclusively', 'excreted', 'excuse', 'execute', 'executed', 'executing', 'execution', 'executions', 'executive', 'executives', 'exemplified', 'exempt', 'exemption', 'exercise', 'exercised', 'exercises', 'exercising', 'exert', 'exerted', 'exhaust', 'exhausted', 'exhaustion', 'exhibit', 'exhibited', 'exhibiting', 'exhibition', 'exhibitions', 'exhibits', 'exile', 'exiled', 'exist', 'existed', 'existence', 'existing', 'exists', 'exit', 'exodus', 'exotic', 'expand', 'expanded', 'expanding', 'expands', 'expansion', 'expatriate', 'expatriates', 'expect', 'expectancy', 'expectation', 'expectations', 'expected', 'expecting', 'expects', 'expedition', 'expeditions', 'expel', 'expelled', 'expenditure', 'expenditures', 'expense', 'expenses', 'expensive', 'experience', 'experienced', 'experiences', 'experiencing', 'experiment', 'experimental', 'experimentally', 'experimentation', 'experimented', 'experimenting', 'experiments', 'expert', 'expertise', 'experts', 'expired', 'explain', 'explained', 'explaining', 'explains', 'explanation', 'explanations', 'explicit', 'explicitly', 'explode', 'exploded', 'exploit', 'exploitation', 'exploited', 'exploiting', 'exploits', 'exploration', 'explore', 'explored', 'explorer', 'explorers', 'explores', 'exploring', 'explosion', 'explosions', 'explosive', 'explosives', 'expo', 'exponential', 'export', 'exported', 'exporter', 'exporting', 'exports', 'expose', 'exposed', 'exposing', 'exposition', 'exposure', 'express', 'expressed', 'expresses', 'expressing', 'expression', 'expressions', 'expressive', 'expulsion', 'extant', 'extend', 'extended', 'extending', 'extends', 'extension', 'extensions', 'extensive', 'extensively', 'extent', 'exterior', 'extermination', 'external', 'externally', 'extinct', 'extinction', 'extra', 'extract', 'extracted', 'extracting', 'extraction', 'extraordinarily', 'extraordinary', 'extras', 'extreme', 'extremely', 'extremes', 'eye', 'eyed', 'eyes', 'fabric', 'face', 'facebook', 'faced', 'faces', 'facial', 'facilitate', 'facilitated', 'facilitates', 'facilitating', 'facilities', 'facility', 'facing', 'fact', 'factbook', 'faction', 'factions', 'facto', 'factor', 'factories', 'factors', 'factory', 'facts', 'factual', 'faculty', 'fade', 'faded', 'fading', 'fail', 'failed', 'failing', 'fails', 'failure', 'failures', 'fair', 'fairly', 'fairy', 'faith', 'faithful', 'faiths', 'fake', 'fall', 'fallen', 'falling', 'fallout', 'falls', 'false', 'falsely', 'fame', 'famed', 'familiar', 'families', 'family', 'famine', 'famines', 'famous', 'famously', 'fan', 'fancy', 'fans', 'fantasies', 'fantastic', 'fantasy', 'far', 'fare', 'fared', 'farewell', 'farm', 'farmer', 'farmers', 'farming', 'farms', 'farther', 'fascinated', 'fascinating', 'fascination', 'fascism', 'fascist', 'fashion', 'fashionable', 'fashioned', 'fast', 'faster', 'fastest', 'fasting', 'fat', 'fatal', 'fatalities', 'fatally', 'fate', 'father', 'fathered', 'fathers', 'fatigue', 'fats', 'fatty', 'fault', 'faults', 'fauna', 'favor', 'favorable', 'favorably', 'favored', 'favoring', 'favorite', 'favorites', 'favors', 'favour', 'favourable', 'favoured', 'favourite', 'fbi', 'fda', 'fear', 'feared', 'fearful', 'fearing', 'fearless', 'fears', 'feasible', 'feast', 'feat', 'feathers', 'feature', 'featured', 'features', 'featuring', 'february', 'fed', 'federal', 'federally', 'federation', 'fee', 'feed', 'feedback', 'feeding', 'feeds', 'feel', 'feeling', 'feelings', 'feels', 'fees', 'feet', 'felix', 'fell', 'fellow', 'fellowship', 'felony', 'felt', 'female', 'females', 'feminine', 'feminism', 'feminist', 'fence', 'ferdinand', 'ferguson', 'fermentation', 'fermented', 'fern', 'fernando', 'ferries', 'ferry', 'fertile', 'fertility', 'fertilization', 'fertilizer', 'festival', 'festivals', 'festivities', 'fetal', 'fetus', 'feud', 'feudal', 'fever', 'fewer', 'fhm', 'fianc', 'fiber', 'fibers', 'fibre', 'fiction', 'fictional', 'fictionalized', 'fictitious', 'fidelity', 'field', 'fields', 'fierce', 'fiercely', 'fifa', 'fifteenth', 'fifth', 'fifths', 'fifty', 'fight', 'fighter', 'fighters', 'fighting', 'fights', 'figure', 'figured', 'figures', 'file', 'filed', 'files', 'filing', 'filipino', 'filled', 'filling', 'fills', 'film', 'filmed', 'filmfare', 'filming', 'filmmaker', 'filmmakers', 'filmmaking', 'films', 'filter', 'filtered', 'filtering', 'filters', 'final', 'finale', 'finalist', 'finalized', 'finally', 'finals', 'finance', 'financed', 'finances', 'financial', 'financially', 'financing', 'finding', 'findings', 'finds', 'fine', 'fined', 'finely', 'fines', 'finest', 'finger', 'fingers', 'finish', 'finished', 'finishes', 'finishing', 'finite', 'finland', 'finnish', 'firearms', 'fired', 'fires', 'fireworks', 'firing', 'firm', 'firmly', 'firms', 'firstly', 'fiscal', 'fish', 'fisher', 'fisheries', 'fishermen', 'fishing', 'fist', 'fit', 'fitness', 'fits', 'fitted', 'fitting', 'fitzgerald', 'fix', 'fixed', 'fixing', 'fixture', 'flag', 'flags', 'flagship', 'flamboyant', 'flame', 'flames', 'flash', 'flat', 'flattened', 'flavor', 'flavors', 'flaw', 'flawed', 'flaws', 'fled', 'fledged', 'flee', 'fleeing', 'fleet', 'fleets', 'flesh', 'flew', 'flexibility', 'flexible', 'flies', 'flight', 'flights', 'flip', 'float', 'floating', 'flood', 'flooded', 'flooding', 'floods', 'floor', 'floors', 'flop', 'flora', 'florence', 'florida', 'flour', 'flourish', 'flourished', 'flourishing', 'flow', 'flowed', 'flower', 'flowering', 'flowers', 'flowing', 'flown', 'flows', 'floyd', 'flu', 'fluctuations', 'fluent', 'fluid', 'fluids', 'flux', 'fly', 'flying', 'flynn', 'focal', 'focus', 'focused', 'focuses', 'focusing', 'fold', 'folded', 'folk', 'folklore', 'follow', 'followed', 'follower', 'followers', 'following', 'follows', 'fond', 'food', 'foods', 'foodstuffs', 'fool', 'foot', 'footage', 'football', 'footsteps', 'foray', 'forbade', 'forbes', 'forbid', 'forbidden', 'forbidding', 'force', 'forced', 'forces', 'forcibly', 'forcing', 'ford', 'fore', 'forecast', 'forecasts', 'forefront', 'forehead', 'foreign', 'foreigners', 'foremost', 'forensic', 'forerunner', 'forest', 'forested', 'forestry', 'forests', 'forever', 'forge', 'forged', 'forget', 'forgotten', 'form', 'formal', 'formalized', 'formally', 'format', 'formation', 'formations', 'formative', 'formats', 'formed', 'formidable', 'forming', 'forms', 'formula', 'formulas', 'formulated', 'formulation', 'fort', 'forth', 'forthcoming', 'fortifications', 'fortified', 'fortress', 'fortresses', 'forts', 'fortune', 'fortunes', 'forum', 'forums', 'forward', 'fossil', 'fossils', 'foster', 'fostered', 'fostering', 'fought', 'foul', 'foundation', 'foundations', 'founded', 'founder', 'founders', 'founding', 'fountain', 'fourteen', 'fourteenth', 'fourth', 'fox', 'fraction', 'fracture', 'fractured', 'fragile', 'fragment', 'fragmentation', 'fragmented', 'fragments', 'fragrance', 'frame', 'framed', 'frames', 'framework', 'frameworks', 'fran', 'franca', 'france', 'frances', 'franchise', 'franchises', 'francis', 'francisco', 'franco', 'frank', 'frankfurt', 'frankish', 'franklin', 'franz', 'fraser', 'fraternity', 'fraud', 'fraudulent', 'fred', 'freddie', 'frederick', 'free', 'freed', 'freedom', 'freedoms', 'freeing', 'freely', 'freeman', 'freeze', 'freezing', 'freight', 'french', 'frequencies', 'frequency', 'frequent', 'frequently', 'fresh', 'freshman', 'freshwater', 'freud', 'friction', 'friday', 'fried', 'friedman', 'friedrich', 'friend', 'friendly', 'friends', 'friendship', 'friendships', 'frightened', 'fritz', 'frog', 'frontal', 'frontier', 'frontiers', 'frontman', 'fronts', 'frost', 'frozen', 'fruit', 'fruits', 'frustrated', 'frustrating', 'frustration', 'fuck', 'fucking', 'fuel', 'fueled', 'fuelled', 'fuels', 'fugitive', 'fulfill', 'fulfilled', 'fulfilling', 'fuller', 'fully', 'fun', 'function', 'functional', 'functionality', 'functioned', 'functioning', 'functions', 'fund', 'fundamental', 'fundamentally', 'funded', 'funding', 'fundraiser', 'fundraising', 'funds', 'funeral', 'funerals', 'fungal', 'fungi', 'funk', 'funny', 'fur', 'furious', 'furniture', 'furthermore', 'fury', 'fuse', 'fused', 'fusion', 'futile', 'future', 'gabriel', 'gag', 'gaga', 'gain', 'gained', 'gaining', 'gains', 'gala', 'galaxy', 'gale', 'galleries', 'gallery', 'gallup', 'gamble', 'gambling', 'game', 'gameplay', 'games', 'gaming', 'gamma', 'gandhi', 'gang', 'gangs', 'gangster', 'gap', 'gaps', 'garage', 'garbage', 'garde', 'garden', 'gardens', 'gardner', 'garlic', 'garments', 'garner', 'garnered', 'garnering', 'garrison', 'garry', 'gary', 'gas', 'gaseous', 'gases', 'gasoline', 'gastrointestinal', 'gate', 'gates', 'gateway', 'gather', 'gathered', 'gatherers', 'gathering', 'gatherings', 'gauge', 'gave', 'gay', 'gazette', 'gdp', 'gear', 'gen', 'gender', 'gene', 'genera', 'general', 'generalized', 'generally', 'generals', 'generate', 'generated', 'generates', 'generating', 'generation', 'generations', 'generator', 'generators', 'generic', 'generous', 'genes', 'genesis', 'genetic', 'genetically', 'genetics', 'geneva', 'genital', 'genitals', 'genius', 'genocide', 'genome', 'genre', 'genres', 'gentle', 'gentleman', 'gently', 'genuine', 'genuinely', 'genus', 'geoffrey', 'geographic', 'geographical', 'geographically', 'geography', 'geological', 'geology', 'geometric', 'geometry', 'georg', 'george', 'georges', 'georgetown', 'georgia', 'georgian', 'gerald', 'gerard', 'german', 'germanic', 'germans', 'germany', 'gestation', 'gesture', 'gestures', 'gets', 'getting', 'ghana', 'ghetto', 'ghost', 'ghosts', 'giant', 'giants', 'gibson', 'gift', 'gifted', 'gifts', 'gig', 'gigantic', 'gigs', 'gilbert', 'giovanni', 'girl', 'girlfriend', 'girls', 'given', 'gives', 'giving', 'glacial', 'glaciers', 'glad', 'glamorous', 'glamour', 'gland', 'glands', 'glasgow', 'glass', 'glasses', 'glen', 'glenn', 'global', 'globalization', 'globally', 'globe', 'globes', 'gloria', 'glorious', 'glory', 'gloves', 'glucose', 'goal', 'goals', 'goat', 'goats', 'god', 'goddess', 'godfather', 'gods', 'goes', 'going', 'gold', 'goldberg', 'golden', 'goldman', 'golf', 'gone', 'gonna', 'good', 'goodbye', 'goodman', 'goodness', 'goods', 'goodwill', 'google', 'gordon', 'gore', 'gospel', 'gossip', 'got', 'gothic', 'gotten', 'gould', 'govern', 'governance', 'governed', 'governing', 'government', 'governmental', 'governments', 'governor', 'governors', 'gps', 'grab', 'grabbed', 'grace', 'grade', 'grades', 'gradient', 'gradual', 'gradually', 'graduate', 'graduated', 'graduates', 'graduating', 'graduation', 'graham', 'grain', 'grains', 'gram', 'grammar', 'grammy', 'grams', 'gran', 'granada', 'grand', 'grandchildren', 'granddaughter', 'grande', 'grandfather', 'grandmother', 'grandparents', 'grandson', 'granite', 'grant', 'granted', 'granting', 'grants', 'grape', 'grapes', 'graph', 'graphic', 'graphical', 'graphics', 'grasp', 'grass', 'grasses', 'grasslands', 'grateful', 'gratitude', 'grave', 'graves', 'gravitational', 'gravity', 'gray', 'grazing', 'great', 'greater', 'greatest', 'greatly', 'greco', 'greece', 'greed', 'greek', 'greeks', 'green', 'greene', 'greenhouse', 'greenland', 'greens', 'greenwich', 'greeted', 'greeting', 'greg', 'gregory', 'grew', 'grey', 'grid', 'grief', 'grievances', 'griffin', 'grim', 'grinding', 'grip', 'groove', 'gross', 'grossed', 'grossing', 'ground', 'grounded', 'grounds', 'group', 'grouped', 'grouping', 'groupings', 'groups', 'grove', 'grow', 'growing', 'grown', 'grows', 'growth', 'guarantee', 'guaranteed', 'guaranteeing', 'guarantees', 'guard', 'guarded', 'guardian', 'guardians', 'guarding', 'guards', 'guatemala', 'guerrilla', 'guess', 'guest', 'guests', 'guidance', 'guide', 'guided', 'guidelines', 'guides', 'guiding', 'guild', 'guilt', 'guilty', 'guinea', 'guinness', 'guitar', 'guitarist', 'guitars', 'gulf', 'gum', 'gun', 'gunpowder', 'guns', 'guru', 'gustav', 'gut', 'guy', 'guyana', 'guys', 'gym', 'gymnastics', 'habit', 'habitat', 'habitation', 'habitats', 'habits', 'habsburg', 'hadn', 'hague', 'hai', 'hail', 'hailed', 'hair', 'haired', 'haiti', 'hal', 'half', 'halfway', 'hall', 'hallmark', 'halls', 'halo', 'halt', 'halted', 'halves', 'ham', 'hamburg', 'hamilton', 'hamlet', 'hammer', 'hampered', 'hampshire', 'han', 'hancock', 'hand', 'handball', 'handed', 'handful', 'handing', 'handle', 'handled', 'handles', 'handling', 'hands', 'handsome', 'handwriting', 'handwritten', 'hang', 'hanged', 'hanging', 'hank', 'hanna', 'hans', 'happen', 'happened', 'happening', 'happens', 'happiness', 'happy', 'harassed', 'harassment', 'harbor', 'harbour', 'hard', 'hardcore', 'hardened', 'harder', 'hardly', 'hardship', 'hardware', 'hardy', 'hare', 'harlem', 'harm', 'harmful', 'harmless', 'harmony', 'harold', 'harper', 'harris', 'harrison', 'harry', 'harsh', 'harshly', 'hart', 'harvard', 'harvest', 'harvested', 'harvesting', 'harvey', 'hasn', 'hassan', 'hat', 'hate', 'hated', 'hatred', 'hats', 'haunted', 'haven', 'having', 'hawaii', 'hawk', 'hawkins', 'hawks', 'hay', 'hazard', 'hazardous', 'hazards', 'hbo', 'head', 'headache', 'headaches', 'headed', 'header', 'heading', 'headline', 'headlined', 'headlines', 'headlining', 'headquartered', 'headquarters', 'heads', 'heal', 'healing', 'health', 'healthcare', 'healthier', 'healthy', 'hear', 'heard', 'hearing', 'hearings', 'hears', 'heart', 'hearted', 'heartland', 'hearts', 'heat', 'heated', 'heath', 'heating', 'heaven', 'heavenly', 'heavier', 'heaviest', 'heavily', 'heavy', 'hebrew', 'hectares', 'heel', 'heels', 'hegemony', 'height', 'heightened', 'heights', 'heinrich', 'heir', 'heirs', 'held', 'helen', 'helena', 'helicopter', 'helicopters', 'helium', 'hell', 'hellenistic', 'hello', 'helmet', 'help', 'helped', 'helpful', 'helping', 'helps', 'hemisphere', 'henri', 'henry', 'hepatitis', 'herald', 'heralded', 'herbert', 'herbs', 'hercules', 'herd', 'hereditary', 'heritage', 'herman', 'hermann', 'hero', 'heroes', 'heroic', 'heroin', 'heroine', 'herzegovina', 'hesitant', 'heterosexual', 'hey', 'hiatus', 'hid', 'hidden', 'hide', 'hides', 'hiding', 'hierarchical', 'hierarchy', 'high', 'higher', 'highest', 'highland', 'highlands', 'highlight', 'highlighted', 'highlighting', 'highlights', 'highly', 'highs', 'highway', 'highways', 'hill', 'hillary', 'hills', 'hilly', 'hilton', 'hinder', 'hindered', 'hindi', 'hindu', 'hinduism', 'hindus', 'hint', 'hinted', 'hints', 'hip', 'hire', 'hired', 'hires', 'hiring', 'hispanic', 'hispanics', 'historian', 'historians', 'historic', 'historical', 'historically', 'histories', 'historiography', 'history', 'hit', 'hitler', 'hits', 'hitting', 'hiv', 'hobby', 'hoc', 'hockey', 'hoffman', 'hold', 'holder', 'holders', 'holding', 'holdings', 'holds', 'hole', 'holes', 'holiday', 'holidays', 'holland', 'hollow', 'holly', 'hollywood', 'holmes', 'holocaust', 'holy', 'homage', 'home', 'homeland', 'homeless', 'homer', 'homes', 'hometown', 'homicide', 'homo', 'homogeneous', 'homosexual', 'homosexuality', 'honduras', 'honest', 'honesty', 'honey', 'hong', 'honor', 'honorable', 'honorary', 'honored', 'honoring', 'honors', 'honour', 'honoured', 'honours', 'hood', 'hook', 'hooks', 'hoover', 'hop', 'hope', 'hoped', 'hopes', 'hoping', 'hopkins', 'horace', 'horizon', 'horizontal', 'horizontally', 'hormonal', 'hormone', 'hormones', 'horn', 'horns', 'horrible', 'horror', 'horse', 'horseback', 'horses', 'hospital', 'hospitalized', 'hospitals', 'host', 'hostage', 'hosted', 'hostile', 'hostilities', 'hostility', 'hosting', 'hosts', 'hot', 'hotel', 'hotels', 'hottest', 'hour', 'hours', 'house', 'housed', 'household', 'households', 'houses', 'housing', 'houston', 'howard', 'html', 'http', 'hub', 'hubert', 'hudson', 'huge', 'hugely', 'hugh', 'hughes', 'hugo', 'hull', 'human', 'humane', 'humanist', 'humanitarian', 'humanities', 'humanity', 'humankind', 'humans', 'humble', 'humid', 'humidity', 'humiliating', 'humiliation', 'humor', 'humorous', 'humour', 'hundreds', 'hung', 'hungarian', 'hungary', 'hunger', 'hungry', 'hunt', 'hunted', 'hunter', 'hunters', 'hunting', 'hurricane', 'hurricanes', 'hurt', 'husband', 'husbands', 'hussein', 'hybrid', 'hybrids', 'hyde', 'hydraulic', 'hydrocarbons', 'hydroelectric', 'hydrogen', 'hygiene', 'hymns', 'hype', 'hypotheses', 'hypothesis', 'hypothesized', 'hypothetical', 'ian', 'iberian', 'ibm', 'ibn', 'icc', 'ice', 'iceland', 'icelandic', 'icon', 'iconic', 'icons', 'idaho', 'idea', 'ideal', 'ideally', 'ideals', 'ideas', 'identical', 'identifiable', 'identification', 'identified', 'identifies', 'identify', 'identifying', 'identities', 'identity', 'ideological', 'ideologies', 'ideology', 'idol', 'idols', 'ieee', 'ign', 'ignited', 'ignorance', 'ignorant', 'ignore', 'ignored', 'ignoring', 'iii', 'ill', 'illegal', 'illegally', 'illegitimate', 'illicit', 'illinois', 'illiterate', 'illness', 'illnesses', 'illuminated', 'illusion', 'illustrate', 'illustrated', 'illustrates', 'illustration', 'illustrations', 'image', 'imagery', 'images', 'imaginary', 'imagination', 'imaginative', 'imagine', 'imagined', 'imaging', 'imbalance', 'imf', 'imitate', 'imitated', 'imitation', 'immature', 'immediate', 'immediately', 'immense', 'immensely', 'immigrant', 'immigrants', 'immigrated', 'immigration', 'imminent', 'immoral', 'immortal', 'immortality', 'immune', 'immunity', 'impact', 'impacted', 'impacts', 'impaired', 'impairment', 'impending', 'imperative', 'imperfect', 'imperial', 'imperialism', 'impetus', 'implement', 'implementation', 'implementations', 'implemented', 'implementing', 'implements', 'implicated', 'implication', 'implications', 'implicit', 'implicitly', 'implied', 'implies', 'imply', 'implying', 'import', 'importance', 'important', 'importantly', 'importation', 'imported', 'importer', 'importing', 'imports', 'impose', 'imposed', 'imposing', 'imposition', 'impossible', 'impoverished', 'impractical', 'impress', 'impressed', 'impression', 'impressions', 'impressive', 'imprint', 'imprisoned', 'imprisonment', 'impromptu', 'improper', 'improve', 'improved', 'improvement', 'improvements', 'improves', 'improving', 'improvised', 'impulse', 'impurities', 'inability', 'inaccurate', 'inactive', 'inadequate', 'inadvertently', 'inappropriate', 'inaugural', 'inaugurated', 'inauguration', 'incapable', 'incarceration', 'incarnation', 'incarnations', 'incentive', 'incentives', 'inception', 'incest', 'inch', 'inches', 'incidence', 'incident', 'incidents', 'inclined', 'include', 'included', 'includes', 'including', 'inclusion', 'inclusive', 'income', 'incomes', 'incoming', 'incompatible', 'incomplete', 'inconclusive', 'inconsistent', 'incorporate', 'incorporated', 'incorporates', 'incorporating', 'incorporation', 'incorrect', 'incorrectly', 'increase', 'increased', 'increases', 'increasing', 'increasingly', 'incredible', 'incredibly', 'incumbent', 'incurred', 'incursions', 'indefinite', 'indefinitely', 'independence', 'independent', 'independently', 'index', 'india', 'indian', 'indiana', 'indianapolis', 'indians', 'indicate', 'indicated', 'indicates', 'indicating', 'indication', 'indications', 'indicative', 'indicator', 'indicators', 'indictment', 'indie', 'indies', 'indifference', 'indifferent', 'indigenous', 'indirect', 'indirectly', 'individual', 'individually', 'individuals', 'indo', 'indonesia', 'indonesian', 'indoor', 'induce', 'induced', 'induces', 'inducing', 'inducted', 'induction', 'industrial', 'industrialisation', 'industrialised', 'industrialization', 'industrialized', 'industries', 'industry', 'ineffective', 'inefficient', 'inequality', 'inert', 'inevitable', 'inevitably', 'inexpensive', 'inexperienced', 'infamous', 'infancy', 'infant', 'infantry', 'infants', 'infect', 'infected', 'infection', 'infections', 'infectious', 'inferior', 'inferred', 'infinite', 'infinity', 'inflammation', 'inflammatory', 'inflation', 'inflicted', 'influence', 'influenced', 'influences', 'influencing', 'influential', 'influenza', 'influx', 'inform', 'informal', 'informally', 'information', 'informed', 'informing', 'informs', 'infrared', 'infrastructure', 'infrequent', 'infringement', 'infused', 'ingested', 'ingestion', 'ingredient', 'ingredients', 'inhabit', 'inhabitants', 'inhabited', 'inherent', 'inherently', 'inherit', 'inheritance', 'inherited', 'inhibit', 'inhibiting', 'inhibition', 'inhibitor', 'inhibitors', 'inhibits', 'initial', 'initially', 'initials', 'initiate', 'initiated', 'initiating', 'initiation', 'initiative', 'initiatives', 'injected', 'injection', 'injured', 'injuries', 'injury', 'injustice', 'ink', 'inland', 'inmates', 'inn', 'innate', 'inner', 'innocence', 'innocent', 'innovation', 'innovations', 'innovative', 'inorganic', 'input', 'inputs', 'inquiry', 'ins', 'insane', 'insanity', 'inscribed', 'inscription', 'inscriptions', 'insect', 'insects', 'insecure', 'insert', 'inserted', 'inserting', 'insertion', 'inside', 'insight', 'insights', 'insignificant', 'insisted', 'insistence', 'insisting', 'insists', 'insomnia', 'inspection', 'inspector', 'inspiration', 'inspire', 'inspired', 'inspiring', 'instability', 'install', 'installation', 'installations', 'installed', 'installing', 'installment', 'instance', 'instances', 'instant', 'instantly', 'instead', 'instigated', 'instinct', 'institute', 'instituted', 'institutes', 'institution', 'institutional', 'institutions', 'instructed', 'instruction', 'instructions', 'instructor', 'instrument', 'instrumental', 'instrumentation', 'instruments', 'insufficient', 'insulin', 'insult', 'insulting', 'insults', 'insurance', 'insurgency', 'insurrection', 'intact', 'intake', 'integral', 'integrate', 'integrated', 'integrating', 'integration', 'integrity', 'intel', 'intellect', 'intellectual', 'intellectuals', 'intelligence', 'intelligent', 'intend', 'intended', 'intending', 'intends', 'intense', 'intensely', 'intensified', 'intensity', 'intensive', 'intent', 'intention', 'intentional', 'intentionally', 'intentions', 'inter', 'interact', 'interacting', 'interaction', 'interactions', 'interactive', 'interacts', 'intercepted', 'interchange', 'interchangeably', 'interconnected', 'intercontinental', 'intercourse', 'interested', 'interesting', 'interests', 'interface', 'interfaces', 'interfere', 'interfered', 'interference', 'interfering', 'intergovernmental', 'interim', 'interior', 'interiors', 'intermediary', 'intermediate', 'intermittent', 'intermittently', 'internal', 'internally', 'international', 'internationally', 'internet', 'interpersonal', 'interpret', 'interpretation', 'interpretations', 'interpreted', 'interpreter', 'interpreting', 'interred', 'interrogation', 'interrupt', 'interrupted', 'intersection', 'interstate', 'intertwined', 'interval', 'intervals', 'intervene', 'intervened', 'intervening', 'intervention', 'interventions', 'interview', 'interviewed', 'interviewer', 'interviews', 'intestinal', 'intestine', 'intestines', 'intimate', 'intimidation', 'intolerance', 'intra', 'intravenous', 'intricate', 'intrigued', 'intrinsic', 'introduce', 'introduced', 'introduces', 'introducing', 'introduction', 'intuitive', 'invade', 'invaded', 'invaders', 'invading', 'invalid', 'invariably', 'invasion', 'invasions', 'invasive', 'invented', 'invention', 'inventions', 'inventor', 'inventors', 'inventory', 'inverse', 'invertebrates', 'inverted', 'invest', 'invested', 'investigate', 'investigated', 'investigating', 'investigation', 'investigations', 'investigative', 'investigators', 'investing', 'investment', 'investments', 'investor', 'investors', 'invisible', 'invitation', 'invite', 'invited', 'invites', 'inviting', 'invoked', 'involuntary', 'involve', 'involved', 'involvement', 'involves', 'involving', 'inward', 'ion', 'ions', 'ios', 'iowa', 'iphone', 'iran', 'iranian', 'iraq', 'iraqi', 'ireland', 'irish', 'iron', 'ironically', 'irony', 'irrational', 'irreconcilable', 'irregular', 'irrelevant', 'irreversible', 'irrigation', 'irritation', 'irving', 'isaac', 'islam', 'islamic', 'islamist', 'island', 'islands', 'isle', 'isn', 'iso', 'isolate', 'isolated', 'isolation', 'isotopes', 'israel', 'israeli', 'issue', 'issued', 'issues', 'issuing', 'istanbul', 'italian', 'italians', 'italy', 'item', 'items', 'itunes', 'itv', 'ivan', 'ivory', 'ivy', 'jack', 'jacket', 'jackie', 'jackson', 'jacob', 'jacqueline', 'jacques', 'jail', 'jailed', 'jake', 'jam', 'jamaica', 'james', 'jamie', 'jan', 'jane', 'janeiro', 'janet', 'january', 'japan', 'japanese', 'jason', 'java', 'jaw', 'jay', 'jazz', 'jealous', 'jealousy', 'jean', 'jeans', 'jeff', 'jefferson', 'jeffrey', 'jehovah', 'jelly', 'jennifer', 'jenny', 'jeopardy', 'jeremy', 'jerome', 'jerry', 'jersey', 'jerusalem', 'jesse', 'jessica', 'jesuit', 'jesus', 'jet', 'jets', 'jew', 'jewellery', 'jewelry', 'jewish', 'jews', 'jfk', 'jim', 'jimmy', 'joan', 'job', 'jobs', 'joe', 'joel', 'joey', 'johann', 'johannes', 'john', 'johnny', 'johns', 'johnson', 'join', 'joined', 'joining', 'joins', 'joint', 'jointly', 'joints', 'joke', 'joked', 'jokes', 'jon', 'jonathan', 'jones', 'jordan', 'jorge', 'jos', 'jose', 'josef', 'joseph', 'josh', 'joshua', 'journal', 'journalism', 'journalist', 'journalists', 'journals', 'journey', 'journeys', 'joy', 'joyce', 'jpg', 'juan', 'judaism', 'judd', 'jude', 'judge', 'judged', 'judgement', 'judges', 'judging', 'judgment', 'judicial', 'judiciary', 'judith', 'judy', 'juice', 'jules', 'julia', 'julian', 'julie', 'juliet', 'julius', 'july', 'jump', 'jumped', 'jumping', 'jumps', 'junction', 'june', 'jung', 'jungle', 'junior', 'jupiter', 'jurisdiction', 'jurisdictions', 'jury', 'just', 'justice', 'justices', 'justification', 'justified', 'justify', 'justin', 'juvenile', 'kane', 'kansas', 'kant', 'kanye', 'kapoor', 'karen', 'karl', 'kate', 'katherine', 'kathleen', 'kathy', 'katie', 'katrina', 'katy', 'kay', 'kazakhstan', 'keen', 'keeper', 'keeping', 'keeps', 'keith', 'kelly', 'ken', 'kennedy', 'kenneth', 'kenny', 'kent', 'kentucky', 'kenya', 'kept', 'kernel', 'kerry', 'kevin', 'key', 'keyboard', 'keyboards', 'keynote', 'keys', 'khan', 'kick', 'kicked', 'kicking', 'kicks', 'kid', 'kidnapped', 'kidnapping', 'kidney', 'kidneys', 'kids', 'kill', 'killed', 'killer', 'killers', 'killing', 'killings', 'kills', 'kilograms', 'kilometer', 'kilometers', 'kilometres', 'kim', 'kind', 'kindergarten', 'kinds', 'kinetic', 'king', 'kingdom', 'kingdoms', 'kings', 'kingston', 'kirk', 'kiss', 'kissing', 'kit', 'kitchen', 'kits', 'klein', 'knee', 'knees', 'knew', 'knife', 'knight', 'knights', 'knock', 'knocked', 'knockout', 'know', 'knowing', 'knowledge', 'known', 'knows', 'kong', 'korea', 'korean', 'kosovo', 'krishna', 'kristen', 'kumar', 'kurt', 'kuwait', 'kyle', 'kyoto', 'lab', 'label', 'labeled', 'labeling', 'labelled', 'labels', 'labor', 'laboratories', 'laboratory', 'laborers', 'labour', 'labourers', 'labs', 'lack', 'lacked', 'lacking', 'lacks', 'ladder', 'laden', 'ladies', 'lady', 'lag', 'laid', 'lake', 'lakes', 'lamb', 'lambert', 'lamp', 'lancaster', 'lance', 'land', 'landed', 'landing', 'landings', 'landmark', 'landmarks', 'landmass', 'landowners', 'lands', 'landscape', 'landscapes', 'landslide', 'lane', 'lanes', 'lang', 'language', 'languages', 'lanka', 'laos', 'lap', 'laptop', 'laptops', 'large', 'largely', 'larger', 'largest', 'larry', 'las', 'laser', 'lasted', 'lasting', 'lastly', 'lasts', 'late', 'latent', 'later', 'lateral', 'latest', 'latin', 'latino', 'latitude', 'latitudes', 'latvia', 'lauded', 'laugh', 'laughing', 'laughs', 'laughter', 'launch', 'launched', 'launching', 'laura', 'laureate', 'lauren', 'laurence', 'lavish', 'law', 'lawful', 'lawn', 'lawrence', 'laws', 'lawsuit', 'lawsuits', 'lawyer', 'lawyers', 'lay', 'layer', 'layered', 'layers', 'laying', 'layout', 'lays', 'lazy', 'lead', 'leader', 'leaders', 'leadership', 'leading', 'leads', 'leaf', 'league', 'leagues', 'leak', 'leaked', 'leaks', 'lean', 'leaning', 'leap', 'learn', 'learned', 'learning', 'learns', 'lease', 'leased', 'leather', 'leave', 'leaves', 'leaving', 'lebanese', 'lebanon', 'lecture', 'lectures', 'led', 'lee', 'leeds', 'left', 'leftist', 'leg', 'legacy', 'legal', 'legality', 'legally', 'legend', 'legendary', 'legends', 'legion', 'legislation', 'legislative', 'legislature', 'legislatures', 'legitimacy', 'legitimate', 'legs', 'leisure', 'lemon', 'lend', 'lending', 'length', 'lengths', 'lengthy', 'lenin', 'lennon', 'lens', 'lent', 'leo', 'leon', 'leonard', 'leonardo', 'leone', 'leopard', 'leopold', 'les', 'lesbian', 'lesions', 'leslie', 'lesser', 'lesson', 'lessons', 'let', 'lethal', 'lets', 'letter', 'letterman', 'letters', 'letting', 'levant', 'level', 'levels', 'leverage', 'levy', 'lewis', 'lexicon', 'lgbt', 'liability', 'liable', 'liaison', 'libel', 'liberal', 'liberalization', 'liberals', 'liberate', 'liberated', 'liberation', 'liberties', 'liberty', 'libraries', 'library', 'libya', 'licence', 'license', 'licensed', 'licenses', 'licensing', 'lie', 'lied', 'lies', 'lieu', 'lieutenant', 'life', 'lifelong', 'lifespan', 'lifestyle', 'lifestyles', 'lifetime', 'lift', 'lifted', 'lifting', 'light', 'lighter', 'lighting', 'lightly', 'lightning', 'lights', 'lightweight', 'like', 'liked', 'likelihood', 'likely', 'likened', 'likeness', 'likes', 'likewise', 'lil', 'lily', 'lima', 'limb', 'limbs', 'lime', 'limestone', 'limit', 'limitation', 'limitations', 'limited', 'limiting', 'limits', 'lincoln', 'linda', 'lindsay', 'line', 'lineage', 'lineages', 'linear', 'lined', 'lines', 'lineup', 'lingua', 'linguistic', 'lining', 'link', 'linked', 'linking', 'links', 'linux', 'lion', 'lionel', 'lions', 'lip', 'lipid', 'lipids', 'lips', 'liquid', 'liquids', 'liquor', 'lisa', 'list', 'listed', 'listen', 'listened', 'listeners', 'listening', 'listing', 'lists', 'lit', 'literacy', 'literal', 'literally', 'literary', 'literate', 'literature', 'lithium', 'lithuania', 'lithuanian', 'litigation', 'little', 'live', 'lived', 'lively', 'liver', 'liverpool', 'lives', 'livestock', 'living', 'liz', 'ller', 'lloyd', 'load', 'loaded', 'loading', 'loads', 'loan', 'loans', 'lobbied', 'lobby', 'lobbying', 'lobe', 'local', 'localized', 'locally', 'locals', 'locate', 'located', 'location', 'locations', 'lock', 'locked', 'locking', 'locks', 'lodge', 'lodged', 'log', 'logan', 'logging', 'logic', 'logical', 'logistics', 'logo', 'logos', 'london', 'lone', 'lonely', 'long', 'longer', 'longest', 'longevity', 'longitude', 'longitudes', 'longitudinal', 'longstanding', 'longtime', 'look', 'looked', 'looking', 'looks', 'loop', 'loops', 'loose', 'loosely', 'lopez', 'lord', 'lords', 'los', 'lose', 'loses', 'losing', 'loss', 'losses', 'lost', 'lot', 'lots', 'lotus', 'lou', 'loud', 'louis', 'louise', 'louisiana', 'love', 'loved', 'lovely', 'lover', 'lovers', 'loves', 'loving', 'low', 'lower', 'lowered', 'lowering', 'lowest', 'lowland', 'lowlands', 'loyal', 'loyalty', 'luc', 'lucas', 'luck', 'lucky', 'lucrative', 'lucy', 'ludwig', 'luis', 'luke', 'lunar', 'lunch', 'lung', 'lungs', 'lure', 'lust', 'luther', 'lutheran', 'luxembourg', 'luxury', 'lying', 'lynch', 'lyndon', 'lynn', 'lyon', 'lyric', 'lyrical', 'lyrics', 'mac', 'macdonald', 'macedonia', 'machine', 'machinery', 'machines', 'mad', 'madagascar', 'madame', 'madison', 'madness', 'madonna', 'madrid', 'mae', 'mafia', 'magazine', 'magazines', 'maggie', 'magic', 'magical', 'magna', 'magnate', 'magnesium', 'magnet', 'magnetic', 'magnificent', 'magnitude', 'magnus', 'maid', 'maiden', 'mail', 'main', 'maine', 'mainland', 'mainly', 'mainstream', 'maintain', 'maintained', 'maintaining', 'maintains', 'maintenance', 'maize', 'majesty', 'major', 'majority', 'make', 'maker', 'makers', 'makes', 'makeup', 'making', 'malaria', 'malay', 'malaysia', 'malcolm', 'male', 'males', 'malibu', 'malicious', 'mall', 'malnutrition', 'malta', 'mammal', 'mammalian', 'mammals', 'man', 'manage', 'managed', 'management', 'manager', 'managers', 'manages', 'managing', 'manchester', 'mandarin', 'mandate', 'mandated', 'mandates', 'mandatory', 'maneuver', 'manga', 'manhattan', 'mania', 'manifest', 'manifestation', 'manifestations', 'manifested', 'manifesto', 'manila', 'manipulate', 'manipulated', 'manipulating', 'manipulation', 'mankind', 'mann', 'manned', 'manner', 'manor', 'manpower', 'mansion', 'mantle', 'manual', 'manually', 'manuel', 'manufacture', 'manufactured', 'manufacturer', 'manufacturers', 'manufacturing', 'manuscript', 'manuscripts', 'mao', 'map', 'maple', 'mapped', 'mapping', 'maps', 'mar', 'marathon', 'marble', 'marc', 'march', 'marched', 'marching', 'marco', 'marcus', 'margaret', 'margin', 'marginal', 'margins', 'maria', 'marie', 'marijuana', 'marilyn', 'marina', 'marine', 'marines', 'mario', 'marion', 'marital', 'maritime', 'mark', 'marked', 'markedly', 'marker', 'markers', 'market', 'marketed', 'marketing', 'marketplace', 'markets', 'marking', 'markings', 'marks', 'marred', 'marriage', 'marriages', 'married', 'marries', 'marry', 'marrying', 'mars', 'marsh', 'marshal', 'marshall', 'mart', 'martha', 'martial', 'martin', 'martyr', 'martyrs', 'marvel', 'marvin', 'marx', 'marxism', 'marxist', 'mary', 'maryland', 'mascot', 'masculine', 'mask', 'masked', 'masks', 'mason', 'mass', 'massachusetts', 'massacre', 'massacres', 'masses', 'massive', 'master', 'mastered', 'masterpiece', 'masters', 'mastery', 'match', 'matched', 'matches', 'matching', 'mate', 'mater', 'material', 'materials', 'maternal', 'mates', 'math', 'mathematical', 'mathematician', 'mathematics', 'mating', 'matrix', 'matt', 'matter', 'matters', 'matthew', 'maturation', 'mature', 'matured', 'maturity', 'maurice', 'max', 'maxim', 'maximize', 'maximum', 'maxwell', 'maya', 'maybe', 'mayer', 'mayor', 'mccain', 'mccarthy', 'mccartney', 'mcdonald', 'mcqueen', 'meal', 'meals', 'mean', 'meaning', 'meaningful', 'meanings', 'means', 'meant', 'meantime', 'measurable', 'measure', 'measured', 'measurement', 'measurements', 'measures', 'measuring', 'meat', 'meats', 'mecca', 'mechanic', 'mechanical', 'mechanics', 'mechanism', 'mechanisms', 'medal', 'medals', 'media', 'median', 'mediate', 'mediated', 'medical', 'medically', 'medication', 'medications', 'medicinal', 'medicine', 'medicines', 'medieval', 'medina', 'mediocre', 'meditation', 'mediterranean', 'medium', 'meet', 'meeting', 'meetings', 'meets', 'mega', 'mel', 'melancholy', 'melbourne', 'melissa', 'melodies', 'melody', 'melt', 'melting', 'member', 'members', 'membership', 'membrane', 'membranes', 'memo', 'memoir', 'memoirs', 'memorable', 'memorandum', 'memorial', 'memorials', 'memories', 'memory', 'memphis', 'men', 'mental', 'mentally', 'mention', 'mentioned', 'mentioning', 'mentions', 'mentor', 'menu', 'mercedes', 'mercenaries', 'merchandise', 'merchandising', 'merchant', 'merchants', 'mercury', 'mercy', 'mere', 'merely', 'merge', 'merged', 'merger', 'merging', 'merit', 'merits', 'merry', 'mesopotamia', 'message', 'messages', 'messaging', 'messenger', 'met', 'meta', 'metabolic', 'metabolism', 'metacritic', 'metal', 'metallic', 'metals', 'metaphor', 'metaphors', 'metaphysical', 'metaphysics', 'meter', 'meters', 'methane', 'method', 'methodist', 'methodology', 'methods', 'metre', 'metres', 'metric', 'metro', 'metropolis', 'metropolitan', 'mexican', 'mexico', 'meyer', 'mgm', 'mia', 'miami', 'mice', 'michael', 'michel', 'michelle', 'michigan', 'mick', 'mickey', 'micro', 'microorganisms', 'microphone', 'microscope', 'microscopic', 'microsoft', 'microwave', 'mid', 'middle', 'midnight', 'midst', 'midway', 'midwest', 'midwestern', 'mighty', 'migrant', 'migrants', 'migrate', 'migrated', 'migrating', 'migration', 'migrations', 'miguel', 'mike', 'mikhail', 'milan', 'mild', 'mildly', 'mile', 'miles', 'milestone', 'militant', 'militarily', 'military', 'militia', 'militias', 'milk', 'millennia', 'millennium', 'miller', 'million', 'millionaire', 'millions', 'mills', 'milton', 'milwaukee', 'mimic', 'mimicking', 'min', 'mind', 'minded', 'minds', 'mineral', 'minerals', 'miners', 'mines', 'ming', 'mini', 'miniature', 'minimal', 'minimize', 'minimum', 'mining', 'miniseries', 'minister', 'ministerial', 'ministers', 'ministries', 'ministry', 'minnesota', 'minor', 'minorities', 'minority', 'minors', 'mint', 'minus', 'minute', 'minutes', 'mir', 'miracle', 'miranda', 'mirror', 'mirrored', 'mirrors', 'mis', 'miscarriage', 'misconception', 'misconduct', 'misdemeanor', 'misleading', 'miss', 'missed', 'missile', 'missiles', 'missing', 'mission', 'missionaries', 'missionary', 'missions', 'mississippi', 'missouri', 'mistake', 'mistaken', 'mistakenly', 'mistakes', 'mistress', 'misunderstood', 'misuse', 'mit', 'mitchell', 'mitigate', 'mitigation', 'mitochondrial', 'mix', 'mixed', 'mixes', 'mixing', 'mixtape', 'mixture', 'mob', 'mobile', 'mobility', 'mobilized', 'mock', 'mocked', 'mode', 'model', 'modeled', 'modeling', 'modelled', 'modelling', 'models', 'moderate', 'moderately', 'modern', 'modernist', 'modernization', 'modernized', 'modes', 'modest', 'modification', 'modifications', 'modified', 'modify', 'modifying', 'module', 'mohammed', 'moist', 'moisture', 'mold', 'molecular', 'molecule', 'molecules', 'molly', 'mom', 'moment', 'moments', 'momentum', 'mon', 'monarch', 'monarchs', 'monarchy', 'monasteries', 'monastery', 'monastic', 'monday', 'monetary', 'money', 'mongol', 'mongolia', 'mongols', 'monica', 'moniker', 'monitor', 'monitored', 'monitoring', 'monitors', 'monk', 'monkey', 'monkeys', 'monks', 'monopoly', 'monroe', 'monsoon', 'monster', 'monsters', 'montana', 'monte', 'montgomery', 'month', 'monthly', 'months', 'montreal', 'monument', 'monumental', 'monuments', 'mood', 'moody', 'moon', 'moore', 'moral', 'morale', 'morality', 'morally', 'morals', 'morgan', 'morning', 'moroccan', 'morocco', 'morphine', 'morphological', 'morphology', 'morris', 'morrison', 'morse', 'mortal', 'mortality', 'mortar', 'mosaic', 'moscow', 'moses', 'mosque', 'mosques', 'moss', 'mother', 'mothers', 'motif', 'motifs', 'motion', 'motions', 'motivated', 'motivation', 'motivations', 'motive', 'motives', 'motor', 'motorcycle', 'motors', 'motto', 'mount', 'mountain', 'mountainous', 'mountains', 'mounted', 'mounting', 'mourning', 'mouse', 'mouth', 'mouths', 'moved', 'movement', 'movements', 'moves', 'movie', 'movies', 'moving', 'mozambique', 'mph', 'mri', 'mrs', 'mtv', 'mud', 'muhammad', 'multi', 'multimedia', 'multinational', 'multiplayer', 'multiple', 'multitude', 'mumbai', 'munich', 'municipal', 'municipalities', 'municipality', 'murder', 'murdered', 'murderer', 'murdering', 'murders', 'murphy', 'murray', 'muscle', 'muscles', 'muscular', 'muse', 'museum', 'museums', 'music', 'musical', 'musically', 'musicals', 'musician', 'musicians', 'muslim', 'muslims', 'mutation', 'mutations', 'mute', 'mutilation', 'mutiny', 'mutual', 'mutually', 'myers', 'myriad', 'myspace', 'mysteries', 'mysterious', 'mystery', 'mystic', 'mystical', 'mysticism', 'myth', 'mythical', 'mythological', 'mythology', 'myths', 'naacp', 'nacional', 'nadu', 'nails', 'naive', 'naked', 'named', 'names', 'namesake', 'namibia', 'naming', 'nancy', 'naples', 'napoleon', 'napoleonic', 'narcotics', 'narrated', 'narration', 'narrative', 'narratives', 'narrator', 'narrow', 'narrowed', 'narrower', 'narrowly', 'nas', 'nasa', 'nasal', 'nascent', 'nashville', 'natalie', 'nathan', 'nation', 'national', 'nationalism', 'nationalist', 'nationalists', 'nationalities', 'nationality', 'nationally', 'nationals', 'nations', 'nationwide', 'native', 'natives', 'nato', 'natural', 'naturalist', 'naturally', 'nature', 'nausea', 'naval', 'navigable', 'navigate', 'navigation', 'navy', 'nazi', 'nazis', 'nba', 'nbc', 'ncaa', 'neal', 'near', 'nearby', 'nearest', 'nearly', 'nebraska', 'necessarily', 'necessary', 'necessity', 'neck', 'need', 'needed', 'needing', 'needle', 'needs', 'negative', 'negatively', 'neglect', 'neglected', 'negligible', 'negotiate', 'negotiated', 'negotiating', 'negotiation', 'negotiations', 'negro', 'neighbor', 'neighborhood', 'neighborhoods', 'neighboring', 'neighbors', 'neighbour', 'neighbourhood', 'neighbouring', 'neighbours', 'neil', 'neill', 'nelson', 'nemesis', 'neo', 'neolithic', 'neon', 'nepal', 'nephew', 'nerve', 'nerves', 'nervous', 'net', 'netherlands', 'nets', 'network', 'networking', 'networks', 'neural', 'neurological', 'neurons', 'neutral', 'neutrality', 'nevada', 'new', 'newborn', 'newcomer', 'newer', 'newest', 'newfoundland', 'newly', 'newman', 'news', 'newspaper', 'newspapers', 'newsweek', 'newton', 'nfl', 'ngo', 'ngos', 'nicaragua', 'nice', 'niche', 'nicholas', 'nicholson', 'nick', 'nickel', 'nickelodeon', 'nickname', 'nicknamed', 'nicknames', 'nicolas', 'nicole', 'niece', 'nielsen', 'nigeria', 'night', 'nightclub', 'nightly', 'nightmare', 'nights', 'nike', 'nikita', 'nile', 'nina', 'nineteen', 'nineteenth', 'ninety', 'nintendo', 'ninth', 'niro', 'nitrogen', 'nixon', 'noah', 'nobel', 'nobility', 'noble', 'nobles', 'node', 'nodes', 'noir', 'noise', 'nokia', 'nolan', 'nomadic', 'nomenclature', 'nominal', 'nominally', 'nominated', 'nomination', 'nominations', 'nominee', 'nominees', 'non', 'nonetheless', 'nonprofit', 'noon', 'nordic', 'norfolk', 'norm', 'normal', 'normally', 'norman', 'normandy', 'norms', 'norse', 'north', 'northeast', 'northeastern', 'northern', 'northernmost', 'northwest', 'northwestern', 'norton', 'norway', 'norwegian', 'nose', 'notable', 'notably', 'notation', 'note', 'noted', 'notes', 'noteworthy', 'notice', 'noticeable', 'noticeably', 'noticed', 'notices', 'notified', 'noting', 'notion', 'notions', 'notoriety', 'notorious', 'notoriously', 'noun', 'nova', 'novel', 'novelist', 'novels', 'novelty', 'november', 'nowadays', 'nuclear', 'nuclei', 'nucleus', 'nude', 'nudity', 'null', 'number', 'numbered', 'numbering', 'numbers', 'numerical', 'numerous', 'nuremberg', 'nurse', 'nursery', 'nurses', 'nursing', 'nutrient', 'nutrients', 'nutrition', 'nutritional', 'nuts', 'nyc', 'oak', 'oakland', 'oath', 'obama', 'obedience', 'obese', 'obesity', 'obey', 'object', 'objected', 'objection', 'objections', 'objective', 'objectives', 'objects', 'obligation', 'obligations', 'obliged', 'obscure', 'obscured', 'obscurity', 'observable', 'observation', 'observational', 'observations', 'observatory', 'observe', 'observed', 'observer', 'observers', 'observes', 'observing', 'obsessed', 'obsession', 'obsessive', 'obsolete', 'obstacle', 'obstacles', 'obtain', 'obtained', 'obtaining', 'obvious', 'obviously', 'occasion', 'occasional', 'occasionally', 'occasions', 'occupation', 'occupational', 'occupations', 'occupied', 'occupies', 'occupy', 'occupying', 'occur', 'occurred', 'occurrence', 'occurrences', 'occurring', 'occurs', 'ocean', 'oceania', 'oceanic', 'oceans', 'october', 'odd', 'odds', 'odyssey', 'oecd', 'offence', 'offended', 'offense', 'offenses', 'offensive', 'offer', 'offered', 'offering', 'offerings', 'offers', 'office', 'officer', 'officers', 'offices', 'official', 'officially', 'officials', 'offs', 'offset', 'offshore', 'offspring', 'ohio', 'oil', 'oils', 'ois', 'okay', 'oklahoma', 'old', 'older', 'oldest', 'olds', 'olive', 'oliver', 'olivia', 'olivier', 'olympia', 'olympic', 'olympics', 'omega', 'omitted', 'ones', 'oneself', 'ongoing', 'online', 'onscreen', 'onset', 'onstage', 'ontario', 'onward', 'onwards', 'open', 'opened', 'opening', 'openings', 'openly', 'openness', 'opens', 'opera', 'operate', 'operated', 'operates', 'operating', 'operation', 'operational', 'operations', 'operative', 'operator', 'operators', 'opined', 'opinion', 'opinions', 'opponent', 'opponents', 'opportunities', 'opportunity', 'oppose', 'opposed', 'opposes', 'opposing', 'opposite', 'opposition', 'oppression', 'oprah', 'opt', 'opted', 'optic', 'optical', 'optics', 'optimal', 'optimism', 'optimistic', 'option', 'optional', 'options', 'oracle', 'oral', 'orally', 'orange', 'orbit', 'orbital', 'orbits', 'orchestra', 'orchestral', 'orchestrated', 'order', 'ordered', 'ordering', 'orders', 'ordinarily', 'ordinary', 'ore', 'oregon', 'org', 'organ', 'organic', 'organisation', 'organisations', 'organise', 'organised', 'organism', 'organisms', 'organization', 'organizational', 'organizations', 'organize', 'organized', 'organizers', 'organizing', 'organs', 'oriental', 'orientation', 'oriented', 'origin', 'original', 'originally', 'originals', 'originate', 'originated', 'originates', 'originating', 'origins', 'orlando', 'orleans', 'orphan', 'orthodox', 'orthodoxy', 'oscar', 'oscars', 'oslo', 'ostensibly', 'oswald', 'otto', 'ottoman', 'ottomans', 'ought', 'ousted', 'outbreak', 'outbreaks', 'outcome', 'outcomes', 'outcry', 'outdated', 'outdoor', 'outdoors', 'outer', 'outfit', 'outfits', 'outgoing', 'outlaw', 'outlawed', 'outlet', 'outlets', 'outline', 'outlined', 'outlines', 'outlook', 'outlying', 'outnumbered', 'output', 'outrage', 'outraged', 'outright', 'outset', 'outside', 'outsiders', 'outskirts', 'outspoken', 'outstanding', 'outward', 'oval', 'ovation', 'overall', 'overcame', 'overcome', 'overcoming', 'overdose', 'overhead', 'overland', 'overlap', 'overlapping', 'overlooked', 'overlooking', 'overly', 'overnight', 'overrun', 'oversaw', 'overseas', 'oversee', 'overseeing', 'overseen', 'oversees', 'overshadowed', 'oversight', 'overt', 'overtaken', 'overthrew', 'overthrow', 'overthrown', 'overtly', 'overtook', 'overturn', 'overturned', 'overview', 'overweight', 'overwhelmed', 'overwhelming', 'overwhelmingly', 'owe', 'owed', 'owen', 'owing', 'owned', 'owner', 'owners', 'ownership', 'owning', 'owns', 'oxford', 'oxidation', 'oxide', 'oxidized', 'oxygen', 'ozone', 'pablo', 'pac', 'pace', 'pacific', 'pack', 'package', 'packaged', 'packages', 'packaging', 'packed', 'packets', 'packs', 'pact', 'pad', 'pagan', 'paganism', 'page', 'pages', 'paid', 'pain', 'painful', 'pains', 'paint', 'painted', 'painter', 'painters', 'painting', 'paintings', 'pair', 'paired', 'pairing', 'pairs', 'pakistan', 'pakistani', 'palace', 'palaces', 'pale', 'paleolithic', 'palestine', 'palestinian', 'palm', 'palmer', 'pamphlet', 'pamphlets', 'pan', 'panama', 'panel', 'panels', 'panic', 'panned', 'pantheon', 'pants', 'papacy', 'papal', 'paper', 'paperback', 'papers', 'par', 'parade', 'parades', 'paradigm', 'paradise', 'paradox', 'paragraph', 'paraguay', 'parallel', 'parallels', 'paralysis', 'parameter', 'parameters', 'paramilitary', 'paramount', 'paranoia', 'paranoid', 'parasites', 'parasitic', 'pardon', 'parent', 'parental', 'parents', 'paris', 'parish', 'parity', 'park', 'parker', 'parking', 'parkinson', 'parks', 'parliament', 'parliamentary', 'parodied', 'parodies', 'parody', 'parsons', 'parted', 'partial', 'partially', 'participant', 'participants', 'participate', 'participated', 'participates', 'participating', 'participation', 'particle', 'particles', 'particular', 'particularly', 'parties', 'partisan', 'partition', 'partitioned', 'partly', 'partner', 'partnered', 'partners', 'partnership', 'partnerships', 'parts', 'party', 'pass', 'passage', 'passages', 'passed', 'passenger', 'passengers', 'passes', 'passing', 'passion', 'passionate', 'passive', 'passport', 'password', 'past', 'paste', 'pastor', 'pat', 'patch', 'patches', 'patent', 'patented', 'patents', 'paternal', 'path', 'pathogens', 'pathological', 'pathology', 'pathophysiology', 'paths', 'pathway', 'pathways', 'patience', 'patient', 'patients', 'patriarch', 'patricia', 'patrick', 'patriot', 'patriotic', 'patriotism', 'patriots', 'patrol', 'patron', 'patronage', 'patrons', 'pattern', 'patterns', 'paul', 'paula', 'pauline', 'paulo', 'pause', 'paved', 'pay', 'paying', 'payment', 'payments', 'payroll', 'pays', 'pbs', 'pcs', 'peace', 'peaceful', 'peacefully', 'peacekeeping', 'peacetime', 'peak', 'peaked', 'peaking', 'peaks', 'pearl', 'pearson', 'peas', 'peasant', 'peasants', 'peculiar', 'pedro', 'peer', 'peers', 'pelvic', 'pen', 'penal', 'penalties', 'penalty', 'pending', 'penetrate', 'penetration', 'peninsula', 'penis', 'penn', 'penned', 'pennsylvania', 'penny', 'pension', 'pensions', 'pentagon', 'people', 'peoples', 'pepper', 'peppers', 'pepsi', 'perceive', 'perceived', 'percent', 'percentage', 'percentages', 'perception', 'perceptions', 'percussion', 'percy', 'perennial', 'perfect', 'perfected', 'perfection', 'perfectly', 'perform', 'performance', 'performances', 'performed', 'performer', 'performers', 'performing', 'performs', 'perimeter', 'period', 'periodic', 'periodically', 'periods', 'peripheral', 'periphery', 'perkins', 'permanent', 'permanently', 'permission', 'permit', 'permits', 'permitted', 'permitting', 'perpendicular', 'perpetrated', 'perpetual', 'perry', 'persecuted', 'persecution', 'persia', 'persian', 'persians', 'persist', 'persisted', 'persistence', 'persistent', 'persists', 'person', 'persona', 'personal', 'personalities', 'personality', 'personally', 'personnel', 'persons', 'perspective', 'perspectives', 'persuade', 'persuaded', 'pertaining', 'peru', 'pervasive', 'pesticides', 'pet', 'peta', 'pete', 'peter', 'petersburg', 'petition', 'petitioned', 'petroleum', 'pets', 'petty', 'pew', 'phantom', 'pharmaceutical', 'pharmaceuticals', 'phase', 'phased', 'phases', 'phd', 'phenomena', 'phenomenon', 'phil', 'philadelphia', 'philanthropic', 'philanthropist', 'philanthropy', 'philip', 'philippe', 'philippine', 'philippines', 'phillip', 'phillips', 'philosopher', 'philosophers', 'philosophical', 'philosophies', 'philosophy', 'phoenix', 'phone', 'phones', 'phosphate', 'phosphorus', 'photo', 'photograph', 'photographed', 'photographer', 'photographers', 'photographic', 'photographs', 'photography', 'photos', 'photosynthesis', 'phrase', 'phrases', 'physical', 'physically', 'physician', 'physicians', 'physicist', 'physicists', 'physics', 'physiological', 'physiology', 'pianist', 'piano', 'pick', 'picked', 'picking', 'picks', 'pictorial', 'picture', 'pictured', 'pictures', 'pie', 'piece', 'pieces', 'pier', 'pierce', 'pierre', 'pig', 'pigs', 'pilgrimage', 'pilgrims', 'pill', 'pillar', 'pillars', 'pills', 'pilot', 'pilots', 'pin', 'pine', 'pink', 'pinnacle', 'pioneer', 'pioneered', 'pioneering', 'pioneers', 'pious', 'pipe', 'pipeline', 'piper', 'pipes', 'piracy', 'pirate', 'pirates', 'pistol', 'pistols', 'pit', 'pitch', 'pitched', 'pits', 'pitt', 'pitted', 'pittsburgh', 'pivotal', 'pizza', 'place', 'placebo', 'placed', 'placement', 'places', 'placing', 'plague', 'plagued', 'plain', 'plains', 'plan', 'plane', 'planes', 'planet', 'planetary', 'planets', 'planned', 'planning', 'plans', 'plant', 'plantation', 'plantations', 'planted', 'planting', 'plants', 'plaque', 'plasma', 'plastic', 'plate', 'plateau', 'plates', 'platform', 'platforms', 'platinum', 'plato', 'plausible', 'play', 'playable', 'playback', 'playboy', 'played', 'player', 'players', 'playhouse', 'playing', 'playoff', 'playoffs', 'plays', 'playstation', 'playwright', 'plaza', 'plea', 'pleaded', 'pleasant', 'pleased', 'pleasure', 'pledge', 'pledged', 'pleistocene', 'plentiful', 'plenty', 'plight', 'pliny', 'plot', 'plots', 'plotted', 'plotting', 'ploy', 'plug', 'plunged', 'plural', 'plurality', 'plus', 'pneumonia', 'png', 'pocket', 'pockets', 'poem', 'poems', 'poet', 'poetic', 'poetry', 'poets', 'point', 'pointed', 'pointing', 'points', 'poison', 'poisoned', 'poisoning', 'poisonous', 'poland', 'polar', 'polarized', 'pole', 'poles', 'police', 'policeman', 'policies', 'policy', 'polish', 'polished', 'political', 'politically', 'politician', 'politicians', 'politics', 'poll', 'polling', 'polls', 'pollutants', 'pollution', 'polo', 'pond', 'pool', 'pools', 'poor', 'poorer', 'poorest', 'poorly', 'pop', 'pope', 'populace', 'popular', 'popularity', 'popularize', 'popularized', 'popularly', 'populated', 'population', 'populations', 'populous', 'pork', 'porn', 'pornographic', 'pornography', 'port', 'portable', 'portal', 'porter', 'portfolio', 'portion', 'portions', 'portland', 'portrait', 'portraits', 'portray', 'portrayal', 'portrayals', 'portrayed', 'portraying', 'portrays', 'ports', 'portugal', 'portuguese', 'pose', 'posed', 'poses', 'posing', 'posited', 'position', 'positioned', 'positioning', 'positions', 'positive', 'positively', 'possess', 'possessed', 'possesses', 'possessing', 'possession', 'possessions', 'possibilities', 'possibility', 'possible', 'possibly', 'post', 'postage', 'postal', 'posted', 'poster', 'posters', 'posthumous', 'posthumously', 'posting', 'postponed', 'posts', 'postulated', 'posture', 'postwar', 'pot', 'potassium', 'potato', 'potatoes', 'potency', 'potent', 'potential', 'potentially', 'potter', 'pottery', 'poultry', 'pound', 'pounds', 'pour', 'poured', 'poverty', 'powder', 'powell', 'power', 'powered', 'powerful', 'powers', 'ppen', 'ppp', 'practical', 'practically', 'practice', 'practiced', 'practices', 'practicing', 'practised', 'practitioner', 'practitioners', 'pradesh', 'pragmatic', 'prague', 'praise', 'praised', 'praising', 'pray', 'prayer', 'prayers', 'pre', 'preached', 'preaching', 'precautions', 'preceded', 'precedence', 'precedent', 'preceding', 'precious', 'precipitated', 'precipitation', 'precise', 'precisely', 'precision', 'precursor', 'precursors', 'predator', 'predators', 'predatory', 'predecessor', 'predecessors', 'predict', 'predictable', 'predicted', 'predicting', 'prediction', 'predictions', 'predicts', 'predominant', 'predominantly', 'predominate', 'prefer', 'preferable', 'preference', 'preferences', 'preferred', 'preferring', 'prefers', 'prefix', 'pregnancies', 'pregnancy', 'pregnant', 'prehistoric', 'prehistory', 'prejudice', 'preliminary', 'prelude', 'premature', 'prematurely', 'premier', 'premiere', 'premiered', 'premise', 'premises', 'premium', 'preparation', 'preparations', 'preparatory', 'prepare', 'prepared', 'prepares', 'preparing', 'prequel', 'presbyterian', 'prescribed', 'prescription', 'presence', 'present', 'presentation', 'presentations', 'presented', 'presenter', 'presenting', 'presently', 'presents', 'preservation', 'preserve', 'preserved', 'preserves', 'preserving', 'presided', 'presidency', 'president', 'presidential', 'presidents', 'presiding', 'presley', 'press', 'pressed', 'pressing', 'pressure', 'pressured', 'pressures', 'prestige', 'prestigious', 'preston', 'presumably', 'presumed', 'pretending', 'pretext', 'pretty', 'prevail', 'prevailed', 'prevailing', 'prevalence', 'prevalent', 'prevent', 'prevented', 'preventing', 'prevention', 'preventive', 'prevents', 'preview', 'previous', 'previously', 'prey', 'price', 'priced', 'prices', 'pricing', 'pride', 'priest', 'priests', 'primarily', 'primary', 'primates', 'prime', 'primetime', 'primitive', 'prince', 'princes', 'princess', 'princeton', 'principal', 'principally', 'principle', 'principles', 'print', 'printed', 'printer', 'printing', 'prints', 'prior', 'priorities', 'priority', 'prison', 'prisoner', 'prisoners', 'prisons', 'privacy', 'private', 'privately', 'privilege', 'privileged', 'privileges', 'prix', 'prize', 'prizes', 'pro', 'probability', 'probable', 'probably', 'probation', 'probe', 'problem', 'problematic', 'problems', 'procedural', 'procedure', 'procedures', 'proceed', 'proceeded', 'proceeding', 'proceedings', 'proceeds', 'process', 'processed', 'processes', 'processing', 'procession', 'processor', 'processors', 'proclaimed', 'proclaiming', 'proclamation', 'produce', 'produced', 'producer', 'producers', 'produces', 'producing', 'product', 'production', 'productions', 'productive', 'productivity', 'products', 'professed', 'profession', 'professional', 'professionally', 'professionals', 'professions', 'professor', 'professors', 'profile', 'profiles', 'profit', 'profitable', 'profits', 'profound', 'profoundly', 'prognosis', 'program', 'programme', 'programmed', 'programmer', 'programmers', 'programmes', 'programming', 'programs', 'progress', 'progressed', 'progresses', 'progression', 'progressive', 'progressively', 'prohibit', 'prohibited', 'prohibiting', 'prohibition', 'prohibits', 'project', 'projected', 'projection', 'projections', 'projects', 'proliferation', 'prolific', 'prolonged', 'prominence', 'prominent', 'prominently', 'promise', 'promised', 'promises', 'promising', 'promote', 'promoted', 'promoter', 'promotes', 'promoting', 'promotion', 'promotional', 'promotions', 'prompt', 'prompted', 'prompting', 'promptly', 'promulgated', 'prone', 'pronounced', 'pronunciation', 'proof', 'proofs', 'prop', 'propaganda', 'propagation', 'propelled', 'proper', 'properly', 'properties', 'property', 'prophecy', 'prophet', 'prophets', 'proponent', 'proponents', 'proportion', 'proportional', 'proportions', 'proposal', 'proposals', 'propose', 'proposed', 'proposes', 'proposing', 'proposition', 'proprietary', 'prose', 'prosecuted', 'prosecution', 'prosecutor', 'prosecutors', 'prospect', 'prospective', 'prospects', 'prosperity', 'prosperous', 'prostate', 'prostitute', 'prostitutes', 'prostitution', 'prot', 'protagonist', 'protect', 'protected', 'protecting', 'protection', 'protections', 'protective', 'protector', 'protectorate', 'protects', 'protein', 'proteins', 'protest', 'protestant', 'protestantism', 'protestants', 'protested', 'protesters', 'protesting', 'protests', 'proto', 'protocol', 'protocols', 'proton', 'prototype', 'protracted', 'proud', 'prove', 'proved', 'proven', 'proves', 'provide', 'provided', 'providence', 'provider', 'providers', 'provides', 'providing', 'province', 'provinces', 'provincial', 'proving', 'provision', 'provisional', 'provisions', 'provocative', 'provoke', 'provoked', 'prowess', 'proximity', 'proxy', 'prussia', 'prussian', 'pseudo', 'pseudonym', 'psychiatric', 'psychiatrist', 'psychological', 'psychologically', 'psychologist', 'psychologists', 'psychology', 'psychosis', 'ptolemy', 'puberty', 'public', 'publication', 'publications', 'publicised', 'publicist', 'publicity', 'publicized', 'publicly', 'publish', 'published', 'publisher', 'publishers', 'publishes', 'publishing', 'puerto', 'pulitzer', 'pull', 'pulled', 'pulling', 'pulp', 'pulse', 'pulses', 'pump', 'pumped', 'pumping', 'pumps', 'punch', 'punched', 'punish', 'punishable', 'punished', 'punishment', 'punishments', 'punjab', 'punjabi', 'punk', 'pupil', 'pupils', 'puppet', 'purchase', 'purchased', 'purchases', 'purchasing', 'pure', 'purely', 'purge', 'purification', 'purified', 'purity', 'purple', 'purported', 'purportedly', 'purpose', 'purposes', 'pursue', 'pursued', 'pursuing', 'pursuit', 'pursuits', 'push', 'pushed', 'pushes', 'pushing', 'puts', 'putting', 'pyramid', 'qaeda', 'qatar', 'qualification', 'qualifications', 'qualified', 'qualify', 'qualifying', 'qualitative', 'qualities', 'quality', 'quantitative', 'quantities', 'quantity', 'quantum', 'quarter', 'quarters', 'quartet', 'quasi', 'quebec', 'queen', 'queens', 'quentin', 'quest', 'question', 'questionable', 'questioned', 'questioning', 'questions', 'quick', 'quickly', 'quiet', 'quietly', 'quit', 'quite', 'quo', 'quota', 'quotas', 'quotation', 'quote', 'quoted', 'quotes', 'rabbi', 'rabbit', 'race', 'races', 'rachel', 'racial', 'racially', 'racing', 'racism', 'racist', 'radar', 'radiation', 'radical', 'radically', 'radicals', 'radio', 'radioactive', 'radius', 'rafael', 'rage', 'rai', 'raid', 'raided', 'raiders', 'raids', 'rail', 'railroad', 'railroads', 'railway', 'railways', 'rain', 'rainbow', 'rainfall', 'rainforest', 'rains', 'rainy', 'raise', 'raised', 'raises', 'raising', 'raj', 'raja', 'rallied', 'rallies', 'rally', 'ralph', 'ram', 'rampant', 'ran', 'ranch', 'random', 'randomly', 'randy', 'range', 'ranged', 'rangers', 'ranges', 'ranging', 'rank', 'ranked', 'ranking', 'rankings', 'ranks', 'rap', 'rape', 'raped', 'rapid', 'rapidly', 'rapper', 'rappers', 'rapping', 'rare', 'rarely', 'rarer', 'rash', 'rat', 'rate', 'rated', 'rates', 'ratification', 'ratified', 'ratify', 'rating', 'ratings', 'ratio', 'ration', 'rational', 'rationale', 'ratios', 'rats', 'ravaged', 'rave', 'raw', 'ray', 'raymond', 'rays', 'rca', 'reach', 'reached', 'reaches', 'reaching', 'react', 'reacted', 'reacting', 'reaction', 'reactions', 'reactive', 'reactor', 'reactors', 'reacts', 'read', 'reader', 'readers', 'readily', 'readiness', 'reading', 'readings', 'reads', 'ready', 'reaffirmed', 'reagan', 'real', 'realised', 'realism', 'realistic', 'realities', 'reality', 'realization', 'realize', 'realized', 'realizes', 'realizing', 'really', 'realm', 'realms', 'rear', 'reason', 'reasonable', 'reasonably', 'reasoned', 'reasoning', 'reasons', 'rebecca', 'rebel', 'rebelled', 'rebellion', 'rebellions', 'rebellious', 'rebels', 'rebirth', 'rebound', 'rebuild', 'rebuilding', 'rebuilt', 'recall', 'recalled', 'recalling', 'recalls', 'recapture', 'recaptured', 'receipts', 'receive', 'received', 'receiver', 'receivers', 'receives', 'receiving', 'recent', 'recently', 'reception', 'receptor', 'receptors', 'recession', 'recipes', 'recipient', 'recipients', 'recited', 'reckless', 'reclaim', 'reclaimed', 'recognise', 'recognised', 'recognises', 'recognition', 'recognizable', 'recognize', 'recognized', 'recognizes', 'recognizing', 'recommend', 'recommendation', 'recommendations', 'recommended', 'recommending', 'recommends', 'reconcile', 'reconciled', 'reconciliation', 'reconnaissance', 'reconstructed', 'reconstruction', 'record', 'recorded', 'recorder', 'recording', 'recordings', 'records', 'recounted', 'recounts', 'recover', 'recovered', 'recovering', 'recovery', 'recreation', 'recreational', 'recruit', 'recruited', 'recruiting', 'recruitment', 'recruits', 'rectangular', 'recurrence', 'recurrent', 'recurring', 'recycled', 'recycling', 'red', 'reddish', 'redemption', 'redesign', 'redesigned', 'rediscovered', 'reduce', 'reduced', 'reduces', 'reducing', 'reduction', 'reductions', 'redundant', 'reed', 'reel', 'refer', 'referee', 'reference', 'referenced', 'references', 'referencing', 'referendum', 'referred', 'referring', 'refers', 'refined', 'refinement', 'refining', 'reflect', 'reflected', 'reflecting', 'reflection', 'reflective', 'reflects', 'reflex', 'reform', 'reformation', 'reformed', 'reformer', 'reformers', 'reformist', 'reforms', 'refrain', 'refuge', 'refugee', 'refugees', 'refusal', 'refuse', 'refused', 'refuses', 'refusing', 'refuted', 'regain', 'regained', 'regard', 'regarded', 'regarding', 'regardless', 'regards', 'regeneration', 'regent', 'reggae', 'regime', 'regimen', 'regiment', 'regimes', 'region', 'regional', 'regionally', 'regions', 'register', 'registered', 'registering', 'registration', 'registry', 'regret', 'regretted', 'regular', 'regularly', 'regulate', 'regulated', 'regulates', 'regulating', 'regulation', 'regulations', 'regulatory', 'rehab', 'rehabilitation', 'rehearsal', 'rehearsals', 'reich', 'reid', 'reign', 'reigned', 'reigning', 'reilly', 'reinforce', 'reinforced', 'reinforcement', 'reinforcements', 'reinstated', 'reintroduced', 'reiterated', 'reject', 'rejected', 'rejecting', 'rejection', 'rejects', 'rejoin', 'rejoined', 'relapse', 'relate', 'related', 'relates', 'relating', 'relation', 'relations', 'relationship', 'relationships', 'relative', 'relatively', 'relatives', 'relativity', 'relaxation', 'relaxed', 'relay', 'release', 'released', 'releases', 'releasing', 'relented', 'relevance', 'relevant', 'reliability', 'reliable', 'reliably', 'reliance', 'reliant', 'relics', 'relied', 'relief', 'relies', 'relieve', 'relieved', 'religion', 'religions', 'religious', 'religiously', 'relocate', 'relocated', 'relocation', 'reluctance', 'reluctant', 'reluctantly', 'rely', 'relying', 'remain', 'remainder', 'remained', 'remaining', 'remains', 'remake', 'remark', 'remarkable', 'remarkably', 'remarked', 'remarking', 'remarks', 'remarried', 'rematch', 'remedy', 'remember', 'remembered', 'remembers', 'remembrance', 'reminded', 'reminder', 'reminiscent', 'remix', 'remixed', 'remnant', 'remnants', 'remote', 'remotely', 'removal', 'remove', 'removed', 'removes', 'removing', 'ren', 'renaissance', 'renal', 'renamed', 'render', 'rendered', 'rendering', 'rendition', 'renew', 'renewable', 'renewal', 'renewed', 'renounced', 'renovated', 'renovation', 'renowned', 'rent', 'rental', 'rented', 'reopened', 'reorganization', 'rep', 'repair', 'repaired', 'repairs', 'repeal', 'repealed', 'repeat', 'repeated', 'repeatedly', 'repeating', 'repertoire', 'repetitive', 'replace', 'replaced', 'replacement', 'replaces', 'replacing', 'replica', 'replicate', 'replicated', 'replication', 'replied', 'replies', 'reply', 'report', 'reported', 'reportedly', 'reporter', 'reporters', 'reporting', 'reports', 'represent', 'representation', 'representations', 'representative', 'representatives', 'represented', 'representing', 'represents', 'repressed', 'repression', 'repressive', 'reprinted', 'reprise', 'reprised', 'reprising', 'reproduce', 'reproduced', 'reproduction', 'reproductive', 'reptiles', 'republic', 'republican', 'republicans', 'republics', 'reputation', 'request', 'requested', 'requesting', 'requests', 'require', 'required', 'requirement', 'requirements', 'requires', 'requiring', 'res', 'rescue', 'rescued', 'research', 'researched', 'researcher', 'researchers', 'researching', 'resemblance', 'resemble', 'resembled', 'resembles', 'resembling', 'resented', 'resentment', 'reservations', 'reserve', 'reserved', 'reserves', 'reservoir', 'reservoirs', 'reside', 'resided', 'residence', 'residences', 'residency', 'resident', 'residential', 'residents', 'resides', 'residing', 'residual', 'residue', 'resign', 'resignation', 'resigned', 'resist', 'resistance', 'resistant', 'resisted', 'resisting', 'resolution', 'resolutions', 'resolve', 'resolved', 'resolving', 'resonance', 'resort', 'resorted', 'resorts', 'resource', 'resources', 'respect', 'respectable', 'respected', 'respective', 'respectively', 'respects', 'respiration', 'respiratory', 'respond', 'responded', 'respondents', 'responding', 'responds', 'response', 'responses', 'responsibilities', 'responsibility', 'responsible', 'responsive', 'rest', 'restaurant', 'restaurants', 'rested', 'resting', 'restless', 'restoration', 'restore', 'restored', 'restoring', 'restrained', 'restraining', 'restraint', 'restrict', 'restricted', 'restricting', 'restriction', 'restrictions', 'restrictive', 'restructuring', 'rests', 'result', 'resultant', 'resulted', 'resulting', 'results', 'resume', 'resumed', 'resurgence', 'resurrected', 'resurrection', 'retail', 'retailer', 'retailers', 'retain', 'retained', 'retaining', 'retains', 'retaliation', 'retention', 'retire', 'retired', 'retirement', 'retiring', 'retreat', 'retreated', 'retreating', 'retrieve', 'retrieved', 'retrospective', 'return', 'returned', 'returning', 'returns', 'reunification', 'reunion', 'reunite', 'reunited', 'reuniting', 'reuters', 'rev', 'revamped', 'reveal', 'revealed', 'revealing', 'reveals', 'revelation', 'revelations', 'revenge', 'revenue', 'revenues', 'revered', 'reverend', 'reversal', 'reverse', 'reversed', 'reversing', 'reverted', 'review', 'reviewed', 'reviewer', 'reviewers', 'reviewing', 'reviews', 'revised', 'revision', 'revisions', 'revisited', 'revival', 'revive', 'revived', 'revoked', 'revolt', 'revolts', 'revolution', 'revolutionaries', 'revolutionary', 'revolutionized', 'revolutions', 'revolved', 'revolver', 'revolves', 'revolving', 'reward', 'rewarded', 'rewards', 'reworked', 'rewrite', 'rewritten', 'rex', 'rey', 'reynolds', 'rez', 'rfc', 'rhetoric', 'rhine', 'rhode', 'rhodes', 'rhyme', 'rhythm', 'rhythmic', 'rhythms', 'riaa', 'ribbon', 'ribs', 'ric', 'rica', 'rican', 'ricardo', 'rice', 'rich', 'richard', 'richards', 'richardson', 'richer', 'riches', 'richest', 'richie', 'richmond', 'rick', 'ricky', 'rico', 'rid', 'ride', 'rider', 'riders', 'rides', 'ridge', 'ridiculed', 'ridiculous', 'riding', 'ridley', 'rifle', 'rifles', 'rift', 'right', 'righteous', 'rights', 'rigid', 'rigorous', 'rihanna', 'riley', 'rim', 'ring', 'rings', 'rio', 'riot', 'riots', 'rise', 'risen', 'rises', 'rising', 'risk', 'risks', 'risky', 'rita', 'rite', 'rites', 'ritual', 'rituals', 'rival', 'rivalry', 'rivals', 'river', 'rivers', 'rna', 'road', 'roads', 'rob', 'robbed', 'robbery', 'robbie', 'robert', 'roberto', 'roberts', 'robertson', 'robin', 'robinson', 'robot', 'robots', 'robust', 'rochester', 'rock', 'rockefeller', 'rocket', 'rockets', 'rocks', 'rocky', 'rod', 'rode', 'rodents', 'rodney', 'rodriguez', 'rods', 'roger', 'rogers', 'rogue', 'roland', 'role', 'roles', 'roll', 'rolled', 'rolling', 'rolls', 'rom', 'roma', 'roman', 'romance', 'romania', 'romanian', 'romans', 'romantic', 'romantically', 'romanticism', 'rome', 'romeo', 'ron', 'ronald', 'roof', 'rookie', 'room', 'roommate', 'rooms', 'roosevelt', 'root', 'rooted', 'roots', 'rope', 'ropes', 'rosa', 'rose', 'roses', 'ross', 'roster', 'rotate', 'rotated', 'rotating', 'rotation', 'rotational', 'rotten', 'rouge', 'rough', 'roughly', 'round', 'rounded', 'rounds', 'route', 'routed', 'routes', 'routine', 'routinely', 'routines', 'row', 'rows', 'roy', 'royal', 'royalties', 'royalty', 'rubber', 'ruby', 'rudimentary', 'rudolf', 'rudolph', 'rugby', 'rugged', 'ruin', 'ruined', 'ruins', 'rule', 'ruled', 'ruler', 'rulers', 'rules', 'ruling', 'rulings', 'rumor', 'rumored', 'rumors', 'rumours', 'run', 'runaway', 'runner', 'runners', 'running', 'runoff', 'runs', 'runway', 'rupert', 'rural', 'rush', 'rushed', 'russell', 'russia', 'russian', 'russians', 'russo', 'ruth', 'rutherford', 'ruthless', 'ryan', 'rye', 'sabotage', 'sack', 'sacked', 'sacred', 'sacrifice', 'sacrifices', 'sad', 'saddam', 'sadness', 'safe', 'safely', 'safer', 'safety', 'saga', 'sage', 'sahara', 'saharan', 'said', 'sail', 'sailed', 'sailing', 'sailor', 'sailors', 'saint', 'saints', 'sake', 'salad', 'salaries', 'salary', 'sale', 'sales', 'salesman', 'saliva', 'sally', 'salman', 'salon', 'salt', 'salts', 'salvador', 'salvation', 'sam', 'samantha', 'sample', 'sampled', 'samples', 'sampling', 'samuel', 'san', 'sanctioned', 'sanctions', 'sanctuary', 'sand', 'sanders', 'sandra', 'sandy', 'sang', 'sanitation', 'sank', 'sanskrit', 'santa', 'santiago', 'santo', 'santos', 'sar', 'sara', 'sarah', 'sat', 'satan', 'satellite', 'satellites', 'satire', 'satirical', 'satisfaction', 'satisfactory', 'satisfied', 'satisfy', 'satisfying', 'saturated', 'saturday', 'saturn', 'sauce', 'saudi', 'savage', 'savannah', 'save', 'saved', 'saves', 'saving', 'savings', 'saw', 'saxon', 'say', 'saying', 'says', 'scale', 'scales', 'scaling', 'scan', 'scandal', 'scandals', 'scandinavia', 'scandinavian', 'scanning', 'scans', 'scar', 'scarce', 'scarcity', 'scare', 'scared', 'scarlet', 'scary', 'scattered', 'scattering', 'scenario', 'scenarios', 'scene', 'scenes', 'scent', 'schedule', 'scheduled', 'schedules', 'scheduling', 'scheme', 'schemes', 'schism', 'schneider', 'scholar', 'scholarly', 'scholars', 'scholarship', 'scholarships', 'school', 'schooling', 'schools', 'schwarzenegger', 'sci', 'science', 'sciences', 'scientific', 'scientifically', 'scientist', 'scientists', 'scope', 'score', 'scored', 'scorer', 'scores', 'scoring', 'scorsese', 'scotia', 'scotland', 'scots', 'scott', 'scottish', 'scout', 'scouts', 'scrapped', 'scratch', 'scream', 'screaming', 'screen', 'screened', 'screening', 'screenplay', 'screens', 'screenwriter', 'script', 'scripted', 'scripts', 'scripture', 'scriptures', 'scrutiny', 'sculptor', 'sculpture', 'sculptures', 'sea', 'seafood', 'seal', 'sealed', 'seals', 'sean', 'search', 'searched', 'searches', 'searching', 'seas', 'season', 'seasonal', 'seasons', 'seat', 'seated', 'seats', 'seattle', 'seawater', 'sebastian', 'sec', 'secession', 'second', 'secondary', 'secondly', 'seconds', 'secrecy', 'secret', 'secretariat', 'secretary', 'secretion', 'secretions', 'secretly', 'secrets', 'sect', 'section', 'sections', 'sector', 'sectors', 'sects', 'secular', 'secure', 'secured', 'securing', 'securities', 'security', 'sediment', 'sediments', 'seed', 'seeds', 'seeing', 'seek', 'seekers', 'seeking', 'seeks', 'seemingly', 'seen', 'sees', 'segment', 'segments', 'segregated', 'segregation', 'seismic', 'seize', 'seized', 'seizing', 'seizure', 'seizures', 'seldom', 'select', 'selected', 'selecting', 'selection', 'selections', 'selective', 'selectively', 'self', 'selfish', 'sell', 'seller', 'sellers', 'selling', 'sells', 'semen', 'semester', 'semi', 'semiconductor', 'seminal', 'semitic', 'sen', 'senate', 'senator', 'senators', 'send', 'sending', 'sends', 'senegal', 'senior', 'sensation', 'sensational', 'sensations', 'sense', 'senses', 'sensible', 'sensing', 'sensitive', 'sensitivity', 'sensors', 'sensory', 'sent', 'sentence', 'sentenced', 'sentences', 'sentencing', 'sentiment', 'sentiments', 'seoul', 'separate', 'separated', 'separately', 'separates', 'separating', 'separation', 'september', 'sequel', 'sequels', 'sequence', 'sequences', 'sequential', 'serbia', 'serbian', 'sergeant', 'serial', 'series', 'seriously', 'serum', 'servant', 'servants', 'serve', 'served', 'server', 'servers', 'serves', 'service', 'services', 'serving', 'session', 'sessions', 'set', 'setbacks', 'seth', 'sets', 'setting', 'settings', 'settle', 'settled', 'settlement', 'settlements', 'settlers', 'settling', 'setup', 'seven', 'seventeen', 'seventeenth', 'seventh', 'seventy', 'severe', 'severed', 'severely', 'severity', 'sewage', 'sex', 'sexes', 'sexiest', 'sexual', 'sexuality', 'sexually', 'sexy', 'seymour', 'shade', 'shades', 'shadow', 'shadows', 'shaft', 'shah', 'shake', 'shaken', 'shakespeare', 'shaking', 'shall', 'shallow', 'shame', 'shane', 'shanghai', 'shannon', 'shape', 'shaped', 'shapes', 'shaping', 'share', 'shared', 'shares', 'sharing', 'shark', 'sharks', 'sharma', 'sharon', 'sharp', 'sharply', 'shattered', 'shaw', 'shed', 'sheen', 'sheep', 'sheer', 'sheet', 'sheets', 'sheffield', 'sheikh', 'shelf', 'shell', 'shelley', 'shells', 'shelter', 'sheltered', 'shelters', 'shelved', 'shelves', 'shepherd', 'sheriff', 'sherman', 'shi', 'shia', 'shield', 'shields', 'shift', 'shifted', 'shifting', 'shifts', 'shine', 'shining', 'ship', 'shipbuilding', 'shipment', 'shipments', 'shipped', 'shipping', 'ships', 'shirley', 'shirt', 'shirts', 'shit', 'shock', 'shocked', 'shocking', 'shoe', 'shoes', 'shook', 'shoot', 'shooter', 'shooting', 'shootout', 'shoots', 'shop', 'shopping', 'shops', 'shore', 'shores', 'short', 'shortage', 'shortages', 'shortcomings', 'shortened', 'shortening', 'shorter', 'shortest', 'shortly', 'shorts', 'shot', 'shots', 'shoulder', 'shoulders', 'shouldn', 'shouted', 'shouting', 'showcase', 'showcased', 'showed', 'showing', 'shown', 'shows', 'showtime', 'shrine', 'shrink', 'shut', 'shuttle', 'shy', 'siberia', 'siberian', 'sibling', 'siblings', 'sicily', 'sick', 'sickness', 'sided', 'sides', 'sidney', 'siege', 'sierra', 'sight', 'sigmund', 'sign', 'signal', 'signaled', 'signaling', 'signalling', 'signals', 'signature', 'signatures', 'signed', 'significance', 'significant', 'significantly', 'signifying', 'signing', 'signs', 'sikh', 'silence', 'silent', 'silicon', 'silk', 'silly', 'silva', 'silver', 'similar', 'similarities', 'similarity', 'similarly', 'simmons', 'simon', 'simple', 'simpler', 'simplest', 'simplicity', 'simplified', 'simplify', 'simply', 'simpson', 'simpsons', 'simulate', 'simulated', 'simulation', 'simulations', 'simultaneous', 'simultaneously', 'sin', 'sinai', 'sinatra', 'sing', 'singapore', 'singer', 'singers', 'singh', 'singing', 'single', 'singled', 'singles', 'sings', 'singular', 'sinister', 'sink', 'sinking', 'sino', 'sins', 'sir', 'sister', 'sisters', 'sit', 'sitcom', 'site', 'sites', 'sits', 'sitting', 'situated', 'situation', 'situations', 'sixteen', 'sixteenth', 'sixth', 'sizable', 'size', 'sizeable', 'sized', 'sizes', 'skating', 'skeletal', 'skeleton', 'skeptical', 'skepticism', 'sketch', 'sketches', 'ski', 'skiing', 'skill', 'skilled', 'skills', 'skin', 'skinned', 'skins', 'skull', 'sky', 'skyscrapers', 'slam', 'slang', 'slash', 'slate', 'slated', 'slaughter', 'slave', 'slavery', 'slaves', 'slavic', 'sleep', 'sleeping', 'slept', 'slide', 'sliding', 'slight', 'slightly', 'slim', 'slip', 'slipped', 'slogan', 'slope', 'slopes', 'slot', 'slots', 'slovakia', 'slovenia', 'slow', 'slowed', 'slower', 'slowing', 'slowly', 'small', 'smaller', 'smallest', 'smallpox', 'smart', 'smartphone', 'smartphones', 'smash', 'smell', 'smile', 'smiling', 'smith', 'smithsonian', 'smoke', 'smoked', 'smoking', 'smooth', 'smoothly', 'smuggling', 'snake', 'snakes', 'snap', 'snl', 'snoop', 'snow', 'snowfall', 'soap', 'soared', 'sober', 'soccer', 'social', 'socialism', 'socialist', 'socialists', 'socially', 'societal', 'societies', 'society', 'socio', 'socioeconomic', 'sociological', 'sociology', 'soda', 'sodium', 'sofia', 'soft', 'softer', 'software', 'soil', 'soils', 'solar', 'sold', 'soldier', 'soldiers', 'sole', 'solely', 'solid', 'solidarity', 'solidified', 'solids', 'solitary', 'solo', 'solomon', 'soluble', 'solution', 'solutions', 'solve', 'solved', 'solvent', 'solving', 'somalia', 'somebody', 'somewhat', 'son', 'song', 'songs', 'songwriter', 'songwriters', 'songwriting', 'sonic', 'sonny', 'sons', 'sony', 'soon', 'sooner', 'sophia', 'sophisticated', 'sophistication', 'sophomore', 'soprano', 'sorrow', 'sorry', 'sort', 'sorts', 'sought', 'soul', 'souls', 'sound', 'sounded', 'sounding', 'sounds', 'soundtrack', 'soundtracks', 'soup', 'sour', 'source', 'sourced', 'sources', 'south', 'southeast', 'southeastern', 'southern', 'southernmost', 'southward', 'southwest', 'southwestern', 'sovereign', 'sovereignty', 'soviet', 'soviets', 'soy', 'space', 'spacecraft', 'spaced', 'spaces', 'spain', 'span', 'spaniards', 'spanish', 'spanned', 'spanning', 'spans', 'spare', 'spared', 'spark', 'sparked', 'sparking', 'sparks', 'sparse', 'sparsely', 'spatial', 'spawned', 'speak', 'speaker', 'speakers', 'speaking', 'speaks', 'spearheaded', 'spears', 'special', 'specialised', 'specialist', 'specialists', 'specialization', 'specialized', 'specializing', 'specially', 'specials', 'specialty', 'species', 'specific', 'specifically', 'specification', 'specifications', 'specified', 'specifies', 'specify', 'specifying', 'specimen', 'specimens', 'spectacle', 'spectacular', 'spectator', 'spectators', 'spectrum', 'speculate', 'speculated', 'speculation', 'speculative', 'speech', 'speeches', 'speed', 'speeds', 'spell', 'spelled', 'spelling', 'spencer', 'spend', 'spending', 'spends', 'spent', 'sperm', 'sphere', 'spheres', 'spherical', 'spice', 'spices', 'spider', 'spielberg', 'spies', 'spike', 'spin', 'spinal', 'spine', 'spinning', 'spiral', 'spirit', 'spirited', 'spirits', 'spiritual', 'spirituality', 'spite', 'split', 'splitting', 'spoke', 'spoken', 'spokesman', 'spokesperson', 'spokeswoman', 'sponsor', 'sponsored', 'sponsors', 'sponsorship', 'spontaneous', 'spontaneously', 'spoof', 'sporadic', 'sport', 'sporting', 'sports', 'spot', 'spotlight', 'spots', 'spotted', 'sprang', 'spray', 'spread', 'spreading', 'spreads', 'spring', 'springs', 'spun', 'spurred', 'spurs', 'spy', 'squad', 'squadron', 'squads', 'square', 'squares', 'squash', 'sri', 'stability', 'stabilization', 'stabilize', 'stabilized', 'stabilizing', 'stable', 'stack', 'stadium', 'stadiums', 'staff', 'stage', 'staged', 'stages', 'staging', 'stagnation', 'stake', 'stalemate', 'stalin', 'stalled', 'stamp', 'stamps', 'stan', 'stance', 'stand', 'standard', 'standardization', 'standardized', 'standards', 'standing', 'standpoint', 'stands', 'stanford', 'stanley', 'staple', 'staples', 'star', 'starch', 'stardom', 'stark', 'starred', 'starring', 'stars', 'start', 'started', 'starter', 'starting', 'starts', 'starvation', 'state', 'stated', 'statement', 'statements', 'states', 'statesman', 'statewide', 'static', 'stating', 'station', 'stationary', 'stationed', 'stations', 'statistical', 'statistically', 'statistics', 'statue', 'statues', 'stature', 'status', 'statute', 'statutes', 'statutory', 'stay', 'stayed', 'staying', 'stays', 'steadily', 'steady', 'steal', 'stealing', 'steals', 'steam', 'steel', 'steep', 'steering', 'stefan', 'stella', 'stellar', 'stem', 'stemmed', 'stemming', 'stems', 'step', 'stepfather', 'stephanie', 'stephen', 'stepped', 'stepping', 'steps', 'stereo', 'stereotypes', 'stereotypical', 'sterile', 'sterling', 'stern', 'steve', 'steven', 'stevens', 'stevenson', 'stevie', 'stewart', 'stick', 'sticks', 'stiff', 'stigma', 'stimulate', 'stimulated', 'stimulates', 'stimulating', 'stimulation', 'stimuli', 'stimulus', 'stint', 'stipulated', 'stir', 'stock', 'stockholm', 'stocks', 'stole', 'stolen', 'stomach', 'stone', 'stones', 'stood', 'stop', 'stopped', 'stopping', 'stops', 'storage', 'store', 'stored', 'stores', 'stories', 'storing', 'storm', 'storms', 'story', 'storyline', 'storylines', 'storytelling', 'straight', 'straightforward', 'strain', 'strained', 'strains', 'strait', 'straits', 'stranded', 'strands', 'strange', 'stranger', 'strangers', 'strategic', 'strategically', 'strategies', 'strategy', 'straw', 'streak', 'stream', 'streamed', 'streaming', 'streams', 'street', 'streets', 'strength', 'strengthen', 'strengthened', 'strengthening', 'strengths', 'stress', 'stressed', 'stresses', 'stretch', 'stretched', 'stretches', 'stretching', 'stricken', 'strict', 'stricter', 'strictly', 'strife', 'strike', 'strikes', 'striking', 'string', 'stringent', 'strings', 'strip', 'stripes', 'stripped', 'strips', 'strive', 'stroke', 'strokes', 'strong', 'stronger', 'strongest', 'stronghold', 'strongly', 'struck', 'structural', 'structurally', 'structure', 'structured', 'structures', 'struggle', 'struggled', 'struggles', 'struggling', 'stuart', 'stuck', 'student', 'students', 'studied', 'studies', 'studio', 'studios', 'study', 'studying', 'stuff', 'stuffed', 'stunned', 'stunning', 'stunt', 'stunts', 'stupid', 'style', 'styled', 'styles', 'stylistic', 'stylized', 'sub', 'subcontinent', 'subdivided', 'subdivisions', 'subdued', 'subgroup', 'subject', 'subjected', 'subjective', 'subjects', 'submarine', 'submarines', 'submerged', 'submission', 'submit', 'submitted', 'subordinate', 'subscribers', 'subscription', 'subsequent', 'subsequently', 'subset', 'subsidiary', 'subsidies', 'subsistence', 'subspecies', 'substance', 'substances', 'substantial', 'substantially', 'substitute', 'substituted', 'substitutes', 'substitution', 'substrate', 'subtle', 'subtropical', 'suburb', 'suburban', 'suburbs', 'subway', 'succeed', 'succeeded', 'succeeding', 'succeeds', 'success', 'successes', 'successful', 'successfully', 'succession', 'successive', 'successor', 'successors', 'sudan', 'sudden', 'suddenly', 'sue', 'sued', 'suez', 'suffer', 'suffered', 'suffering', 'suffers', 'sufficient', 'sufficiently', 'suffix', 'suffrage', 'sugar', 'sugars', 'suggest', 'suggested', 'suggesting', 'suggestion', 'suggestions', 'suggestive', 'suggests', 'suicidal', 'suicide', 'suit', 'suitable', 'suite', 'suited', 'suits', 'sulfate', 'sulfur', 'sullivan', 'sultan', 'sum', 'summarized', 'summary', 'summed', 'summer', 'summers', 'summit', 'summoned', 'sums', 'sun', 'sundance', 'sunday', 'sung', 'sunlight', 'sunni', 'sunny', 'sunrise', 'sunset', 'sunshine', 'super', 'superficial', 'superhero', 'superior', 'superiority', 'superiors', 'superman', 'supermarket', 'supernatural', 'superseded', 'superstar', 'supervised', 'supervising', 'supervision', 'supervisor', 'supplanted', 'supplement', 'supplemented', 'supplements', 'supplied', 'supplier', 'suppliers', 'supplies', 'supply', 'supplying', 'support', 'supported', 'supporter', 'supporters', 'supporting', 'supportive', 'supports', 'suppose', 'supposed', 'supposedly', 'suppress', 'suppressed', 'suppressing', 'suppression', 'supremacy', 'supreme', 'sur', 'sure', 'surely', 'surface', 'surfaced', 'surfaces', 'surge', 'surgeon', 'surgeons', 'surgery', 'surgical', 'surname', 'surpass', 'surpassed', 'surpassing', 'surplus', 'surprise', 'surprised', 'surprising', 'surprisingly', 'surrender', 'surrendered', 'surrogate', 'surround', 'surrounded', 'surrounding', 'surroundings', 'surrounds', 'surveillance', 'survey', 'surveyed', 'surveys', 'survival', 'survive', 'survived', 'survives', 'surviving', 'survivor', 'survivors', 'susan', 'susceptibility', 'susceptible', 'suspect', 'suspected', 'suspects', 'suspend', 'suspended', 'suspension', 'suspicion', 'suspicions', 'suspicious', 'sustain', 'sustainability', 'sustainable', 'sustained', 'sustaining', 'svg', 'swan', 'sway', 'swear', 'swearing', 'sweat', 'sweden', 'swedish', 'sweep', 'sweeping', 'sweet', 'swelling', 'swept', 'swift', 'swiftly', 'swim', 'swimming', 'swing', 'swinging', 'swings', 'swiss', 'switch', 'switched', 'switches', 'switching', 'switzerland', 'sword', 'swords', 'sworn', 'sydney', 'sylvester', 'symbol', 'symbolic', 'symbolically', 'symbolism', 'symbolize', 'symbols', 'sympathetic', 'sympathy', 'symphony', 'symptom', 'symptoms', 'synagogue', 'sync', 'syndicated', 'syndication', 'syndrome', 'synonym', 'synonymous', 'synopsis', 'syntax', 'synthesis', 'synthesize', 'synthesized', 'synthetic', 'syphilis', 'syria', 'syrian', 'syrup', 'systematic', 'systematically', 'systemic', 'systems', 'table', 'tables', 'tablet', 'tablets', 'tabloid', 'taboo', 'tackle', 'tactic', 'tactical', 'tactics', 'tag', 'tags', 'tail', 'tailor', 'tails', 'taiwan', 'taken', 'takeover', 'takes', 'taking', 'tale', 'talent', 'talented', 'talents', 'tales', 'taliban', 'talk', 'talked', 'talking', 'talks', 'tall', 'taller', 'tallest', 'tally', 'tamil', 'tan', 'tang', 'tangible', 'tank', 'tanks', 'tanzania', 'tap', 'tape', 'taped', 'tapes', 'taping', 'tapped', 'taran', 'target', 'targeted', 'targeting', 'targets', 'tariff', 'tariffs', 'task', 'tasked', 'tasks', 'taste', 'tastes', 'tat', 'tate', 'tattoo', 'tattoos', 'taught', 'tax', 'taxation', 'taxes', 'taxi', 'taxonomy', 'taylor', 'tea', 'teach', 'teacher', 'teachers', 'teaches', 'teaching', 'teachings', 'team', 'teamed', 'teaming', 'teammate', 'teammates', 'teams', 'tear', 'tearing', 'tears', 'teaser', 'tech', 'technical', 'technically', 'technicians', 'technique', 'techniques', 'techno', 'technological', 'technologies', 'technology', 'tectonic', 'ted', 'teddy', 'teen', 'teenage', 'teenager', 'teenagers', 'teens', 'teeth', 'tel', 'telecast', 'telecom', 'telecommunication', 'telecommunications', 'telegram', 'telegraph', 'telephone', 'telephones', 'telescope', 'televised', 'television', 'televisions', 'tell', 'telling', 'tells', 'telugu', 'temper', 'temperament', 'temperate', 'temperature', 'temperatures', 'tempered', 'template', 'temple', 'temples', 'tempo', 'temporal', 'temporarily', 'temporary', 'temptation', 'tend', 'tended', 'tendencies', 'tendency', 'tender', 'tends', 'tenets', 'tennessee', 'tennis', 'tenor', 'tens', 'tense', 'tension', 'tensions', 'tent', 'tentative', 'tentatively', 'tenth', 'tenure', 'teresa', 'term', 'termed', 'terminal', 'terminals', 'terminate', 'terminated', 'termination', 'terminology', 'terminus', 'terms', 'terrain', 'terrestrial', 'terrible', 'territorial', 'territories', 'territory', 'terror', 'terrorism', 'terrorist', 'terrorists', 'terry', 'tertiary', 'test', 'testament', 'tested', 'testified', 'testify', 'testimony', 'testing', 'tests', 'texas', 'text', 'textbook', 'textbooks', 'textile', 'textiles', 'texts', 'textual', 'texture', 'tha', 'thai', 'thailand', 'thames', 'thank', 'thanked', 'thanks', 'thanksgiving', 'thatcher', 'theater', 'theaters', 'theatre', 'theatres', 'theatrical', 'theatrically', 'theft', 'theme', 'themed', 'themes', 'theodore', 'theologians', 'theological', 'theology', 'theorem', 'theoretical', 'theoretically', 'theories', 'theorist', 'theorists', 'theorized', 'theory', 'therapeutic', 'therapies', 'therapist', 'therapy', 'thereof', 'thermal', 'thesis', 'thicker', 'thickness', 'thief', 'thieves', 'thigh', 'thing', 'things', 'think', 'thinkers', 'thinking', 'thinks', 'thinner', 'thirds', 'thirteen', 'thirteenth', 'thirty', 'thomas', 'thompson', 'thomson', 'thor', 'thorough', 'thoroughly', 'thought', 'thoughts', 'thousand', 'thousands', 'thread', 'threat', 'threaten', 'threatened', 'threatening', 'threatens', 'threats', 'threshold', 'threw', 'thriller', 'thrive', 'thriving', 'throat', 'throne', 'throw', 'throwing', 'thrown', 'throws', 'thrust', 'thumb', 'thunder', 'thunderstorms', 'thursday', 'thwarted', 'tibet', 'tibetan', 'ticket', 'tickets', 'tidal', 'tide', 'tie', 'tied', 'tier', 'ties', 'tiger', 'tigers', 'tight', 'tightly', 'till', 'tim', 'timber', 'timberlake', 'time', 'timed', 'timeline', 'times', 'timing', 'timothy', 'tin', 'tina', 'tiny', 'tip', 'tips', 'tired', 'tissue', 'tissues', 'titan', 'titans', 'title', 'titled', 'titles', 'titular', 'tnt', 'tobacco', 'today', 'todd', 'toe', 'toes', 'token', 'tokyo', 'told', 'tolerance', 'tolerant', 'tolerate', 'tolerated', 'toll', 'tom', 'tomatoes', 'tomb', 'tombs', 'tommy', 'tomorrow', 'ton', 'tone', 'tones', 'tongue', 'toni', 'tonight', 'tonnes', 'tons', 'tony', 'took', 'tool', 'tools', 'tooth', 'topic', 'topical', 'topics', 'topography', 'topped', 'topping', 'tops', 'torch', 'tore', 'torn', 'tornado', 'toronto', 'torres', 'torture', 'tortured', 'toss', 'total', 'totaled', 'totaling', 'totally', 'totals', 'touch', 'touched', 'touches', 'touching', 'tough', 'tour', 'toured', 'touring', 'tourism', 'tourist', 'tourists', 'tournament', 'tournaments', 'tours', 'tower', 'towers', 'town', 'towns', 'toxic', 'toxicity', 'toxins', 'toy', 'toys', 'trace', 'traced', 'traces', 'tracing', 'track', 'tracked', 'tracking', 'tracks', 'tract', 'tracts', 'tracy', 'trade', 'traded', 'trademark', 'traders', 'trades', 'trading', 'tradition', 'traditional', 'traditionally', 'traditions', 'traffic', 'trafficking', 'tragedy', 'tragic', 'trail', 'trailer', 'trailers', 'trailing', 'train', 'trained', 'trainer', 'training', 'trains', 'trait', 'traits', 'trajectory', 'trans', 'transaction', 'transactions', 'transcription', 'transfer', 'transferred', 'transferring', 'transfers', 'transform', 'transformation', 'transformed', 'transforming', 'transforms', 'transient', 'transit', 'transition', 'transitional', 'transitions', 'translate', 'translated', 'translates', 'translating', 'translation', 'translations', 'transmission', 'transmissions', 'transmit', 'transmitted', 'transmitting', 'transparency', 'transparent', 'transport', 'transportation', 'transported', 'transporting', 'transports', 'trap', 'trapped', 'trauma', 'traumatic', 'travel', 'traveled', 'travelers', 'traveling', 'travelled', 'travelling', 'travels', 'treason', 'treasure', 'treasures', 'treasury', 'treat', 'treated', 'treaties', 'treating', 'treatise', 'treatment', 'treatments', 'treats', 'treaty', 'tree', 'trees', 'trek', 'tremendous', 'trench', 'trend', 'trends', 'trent', 'trevor', 'tri', 'trial', 'trials', 'triangle', 'triangular', 'tribal', 'tribe', 'tribes', 'tribunal', 'tribune', 'tribute', 'tributes', 'trick', 'tricks', 'tried', 'tries', 'trigger', 'triggered', 'triggering', 'triggers', 'trillion', 'trilogy', 'trinidad', 'trinity', 'trio', 'trip', 'triple', 'trips', 'triumph', 'trivial', 'troop', 'troops', 'trophies', 'trophy', 'tropical', 'tropics', 'trouble', 'troubled', 'troubles', 'troupe', 'troy', 'truce', 'truck', 'trucks', 'true', 'truly', 'truman', 'trunk', 'trust', 'trusted', 'truth', 'truths', 'try', 'trying', 'tsunami', 'tube', 'tuberculosis', 'tubes', 'tucker', 'tudor', 'tuesday', 'tuition', 'tumor', 'tune', 'tuned', 'tunes', 'tunisia', 'tunnel', 'tunnels', 'turbulent', 'turkey', 'turkish', 'turks', 'turmoil', 'turn', 'turned', 'turner', 'turning', 'turnout', 'turnover', 'turns', 'tutor', 'tutors', 'tweeted', 'twelfth', 'twenties', 'twentieth', 'twice', 'twilight', 'twin', 'twins', 'twist', 'twisted', 'twitter', 'tying', 'tyler', 'type', 'types', 'typical', 'typically', 'tyranny', 'ubiquitous', 'ucla', 'uefa', 'uganda', 'ugly', 'ukraine', 'ukrainian', 'ultimate', 'ultimately', 'ultimatum', 'ultra', 'ultrasound', 'ultraviolet', 'ulysses', 'umbrella', 'unable', 'unacceptable', 'unaffected', 'unanimous', 'unanimously', 'unauthorized', 'unavailable', 'unaware', 'uncertain', 'uncertainty', 'unchanged', 'uncle', 'unclear', 'uncomfortable', 'uncommon', 'unconscious', 'unconstitutional', 'unconventional', 'uncovered', 'uncredited', 'und', 'undefeated', 'undercover', 'underdeveloped', 'undergo', 'undergoes', 'undergoing', 'undergone', 'undergraduate', 'underground', 'underlying', 'undermine', 'undermined', 'undermining', 'underneath', 'understand', 'understanding', 'understood', 'undertake', 'undertaken', 'undertaking', 'undertook', 'underwater', 'underway', 'underwent', 'underworld', 'undesirable', 'undisclosed', 'undisputed', 'undoubtedly', 'unemployed', 'unemployment', 'unequal', 'unesco', 'uneven', 'unexpected', 'unexpectedly', 'unfair', 'unfamiliar', 'unfavorable', 'unfinished', 'unfit', 'unfortunate', 'unfortunately', 'unhappy', 'unhealthy', 'unheard', 'unicameral', 'unicef', 'unidentified', 'unification', 'unified', 'uniform', 'uniformly', 'uniforms', 'unify', 'unifying', 'unilateral', 'uninhabited', 'unintended', 'union', 'unions', 'unique', 'uniquely', 'unit', 'unitary', 'unite', 'united', 'uniting', 'units', 'unity', 'universal', 'universally', 'universe', 'universities', 'university', 'unix', 'unknown', 'unlawful', 'unless', 'unlike', 'unlikely', 'unlimited', 'unmarried', 'unnamed', 'unnatural', 'unnecessary', 'unofficial', 'unofficially', 'unorthodox', 'unpaid', 'unpleasant', 'unpopular', 'unprecedented', 'unpredictable', 'unpublished', 'unrealistic', 'unrelated', 'unreleased', 'unreliable', 'unresolved', 'unrest', 'unsafe', 'unseen', 'unspecified', 'unstable', 'unsuccessful', 'unsuccessfully', 'unsuitable', 'unsure', 'unto', 'untreated', 'unused', 'unusual', 'unusually', 'unveiled', 'unwanted', 'unwilling', 'upbringing', 'upcoming', 'update', 'updated', 'updates', 'upgrade', 'upgraded', 'upheaval', 'upheld', 'uphold', 'upload', 'uploaded', 'upper', 'upright', 'uprising', 'uprisings', 'ups', 'upset', 'upside', 'uptake', 'upward', 'upwards', 'uranium', 'urban', 'urbanization', 'urdu', 'urge', 'urged', 'urgent', 'urging', 'urinary', 'urine', 'uruguay', 'usa', 'usable', 'usage', 'usb', 'usd', 'use', 'used', 'useful', 'usefulness', 'useless', 'user', 'users', 'uses', 'usher', 'ushered', 'using', 'uss', 'ussr', 'usual', 'usually', 'utah', 'utc', 'uterus', 'utilities', 'utility', 'utilization', 'utilize', 'utilized', 'utilizes', 'utilizing', 'vacant', 'vacation', 'vaccination', 'vaccine', 'vaccines', 'vacuum', 'vagina', 'vaginal', 'vague', 'vain', 'valentine', 'valid', 'validated', 'validity', 'valley', 'valleys', 'valuable', 'value', 'valued', 'values', 'valve', 'vampire', 'van', 'vancouver', 'vanessa', 'vanguard', 'vanilla', 'vanished', 'vanity', 'vapor', 'var', 'variability', 'variable', 'variables', 'variance', 'variant', 'variants', 'variation', 'variations', 'varied', 'varies', 'varieties', 'variety', 'various', 'variously', 'vary', 'varying', 'vascular', 'vast', 'vastly', 'vatican', 'vault', 'vector', 'vegan', 'vegas', 'vegetable', 'vegetables', 'vegetarian', 'vegetation', 'vehicle', 'vehicles', 'vein', 'veins', 'velocity', 'velvet', 'vendor', 'vendors', 'venetian', 'venezuela', 'vengeance', 'venice', 'venture', 'ventured', 'ventures', 'venue', 'venues', 'venus', 'verb', 'verbal', 'verbally', 'verdict', 'verge', 'verification', 'verified', 'verify', 'vermont', 'vernacular', 'veronica', 'versa', 'versailles', 'versatile', 'versatility', 'verse', 'verses', 'version', 'versions', 'versus', 'vertebrate', 'vertebrates', 'vertical', 'vertically', 'vessel', 'vessels', 'vested', 'veteran', 'veterans', 'veto', 'vetoed', 'viability', 'viable', 'vibe', 'vibrant', 'vibration', 'vic', 'vice', 'vicinity', 'vicious', 'victim', 'victims', 'victor', 'victoria', 'victorian', 'victories', 'victorious', 'victory', 'video', 'videos', 'vienna', 'vietnam', 'vietnamese', 'view', 'viewed', 'viewer', 'viewers', 'viewing', 'viewpoint', 'views', 'vigorous', 'vigorously', 'vii', 'viii', 'viking', 'villa', 'village', 'villages', 'villain', 'villainous', 'villains', 'vince', 'vincent', 'vinci', 'vintage', 'vinyl', 'violate', 'violated', 'violating', 'violation', 'violations', 'violence', 'violent', 'violently', 'violet', 'viral', 'virgin', 'virginia', 'virtual', 'virtually', 'virtue', 'virtues', 'virus', 'viruses', 'visa', 'visibility', 'visible', 'visibly', 'vision', 'visions', 'visit', 'visited', 'visiting', 'visitor', 'visitors', 'visits', 'vista', 'visual', 'visually', 'vital', 'vitamin', 'vitamins', 'vitro', 'vivid', 'vladimir', 'vocabulary', 'vocal', 'vocalist', 'vocals', 'vocational', 'vogue', 'voice', 'voiced', 'voices', 'voicing', 'void', 'vol', 'volatile', 'volcanic', 'volcano', 'volcanoes', 'volleyball', 'voltage', 'volume', 'volumes', 'voluntarily', 'voluntary', 'volunteer', 'volunteered', 'volunteers', 'vomiting', 'von', 'vote', 'voted', 'voter', 'voters', 'votes', 'voting', 'vowed', 'vows', 'voyage', 'voyages', 'vulnerability', 'vulnerable', 'wade', 'wage', 'waged', 'wages', 'wagner', 'wagon', 'waist', 'wait', 'waited', 'waiting', 'waitress', 'wake', 'waking', 'wales', 'walk', 'walked', 'walker', 'walking', 'walks', 'wall', 'wallace', 'walls', 'walsh', 'walt', 'walter', 'wanna', 'want', 'wanted', 'wanting', 'wants', 'war', 'ward', 'wardrobe', 'warehouse', 'warfare', 'warm', 'warmer', 'warmest', 'warming', 'warmth', 'warn', 'warned', 'warner', 'warning', 'warnings', 'warns', 'warrant', 'warren', 'warring', 'warrior', 'warriors', 'wars', 'warsaw', 'warships', 'wartime', 'wary', 'wash', 'washed', 'washing', 'washington', 'wasn', 'waste', 'wasted', 'wastes', 'watch', 'watched', 'watches', 'watching', 'water', 'waters', 'watershed', 'waterways', 'watson', 'watts', 'wave', 'wavelength', 'waves', 'wax', 'way', 'wayne', 'ways', 'weak', 'weaken', 'weakened', 'weakening', 'weaker', 'weakness', 'weaknesses', 'wealth', 'wealthiest', 'wealthy', 'weapon', 'weaponry', 'weapons', 'wear', 'wearing', 'wears', 'weather', 'weaving', 'web', 'weber', 'website', 'websites', 'webster', 'wed', 'wedding', 'weddings', 'wednesday', 'week', 'weekend', 'weekends', 'weekly', 'weeks', 'weigh', 'weighed', 'weighing', 'weighs', 'weight', 'weighted', 'weights', 'weird', 'welcome', 'welcomed', 'welfare', 'wells', 'welsh', 'wembley', 'went', 'weren', 'werner', 'wesley', 'west', 'western', 'westminster', 'westward', 'wet', 'wetlands', 'whale', 'whales', 'whatsoever', 'wheat', 'wheel', 'wheelchair', 'wheels', 'whilst', 'whip', 'white', 'whites', 'whitney', 'wholesale', 'wholly', 'wicked', 'wide', 'widely', 'wider', 'widespread', 'widow', 'widowed', 'width', 'wielding', 'wife', 'wii', 'wild', 'wilde', 'wilderness', 'wildlife', 'wildly', 'wilhelm', 'willem', 'william', 'williams', 'willie', 'willing', 'willingness', 'willis', 'wilson', 'win', 'wind', 'window', 'windows', 'winds', 'windsor', 'wine', 'winfrey', 'wing', 'wings', 'winner', 'winners', 'winning', 'wins', 'winston', 'winter', 'winters', 'wiped', 'wire', 'wired', 'wireless', 'wires', 'wisconsin', 'wisdom', 'wise', 'wish', 'wished', 'wishes', 'wishing', 'wit', 'witch', 'witches', 'withdraw', 'withdrawal', 'withdrawing', 'withdrawn', 'withdrew', 'withstand', 'witness', 'witnessed', 'witnesses', 'wives', 'wizard', 'wolf', 'wolfgang', 'wolves', 'woman', 'women', 'won', 'wonder', 'wonderful', 'wonders', 'wood', 'wooden', 'woodrow', 'woods', 'woody', 'wool', 'word', 'wording', 'words', 'wore', 'work', 'worked', 'worker', 'workers', 'workforce', 'working', 'workings', 'workplace', 'works', 'workshop', 'workshops', 'world', 'worlds', 'worldwide', 'worm', 'worms', 'worn', 'worried', 'worry', 'worse', 'worsened', 'worsening', 'worship', 'worshipped', 'worst', 'worth', 'worthy', 'wouldn', 'wound', 'wounded', 'wounds', 'woven', 'wrap', 'wrapped', 'wrestler', 'wrestling', 'wright', 'wrist', 'write', 'writer', 'writers', 'writes', 'writing', 'writings', 'written', 'wrong', 'wrongly', 'wrote', 'wto', 'www', 'wyoming', 'xavier', 'xbox', 'yacht', 'yahoo', 'yale', 'yang', 'yard', 'yards', 'yeah', 'year', 'yearly', 'years', 'yeast', 'yellow', 'yemen', 'yes', 'yesterday', 'yield', 'yielded', 'yielding', 'yields', 'yoga', 'york', 'yorker', 'yorkshire', 'young', 'younger', 'youngest', 'youth', 'youthful', 'youths', 'youtube', 'yugoslavia', 'zealand', 'zenith', 'zeppelin', 'zero', 'zeus', 'zimbabwe', 'zinc', 'zone', 'zones', 'zoo']

```


```python
# Importing pandas
import pandas as pd

# Creating a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns=words)

# Printing the shape of the DataFrame
print(components_df.shape)

# Selecting row 3: component
component = components_df.iloc[3]

# Printing result of nlargest
print(component.nlargest())

```

    (6, 13125)
    film       0.627873
    award      0.253130
    starred    0.245283
    role       0.211450
    actress    0.186397
    Name: 3, dtype: float64


###### Exploring the LED digits dataset

In the following exercises, you'll use NMF to decompose grayscale images into their commonly occurring patterns. Firstly, explore the image dataset and see how it is encoded as an array. You are given 100 images as a 2D array samples, where each row represents a single 13x8 image. The images in your dataset are pictures of a LED digital display.


```python
fp4 = '/Users/MuhammadBilal/Desktop/Data Camp/Unsupervised learning in Python/Data/lcd-digits.csv'
```


```python
samples = pd.read_csv(fp4, header=None)
```


```python
samples.shape
```




    (100, 104)




```python
# Importing pyplot
from matplotlib import pyplot as plt

# Selecting the 0th row: digit
digit = samples.iloc[0,:].values

# Printing digit
print(digit)

# Reshaping digit to a 13x8 array: bitmap
bitmap = digit.reshape(13,8)

# Printing bitmap
print(bitmap)

# Using plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()

```

    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.
     0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0.
     0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0.]
    [[0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 1. 1. 1. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0.]]



![output_133_1](https://user-images.githubusercontent.com/49030506/83438775-fc257780-a40f-11ea-986e-fd755264c597.png)


I'll explore this dataset further in the next exercise and see how NMF can learn the parts of images.

###### NMF learns the parts of images

Now use what you've learned about NMF to decompose the digits dataset. You are again given the digit images as a 2D array samples. This time, you are also provided with a function show_as_image() that displays the image encoded by any 1D array:

def show_as_image(sample):
    
    bitmap = sample.reshape((13, 8))
    
    plt.figure()
    
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    
    plt.colorbar()
    
    plt.show()
    
After you are done, take a moment to look through the plots and notice how NMF has expressed the digit as a sum of the components!


```python
def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()
```


```python
# Importing NMF
from sklearn.decomposition import NMF 

# Creating an NMF model: model
model = NMF(n_components=7)

# Applying fit_transform to samples: features
features = model.fit_transform(samples)

# Calling show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Assigning the 0th row of features: digit_features
digit_features = features[0,:]

# Printing digit_features
print(digit_features)

```


![output_137_0](https://user-images.githubusercontent.com/49030506/83438776-fc257780-a40f-11ea-8639-26ae92d7f027.png)



![output_137_1](https://user-images.githubusercontent.com/49030506/83438778-fcbe0e00-a40f-11ea-983b-b1f7bd2d9edf.png)



![output_137_2](https://user-images.githubusercontent.com/49030506/83438780-fcbe0e00-a40f-11ea-8782-9f28bd22008d.png)



![output_137_3](https://user-images.githubusercontent.com/49030506/83438781-fcbe0e00-a40f-11ea-8443-13cf84f167c1.png)



![output_137_4](https://user-images.githubusercontent.com/49030506/83438782-fd56a480-a40f-11ea-9218-8d9a1a84e5c3.png)



![output_137_5](https://user-images.githubusercontent.com/49030506/83438783-fd56a480-a40f-11ea-96b0-b008362b412b.png)



![output_137_6](https://user-images.githubusercontent.com/49030506/83438784-fd56a480-a40f-11ea-9872-8baa43560c49.png)


    [4.76823559e-01 0.00000000e+00 0.00000000e+00 5.90605054e-01
     4.81559442e-01 0.00000000e+00 7.37551667e-16]


Take a moment to look through the plots and notice how NMF has expressed the digit as a sum of the components!

###### PCA doesn't learn parts

Unlike NMF, PCA doesn't learn the parts of things. Its components do not correspond to topics (in the case of documents) or to parts of images, when trained on images. It can be verified by inspecting the components of a PCA model fit to the dataset of LED digit images from above.



```python
# Importing PCA
from sklearn.decomposition import PCA

# Creating a PCA instance: model
model = PCA(n_components=7)

# Applying fit_transform to samples: features
features = model.fit_transform(samples)

# Calling show_as_image on each component
for component in model.components_:
    show_as_image(component)

```


![output_140_0](https://user-images.githubusercontent.com/49030506/83438785-fd56a480-a40f-11ea-924d-dd7a71a7ae7a.png)



![output_140_1](https://user-images.githubusercontent.com/49030506/83438786-fdef3b00-a40f-11ea-98c4-3c5f724fac1a.png)



![output_140_2](https://user-images.githubusercontent.com/49030506/83438788-fdef3b00-a40f-11ea-98a4-455546f5dd0a.png)



![output_140_3](https://user-images.githubusercontent.com/49030506/83438791-fdef3b00-a40f-11ea-8bd2-b7cbb814be05.png)



![output_140_4](https://user-images.githubusercontent.com/49030506/83438792-fdef3b00-a40f-11ea-9ded-db418f7366a2.png)



![output_140_5](https://user-images.githubusercontent.com/49030506/83438793-fe87d180-a40f-11ea-8e6a-9ef9818a8684.png)



![output_140_6](https://user-images.githubusercontent.com/49030506/83438794-fe87d180-a40f-11ea-9cc0-d900dd06dc20.png)


Noticing that the components of PCA do not represent meaningful parts of images of LED digits!

###### Which articles are similar to 'Cristiano Ronaldo'?

In the video, you learned how to use NMF features and the cosine similarity to find similar articles. Apply this to your NMF model for popular Wikipedia articles, by finding the articles most similar to the article about the footballer Cristiano Ronaldo. The NMF features you obtained earlier are available as nmf_features, while titles is a list of the article titles.


```python
# Performing the necessary imports
import pandas as pd
from sklearn.preprocessing import normalize

# Normalizing the NMF features: norm_features
norm_features = normalize(nmf_features)

# Creating a DataFrame: df
df = pd.DataFrame(norm_features, index=titles)

# Selecting the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo']

# Computing the dot products: similarities
similarities = df.dot(article)

# Displaying those with the largest cosine similarity
print(similarities.nlargest())

```

    Cristiano Ronaldo                1.000000
    Franck Ribéry                    0.999972
    Radamel Falcao                   0.999942
    Zlatan Ibrahimović               0.999942
    France national football team    0.999923
    dtype: float64


Although one may need to know a little about football (or soccer, depending on where one is from!) to be able to evaluate for oneself the quality of the computed similarities!


```python
fp5 = '/Users/MuhammadBilal/Desktop/Data Camp/Unsupervised learning in Python/Data/Musical artists/artists_sample.csv'
```


```python
artists = pd.read_csv(fp5)
```


```python
artists.shape
```




    (111, 3)




```python
artist_names = ['Massive Attack',
 'Sublime',
 'Beastie Boys',
 'Neil Young',
 'Dead Kennedys',
 'Orbital',
 'Miles Davis',
 'Leonard Cohen',
 'Van Morrison',
 'NOFX',
 'Rancid',
 'Lamb',
 'Korn',
 'Dropkick Murphys',
 'Bob Dylan',
 'Eminem',
 'Nirvana',
 'Van Halen',
 'Damien Rice',
 'Elvis Costello',
 'Everclear',
 'Jimi Hendrix',
 'PJ Harvey',
 'Red Hot Chili Peppers',
 'Ryan Adams',
 'Soundgarden',
 'The White Stripes',
 'Madonna',
 'Eric Clapton',
 'Bob Marley',
 'Dr. Dre',
 'The Flaming Lips',
 'Tom Waits',
 'Moby',
 'Cypress Hill',
 'Garbage',
 'Fear Factory',
 '50 Cent',
 'Ani DiFranco',
 'Matchbox Twenty',
 'The Police',
 'Eagles',
 'Phish',
 'Stone Temple Pilots',
 'Black Sabbath',
 'Britney Spears',
 'Fatboy Slim',
 'System of a Down',
 'Simon & Garfunkel',
 'Snoop Dogg',
 'Aimee Mann',
 'Less Than Jake',
 'Rammstein',
 'Reel Big Fish',
 'The Prodigy',
 'Pantera',
 'Foo Fighters',
 'The Beatles',
 'Incubus',
 'Audioslave',
 'Bright Eyes',
 'Machine Head',
 'AC/DC',
 'Dire Straits',
 'MotÃ¶rhead',
 'Ramones',
 'Slipknot',
 'Me First and the Gimme Gimmes',
 'Bruce Springsteen',
 'Queens of the Stone Age',
 'The Chemical Brothers',
 'Bon Jovi',
 'Goo Goo Dolls',
 'Alice in Chains',
 'Howard Shore',
 'Barenaked Ladies',
 'Anti-Flag',
 'Nick Cave and the Bad Seeds',
 'Static-X',
 'Misfits',
 '2Pac',
 'Sparta',
 'Interpol',
 'The Crystal Method',
 'The Beach Boys',
 'Goldfrapp',
 'Bob Marley & the Wailers',
 'Kylie Minogue',
 'The Blood Brothers',
 'Mirah',
 'Ludacris',
 'Snow Patrol',
 'The Mars Volta',
 'Yeah Yeah Yeahs',
 'Iced Earth',
 'Fiona Apple',
 'Rilo Kiley',
 'Rufus Wainwright',
 'Flogging Molly',
 'Hot Hot Heat',
 'Dredg',
 'Switchfoot',
 'Tegan and Sara',
 'Rage Against the Machine',
 'Keane',
 'Jet',
 'Franz Ferdinand',
 'The Postal Service',
 'The Dresden Dolls',
 'The Killers',
 'Death From Above 1979']

```

##### Recommending musical artists part I

In this exercise and the next, you'll use what you've learned about NMF to recommend popular music artists! You are given a sparse array artists whose rows correspond to artists and whose columns correspond to users. The entries give the number of times each artist was listened to by each user.

In this exercise, build a pipeline and transform the array into normalized NMF features. The first step in the pipeline, MaxAbsScaler, transforms the data so that all users have the same influence on the model, regardless of how many different artists they've listened to. In the next exercise, you'll use the resulting normalized NMF features for recommendation!


```python
# Performing the necessary imports
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

# Creating a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Creating an NMF model: nmf
nmf = NMF(n_components=20)

# Creating a Normalizer: normalizer
normalizer = Normalizer()

# Creating a pipeline: pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)

# Applying fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)

```

###### Recommending musical artists part II

Suppose you were a big fan of Bruce Springsteen - which other musicial artists might you like? Use your NMF features from the previous exercise and the cosine similarity to find similar musical artists. A solution to the previous exercise has been run, so norm_features is an array containing the normalized NMF features as rows. The names of the musical artists are available as the list artist_names.



```python
# Importing pandas
import pandas as pd

# Creating a DataFrame: df
df = pd.DataFrame(norm_features, index=artist_names)

# Selecting row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Computing cosine similarities: similarities
similarities = df.dot(artist)

# Displaying those with highest cosine similarity
print(similarities.nlargest())

```

    Bruce Springsteen    1.000000
    Interpol             0.996395
    Fiona Apple          0.995637
    The Mars Volta       0.995403
    Yeah Yeah Yeahs      0.992239
    dtype: float64


Above are all the similar articles. If a reader is interested to read one of them, he/she might find the others interesing as well. 

# Customer Segmentation 


```python
fp6 = '/Users/MuhammadBilal/Desktop/Machine Learning/Machine Learning A-Z New /Part 4 - Clustering/Section 24 - K-Means Clustering/Mall_Customers.csv'
```


```python
dataset = pd.read_csv(fp6)
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>Genre</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>Male</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>Male</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>Female</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>Female</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>Female</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
""""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```


![output_157_0](https://user-images.githubusercontent.com/49030506/83438796-fe87d180-a40f-11ea-874c-b6a6168be956.png)



![output_157_1](https://user-images.githubusercontent.com/49030506/83438798-fe87d180-a40f-11ea-8d0e-11d68d90ea7c.png)


It can be seen from the chart that customers with lowest and highest incomes are spending the most. Poeple with middle income are spending money modelately. Approximately half of the high income customers are not spending much. This information can be really insightful for any company to develop strategies and target their customers more effectively.  

I have applied unsupervised techniques on real world datasets and also built the knowledge of python along the way. In particular, I used scikit-learn and scipy for unsupervised learning challenges. I harnessed both clustering and dimension reduction techniques to tackle problems with real world datasets such as clustering wikipedia documents by the words they contain and recommending musical artists to consumers. 

## Thanks a lot for your attention. 


```python

```
