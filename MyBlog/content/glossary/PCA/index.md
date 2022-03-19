---
title: "Principal Component Analysis (PCA)"
date: 2021-09-02T17:46:41+05:30
tags: ["statistics", "machine learning", "linear-algebra"]
mathjax: true

---


PCA is very useful tool in machine learning. When it comes to dimensionality reduction, PCA is one of the basic and mostly used tools by ML practitioners. PCA is linear combination of variables which best explains the data.Theoretical explanation can be found [here](https://en.wikipedia.org/wiki/Principal_component_analysis).

#### Some assumptions that are used in PCA:
* __Linearity:__  pca is linear combination of variables.
* __Large variance implies more information:__ The direction of high variance accounts for most of the information in the dataset.
* __Orthogonality:__ principal components are orthogonal to each other.

<div style="text-align: center;"><h4>PCA in Python🐍</h4></div>

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
```

```python
# toy dataset (n_rows= 1000, n_cols =15 )
x_toy = np.random.random(size =(1000, 15))
x_toy.shape
```
output:
```txt
(1000, 15)
```



### Important Points

1. eigen vectors corresponding to highest eigen vector
contains most of the variance in the dataset, second eigen value contains the second highest variance in the dataset and so on.
2. eigen vectors represent the linear combination of original feature vectors.
3. Eigen vectors do not change their directions, when linear rotation(scaling) is applied to them.
   

### Steps :
1. Sandardizing (if the scale of the data is not uniform).

2. Zero center the data (substract the mean value)

3. find the covariance matrix.

4. find the eigen values and eigen vectors.

5. sort the eigen values and corresponding eigen vectors in descending order of eigen values.

6. retain the first `k` components of sorted eigen values to meet the `k` new feature dimension requirement.

7. project the original datapoints onto the line in the direction of eigen vectors.

  

Let's see these steps one by one.
##### 1. Standardization

Since our data already follows a uniform scale, Let's not do that here.

```python
x_scaled = StandardScaler().fit_transform(x_toy) 
```
**2.** **Zero centering**

 Mean centering ensures that the first component is proportional to the direction of maximum variance.

```python
x_toy = x_toy - x_toy.mean(axis = 0)
```





##### 2. covariance matrix


```python
# covariance matrix dimension: (n_clos, n_cols)
n_cols = 15
cov = np.cov(x_toy, rowvar = False, bias = True)
assert cov.shape == (n_cols, n_cols)
```
<br>

##### 3. eigen values and eigen vectors


```python
eig_val, eig_vect = np.linalg.eig(cov)
eig_val.shape, eig_vect.shape
```
output:
```txt
 ((15,), (15, 15))
```


There are 15 eigen values and corresponding to each eigen value, there is an eigen vector of shape (1, 15) e.i. for each `eig_val[i]`, there is an eigen vector `eig_vect[:, i]`.
<br>

##### 4. sorting in descending order of eigen values


```python
indices = [i for i in range(len(eig_val))]

#sort the index of eig_val in descending order
indices.sort(key = eig_val.__getitem__, reverse=True)

#sorted eigen vectors
sorted_eig_vect = eig_vect[:, indices]

#sorted eigen values
sorted_eig_val = eig_val[indices]
```
<br>

##### 5. Select first k eigen values and eigen vectors


```python
k = 2
k_eigen_vals = sorted_eig_val[:2]
k_eigen_vectors = sorted_eig_vect[:, :2]
```
<br>

##### 6. Projection of original datapoints along the principal axis
```python
new_dataset = x_toy.dot(k_eigen_vectors)
new_dataset.shape
```
output:
```txt
(1000, 2)
```
Voila! You have just learnt PCA🥳


Since, PCA is fairly old technique, there are some drawbacks.\
__Drawbacks:__
* It assumes the linear combination of features, which is not always true for every datasets.
* Some information is lost. Even though it does not cause much loss in the actual outcome, sometimes, it can be bothering if you are working on very sensitive issue like medical diagnostic.
* It is sensitive to the scaling of variables.
---
Happy learning📖\
keep smiling😊