---
title: "Singular Value Decomposition(SVD) in python from scratch"
date: 2021-09-24T17:46:41+05:30
tags: ["statistics", "machine learning", "linear-algebra"]
comments: true
mathjax: true

---

## Singular Value Decomposition in python from scratch

According to wikipedia, SVD is a factorization of a real or complex matrix. For more information, please go through this [wiki](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi1nNHegZbzAhVQ8XMBHdKfB8gQFnoECAYQAQ&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FSingular_value_decomposition&usg=AOvVaw324yDU0a2HghF_7qvPPR8t) link.

A matrix M can be written  in terms of factorization as under.
$$
M = UΣV^T
$$


Where M is (mxn) matrix.

**Steps to compute SVD** : 

1. Columns of V are eigen vectors of $M^TM$.
2. Columns of U are eigne vectors of $MM^T$.
3. The singular values in $Σ$ are the square root of eignen values from $M^TM$ or $MM^T$. They are in the diagonal of Σ matrix and arranged in decreasing order.

**Python implementation**:

```python
A = np.asarray([[2, 4], [1,3], [0, 0], [0, 0]])
```

```python
# write A as  A=UΣ(V.T)
# Assume that A is a full rank matrix.
# For more information, I have given a link in the reference section. Do checkout.
AA_T = A.dot(A.T)
print(AA_T.shape)
u_eval, u_evect = np.linalg.eig(AA_T)
print(u_eval.shape, u_evect.shape)
U = u_evect
```



```python
# V.T
A_TA = A.T.dot(A)
print(A_TA.shape)
v_eval, v_evect = np.linalg.eig(A_TA)
print(v_eval.shape, v_evect.shape)
V_T = v_evect
```

outputs:

```tex
U=
array([[ 0.81741556, -0.57604844,  0.        ,  0.        ],
       [ 0.57604844,  0.81741556,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  1.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
```

```tex
V=
array([[-0.9145143 , -0.40455358],
       [ 0.40455358, -0.9145143 ]])
```

Note that V is not transposed yet. (in case you need a transpose one, make sure you do that yourself.)



eigen values of $A^TA$ : (you can take either of $A^TA$ or $AA^T$)

```tex
 array([0.36596619, 5.4649857 ]))
```

<br>

#### Let's cross check using numpy inbuilt linear algebra module.

```python
u, sigma, vh = np.linalg.svd(a)
```

Output:

```tex
U=
array([[-0.81741556, -0.57604844,  0.        ,  0.        ],
       [-0.57604844,  0.81741556,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  1.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
 
V.T =
array([[-0.40455358, -0.9145143 ],
       [-0.9145143 ,  0.40455358]])
       
singular values=
array([5.4649857 , 0.36596619])
```



Negative sign means the direction of eigen vectors gets reversed while keeping the magnitude as a liinear function of singular values. 

##### References:

* https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm

  