---

title: "Sentence Autocompletion Using N-Gram Language Model"
date: 2021-10-20T11:23:05+05:30
tags: ['natural language processing', 'data science']
Mathjax: true
---

---

We almost use Google everyday for web surfing. And you would must have come across the thing that you see in the above picture. Its amazing, isn't it! Let me ask you one thing, how Google knows everything that we are looking for?. Well, we call it ***autocompletion*** in technical term. So, Let's be Byomkesh Bakshi and solve this mystery together. By the end of this post, you will be able to build your own simple and powerful *autocompletion* system.

**Table of contents:**

1. [Probability refresher](#probability_refresher)
2. [N-grams](#n_grams)
6. [Python implementation](#python_implementation)
4. [Evaluation](#evaluation)

we will see all ot these one by one. So buckle up for some interesting stuff that  I am going to walk you through for next 5-10 minutes.



<a id='probability_refresher'></a>

# Probability refresher

Probability is the building block for many language models and so is for autocompletion. A language model assign the probability to sequence of words and the most probable sequence is suggested by the search engines. If you know about the basics of conditional probability and Bayes theorem then you would have guessed what I meant by the probability of sequence of words. If you haven't heard about it or have forgotten, then don't worry. I have got your back. I will cover about these concepts in brief  that will be enough for you to understand the entire workflow of language model based autocorrection system. So, let's get started.

**Conditional probability**  of an event A given that event B has already happend is given by
$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)} \tag{1}
$$
Where, $P(A \cap B)$ Is the joint probability (chance of happening event A and B together) and  $P(B)$ is the probablity of the event B.    From equation (1)  Joint probability of A and B can be written as
$$
P(A \cap B)  = P(A|B)P(B) \tag{2}
$$
Now, we will see how can we apply conditional probability to a languagae model.

For a given sentence 

> I am a man of science, not someone's snuggle bunny!

Suppose a user types "I am a man", then the next word X is suggested such that the sentence "I am a man X" has the highest probability.  Mathematically  speaking, we just need to find a word X that maximize the joint probability of "I am a man" and X. That is for a suggested X, $P(I \ am \ a \ man \ X)$ is maximum and we can use equation (2) to calculate joint priobaility. We will now dive deeper into probability calculation for sequence of words.

<a id='n_grams'></a>

# N-Gram language model

In simple words, Language model are those that assign probability to sequence of words. **N-gram** LM is a simplest language model that assigns probability to sequecne of words. An **N-gram** is a squence of n words. one-gram is the sequence of one word, bi-gram is sequence of 2 words and so on. For clarity, take the example sentence from porevious section. 

The corresponding 1-gram would be 

`[I, am, a, man, of, science, not, someone's, snuggle, bunny]`

The corresponding bi-gram would be

`[I am, am a, a man, man of, of science, science not, not  someone's, someone's snuggle, snuggle bunny]`

Similarly trigram means sequence of three words at a time e.i. `I am a, am a man, ...`

Now that we know the meaning of n-gram, let's see how can we laverage them to calculate the probability of a sentence. We will use equation (2) to calculate the probability of a sentence.

Pbrobability of next word w given that it has some history h is given by
$$
P(w|j) = \frac{C(w,h)}{C(h)} \tag{3}
$$
where $C(w, h)$ is the count of word $w$ and $h$ occuring together in the corpus and $C(h)$ is the count of history $h$.  Let's understand it with an example.

We will again take the same sentence from section [1](probability_refresher) for illustration. Suppose that you have a sequence "I am a" and you want to calculate the probability of next word "man". Then the probability of word w as `man` given its history h as `I am a` can be written as
$$
P(man|I \ am \ a) = \frac{C(I \ am \ a \ man)}{C(I \ am \ a)}
$$
For a small corpora,  using count for probability calculation can be feasible but text has no boundary. It's dynamic in nature. New sentences ae created all the time and we won't always be able to count the entire sentences. It seems rather much hardwork that we as an eneginneer always avoid by using proxies. We will use our probabilty knowledge to make modification to equation (3). But before that we need to juggle around with couple of notations. 

we will represent the joint probaility of a sequence of words $w_1, w_2, w_3...w_n$  by $P(w_1, w_2, w_3..., w_{n-1}, w_n)$ .We will represent the sequence of n words as either $w_1, w_2, w_3...w_n$ or $w_{1:n}$ . We will also laverage the **chain rule of probability** to calculate the probability of sequence of words, that is given by
$$
\begin{aligned}
P(w_1, w_2, w_3, ...w_n)= 
&P(w_1)P(w_2|w_1)p(w_3|w_{1:2})...P(w_n|w_{1:n-1})\newline
=&\prod_{k=1}^{n}P(w_k|w_{1:k-1}) \end{aligned} \tag{4}
$$
Let's see how the equation (3) works in text scenario for calculating the probability of sequence of words. Again taking the same example as in section 1, we will see how can we estimate the probability of a sentence. 

We will slice the sentence a little bit because of space constraint and apply the equation (4) on it.

***P (I am a man of science)*** can be written as 
$$
\begin{aligned}
=P(I)P(am | I)P(a|I \ am)p(man | I \ am \ a) P(of |I \ am \ a \ man)P(science| I \ am \ a \ man \ of)
\end{aligned}
$$
We see that probaility of next word is dependent on all the previous word. Did you sense something here🤔? If not, let me again remind you of engineering thing and try to find a easier wat out of such lengthy and messy probability calculation. There comes the **Markov assumption**, it says that we don't really need to look too far in the past to calculate the probability of a word in the future. We can generalize the assumption that for a bigram model that looks for one word in the past and trigram model that looks for two words in the past thus for **n-gram**, we just need to look for **n-1** words in the past.  For bigram assumption, equation (4) can be rewritten as
$$
\begin{aligned}P(w_1, w_2, w_3, ...w_n)= \prod_{k=1}^{n}P(w_k|w_{k-1}) \end{aligned} \tag{4.1}
$$


But how do estimate the n-gram probabilities? If you can recall the **MLE** or **Maximum Likelihood Estimation**, we can get the estimate of probability for  n-gram model using the count of occurence of n-gram in the corpus normalized by the count of n-1 gram so that the value lies between 0 and 1. Say, for bigram ***xy***, the probability of **y** given **x** can be calculated using the count of bigram **xy** normalized by the sum of count of all the words that share the same first word **x**. Mathematically, we can represent the probability of bigram ***xy*** as
$$
P(xy) = \frac{C(xy)}{\sum_wxw}
$$
In which the denominator is same as the count of unigram **x**.
$$
P(xy) = \frac{C(xy)}{C(x)}
$$
In general, the probablity of $n^{th}$ word of a bigram grammar can be calculated as
$$
P(w_n|w_{n-1}) = \frac{C(w_{n-1}w_n)}{C(w_{n-1})}\tag{5}
$$
Let's workout these with an example of a small corpus having three sentences.

```tex
<s>I am Sheldon</s>
<s>Sheldon I am</s>
<s>I make every conversation a jam</s>
```

**Note** that we have put a prefix \<s> to give us the bigram context of the first word and suffix \</s>  for every sentence in the corpus to make bigram grammar a true probaility distribution. If we do not add \</s> symbol , then the probabilities for all the sentence of a given length would sum to one. This model would define an infinite set of probabilities distribution per sentence length. 

In the above example, $P(I|am)$ can be written as
$$
P(am|I)= \frac{Count(I am)}{Count(I)}
$$
that is
$$
P(am|I) = \frac{2}{3} \ = 0.67
$$
and the probability of the first sentence can be calculated as
$$
\begin{align*}
P(I am Sheldon) =
&P(\<s\>|I)P(am|I)P(Sheldon|am)P(\</s\>|Sheldon) \newline
=&\frac{2}{3}*\frac{2}{3}*\frac{1}{2}*\frac{1}{2} \newline
=&\frac{1}{9} = 0.11
\end{align*}
$$
So, This is how, probability of sequence of words is calculated. Though it seems straightforward but there are some practical issue when delaing with real  life text dataset. Let' see what are those issues and how can we diagnose them.

#### Some practial issues

Although we have used bigram model, but for large training dataset, It is more common to use **trigram** model conditioned on previous two words and **4-gram** model conditioned on previous three word. Also keep in mind that to include the context of first word in probability calculation , for **trigram** model, you wll have to augment the sentence with 2 start token i.e. first trigram would be $P(I|\<s\>\<s\>)$ And first 4-gram would be $P(I|\<s\>\<s\>\<s\>)$.

In practice, we use **log probability** to avoid numerical underflow. If you have a longer sequence of words, then the traditional probability calculation can leads to underflow as the probabilies are between 0 and 1 and if we multiply it for n number of times where n is very very large then the probability will convere to 0 and will cause numerical underflow.  Therefore we use summation of log prbabilities instead of multiplication of traditional probabilities to calculate the probability for sequence of words.

Now we will implement in python what have we learnt so far.

<a id='python_implementation'></a>

# Python Implementation

We will use twitter.txt dataset. I have modified it for personal use. Feel free to experiment with custom dataset. Withput any further due, let's get started. You can download the dataset from [here](https://github.com/chandan5362/Data-Science-Hackathons/blob/a2f5fc7b68b66c524d0e33fb1a1f2bd157121cb2/N%20gram%20based%20Autocomplete%20language%20Model/twitter.txt).

Let us write down the necessary step that will help us to write the concise logic in sequential manner. Our implementation consists of following steps.

* Data Pre-preprocessing.
* split the data into train and test set (4:1).
* create a closed from vocabulary using training set only.
* Transform  the train and test set based on newly created vocabulary.
* Build n-gram language model.

we will see all of it one by one. So hold on to your cofee and bare with me for some more time.  We will test each and every code blocks with our custom small corpora from the previous section.

Firstly, we will read the text data using following code.

Import the required libraries.

```python
import numpy as np
import pandas as pd
import nltk
import random
```

```python
# read the text data
with open('twitter.txt', 'r') as f:
    data = f.read()
```

There are total of three million characters in our dataset including linebreaks and whitespaces. Now our next step is to clean it for our purpose. Let's do that.

### Data pre-processing

Codes are very well annotated and documented, so I think that will be sufficient for you to understand every functionality of implemented codes. For better clarification, I will also share the notebook at the end of this post.

```python
 def Preprocessing(data):
    """
    Preprocess the text corpus
    split the data by line breaks '\n'

    Parameters
    ----------
    data : str
        corpus in raw format

    Returns
    -------
    list_sentences :  list
        preprocessed list of list of tokens for each sentence.

    """

    # split by linebreaks
    sentences = data.split("\n")

    # strip whitespaces
    sentences = [s.strip() for s in sentences]

    #remove empty sentence
    sentences = [s for s in sentences if len(s) > 0]

    #tokenize, # and punctuation
    list_sentences = [s.lower().split(" ") for s in sentences]
    
    
    return list_sentences
```

```python
tokenized_sentences = Preprocessing(data)
len(tokenized_sentences)
```

### Dataset splitting


```python
# split the data into train and test

random.seed(42)
random.shuffle(tokenized_sentences)

#80/20 train and test
lt = int(len(tokenized_sentences)*0.8)
train_data = tokenized_sentences[:lt]
test_data = tokenized_sentences[lt:]

print(f'size of train data:{len(train_data)}\nsize of test data:{len(test_data)}')
```

```tex
size of train data:38368
size of test data:9593
```

Now we will create a count dictionary where we will store the count of every words of our training dataset.

```python
from collections import defaultdict, Counter
# get the count of all the words in the corpus
flat_list = [el for sentence in tokenized_sentences for el in sentence]
```


```python
word_count = Counter(flat_list)
word_count.most_common(10)
```


```tex
[('the', 18863),
 ('to', 15647),
 ('i', 14263),
 ('a', 12254),
 ('you', 9732),
 ('and', 8660),
 ('for', 7716),
 ('in', 7469),
 ('is', 7268),
 ('of', 7221)]
```

You can access the count of a given word using `get` keyword.


```python

word_count.get('the')

```


```tex
18863
```



#### Handle OoV(Out of Vocabulary) word

1. create a closed vocabulary( Vocabulary consisting only of training data words)
2. words that does not appear more frequently, assign then an unknows token \<oov>



```python
def create_closed_vocab(tokens, freq_threshold):
    """
    create a closed vocabulary for a given text corpus
    
    Parameters
    ----------
    tokens: list
        list of sentence tokens
    freq_threshold: int
        word frequemcy threshold
        
    Returns
    -------
    closed_vocab : list
        list of tokens having frequency greater than the threshold
        
    """
    
    closed_vocab = []
    for sentence_t in tokens:
        for word in sentence_t:
            if word_count.get(word) > freq_threshold:
                closed_vocab.append(word)
                
    return set(closed_vocab)
    
```

In the above code block, we have used a `freq_threshold` keyword. It has been used to ignore the word tokens whose count is less than the given threshold (1 in this case). And later we can replace these words using out-of-vocabulary token \<oov> token.

#### Steps to create a closed form vocabulary

1. Get the word count for each of the tokens in training corpus.
2. Replace the unseen word in training set (from closed from vocab) with \<oov> token.




```python
#get closed  form vocab using training set
closed_vocab =  create_closed_vocab(train_data, freq_threshold=2)
```


```python
len(closed_vocab)
```


```tex
14905
```

So, there are only 14905 word tokens in the closed form vocabulary.


```python
def PreprocessTrainTest(data, vocab):
    """
    Handles unknown/missing word in training corpus with <oov> token
    
    parameters
    ----------
    data: list 
        list of list of tokens
    vocab: set
        set containing unique words in traiing data
        
    Returns
    -------
    replaced_sentence: list
        preprocessed list of list of tokens
    
    """
    #replace the token not in vocab with <oov>
    replaced_sentence = []
    for sentence in data:
        temp_sentence = []
        for token in sentence:
            if token not in vocab:
                temp_sentence.append('<oov>')
            else:
                temp_sentence.append(token)
        replaced_sentence.append(temp_sentence)
    
    return replaced_sentence  
```


```python
# preprocess train data
processed_train_data = PreprocessTrainTest(train_data, closed_vocab)
```


```python
#preproces test data
processed_test_data = PreprocessTrainTest(test_data, closed_vocab)
```

Now we will write a function `create_n_gram`  to create n-gram dictionary.


```python
def create_n_gram(data, n=2, start_token= '<s>', end_token='<e>'):
    """
    Create n-gram dictionary.
    
    Parameters
    ----------
    data: list
    	list of list of word tokens
 		
 		Returns
 		-------
 		n-gram based dictionary.
 		
    """
    
    n_gram_dict = {}
    for sentence in data:
        # append n start token in the begining of each tokens
        tokens = sentence.copy()
        if n==1:
            tokens.insert(0, start_token)

        for i in range(n-1):
            tokens.insert(i, start_token)
        tokens.append(end_token)
        
        #create n gram count dictionary
        for i in range(len(tokens)):
            key = tuple(tokens[i:i+n])
            if len(key) == n:

                if key in n_gram_dict:
                    n_gram_dict[key]+=1

                else:
                    n_gram_dict[key]= 1
        
    return n_gram_dict
```

Now we will see how n-gram dictionary looks on our custom corpora.

```python
sentences = [['I' , 'am', 'Sheldon'],
                 ['Sheldon', 'I', 'am'],
             ['I', 'make', 'every', 'conversation', 'a', 'jam']]
```




```python
 unique_words = list(set(sentences[0] + sentences[1] + sentences[2]))
```


```python
create_n_gram(sentences, n=2)
```

Bsically, our n-gram dictionary is the dictionary containing the n-gram word as key and their count as you can see below.

Output for bigram-


```tex
{('<s>', 'I'): 2,
 ('I', 'am'): 2,
 ('am', 'Sheldon'): 1,
 ('Sheldon', '<e>'): 1,
 ('<s>', 'Sheldon'): 1,
 ('Sheldon', 'I'): 1,
 ('am', '<e>'): 1,
 ('I', 'make'): 1,
 ('make', 'every'): 1,
 ('every', 'conversation'): 1,
 ('conversation', 'a'): 1,
 ('a', 'jam'): 1,
 ('jam', '<e>'): 1}
```


```python
#one gram dictionary
create_n_gram(sentences, 1)
```


```tex
{('<s>',): 3,
 ('I',): 3,
 ('am',): 2,
 ('Sheldon',): 2,
 ('<e>',): 3,
 ('make',): 1,
 ('every',): 1,
 ('conversation',): 1,
 ('a',): 1,
 ('jam',): 1}
```

We will be using laplacian-k smooting to handle the probability underflow of missing n-gram. You can go through this [StackExchange](https://stats.stackexchange.com/questions/108797/in-naive-bayes-why-bother-with-laplace-smoothing-when-we-have-unknown-words-in) thread to know more of its use cases and calculation.


```python
## Estimate the porobaility
# use laplacian-k smoothing to handle the probability underflow for missing N gram

def calculate_probabilities(previous_n_gram, vocab, prev_n_gram_count, n_gram_count, end_token ='<e>', oov_token='<oov>', k=1 ):
    """
    Parameters
    ----------

    previous_n_gram: tuple 
        sequenc of previous words of length n
    vocab: set
        set of unique word in training corpus
    prev_n_gram_count : dict
        dictionary for prev n-gram count
    n_gram_count : dict
        dictionary for n+1 gram count
        
    Return
    ------
    dictionary of joint probability of each word and previous n-gram word
    
    """
    # since start token ca not be the end of a n-gram
    #we didn't include start token in vocab
    vocab_new = vocab + [oov_token, end_token]
    probabilities = {}
    for word in vocab_new:
        joint_words = previous_n_gram +(word,)

        count = n_gram_count[joint_words] if joint_words in n_gram_count else 0
        prev_count = prev_n_gram_count[previous_n_gram] if previous_n_gram in prev_n_gram_count else 0
      
        #apply k smoothing
        prob = (count + k)/(prev_count + len(vocab)*k)
        probabilities[word] = prob
    
    return probabilities
```


```python
prev_n_gram_count = create_n_gram(sentences, 1)
n_gram_count = create_n_gram(sentences, 2)
calculate_probabilities(('a',), unique_words.copy(),  prev_n_gram_count, n_gram_count)
```

We check the above code on our custom corpora to see whether the following word given a previous n-1 gram word makes sense or not. Since we re using bi-gram languag model for our language model, We pass a one-gram word `a` to the `calculate_probailities` fucntion to predict the highly probable next word. In the following output snippet, you can see that the word `jam` has the highest probabilty after `a`.  And also, you can verify from the corpora that `a jam` is a bigram that is actually there in the third sentence.


```tex
{'Sheldon': 0.1111111111111111,
 'am': 0.1111111111111111,
 'a': 0.1111111111111111,
 'conversation': 0.1111111111111111,
 'jam': 0.2222222222222222,
 'I': 0.1111111111111111,
 'make': 0.1111111111111111,
 'every': 0.1111111111111111,
 '<oov>': 0.1111111111111111,
 '<e>': 0.1111111111111111}
```

For better understanding, we will see the count matrix (matrix containoing the row as the word of the sentence and columns as the each tokens of the vocabulary and the value at (row, col) is the count of the word (prev_n-1_gram, words_of_vocab )) and probability matrix(probability of every words of the vocabulary following a given word). It will become more clear once we look a the matrices.


```python
# lets plot them using matrix
def show_as_matrix(sentence, 
                   vocab, 
                   sentences,
                   start_token='<s>', 
                   end_token ='<e>', 
                   oov_token='<oov>',
                   n=2, 
                   k=1):
    """
    Prints the count matrix and probability matrix as dataframe for better visualization.
    
    """
  
    
    n_gram = []
    sentence = [start_token]*(n-1) + sentence

     
    vocab_new = list(set(vocab + [oov_token, end_token]))
    
    prev_n_gram_count = create_n_gram(sentences, n-1)
    n_gram = list(set(prev_n_gram_count.keys())) 
    n_gram_count = create_n_gram(sentences, n)
    e 
    # zero initialize the count matrix and probability matrix
    count_matrix = np.zeros((len(n_gram), len(vocab_new)))
    prob_matrix = np.zeros((len(n_gram), len(vocab_new)))
    
    for i in range(len(n_gram)):
        for j in range(len(vocab_new)):
            n_token = n_gram[i] + (vocab_new[j],)
            count_n_token = n_gram_count[n_token] if n_token in n_gram_count else 0
            count_matrix[i,j] = count_n_token
            prob_matrix[i,j] = calculate_probabilities(n_gram[i], vocab_new,prev_n_gram_count, n_gram_count)[vocab_new[j]]
    
    count_matrix = pd.DataFrame(count_matrix, index=n_gram, columns=vocab_new)
    prob_matrix = pd.DataFrame(prob_matrix, index=n_gram, columns=vocab_new)
    
    return count_matrix, prob_matrix
    
```

We are considering trigram for our model. You can play around woth the value of `n` if you would like to.

```python
# we re considering a trigram language model
cm, pm = show_as_matrix(unique_words, unique_words,sentences, n=3 )
```

Now, Since we have both the required matrices namely, **probability matrix** and **count matrix**,  we are ready to develop an autocomplete model. Let's jump right onto that.  We will define a function `predict_nextword` that returns the next probable word given previous **n-1** gram.


```python
def predict_nextword(sentences, sentence, vocab, n = 2, start_token ='<s>', end_token ='<e>', begins_with='d', suggestion_freq=2):
    """
    parameters
    ----------
    sentences: list of list of sentence tokens
        list of words of sentence
    sentence :  list 
        list of toknes whose pp is to be calculated
    vocab :  list 
        list of unique words in the corpus
    n : int
        expected n gram
        
    Returns
    -------
    a list of predicted k words where k is suggestion_freq
    
    """
    
    #get previous n-1 word
    sentence = tuple(sentence)
    prev_word = sentence[-(n-1):]
    
    new_vocab = list(set(vocab + [end_token]))
    prev_n_gram_count = create_n_gram(sentences, n-1)
    n_gram_count = create_n_gram(sentences, n)
    
    
    # get the probability for prev word and next candidate word
    prob = calculate_probabilities(prev_word, new_vocab, prev_n_gram_count, n_gram_count)
    prob =  {k: v for k, v in sorted(prob.items(), reverse=True,  key=lambda item: item[1])}
    
    
    
    res = []
    for k, v in prob.items():
        if len(begins_with) > 0:
            if k.startswith(begins_with):
                res.append({k:v})
        else:
            res.append({k:v})
                
                
    return res[:suggestion_freq]
    print(res[:suggestion_freq])
```

Now that we have our function, we will evaluate it on the subset of sentence `['I', 'am']` and it should predict `sheldon` as `I am Sheldon`  is a sentence in our corpora. Let's check whether we have got our model right or not.




```python
#Test sentence
sentence = ['i', 'am']
```




```python
# predict the next word
predict_nextword(sentences, sentences[0][:-1],unique_words, begins_with='' )
```


```tex
[{'Sheldon': 0.18181818181818182}, {'<e>': 0.18181818181818182}]
```

Yay!😀 we got it right. it actually predicts the next two words (as the `suggestion_freq argument` is 2) correctly. `<e>` token denotes the end of the sentence.

Let's define another function `autocompleteSentence` for our convenience that will give us the correct sentence given an incorrect sentence as the argument.


```python
def autocompleteSentence(sentences, sentence, vocab, n = 2, start_token ='<s>', end_token ='<e>'):
    """
    parameters
    ----------
    sentences: list of list of sentence tokens
        list of words of sentence
    sentence :  list 
        list of toknes whose pp is to be calculated
    vocab :  list 
        list of unique words in the corpus

        
    Returns
    -------
        complete sentence based on n-gram probability
    """
    sentence = sentence.copy()
    curr_word = start_token
    while curr_word != end_token:
        prev_word = list(sentence[-(n-1):])

        curr_word = list(predict_nextword(sentences, prev_word, vocab, n, begins_with='', suggestion_freq=1)[0])[0]      

        sentence.append(curr_word)
        
    return " ".join(sentence[:-1])
    
```


```python
input_sentence = 'I make'
sentence = input_sentence.split(" ")
sentence
```

```tex
['I', 'make']
```

```python
# predict the next word for a given a particular sequence of a sentence
pred = autocompleteSentence(sentences, sentence, unique_words, n=3 )
pred
```

```tex
'I make every conversation a jam'
```

You can see that our model has successfully predicted the correct sentence as this is one of the original sentence of our corpora. One of the drawback of this n-gram lanaguage model is that it works well for smller corpora but it will miserably fail to predict on huge corpus. In fact, we will check this model against out test data and will see that it is not able to predict the next word correctly as it fails to capture the long term dependency. For suh complex tasks, you can use sequential models like RNN, LSTM etc to correctly predict the next word given some context.

#### Test on our custom dataset

```python
#original sentence
processed_test_data[3]
```

```tex
['thanks', 'for', 'the', 'follow', 'and', 'mention', '!']
```

```python
#test it
sentence = processed_test_data[3][:-2]
autocompleteSentence(processed_train_data, sentence, list(closed_vocab), n=3)
```

```tex
'thanks for the follow and support across the <oov>'
```

As mentioned previously, it has failed to predict the correct word.   

For every model it is always advisable to fix some evlation metric. For our language model, we will be using **Perplexity** score to evaluate on predicted sentences.

<a id='evaluation'></a>

# Model Evaluation

In practice we don't use raw probability as a metric to evaluate a language model but a different version of it called as **Perplexity**. The perplexity (sometime know as PP in short) of a languge model on a test set is the inverse probability of the test set, normalised by the number of words. For a test set $W = w_1, w_2, w_3....w_n$ , PP is given by
$$
PP(W) = P(w_1w_2w_3...w_n)^{-\frac{1}{N}}
$$
 Lower is the PP, higher is the chance of predicted sentence to be an actual sentence or more likely to make sense based on the context of corpus. There is a nice [pdf](https://web.stanford.edu/~jurafsky/slp3/3.pdf) by Stanford that explains best about the **Perplexity** . Infact the aforementioned abstract is taken from this pdf only. Let's code it down.

```python
def perplexity_of_sentence(sentences, sentence, vocab, n = 2, start_token ='<s>', end_token ='<e>'):
    """
    parameters
    ----------
    sentences: list of list of sentence tokens
        list of words of sentence
    sentence :  list 
        list of toknes whose pp is to be calculated
    vocab :  list 
        list of unique words in the corpus
    n : int
        expected n gram
        
    Returns
    -------
    pp : float
        perplexity score for a given sentence
        
    """
		prev_n_gram_count = create_n_gram(sentences, n-1)
    n_gram_count = create_n_gram(sentences, n)
    
    
    sentence = [start_token]*(n-1) + sentence +[end_token]
    sentence = tuple(sentence)
    
    N = len(sentence)
    
    pp_prob = 1
    for i in range(n-1, N):
        n_gram = sentence[i-n+1:i]
        
        word = sentence[i]
        
        #include start and end token in calculating prob
        # count the start and end token in sequence lenght also
        try:
            prob = calculate_probabilities(n_gram, vocab, prev_n_gram_count, n_gram_count)[word]
        except KeyError:
            prob = calculate_probabilities(n_gram, vocab, prev_n_gram_count, n_gram_count)['<oov>']
        pp_prob+=math.log((1/prob))
    
    
    pp = pp_prob**(float(1/N))
    return pp
    
```




```python
# check the perplexity score for different n gram predicted sentence
for n in range(2, 6):
    
    pred = autocompleteSentence(sentences, sentence, unique_words, n=n )
    score = perplexity_of_sentence(sentences, pred.split(" "), unique_words, n=n)
    
    print(f"for {n}-gram, PP: {score}")

```

```tex
for 2-gram, PP: 1.3328333831401098
for 3-gram, PP: 1.2914802650430801
for 4-gram, PP: 1.2624417182398258
for 5-gram, PP: 1.2390810140561817
```

We can see in the output section that as we increase the **n** in n-gram, PP decreases. In other words, predictions are likely to make much more sense. It is also obvious that if we increse the n, it captures the long term dependencies of the word in the past. Therefore the predicted words are likely to make much more sense.

That's all for now. I am really sorry for making this blog lengthy and some grammetical errors in the blog( I am trying to be good at it😌). Thanks for taking out your valuable time for reading this blog. 

You can find the implemented notebook [here](https://github.com/chandan5362/Data-Science-Hackathons/blob/master/N%20gram%20based%20Autocomplete%20language%20Model/Autocomplete%20Language%20Model.ipynb).

---

Happy learning📚

happy reading📖



