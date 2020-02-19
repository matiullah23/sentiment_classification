
# import dependencies
import pandas as pd
from collections import defaultdict
from pathlib import Path
import nltk as nl
from nltk.tokenize import word_tokenize
import re
from nltk.tokenize.casual import TweetTokenizer
import numpy as np
import math
import os
import sys 
import operator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

nl.download('punkt')

# pass path as train and test folder


################ Functions######################
def get_data(path):
    my_dir_path_pos = path + '/positive'
    # create list to store text
    results = defaultdict(list)

    # loop through files and append text to list
    for file in Path(my_dir_path_pos).iterdir():
        with open(file, "r", encoding="utf8") as file_open:
            results["text"].append(file_open.read())

    # read the list in as a dataframe
    df_pos = pd.DataFrame(results)

    #set directory path
    my_dir_path_neg = path + '/negative'

    # create list to store text
    results_neg = defaultdict(list)

    # loop through files and append text to list
    for file in Path(my_dir_path_neg).iterdir():
        with open(file, "r", encoding="utf8") as file_open:
            results_neg["text"].append(file_open.read())
    # read the list in as a dataframe
    df_neg = pd.DataFrame(results_neg)
    df_neg.head()

    #add sentiment to both datasets and then combine them for test data 1 for positive and 0 for negative
    df_pos['Sentiment']=1
    df_neg['Sentiment']=0
    frames = [df_pos, df_neg]
    df = pd.concat(frames)

    # increase column width to see more of the tweets
    pd.set_option('max_colwidth', 140)

    # reshuffle the tweets to see both pos and neg in random order
    df = df.sample(frac=1).reset_index(drop=True)

    # explore top 5 rows
    df.head(5)
    return df

def clean(df):

    # Remove any markup tags (HTML), all the mentions of handles(starts with '@') and '#' character
    def cleantweettext(raw_html):
        pattern = re.compile('<.*?>')
        cleantext = re.sub(pattern, '', raw_html)
        cleantext = " ".join(filter(lambda x:x[0]!='@', cleantext.split()))
        cleantext = cleantext.replace('#', '')
        return cleantext

    def removeat(text):
        atlist=[]
        for word in text:
            pattern = re.compile('^@')
            if re.match(pattern,word):
                #cleantext1 = re.sub(pattern, word[1:], word)
                atlist.append(word[1:])
            else:
                atlist.append(word)
        return atlist

    def tolower(text):
        lowerlist=[]
        for word in text:
            pattern = re.compile('[A-Z][a-z]+')
            if re.match(pattern,word):
                cleantext1 = re.sub(pattern, word.lower(), word)
                lowerlist.append(cleantext1)
            else:
                lowerlist.append(word)
        return lowerlist

    cleantweet= []
    for doc in df.text:
        cleantweet.append(cleantweettext(doc))


    tokentweet=[]
    df.text= cleantweet
    for doc in df.text:
        tokentweet.append(TweetTokenizer().tokenize(doc))
    df.text= tokentweet

    removeattweet=[]
    for doc in df.text:
        removeattweet.append(removeat(doc))
    df.text =removeattweet

    lowertweet=[]
    for doc in df.text:
        lowertweet.append(tolower(doc))
    df.text = lowertweet

    tweets=[]
    for x in df.text:
        tweet = ''
        for word in x:
            tweet += word+' '
        tweets.append(word_tokenize(tweet))
    df.text= tweets

    #stemming
    stemtweets=[]
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english", ignore_stopwords=False)
    #ps= PorterStemmer()
    for x in df.text:
        stemtweet=''
        for word in x:
            stemtweet=stemtweet+stemmer.stem(word)+' '
        stemtweets.append(word_tokenize(stemtweet))
    df['stemmed']=stemtweets

    df_unstemmed = pd.DataFrame()
    df_unstemmed['text'] = df['text']
    df_unstemmed['Sentiment'] = df['Sentiment']
    df_stemmed = pd.DataFrame()
    df_stemmed['text'] = df['stemmed']
    df_stemmed['Sentiment'] = df['Sentiment']
    
    ### Finalize both the stemmed and unstemmed dataframes
    #df_unstemmed = df.drop(['stemmed'], axis=1)
    #df_unstemmed.head()

    # create a df with stemmed text
    #df_stemmed = df.drop(['text'], axis=1)
    
    return df_stemmed,df_unstemmed


# initialize count vectorizer
def dummy_fun(doc):
    return doc

def getrep(df, rep):
    if rep == 'binary':
        vectorizer = CountVectorizer(binary = True, analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None)  
    else: #for freq
        vectorizer = CountVectorizer(analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None)          
    
    text = df.text
    vectorizer.fit(text)
    freqVocab = vectorizer.vocabulary_
    train_vector = vectorizer.transform(text)
    #Create bigdoc that contains words in V, their corresponding frequencies for each class

    #1.Transform pos and neg tweets into seprate vectors
    train_pos_vector1 = vectorizer.transform(df[df['Sentiment']==1]['text'])
    train_neg_vector1 = vectorizer.transform(df[df['Sentiment']==0]['text'])

    #2. column sum of vectors(word per column)
    sum_pos = train_pos_vector1.sum(axis = 0)
    sum_neg = train_neg_vector1.sum(axis = 0)

    #3. Initialize bigdoc as a dataframe
    bigdoc = pd.DataFrame(index = list(set(freqVocab.keys())), columns = ['pos', 'neg'])

    #4. get the corresponding frequency from the above matrx and set it to bigdoc
    for word in freqVocab.keys():
        index = freqVocab.get(word)
        bigdoc.at[word, 'pos'] = sum_pos[:, index].item()
        bigdoc.at[word, 'neg'] = sum_neg[:, index].item()
#here
    print("length of vocab: ")
    print(len(freqVocab))
    return bigdoc, freqVocab

   # https://www.geeksforgeeks.org/g-fact-41-multiple-return-values-in-python/

####### NAIVE BAYES#######
def Naivebayes(data,category,vector,bigvec):
    logprob = bigvec.copy()
    priors = []
    for cat in category:
        ndoc= len(data)
        nc= len(data[data['Sentiment']== cat])
        prior = math.log(nc/ndoc)
        priors.append(prior)
        if cat == 0:
            colname = 'neg'
        else:
            colname = 'pos'
        denominator = bigvec[colname].sum() + len(bigvec) #denominator for likelihood
        logprob[colname] = bigvec[colname].apply(lambda x:math.log((x+1)/denominator)) #likelihood
    return logprob,priors

def TestNaiveBayes(testdf, logprior, loglikelihood, category, V):
    y_pred = []
    y_true= testdf['Sentiment']
    print("testdf len = " + str(len(testdf)))
    for testdoc in testdf.text:
      mle= {0:0, 1:0}
      for cat in category:
        if cat == 0:
            colname = 'neg'
        else:
            colname = 'pos'
        mle[cat] = logprior[cat]
        #print(testdoc)
        for word in testdoc:
            if word in V.keys():
                #print(word)
                mle[cat] = mle[cat] + loglikelihood.loc[word,colname]
                #print(mle)
      y_pred.append(max(mle.items(), key=operator.itemgetter(1))[0])
    
    testdf['pred']= y_pred
    #confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=None)
    print(cm)
    #accuracy
    accuracy = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    print("accuracy is " + str(accuracy))
    return testdf
     


def main():
    # print command line arguments
    train= get_data(sys.argv[1])
    test= get_data(sys.argv[2])

    print("cleaning data...")
    clean_train_stem,clean_train_nostem= clean(train)
    clean_test_stem, clean_test_nostem= clean(test)
    print("cleaning done")
    print(clean_train_stem.head(5))
    print(clean_train_nostem.head(5))
    
    print("create vectors...")
    #Binary and Freq representations for stemmed and nonstemmed data
    trainvector_stem_bin, stem_vocab= getrep(clean_train_stem, 'binary')
    trainvector_stem_freq, stem_vocab= getrep(clean_train_stem, 'freq')
    trainvector_nostem_bin, nostem_vocab = getrep(clean_train_nostem, 'binary')
    trainvector_nostem_freq, nostem_vocab = getrep(clean_train_nostem, 'freq')

    print("training...")
    #calling naive bayes and logistic regression 4 times each to train on the 4 data vectors above
    #Naive Bayes
    likelihood1,priors1 = Naivebayes(clean_train_stem, [0,1], stem_vocab,trainvector_stem_bin)
    likelihood2,priors2 = Naivebayes(clean_train_stem, [0,1], stem_vocab,trainvector_stem_freq)
    likelihood3,priors3 = Naivebayes(clean_train_nostem, [0,1], nostem_vocab,trainvector_nostem_bin)
    likelihood4,priors4 = Naivebayes(clean_train_nostem, [0,1], nostem_vocab,trainvector_nostem_freq)
    
    print("testing...")
    print("****stem & Binary*******")    
    TestNaiveBayes(clean_test_stem, priors1, likelihood1,  [0,1], stem_vocab)
    print("******stem & frequency*******")
    TestNaiveBayes(clean_test_stem, priors2, likelihood2,  [0,1], stem_vocab)
    print("******no-stem & binary*******")
    TestNaiveBayes(clean_test_nostem, priors3, likelihood3,  [0,1], nostem_vocab)
    print("******no-stem & frequency*******")
    TestNaiveBayes(clean_test_nostem, priors4, likelihood4,  [0,1], nostem_vocab)
    
    
    #Logistic Regression
   # result5 = logisticreg(trainvector_stem_bin)#clean_train_stem, [0,1], freqVector,bigdoc)
   # result6 = logisticreg(trainvector_stem_freq)#clean_train_stem, [0,1], freqVector,bigdoc)
   # result7 = logisticreg(trainvector_nostem_bin)#clean_train_stem, [0,1], freqVector,bigdoc)
   # result8 = logisticreg(trainvector_nostem_freq)#clean_train_stem, [0,1], freqVector,bigdoc)

    #calling test for each of the  models created: Print the accuracy and confusion matrix for each of them

if __name__ == "__main__":
    main()