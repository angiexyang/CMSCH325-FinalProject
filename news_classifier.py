from nltk.classify import apply_features, accuracy
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.corpus import stopwords
import nltk
import random
import string
import os
import sys

def article_features(article_words, article_bigrams):
    #create features for each article using words and bigrams
    
    features = {}

    article_bigrams = set(article_bigrams) 
    for bigram in bigram_features:
        features['contains({})'.format(bigram)] = (bigram in article_bigrams)
        
    article_words = set(article_words)
    for word in word_features:
        features['contains({})'.format(word)] = (word in article_words)
    return features
        
if __name__ == '__main__':
    #set up path to data
    data_folder_name = sys.argv[1]
    data_path = os.path.join(os.getcwd(), '', data_folder_name)

    #make article object to read in files
    article = CategorizedPlaintextCorpusReader(data_path, r'.*\.*\.txt', cat_pattern=r'(\w+).*\.txt')
    
    #make list of all articles with labels based on what folder the file is in
    all_articles = []
    for category in article.categories():
        for fileid in article.fileids(category):
            #lowercases words and takes out stopwords
            process = list(w.lower() for w in list(article.words(fileid)) if w.isalpha() and w not in stopwords.words('english'))
            entry = [process, category]
            all_articles.append(entry)

    random.shuffle(all_articles)

    #make bigrams for every article
    word_bigrams = [(nltk.bigrams(all_articles[i][0])) for i in range(len(all_articles))]

    #create frequency distribution for all words and select top 2000 for features
    all_words = nltk.FreqDist(article.words())
    word_features = list(all_words)[:2000]

    #create list holding all bigrams
    all_bigrams_list = []
    for bigram_list in word_bigrams:
        temp = list(bigram_list)
        for bigram in temp:
            all_bigrams_list.append(bigram)

    #frequency distribution and feature selection for bigrams
    all_bigrams = nltk.FreqDist(all_bigrams_list)
    bigram_features = list(all_bigrams)[:2000]

    #make feature, train, and test sets, train classifier
    featuresets = [(article_features(all_articles[i][0],word_bigrams[i]),all_articles[i][1]) for i in range(len(all_articles))]
    train_set, test_set = featuresets[10:], featuresets[:10]
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    #print accuracy and most informative features
    print("accuracy: ", str(nltk.classify.accuracy(classifier, test_set)*100), "%")
    print(classifier.show_most_informative_features(15))

