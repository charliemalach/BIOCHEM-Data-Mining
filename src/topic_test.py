"""
https://honingds.com/blog/topic-modeling-latent-dirichlet-allocation-lda/
"""

import pandas as pd
import numpy as np
#read the csv file with amazon reviews
data = pd.read_csv('data/data.csv', encoding='utf-8', error_bad_lines=False)
data['Descriptions'] = data['Descriptions'].astype(str)
"""
reviews_df=pd.read_csv('reviews.csv',error_bad_lines=False)
reviews_df['Reviews'] = reviews_df['Reviews'].astype(str) 
reviews_df.head(6)
"""
#text processing
import re
import string
import nltk
nltk.download('stopwords')
from gensim import corpora, models, similarities 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def initial_clean(text):
     """
     Function to clean text-remove punctuations, lowercase text etc.    
     """
     text = re.sub("[^a-zA-Z ]", "", text)
     text = text.lower() # lower case text
     text = nltk.word_tokenize(text)
     return text
 
stop_words = stopwords.words('english')
stop_words.extend(['news', 'say','use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do','took','time','year',
'done', 'try', 'many', 'some','nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line','even', 'also', 'may', 'take', 'come', 'new','said', 'like','people'])
def remove_stop_words(text):
     return [word for word in text if word not in stop_words]

stop_words = stopwords.words('english')
stop_words.extend(['news', 'say','use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do','took','time','year',
'done', 'try', 'many', 'some','nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line','even', 'also', 'may', 'take', 'come', 'new','said', 'like','people'])
def remove_stop_words(text):
     return [word for word in text if word not in stop_words]
 
stemmer = PorterStemmer()
def stem_words(text):
    """
    Function to stem words
    """
    try:
        text = [stemmer.stem(word) for word in text]
        #text = [word for word in text if len(word) &gt; 1] # no single letter words
    except IndexError:
        pass
    return text 

def apply_all(text):
     """
     This function applies all the functions above into one
     """
     return stem_words(remove_stop_words(initial_clean(text)))
 

# clean reviews and create new column "tokenized" 
import time   
t1 = time.time()   
data['tokenized_reviews'] = data['Descriptions'].apply(apply_all)    
t2 = time.time()  
print("Time to clean and tokenize", len(data), "reviews:", (t2-t1)/60, "min") #Time to clean and tokenize 3209 reviews: 0.21254388093948365 min

#LDA
import gensim
import pyLDAvis.gensim

#Create a Gensim dictionary from the tokenized data 
tokenized = data['tokenized_reviews']
#Creating term dictionary of corpus, where each unique term is assigned an index.
dictionary = corpora.Dictionary(tokenized)
#Filter terms which occurs in less than 1 review and more than 80% of the reviews.
dictionary.filter_extremes(no_below=1, no_above=0.8)
#convert the dictionary to a bag of words corpus 
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
print(corpus[:1])

[[(dictionary[id], freq) for id, freq in cp] for cp in corpus[:1]]

#LDA
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 7, id2word=dictionary, passes=15)
ldamodel.save('model_combined.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
   print(topic)
   
get_document_topics = ldamodel.get_document_topics(corpus[0])
print(get_document_topics)

#visualizing topics
lda_viz = gensim.models.ldamodel.LdaModel.load('model.gensim')
lda_display = pyLDAvis.gensim.prepare(lda_viz, corpus, dictionary, sort_topics=True)
pyLDAvis.display(lda_display)


def dominant_topic(ldamodel, corpus, texts):
     #Function to find the dominant topic in each review
     sent_topics_df = pd.DataFrame() 
     # Get main topic in each review
     for i, row in enumerate(ldamodel[corpus]):
         row = sorted(row, key=lambda x: (x[1]), reverse=True)
         # Get the Dominant topic, Perc Contribution and Keywords for each review
         for j, (topic_num, prop_topic) in enumerate(row):
             if j == 0:  # =&gt; dominant topic
                 wp = ldamodel.show_topic(topic_num,topn=4)
                 topic_keywords = ", ".join([word for word, prop in wp])
                 sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
             else:
                 break
     sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
     contents = pd.Series(texts)
     sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
     return(sent_topics_df)
 
#df_dominant_topic = dominant_topic(ldamodel=ldamodel, corpus=corpus, texts=reviews_df['Reviews']) 
#df_dominant_topic.head()