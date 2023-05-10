# Importing modules
import pandas as pd
import os

os.chdir('.')

print(os.getcwd())

data = pd.read_csv('./data/data.csv', encoding='utf-8')

# Remove the columns
#data = data.drop(columns=['Matches', 'Comments'], axis=1)
data['Descriptions'].fillna("?", inplace = True) 
data['Origin'].fillna("?", inplace = True) 

# Load the regular expression library
import re
# Remove punctuation
data['Descriptions_Processed'] = data['Descriptions'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the descriptions to lowercase
data['Descriptions_Processed'] = data['Descriptions_Processed'].map(lambda x: x.lower())
# Print out the first rows of data
#print(data['Descriptions_Processed'].head())

# List extra words to ignore from our dataframes
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['news', 'say','use', 'not', 'would', 'say', 'could', 
                   '_', 'be', 'know', 'good', 'go', 'get', 'do','took',
                   'time','year', 'done', 'try', 'many', 'some','nice', 
                   'thank', 'think', 'see', 'rather', 'easy', 'easily', 
                   'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 
                   'even', 'right', 'line','even', 'also', 'may', 'take', 
                   'come', 'new','said', 'like','people', 'chm', 'chem',
                   'biol', 'week', 'hour', 'students', 'lectures', 
                   'recommended', 'course', 'biochemistry', 'prerequisite'])
def remove_stop_words(text):
    # FIX!!!
    return [word for word in text if word not in stop_words]

def clean_data_frame(df):
    for i in range(0, len(df['Descriptions_Processed'])):
        desc = remove_stop_words(df['Descriptions_Processed'][i])
        df['Descriptions_Processed'][i] = desc

            
clean_data_frame(data)

pd.DataFrame.to_csv(data, 'test.csv')

acc_schools = []
acc_origin = []
acc_course_ids = [] 
acc_descriptions = []

non_acc_schools = []
non_acc_origin = []
non_acc_course_ids = []
non_acc_descriptions = []

valpo_course_ids = []
valpo_descriptions = []

# Filter data into different lists
for i in range(0, len(data['Accredited'])):
    if(data['School'][i] == 'Valparaiso University'):
        #print('Found ' + data['CourseID'][i])
        valpo_course_ids.append(data['CourseID'][i])
        valpo_descriptions.append(data['Descriptions_Processed'][i])
    else:
        if(data['Accredited'][i]):
            acc_schools.append(data['School'][i])
            acc_origin.append(data['Origin'][i])
            acc_course_ids.append(data['CourseID'][i])
            acc_descriptions.append(data['Descriptions_Processed'][i])
        else:
            non_acc_schools.append(data['School'][i])
            non_acc_origin.append(data['Origin'][i])
            non_acc_course_ids.append(data['CourseID'][i])
            non_acc_descriptions.append(data['Descriptions_Processed'][i])
            
            
# Get some stastics of our data
print('Total number of accredited courses:', len(acc_course_ids))
print('Total number of non accredited courses:', len(non_acc_course_ids))
print('Total number of Valpo courses:', len(valpo_course_ids))

            
# Create 3 separate dataframes with filtered data
acc_data = {
    'School': acc_schools,
    'Origin': acc_origin,
    'CourseID': acc_course_ids,
    'Descriptions': acc_descriptions
}

non_acc_data = {
    'School': non_acc_schools,
    'Origin': non_acc_origin,
    'CourseID': non_acc_course_ids,
    'Descriptions': non_acc_descriptions
}

valpo_data = {
    'CourseID': valpo_course_ids,
    'Descriptions': valpo_descriptions
}

"""
#https://anaconda.org/pace-u-cs632v/wordcloud-python-s-wordle/notebook

def showWordCloud(list):
    from os import path
    # Import necessary libraries
    #from scipy.misc import imread
    import matplotlib.pyplot as plt
    #import random
    from wordcloud import WordCloud, STOPWORDS
    # Join the different processed titles together.
    long_string = ','.join(list)
    # Visualize the word cloud
    wordcloud = WordCloud(font_path='C:/Windows/WinSxS/amd64_microsoft-windows-font-truetype-verdana_31bf3856ad364e35_10.0.14393.0_none_ebed8e9b15e088a7/Verdana.ttf',
                      relative_scaling = 1.0,
                      stopwords = 'to of'
                      ).generate(long_string) 
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
showWordCloud(acc_data['Descriptions'])
print('We have generated a word cloud!')
"""

# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#%matplotlib inline

# Helper function
def plot_10_most_common_words(count_data, count_vectorizer, data_name):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title=(data_name + ' 10 Most Common Words'))
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()

# Starting with pyldavis
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
 
# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below (use int values below 15)
number_topics = 5
number_words = 3


def gen_topics(df_name, count_data, count_vectorizer, number_topics, number_words):
    
    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)
    # Print the topics found by the LDA model
    print("\nTopics found via LDA for "+df_name+":")
    print_topics(lda, count_vectorizer, number_words)
        
    from pyLDAvis import sklearn as sklearn_lda
    import pickle 
    import pyLDAvis
    
    # Visualize the topics
    pyLDAvis.enable_notebook()
    
    LDAvis_data_filepath = os.path.join('./data/lda_files/'+df_name+'_ldavis_prepared_'+str(number_topics))
    
    try:
        # # this is a bit time consuming - make the if statement True
        # # if you want to execute visualization prep yourself
        if 1 == 1:
        
            LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
        
            with open(LDAvis_data_filepath, 'wb') as f:
                pickle.dump(LDAvis_prepared, f)
                
        # load the pre-prepared pyLDAvis data from disk
        with open(LDAvis_data_filepath, 'rb') as f:
            LDAvis_prepared = pickle.load(f)
    
        filepath = './data/lda_outputs/'+df_name+'_ldavis_prepared_'+str(number_topics)+'.html'
        #filepath = './' + df_name + '_ldavis_prepared_' + str(number_topics) + '.html'
    
        pyLDAvis.save_html(LDAvis_prepared, filepath)
    except Exception as e: print(e)
    
# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed courses
count_data = count_vectorizer.fit_transform(valpo_data['Descriptions'])
# Visualise the 10 most common words
#plot_10_most_common_words(count_data, count_vectorizer, 'Valpariso University')
gen_topics('valpo', count_data, count_vectorizer, number_topics, number_words)

count_vectorizer = CountVectorizer(stop_words='english')
count_data = count_vectorizer.fit_transform(acc_data['Descriptions'])
#plot_10_most_common_words(count_data, count_vectorizer, 'Accredited Schools')
gen_topics('acc_data', count_data, count_vectorizer, number_topics, number_words)

count_vectorizer = CountVectorizer(stop_words='english')
count_data = count_vectorizer.fit_transform(non_acc_data['Descriptions'])
#plot_10_most_common_words(count_data, count_vectorizer, 'Non Accredited Schools')
gen_topics('non_acc_data', count_data, count_vectorizer, number_topics, number_words)

