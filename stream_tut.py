import streamlit as st
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
import matplotlib.pyplot as plt
import time
import joblib,os
#'''Will need to do POS tagging to filter tweets by organizations '''
# Load Vectorizers
count_vectorizer = open('resources/pickles/countvec.pkl','rb')
count_vec = joblib.load(count_vectorizer)
tfidf_vectorizer = open('resources/pickles/tfidf.pkl','rb')
tfidf_vec = joblib.load(tfidf_vectorizer)

#Load Models
logistic_reg = joblib.load(open(os.path.join('resources/pickles/logisticreg.pkl'),"rb"))
linearsvc = joblib.load(open(os.path.join('resources/pickles/linearsvc.pkl'),"rb"))
randomforest = joblib.load(open(os.path.join('resources/pickles/randomforest.pkl'),"rb"))

train_df = pd.read_csv('resources/train.csv')
labels_dict = {-1: 'Agnostic',0: 'Neutral',1: 'Believer',2: 'News'}
train_df.replace({'sentiment': labels_dict}, inplace=True)
def cleaning_fun(tweet):
    """This function takes a tweet and extracts important text"""
    added_stop_words = ['rt','dm']
    stop_words = set(stopwords.words("english")+added_stop_words)
    removed_stop_words = ['not','do']
    for i in removed_stop_words:
        stop_words.remove(i)

    tweet = tweet.lower()
    tweet = re.sub(r'https?://\S+|www\.\S+','',tweet) # Remove URLs
    tweet = re.sub(r'<.*?>','',tweet) # Remove html tags
    tweet = re.sub(r'abc|cnn|fox|sabc','news',tweet) # Replace tags with news
    tweet = re.sub(r'climatechange','climate change',tweet)
#   Tokenize tweet
    tokenizer = TreebankWordTokenizer()
    tweet = tokenizer.tokenize(tweet)
    tweet = [word for word in tweet if word.isalnum()] #Remove punctuations
#   Remove numbers
    tweet = [word for word in tweet if not any(c.isdigit() for c in word)]
#   Replace News if news is in the words
    tweet = ['news' if 'news' in word else word for word in tweet]
#   Replace word with trump if trump is in the word
    tweet = ['trump' if 'trump' in word else word for word in tweet]
#   Remove stop words
    tweet = ' '.join([word for word in tweet if word not in stop_words])
    return(tweet)
@st.cache(show_spinner=False)
def clean_message(input_df):
    df = input_df.copy()
    df.message = df.message.apply(lambda x: cleaning_fun(x))
    return(df)
@st.cache(show_spinner=False)
def corpus(input_df):
    df = input_df.copy()
    corpus = [tweet for tweet in df.message]
    for tweet in corpus:
        tweet = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", tweet)
    joined_corpus =' '.join(corpus)
    num_of_words = len(joined_corpus.split())
    return (num_of_words)
#@st.cache(show_spinner=False,suppress_st_warning=True)
def Word_count(df,num_feat):
    """Output graph of most frequent words in each class
       given a dataframe with a class column and a corpus """
    Corpus = [tweet.lower() for tweet in df.message]
    for tweet in Corpus:
        tweet = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", tweet)
    fig, axs = plt.subplots(2,2, figsize=(16,8),)
    fig.subplots_adjust(hspace = 0.5, wspace=.2)
    axs = axs.ravel()
    for index, stance in enumerate(df.sentiment.unique()):
        corpus = np.array(Corpus)[df[df.sentiment == stance].index.values]
        corpus = ' '.join(corpus).split(' ')
        word_counts = {}
        for word in corpus:
            if word in word_counts.keys():
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        word_val_pair = []
        for word,word_freq in word_counts.items():
            word_val_pair.append((word,word_freq))
        word_val_pair.sort(key = lambda x: x[1],reverse=True)
        words = []
        frequency = []
        for word_val in word_val_pair[:num_feat]:
            words.append(word_val[0])
            frequency.append(word_val[1])
        axs[index].set_title(f'{stance}',fontsize=8)
        axs[index].bar(x=words,height=frequency,edgecolor='k')
        return(axs)


models = {'LogisticRegressor':'Model_1','SupportVector':'model_2','RandomForrest':'Model_3'}

def main():
    '''main function'''
    st.title('Tweet Classifier')
    st.subheader('Climate change tweet classification')
    
    #Create a sidebar of options
    options = ['Prediction','Information']
    selection = st.sidebar.selectbox('Choose option',options)
    
    if selection == 'Information':
        st.sidebar.selectbox('type of info wanted',['models','presentation'])
        st.info('General Information')
        app_info =open('resources/streamlit_markdown/tweet_classifier_info.md')
        st.write(app_info.read())
        
        st.subheader('Raw Twitter data and label')
        if st.checkbox('Show raw data'):
            number_of_words = corpus(train_df)
            number_of_entries = len(train_df)
            st.write('The data has',number_of_entries,'entries and',number_of_words,'words in total.')
            st.write(train_df[['sentiment','message']])
            Word_count(train_df,10)
        if st.checkbox('show lemmatized data'):
            with st.spinner('Cleaning Data..Please wait'):
                clean_df = clean_message(train_df)
                number_of_words = corpus(clean_df)
                number_of_entries = len(clean_df)
                time.sleep(5)
                st.write('The cleaned data has',number_of_entries,'entries and',number_of_words,'words in total.')
                time.sleep(3)
                st.write(clean_df[['sentiment','message']])
    if selection == 'Prediction':
        st.info('Prediction with ML models')
        tweet_text = st.text_area('Enter text','Type here')
        
        if st.button('Classify'):
            # clean User tweet
            tweet_text = cleaning_fun(tweet_text)
            tweet_text = count_vec.transform([tweet_text]).toarray()
            model_select = st.radio('select models',list(models.keys()))
            if model_select == 'LogisticRegressor':
                prediction = logistic_reg.predict(tweet_text)
                st.success("Text Categorized as: {}".format(prediction))                
            if model_select == 'SupportVector':
                prediction = linearsvc.predict(tweet_text)
                st.success("Text Categorized as: {}".format(prediction))
            if model_select == 'RandomForrest':
                prediction = randomforest.predict(tweet_text)
                st.success("Text Categorized as: {}".format(prediction))

if __name__ == '__main__':
    main()

