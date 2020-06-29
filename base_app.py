"""
    Simple Streamlit webserver application for serving developed classification
	models.

    Author: TEAM_4_DBN_DataPlanet

    Description:    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""


# Streamlit imports
import streamlit as st
import joblib,os


# Data imports
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from datetime import date, timedelta


# Plotting imports
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.style as style 
sns.set(font_scale=1)

#nlp packages
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
import re


#Load models
logistic_reg = joblib.load(open(os.path.join('resources/pickles/logisticreg.pkl'),"rb"))
linearsvc = joblib.load(open(os.path.join('resources/pickles/linearsvc.pkl'),"rb"))
randomforest = joblib.load(open(os.path.join('resources/pickles/randomforest.pkl'),"rb"))

### Load Vectorizers
count_vectorizer = open('resources/pickles/countvec.pkl','rb')
count_vec = joblib.load(count_vectorizer)
tfidf_vectorizer = open('resources/pickles/tfidf.pkl','rb')
tfidf_vec = joblib.load(tfidf_vectorizer)

### Loading raw data
raw = pd.read_csv("resources/train.csv")
labels_dict = {-1: 'Agnostic',0: 'Neutral',1: 'Believer',2: 'News'}
raw.replace({'sentiment': labels_dict}, inplace=True)

# Models used

models = ['Linear SVC' , 'Logistic Regression','Random Forest']

# EDA functions
# Document Corpus
raw_corpus = [statement.lower() for statement in raw.message]
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

def word_count(df,Corpus,len1):
    """Output graph of most frequent words in each class
       given a dataframe with a class column and a corpus """
    corpus = np.array(Corpus)[df.index.values]
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
    for word_val in word_val_pair[:len1]:
        words.append(word_val[0])
        frequency.append(word_val[1])
    word_count_plt = sns.barplot(x=words,y=frequency,edgecolor='k')
    word_count_plt.set(ylabel='word_frequency')
    # word_count_plt.set_title(f'{stance}',fontsize=15)
    return(word_count_plt.figure)
@st.cache(show_spinner=False)

## Word Cloud
def word_cloud(input_df,Corpus):
    """Function output the wordcloud of a class given
       a dataframe with a sentiment column and a corpus"""
    df = input_df.copy()
    corpus = np.array(Corpus)[df.index.values]
    corpus = ' '.join(corpus)
    word_cloud = WordCloud(background_color='white', max_font_size=80).generate(corpus)
    image = plt.imshow(word_cloud,interpolation='bilinear')
    # plt.title(df.index.unique(),fontsize=15)
    plt.axis('off')
    return(image.figure)
    
clean_corpus = [cleaning_fun(tweet) for tweet in raw_corpus]


def main():
    """Streamlit App """
    html_temp = """
	<div 
    style = "background-color:white;padding:20px">
	<h2 style="color:green;text-align:center;">PLANET DATA TWEET CLASSIFIER</h2>
	</div>
    
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    
    # Creating sidebar 
    
    options = ['Information','Analyse Tweet','View Raw Data', 'Top words EDA','View Twitter Hashtags',]
    selection = st.sidebar.selectbox(label="Choose Tab", options=options)
                                    
                                    
    # Build Welcome page
    
    if selection == 'Information' :
        st.subheader("About")
        st.markdown('')
        st.markdown('Is your organisation buying into the green deal? Our insights can help your '\
                    'orgarnistion achieve sustainable growth. The Twitter sentiment analysis allows you '\
                    'to maintain guided business practices which will in turn assist identify niches in the market '\
                    'through streamlined marketing compaigns for your target audience.')
    ## EDA
    
    if selection == 'Top words EDA' :
        ##st.subheader("Top words visualiser with a word coud functionality")
        empty_df = pd.DataFrame()
        classes = st.multiselect('Which classes would you like to view?',['Agnostic', 'Neutral', 'News', 'Believer'])
        for sentiment in classes:
            empty_df = pd.concat([empty_df,raw[raw.sentiment==sentiment]])
        if len(classes) == 0:
            st.write('')
        if st.checkbox('Top words per class'):
            if len(classes) == 0:
                st.warning('Please choose and option')
            if len(classes) > 0:
                word_hash = st.slider('Drag the slider for top words' , 1, 15 )
                st.write(word_count(empty_df,clean_corpus,word_hash))
        if st.checkbox('word_cloud'):
            if len(classes) == 0:
                st.warning('Please choose and option')
            if len(classes) > 0:
                st.write(word_cloud(empty_df,clean_corpus))
    
    # Build Tweet analyser page
    
    if selection == "Analyse Tweet":
        st.subheader("Sentiment translation based on an individual's tweet")
        
        # Creating a text box for user input
        tweet_text = st.text_area("Input Tweet")
        tweet_text = cleaning_fun(tweet_text)
        vect_text = count_vec.transform([tweet_text]).toarray()
        
        
        model_list = st.selectbox('Classification model  ' , models)

        if model_list == 'Linear SVC' :
            picked_model = linearsvc
        elif model_list == 'Logistic Regression':
            picked_model = logistic_reg
        elif model_list == 'Random Forest':
            picked_model = randomforest

               
        if st.button("Classify"):
                  
            # Transform user input with vectorizer
            # vect_text = [tweet_text]
            prediction = picked_model.predict(vect_text)
            st.success(f'Tweet is : {prediction[0]}')
              

    if selection == "View Twitter Hashtags":
        st.subheader('Green Speak Agnostic Hashtags')
        num_hash = st.slider('Drag the slider' , 1, 20 )
        if st.button("Show"):           
            rand_hash = random.sample(agnostic,num_hash) 
            for i in range(len(rand_hash)):                
                st.success(rand_hash[i])


    if selection == "View Raw Data":
        st.subheader("Raw Twitter data with labels")
        st.markdown('''Take a look at the the raw data that is used to train whichever model you choose to use to predict the sentiment''')
        if st.checkbox('Show raw data'):
            st.write(raw[['sentiment', 'message']])
            st.markdown('''The sentiment classification is defined as as follows''')

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write(data[['message']])
    
# Popular Hashtags

agnostic = ['#Climate','#globalwarming','endangered','#Trump', '#climatechange',
            '#ClimateVoter','#Iamwithher','#SaveOurOcean','#Africa',
            '#ParisAgreement','#ActOnClimate','#globalwarming', '#Climatechange',
            '#BeforetheFlood','#sustainability','#science','#ClimateCounts','#ClimateAction','#sea',
            '#EarthDay','#EarthToMarrakech','#EPA','#ClimateChangeIsReal', '#deforestation'
            '#EarthHour','#Women4Climate','#ClimateMarch','#Africa','#climatemarch',
            '#Cities4Climate','#actonclimate','#itstimetochange','#SDGs','#CleanPowerPlan',
            '#SaveTheEPA','#vegan','#WhyIMarch','#WorldVeganDay',
            '#health','#ClimateFacts','#StandUp','#ClimateofHope','#GreenSummit','#ThursdayThoughts',
            '#cleanenergy','#showthelove','#MyClimateAction',
            '#NatGeo','#beforetheflood','#G20','#QandA','#green','#eco',
            '#GreenNewDeal','#UniteBlue','#MarchForScience','#SDG13','#WEF','#Analytics',
            ]

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()
