import streamlit as st
import pandas as pd
import numpy as np
import nltk

#'''Will need to do POS tagging to filter tweets by organizations '''
train_df = pd.read_csv('resources/train.csv')


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
            st.write(train_df[['sentiment','message']])
            st.checkbox('show lemmatized data')
    if selection == 'Prediction':
        st.info('Prediction with ML models')
        tweet_text = st.text_area('Enter text','Type here')
        
        if st.button('Classify'):
            model_select = st.selectbox('select models',list(models.keys()))
    
            if model_select == 'LogisticRegressor':
                st.write(models[model_select]+str(3))
            if model_select == 'SupportVector':
                st.write(models[model_select]+str(2))
            if model_select == 'RandomForrest':
                st.write(models[model_select] + str(1))

if __name__ == '__main__':
    main()

