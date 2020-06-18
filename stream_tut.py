import streamlit as st
import pandas as pd
import numpy as np
import spacy

#'''Will need to do POS tagging to filter tweets by organizations '''
train_df = pd.read_csv('resources/train.csv')

# models = ['model_1','model_2','model_3']
# mod_select = st.selectbox('choose_option',models)

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
        st.markdown('read markdown file here')
        
        st.subheader('Raw Twitter data and label')
        if st.checkbox('Show raw data'):
            st.write(train_df[['sentiment','message']])
    if selection == 'Prediction':
        st.info('Prediction with ML models')
        tweet_text = st.text_area('Enter text','Type here')
        
        if st.button('Classify'):
            models = ['model_1','model_2','model_3']
            st.selectbox('select options',models)

if __name__ == '__main__':
    main()

