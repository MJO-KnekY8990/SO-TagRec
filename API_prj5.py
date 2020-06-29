import numpy as np
import pandas as pd
import re
import streamlit as st
from bs4 import BeautifulSoup

from scipy import sparse
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pickle

import gensim
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression


#Imports
mlb = pickle.load(open("final_mlb.pkl", 'rb'))
tfvec = pickle.load(open("final_tfvect.pkl", 'rb'))
tfidf = pickle.load(open("final_tfidf.pkl", 'rb'))
supervised = pickle.load(open("final_supervised.pkl", 'rb'))
unsupervised = pickle.load(open("final_unsupervised.pkl", 'rb'))
w2v = pickle.load(open("final_w2v.pkl", 'rb'))
it_tags_dict = pd.read_csv('it_dict_syno.csv',header=None,dtype='str')[0].values

class WordReplacer(object):
    def __init__(self, word_map):
        self.word_map = word_map

    def replace(self, word):
        return self.word_map.get(word, word)
    
tags_synonyms = pd.read_csv("StackExchange_Synonyms.csv")
replacer = WordReplacer(pd.DataFrame(tags_synonyms.drop(columns=["Id"]).to_dict('split')["data"]).set_index(0).to_dict()[1])

lemmatizer = nltk.stem.WordNetLemmatizer()

words_to_remove = list(pd.Series(pd.read_csv("my_stop_words.csv",header=None)[0]).values)

def text_preparation(txt_title, txt_body):
    text = re.sub("\\n",
                  " ",
                  txt_title + 
                  " " + 
                  BeautifulSoup(txt_body, "html.parser").get_text().lower())
    sw = set()
    sw.update(words_to_remove)
    sw.update(tuple(stopwords.words('english')))
    to_space = re.compile('[/(){}\[\]\|\@,;:\-_\*]')
    to_keep_ = re.compile('[^a-z #+.]')
    to_remove = list(sw)
    
    text = text.lower()
    text = re.sub("\\n"," ",text)
    text = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', '', text, flags=re.MULTILINE)
    text = re.sub(r'http?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', '', text, flags=re.MULTILINE)
    text = " ".join([str(replacer.replace(words)) for words in text.split()])
    text = re.sub(to_space," ",text)
    text = re.sub(to_keep_," ",text)
    text = re.sub("\.\n"," ",text)
    text = re.sub("\. "," ",text)
    text = " ".join([words for words in text.split() if (words not in to_remove) and (len(words)>1)])
    
    text_l = lemmatizer.lemmatize(text)
    
    return (text, text_l)

def wstokenize(text):
    return nltk.WhitespaceTokenizer().tokenize(text)

def show_topics(vectorizer, lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        topic_words_in_it = []
        for ind in (-topic_weights).argsort():
            if len(topic_words_in_it) >= n_words:
                break
            else:
                if keywords[ind] in it_tags_dict:
                    topic_words_in_it.append(ind)
        top_keyword_locs = topic_words_in_it
        #top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=tfvec, lda_model=unsupervised, n_words=20)        

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i+1) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i+1) for i in range(df_topic_keywords.shape[0])]

def tagging(text,text_lem,number_of_tags):
    #tf_vect
    text_vect = tfvec.transform([text])
    #tfidf
    text_tfv = tfidf.transform([text_lem])
    #whitespace token
    wtext = wstokenize(text_lem)
    
    #tagz_from_model
    threshold = 0.265
    tagz_from_model = supervised.predict_proba(text_tfv)
    tagz_from_model = (tagz_from_model>threshold).astype('int')
    if np.sum(tagz_from_model) < 1:
        rez = pd.DataFrame(index=["tag","value"])
        #tagz_from_it
        tagz_from_it = list([words for words in wtext if words in it_tags_dict])
        if len(tagz_from_it) > 0:
            for tag in tagz_from_it:
                ind = rez.shape[1]
                rez.at["tag",ind] = tag
                rez.at["value",ind] = 0.3194
        
        #tagz_from_topics
        tagz_from_topics = df_topic_keywords.iloc[np.argmax(unsupervised.transform(text_vect))].values
        temp = []
        for w in it_tags_dict:
            if w in tagz_from_topics:
                temp.append(w)
        tagz_from_topics = temp
        for n, tag in enumerate(tagz_from_topics):
            ind = rez.shape[1]
            rez.at["tag",ind] = tag
            rez.at["value",ind] = 0.0976 - 1e-5*n
            
        return(list(rez.T.groupby("tag").sum().sort_values(by="value",ascending=False).index)[:number_of_tags])
        
    else:
        tagz_from_model = mlb.inverse_transform(sparse.csr_matrix(tagz_from_model))[0]
        return(tagz_from_model)

def tag_w2v(tagz,number_reco):
    #word2vec
    rez_w2v = pd.DataFrame()
    for u, w in enumerate(tagz):
        reco=[]
        if w in w2v.wv.vocab:
            for i, j in w2v.wv.most_similar(w,topn=len(it_tags_dict)):
                if i in it_tags_dict:
                    reco.append(i)
            for l in range(len(reco[:number_reco])):
                rez_w2v.at[w,str("Word_"+str(l+1))]=reco[l]
    rez=[]
    for col in rez_w2v.columns:
        rez.extend(list(rez_w2v[col].value_counts().index))
    return(list(pd.Series(rez).value_counts().index))

st.title('Stackoverflow tag recommendation')

title = st.text_input("Title", "Post title")
body = st.text_area("Body", "Post body", height=350)
min_num_tags = st.number_input("Minimum number of tags (the more you ask for, the less relevant they might be)", min_value=1, max_value=5, value=1)

if st.button('Go!'):
    if (body != "") & (body != "Post body"):
        text, text_lem = text_preparation(title,body)
        tags = tagging(text,text_lem,min_num_tags)
        tags_w2v = tag_w2v(tags,min_num_tags)
        
        rez1 = ""
        for w in tags:
            rez1 = w + " - " + rez1 
        rez1 = rez1[:-2]
            
        rez2 = ""
        x = 2*min_num_tags
        if x < 4:
            x = 4
        for w in tags_w2v[:x]:
            rez2 = w + " - " + rez2
        rez2 = rez2[:-2]
        
        st.markdown('**Recommended tags:**')
        st.write(rez1)
        to_print = '**' + str(2*min_num_tags) + ' Related tags** (if possible):'
        st.markdown(to_print)
        st.write(rez2)
    else:
        st.write("Write something in the Body")