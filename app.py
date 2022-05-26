import streamlit as st
import nltk
import spacy
import spacy_streamlit
from nltk.corpus import wordnet
from textblob import TextBlob
import streamlit.components.v1 as components
from pattern.web import Google
import codecs
import streamlit.components.v1 as stc
import webbrowser
# Text to speech Pkgs
import os
import time
import glob
from gtts import gTTS

# Text cleaning Pkg
import neattext.functions as nfx
# Web Scraping Pkg
from bs4 import BeautifulSoup
from urllib.request import urlopen

# File Processing Pkgs
import pandas as pd
import docx2txt
import PyPDF2
from PyPDF2 import PdfFileReader, PdfFileWriter
import pdfplumber

# Text downloading
import base64 
timestr = time.strftime("%Y%m%d-%H%M%S")

import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

# NLP Pkgs
from textblob import TextBlob
import pandas as pd 
# Emoji for tone checker
import emoji

import io

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
# let's begin


from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
import numpy as np


nlp = English()
nlp.add_pipe('sentencizer')

nlp = spacy.load('en_core_web_sm')

try:
    os.mkdir("temp")
except:
    pass
input_language = "en"
output_language = "en"
tld = "com"

st.set_page_config(page_title='Text Summarization',
                   page_icon=':book:',
                   layout='wide')

st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #D3D3D3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

url = 'http://127.0.0.1:8000/'

if st.button('HomePage'):
    webbrowser.open_new_tab(url)

# Custom Components Fxn
def st_calculator(calc_html,width=1000,height=1350):
	calc_file = codecs.open(calc_html,'r')
	page = calc_file.read()
	components.html( calc_file ,width=width,height=height,scrolling=False)

def read_pdf_with_pdfplumber(file):
	with pdfplumber.open(file) as pdf:
	    page = pdf.pages[0]
	    return page.extract_text()
   
def get_text(raw_url):
	page = urlopen(raw_url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text

def read_pdf(file):
	pdfReader = PdfFileReader(file)
	count = pdfReader.numPages
	all_page_text = ""
	for i in range(count):
		page = pdfReader.getPage(i)
		all_page_text += page.extractText()

	return all_page_text


def convert_pdf_to_txt(path):
    '''Convert pdf content from a file path to text

    :path the file path
    '''
    rsrcmgr = PDFResourceManager()
    codec = 'utf-8'
    laparams = LAParams()

    with io.StringIO() as retstr:
        with TextConverter(rsrcmgr, retstr, codec=codec,
                           laparams=laparams) as device:
            with open(path, 'rb') as fp:
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                password = ""
                maxpages = 0
                caching = True
                pagenos = set()

                for page in PDFPage.get_pages(fp,
                                              pagenos,
                                              maxpages=maxpages,
                                              password=password,
                                              caching=caching,
                                              check_extractable=True):
                    interpreter.process_page(page)

                return retstr.getvalue()

# Downloading text as .txt file
def text_downloader(raw_text):
	b64 = base64.b64encode(raw_text.encode()).decode()
	new_filename = "new_text_file_{}_.txt".format(timestr)
	st.markdown("#### Download File ###")
	href = f'<a href="data:file/txt;base64,{b64}" download="{new_filename}">Click Here!!</a>'
	st.markdown(href,unsafe_allow_html=True)

def main():
    activites = ["Summary with text","Analyse with URL","Summarizing a file","Comparing different articles","Best of All","Text Cleaning",
    "Tokenization","Grammar","Search"]
    choice = st.sidebar.selectbox("Select Activity",activites)

    #Summary with text
    if choice == "Summary with text":
        html_temp = """
      <div style= "background-color:#778899;"><p style="color:white; font-size:60px; width:70%;">Text Summarization</p></div>
	"""
        components.html(html_temp)
        text5 = st.text_area("Input Text For Summary",height=300)


        if st.button("summarize"):
            doc = nlp(text5.replace("\n", ""))
            sentences = [sent.text.strip() for sent in doc.sents]
            sentence_organizer = {k:v for v,k in enumerate(sentences)}
            tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                        strip_accents='unicode', 
                                        analyzer='word',
                                        token_pattern=r'\w{1,}',
                                        ngram_range=(1, 3), 
                                        use_idf=1,smooth_idf=1,
                                        sublinear_tf=1,
                                        stop_words = 'english')
            tf_idf_vectorizer.fit(sentences)
            sentence_vectors = tf_idf_vectorizer.transform(sentences)
            sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
            N = 3
            top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
            mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
            # Ordering our top-n sentences in their original ordering
            mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
            ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
            summary2 = " ".join(ordered_scored_sentences)
            def summarizer(text, tokenizer, max_sent_in_summary=3):
                # Create spacy document for further sentence level tokenization
                doc = nlp(text5.replace("\n", ""))
            #     sentences = [sent.string.strip() for sent in doc.sents]
                sentences = [sent.text.strip() for sent in doc.sents]
                # Let's create an organizer which will store the sentence ordering to later reorganize the 
                # scored sentences in their correct order
                sentence_organizer = {k:v for v,k in enumerate(sentences)}
                # Let's now create a tf-idf (Term frequnecy Inverse Document Frequency) model
                tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                    strip_accents='unicode', 
                                                    analyzer='word',
                                                    token_pattern=r'\w{1,}',
                                                    ngram_range=(1, 3), 
                                                    use_idf=1,smooth_idf=1,
                                                    sublinear_tf=1,
                                                    stop_words = 'english')
                # Passing our sentences treating each as one document to TF-IDF vectorizer
                tf_idf_vectorizer.fit(sentences)
                # Transforming our sentences to TF-IDF vectors
                sentence_vectors = tf_idf_vectorizer.transform(sentences)
                # Getting sentence scores for each sentences
                sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                # Getting top-n sentences
                N = max_sent_in_summary
                top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                # Let's now do the sentence ordering using our prebaked sentence_organizer
                # Let's map the scored sentences with their indexes
                mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                # Ordering our top-n sentences in their original ordering
                mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                # Our final summary
                summary = " ".join(ordered_scored_sentences)
                return summary

            st.success(summarizer(text=text5, tokenizer=nlp, max_sent_in_summary=3))
        text_range= st.sidebar.slider("Summarize sentences Range",5,50)
        text1 = st.text_area("Input Text For Summary",height=250)


        if st.button("summarize_text_with_wordcount"):
            doc = nlp(text1.replace("\n", ""))
            sentences = [sent.text.strip() for sent in doc.sents]
            sentence_organizer = {k:v for v,k in enumerate(sentences)}
            tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                        strip_accents='unicode', 
                                        analyzer='word',
                                        token_pattern=r'\w{1,}',
                                        ngram_range=(1, 3), 
                                        use_idf=1,smooth_idf=1,
                                        sublinear_tf=1,
                                        stop_words = 'english')
            tf_idf_vectorizer.fit(sentences)
            sentence_vectors = tf_idf_vectorizer.transform(sentences)
            sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
            N = 3
            top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
            mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
            # Ordering our top-n sentences in their original ordering
            mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
            ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
            summary2 = " ".join(ordered_scored_sentences)
            def summarizer(text, tokenizer, max_sent_in_summary=3):
                # Create spacy document for further sentence level tokenization
                doc = nlp(text1.replace("\n", ""))
            #     sentences = [sent.string.strip() for sent in doc.sents]
                sentences = [sent.text.strip() for sent in doc.sents]
                # Let's create an organizer which will store the sentence ordering to later reorganize the 
                # scored sentences in their correct order
                sentence_organizer = {k:v for v,k in enumerate(sentences)}
                # Let's now create a tf-idf (Term frequnecy Inverse Document Frequency) model
                tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                    strip_accents='unicode', 
                                                    analyzer='word',
                                                    token_pattern=r'\w{1,}',
                                                    ngram_range=(1, 3), 
                                                    use_idf=1,smooth_idf=1,
                                                    sublinear_tf=1,
                                                    stop_words = 'english')
                # Passing our sentences treating each as one document to TF-IDF vectorizer
                tf_idf_vectorizer.fit(sentences)
                # Transforming our sentences to TF-IDF vectors
                sentence_vectors = tf_idf_vectorizer.transform(sentences)
                # Getting sentence scores for each sentences
                sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                # Getting top-n sentences
                N = max_sent_in_summary
                top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                # Let's now do the sentence ordering using our prebaked sentence_organizer
                # Let's map the scored sentences with their indexes
                mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                # Ordering our top-n sentences in their original ordering
                mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                # Our final summary
                summary = " ".join(ordered_scored_sentences)
                return summary
            st.markdown(f"## Summary :")
            # Summarizing
            st.warning(summarizer(text=text1, tokenizer=nlp, max_sent_in_summary=text_range))

            def text_to_speech(input_language, output_language, text, tld):
                tts = gTTS(text, lang=output_language, tld=tld, slow=False)
                try:
                    my_file_name = text2[0:20]
                except:
                        my_file_name = "audio"
                        tts.save(f"temp/{my_file_name}.mp3")
                        return my_file_name, text
            def remove_files(n):
                mp3_files = glob.glob("temp/*mp3")
                if len(mp3_files) != 0:
                    now = time.time()
                    n_days = n * 86400
                    for f in mp3_files:
                        if os.stat(f).st_mtime < now - n_days:
                            os.remove(f)
                            print("Deleted ", f)
            remove_files(7)

            # Text to speech 
            text= summarizer(text=text1, tokenizer=nlp, max_sent_in_summary=text_range)
            result,output_text = text_to_speech(input_language, output_language, text, tld)
            audio_file = open(f"temp/{result}.mp3", "rb")
            audio_bytes = audio_file.read()
            st.markdown(f"## Summarized text audio :")
            st.audio(audio_bytes, format="audio/mp3", start_time=0)
            
            # Downloading summary content
            text_downloader(text)


        # Summarizing the text using url
    elif choice == "Analyse with URL":
        html_temp1 = """
	<div style= "background-color:#778899;"><p style="color:white;font-size:60px;">Summarizing the content in given URL</p></div>
	""" 
        components.html(html_temp1)
        text_range= st.sidebar.slider("Summarize sentences Range",5,50)
        raw_url= st.text_input("Type URL")
        if st.button("Extract"):
            result=get_text(raw_url)
            doc = nlp(result.replace("\n", ""))
            sentences = [sent.text.strip() for sent in doc.sents]
            sentence_organizer = {k:v for v,k in enumerate(sentences)}
            tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                        strip_accents='unicode', 
                                        analyzer='word',
                                        token_pattern=r'\w{1,}',
                                        ngram_range=(1, 3), 
                                        use_idf=1,smooth_idf=1,
                                        sublinear_tf=1,
                                        stop_words = 'english')
            tf_idf_vectorizer.fit(sentences)
            sentence_vectors = tf_idf_vectorizer.transform(sentences)
            sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
            N = 3
            top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
            mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
            # Ordering our top-n sentences in their original ordering
            mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
            ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
            summary2 = " ".join(ordered_scored_sentences)
            def summarizer(text, tokenizer, max_sent_in_summary=3):
                # Create spacy document for further sentence level tokenization
                doc = nlp(result.replace("\n", ""))
            #     sentences = [sent.string.strip() for sent in doc.sents]
                sentences = [sent.text.strip() for sent in doc.sents]
                # Let's create an organizer which will store the sentence ordering to later reorganize the 
                # scored sentences in their correct order
                sentence_organizer = {k:v for v,k in enumerate(sentences)}
                # Let's now create a tf-idf (Term frequnecy Inverse Document Frequency) model
                tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                    strip_accents='unicode', 
                                                    analyzer='word',
                                                    token_pattern=r'\w{1,}',
                                                    ngram_range=(1, 3), 
                                                    use_idf=1,smooth_idf=1,
                                                    sublinear_tf=1,
                                                    stop_words = 'english')
                # Passing our sentences treating each as one document to TF-IDF vectorizer
                tf_idf_vectorizer.fit(sentences)
                # Transforming our sentences to TF-IDF vectors
                sentence_vectors = tf_idf_vectorizer.transform(sentences)
                # Getting sentence scores for each sentences
                sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                # Getting top-n sentences
                N = max_sent_in_summary
                top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                # Let's now do the sentence ordering using our prebaked sentence_organizer
                # Let's map the scored sentences with their indexes
                mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                # Ordering our top-n sentences in their original ordering
                mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                # Our final summary
                summary = " ".join(ordered_scored_sentences)
                return summary
            st.markdown(f"## Summary :")
            # Summarizing
            st.warning(summarizer(text=result, tokenizer=nlp, max_sent_in_summary=text_range))
        
            result=get_text(raw_url)
            text= summarizer(text=result, tokenizer=nlp, max_sent_in_summary=text_range)
            def text_to_speech(input_language, output_language, text, tld):
                tts = gTTS(text, lang=output_language, tld=tld, slow=False)
                try:
                    my_file_name = text2[0:20]
                except:
                        my_file_name = "audio"
                        tts.save(f"temp/{my_file_name}.mp3")
                        return my_file_name, text


            # Text to speech 
            result,output_text = text_to_speech(input_language, output_language, text, tld)
            audio_file = open(f"temp/{result}.mp3", "rb")
            audio_bytes = audio_file.read()
            st.markdown(f"## Summarized text audio :")
            st.audio(audio_bytes, format="audio/mp3", start_time=0)
                

            def remove_files(n):
                mp3_files = glob.glob("temp/*mp3")
                if len(mp3_files) != 0:
                    now = time.time()
                    n_days = n * 86400
                    for f in mp3_files:
                        if os.stat(f).st_mtime < now - n_days:
                            os.remove(f)
                            print("Deleted ", f)
            remove_files(7)

            # Downloading summary content
            text_downloader(text)


    elif choice == "Summarizing a file":
        html_temp2 = """
	    <div style= "background-color:#778899;"><p style="color:white;font-size:60px;">Summarizing a file</p></div>
	    """    
        components.html(html_temp2)
        text_range= st.sidebar.slider("Summarize sentences Range",5,50)
        docx_file = st.file_uploader("Upload File",type=['docx','pdf'])
        if st.button("Process"):
            if docx_file is not None:
                    file_details = {"Filename":docx_file.name,"FileType":docx_file.type,"FileSize":docx_file.size}
                    st.write(file_details)
                    if docx_file.type == "application/pdf":
                        try:
                            with pdfplumber.open(docx_file) as pdf:
                                    result=convert_pdf_to_txt(docx_file.name)
                                    # st.warning(docx_file.name)
                                    # st.warning(result)
                                    doc = nlp(result.replace("\n", ""))
                                    sentences = [sent.text.strip() for sent in doc.sents]
                                    sentence_organizer = {k:v for v,k in enumerate(sentences)}
                                    tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                                strip_accents='unicode', 
                                                                analyzer='word',
                                                                token_pattern=r'\w{1,}',
                                                                ngram_range=(1, 3), 
                                                                use_idf=1,smooth_idf=1,
                                                                sublinear_tf=1,
                                                                stop_words = 'english')
                                    tf_idf_vectorizer.fit(sentences)
                                    sentence_vectors = tf_idf_vectorizer.transform(sentences)
                                    sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                                    N = 3
                                    top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                                    mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                                    # Ordering our top-n sentences in their original ordering
                                    mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                                    ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                                    summary2 = " ".join(ordered_scored_sentences)
                                    def summarizer(text, tokenizer, max_sent_in_summary=3):
                                        # Create spacy document for further sentence level tokenization
                                        doc = nlp(result.replace("\n", ""))
                                    #     sentences = [sent.string.strip() for sent in doc.sents]
                                        sentences = [sent.text.strip() for sent in doc.sents]
                                        # Let's create an organizer which will store the sentence ordering to later reorganize the 
                                        # scored sentences in their correct order
                                        sentence_organizer = {k:v for v,k in enumerate(sentences)}
                                        # Let's now create a tf-idf (Term frequnecy Inverse Document Frequency) model
                                        tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                                            strip_accents='unicode', 
                                                                            analyzer='word',
                                                                            token_pattern=r'\w{1,}',
                                                                            ngram_range=(1, 3), 
                                                                            use_idf=1,smooth_idf=1,
                                                                            sublinear_tf=1,
                                                                            stop_words = 'english')
                                        # Passing our sentences treating each as one document to TF-IDF vectorizer
                                        tf_idf_vectorizer.fit(sentences)
                                        # Transforming our sentences to TF-IDF vectors
                                        sentence_vectors = tf_idf_vectorizer.transform(sentences)
                                        # Getting sentence scores for each sentences
                                        sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                                        # Getting top-n sentences
                                        N = max_sent_in_summary
                                        top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                                        # Let's now do the sentence ordering using our prebaked sentence_organizer
                                        # Let's map the scored sentences with their indexes
                                        mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                                        # Ordering our top-n sentences in their original ordering
                                        mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                                        ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                                        # Our final summary
                                        summary = " ".join(ordered_scored_sentences)
                                        return summary
                                    st.success(summarizer(text=result, tokenizer=nlp, max_sent_in_summary=text_range))
                        except:
                            st.write("None")

                # elif docx_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                #     raw_text = docx2txt.process(docx_file) # Parse in the uploadFile Class directory
                #     st.warning(summarizer(text=raw_text, tokenizer=nlp, max_sent_in_summary=text_range))

    # Text Cleaning
    elif choice == "Text Cleaning":
        html_temp5 = """
	<div style= "background-color:#778899;"><p style="color:white;font-size:60px;">Text Cleaning</p></div>
	""" 
        components.html(html_temp5)
        text_file = st.file_uploader("Upload text file", type=['txt'])
        normalize_case = st.sidebar.checkbox("Normalize Case")
        clean_stopwords = st.sidebar.checkbox("Stopwords")
        clean_punctuations = st.sidebar.checkbox("Punctuations")
        clean_emails = st.sidebar.checkbox("Emails")
        clean_special_char = st.sidebar.checkbox("Special Characters")
        clean_numbers = st.sidebar.checkbox("Numbers")
        clean_urls = st.sidebar.checkbox("URLS")

        if text_file is not None:
            file_details = {"Filename" : text_file.name , "Filesize":text_file.size,"Filetype":text_file.type}
            st.write(file_details)
            raw_text=text_file.read().decode('utf-8')
            col1,col2 = st.columns(2)

            with col1:
                with st.expander("Original text"):
                    st.write(raw_text)

            with col2:
                with st.expander("Processed text"):
                    if normalize_case:
                        raw_text=raw_text.lower()

                    if clean_stopwords:
                        raw_text= nfx.remove_stopwords(raw_text)

                    if clean_punctuations:
                        raw_text= nfx.remove_punctuations(raw_text)

                    if clean_emails:
                        raw_text= nfx.remove_emails(raw_text)

                    if clean_special_char:
                        raw_text= nfx.remove_emails(raw_text)

                    if clean_numbers:
                        raw_text= nfx.remove_numbers(raw_text)

                    if clean_urls:
                        raw_text= nfx.remove_urls(raw_text)

                    st.write(raw_text)

                    text_downloader(raw_text)


    # Tokenizer,NER,Text Relation 
    elif choice == "Tokenization":
        html_temp6 = """
    <div style="background-color:#778899;"><p style="color:white;font-size:60px;">Text Tokenizer</p></div>
	"""
  
        components.html(html_temp6)
        row_data = st.text_area("Write Text For Tokenizer")
        docx= nlp(row_data)
        if st.button("Tokenizer"):
            spacy_streamlit.visualize_tokens(docx,attrs=['text','pos_','dep_','ent_type_'])
        if st.button("NER"):
            spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)
        if st.button("Text Relationship"):
            spacy_streamlit.visualize_parser(docx)


    # Spell Correction, Synonyms, Antonyms , Defination , Examples
    elif choice == "Grammar":
        html_temp7 = """
	<div style="background-color:#778899;"><p style="color:white;font-size:60px;">Spell Correction and Synonyms</p></div>
	"""
        components.html(html_temp7)
        # Spell Correction
        st.subheader("SPELL CORRECTION")
        text_data = st.text_area("Enter Text Here")
        a = TextBlob(text_data)
        if st.button("Correct"):
            st.success(a.correct())

        # Synonyms
        st.subheader("SYNONYMS")
        text = st.text_area("Enter Text")
        if st.button("Find"):
            for syn in wordnet.synsets(text):
                for i in syn.lemmas():
                    st.success(i.name())
        # Defination
        if st.checkbox("Defination"):
            for syn in wordnet.synsets(text):
                st.warning(syn.definition()) 
        # Examples
        if st.checkbox("Example"):
            for syn in wordnet.synsets(text):
                st.success(syn.examples())

        # Antonyms
        antonyms =[]
        st.subheader("ANTONYMS")
        text1 = st.text_area("Enter Text here")
        if st.button("Find "):
            for syn in wordnet.synsets(text1):
                for l in syn.lemmas():
                    if l.antonyms():
                            antonyms.append(l.antonyms()[0].name())
            st.success(antonyms)
        if st.checkbox("Defination "):
            for syn in wordnet.synsets(text1):
                st.warning(syn.definition()) 
        if st.checkbox("Examples"):
            for syn in wordnet.synsets(text1):
                st.success(syn.examples())


    # Search Anything
    elif choice == "Search":
        html_temp8 = """
	<div style="background-color:#778899;"><p style="color:white;font-size:60px;">Search</p></div>
	"""
        components.html(html_temp8)
        row_text= st.text_input("Search Anything")
        google = Google(license=None)
        if st.button("search"):
            for search_result in google.search(row_text):
                st.write(search_result.text)
                st.warning(search_result.url)


    # Identifying the best pdf amoung them 
    elif choice == "Comparing different articles":
        html_temp3 = """
	<div style= "background-color:#778899;"><p style="color:white;font-size:60px;">Comparing different articles</p></div>
	""" 
        components.html(html_temp3)
        text_range= st.sidebar.slider("Summarize sentences Range",5,50)
        uploaded_files =st.file_uploader("Upload txt files",type=['pdf'], accept_multiple_files=True)
        max = -1
        filename = ""
        if uploaded_files :
            for file in uploaded_files:
                file.seek(0)
                with pdfplumber.open(file) as pdf:
                            result=convert_pdf_to_txt(file.name)
                            doc = nlp(result.replace("\n", ""))
                            sentences = [sent.text.strip() for sent in doc.sents]
                            sentence_organizer = {k:v for v,k in enumerate(sentences)}
                            tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                        strip_accents='unicode', 
                                                        analyzer='word',
                                                        token_pattern=r'\w{1,}',
                                                        ngram_range=(1, 3), 
                                                        use_idf=1,smooth_idf=1,
                                                        sublinear_tf=1,
                                                        stop_words = 'english')
                            tf_idf_vectorizer.fit(sentences)
                            sentence_vectors = tf_idf_vectorizer.transform(sentences)
                            sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                            N = 3
                            top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                            mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                            # Ordering our top-n sentences in their original ordering
                            mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                            ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                            summary2 = " ".join(ordered_scored_sentences)
                            def summarizer(text, tokenizer, max_sent_in_summary=3):
                                # Create spacy document for further sentence level tokenization
                                doc = nlp(result.replace("\n", ""))
                            #     sentences = [sent.string.strip() for sent in doc.sents]
                                sentences = [sent.text.strip() for sent in doc.sents]
                                # Let's create an organizer which will store the sentence ordering to later reorganize the 
                                # scored sentences in their correct order
                                sentence_organizer = {k:v for v,k in enumerate(sentences)}
                                # Let's now create a tf-idf (Term frequnecy Inverse Document Frequency) model
                                tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                                    strip_accents='unicode', 
                                                                    analyzer='word',
                                                                    token_pattern=r'\w{1,}',
                                                                    ngram_range=(1, 3), 
                                                                    use_idf=1,smooth_idf=1,
                                                                    sublinear_tf=1,
                                                                    stop_words = 'english')
                                # Passing our sentences treating each as one document to TF-IDF vectorizer
                                tf_idf_vectorizer.fit(sentences)
                                # Transforming our sentences to TF-IDF vectors
                                sentence_vectors = tf_idf_vectorizer.transform(sentences)
                                # Getting sentence scores for each sentences
                                sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                                # Getting top-n sentences
                                N = max_sent_in_summary
                                top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                                # Let's now do the sentence ordering using our prebaked sentence_organizer
                                # Let's map the scored sentences with their indexes
                                mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                                # Ordering our top-n sentences in their original ordering
                                mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                                ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                                # Our final summary
                                summary = " ".join(ordered_scored_sentences)
                                return summary
                            st.success(summarizer(text=result, tokenizer=nlp, max_sent_in_summary=text_range))
                            blob = TextBlob(summarizer(text=result, tokenizer=nlp, max_sent_in_summary=text_range))
                            result = blob.sentiment.polarity
                            if(max < result):
                                    max=result
                                    filename = file.name
                            if result > 0.0:
                                custom_emoji = ':smile:'
                                # st.write(emoji.emojize(custom_emoji,use_aliases=True))
                            elif result < 0.0:
                                custom_emoji = ':disappointed:'
                                # st.write(emoji.emojize(custom_emoji,use_aliases=True))
                            # else:
                                # st.write(emoji.emojize(':expressionless:',use_aliases=True))
                            # st.info("Polarity Score is:: {}".format(result))
            st.warning("{} is best among them".format(filename))     
            
            
    # Best of all
    elif choice == "Best of All":
        html_temp10 = """
	<div style= "background-color:#778899;"><p style="color:white;font-size:60px;">Best of All</p></div>
	""" 
        components.html(html_temp10)
        text_range= st.sidebar.slider("Summarize sentences Range",5,50)
        uploaded_files1 =st.file_uploader("Upload txt files",type=['pdf'], accept_multiple_files=True)
        max = -1
        filename = ""
        filename1 = ""
        
        if uploaded_files1 :
            for file in uploaded_files1:
                file.seek(0)
                with pdfplumber.open(file) as pdf:
                        page = pdf.pages[0]
                        result1=page.extract_text()
                        pdf_file_path = file.name
                        # st.success(pdf_file_path)
                        file_base_name = pdf_file_path.replace('.pdf', '')
                        output_folder_path = os.path.join(os.getcwd(), 'Output')

                        pdf = PdfFileReader(pdf_file_path)

                        for page_num in range(pdf.numPages):
                            pdfWriter = PdfFileWriter()
                            pdfWriter.addPage(pdf.getPage(page_num))

                            with open(os.path.join(output_folder_path, '{0}_Page{1}.pdf'.format(file_base_name, page_num+1)), 'wb') as f:
                                pdfWriter.write(f)
                                f.close()

                        doc = nlp(result1.replace("\n", ""))
                        sentences = [sent.text.strip() for sent in doc.sents]
                        sentence_organizer = {k:v for v,k in enumerate(sentences)}
                        tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                        strip_accents='unicode', 
                                                        analyzer='word',
                                                        token_pattern=r'\w{1,}',
                                                        ngram_range=(1, 3), 
                                                        use_idf=1,smooth_idf=1,
                                                        sublinear_tf=1,
                                                        stop_words = 'english')
                        tf_idf_vectorizer.fit(sentences)
                        sentence_vectors = tf_idf_vectorizer.transform(sentences)
                        sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                        N = 3
                        top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                        mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                            # Ordering our top-n sentences in their original ordering
                        mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                        ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                        summary2 = " ".join(ordered_scored_sentences)

                        def summarizer(text, tokenizer, max_sent_in_summary=3):
                                # Create spacy document for further sentence level tokenization
                            doc = nlp(result1.replace("\n", ""))
                                # sentences = [sent.string.strip() for sent in doc.sents]
                            sentences = [sent.text.strip() for sent in doc.sents]
                                # Let's create an organizer which will store the sentence ordering to later reorganize the 
                                # scored sentences in their correct order
                            sentence_organizer = {k:v for v,k in enumerate(sentences)}
                                # Let's now create a tf-idf (Term frequnecy Inverse Document Frequency) model
                            tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                                    strip_accents='unicode', 
                                                                    analyzer='word',
                                                                    token_pattern=r'\w{1,}',
                                                                    ngram_range=(1, 3), 
                                                                    use_idf=1,smooth_idf=1,
                                                                    sublinear_tf=1,
                                                                    stop_words = 'english')
                                # Passing our sentences treating each as one document to TF-IDF vectorizer
                            tf_idf_vectorizer.fit(sentences)
                                # Transforming our sentences to TF-IDF vectors
                            sentence_vectors = tf_idf_vectorizer.transform(sentences)
                                # Getting sentence scores for each sentences
                            sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                                # Getting top-n sentences
                            N = max_sent_in_summary
                            top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                                # Let's now do the sentence ordering using our prebaked sentence_organizer
                                # Let's map the scored sentences with their indexes
                            mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                                # Ordering our top-n sentences in their original ordering
                            mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                            ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                                # Our final summary
                            summary = " ".join(ordered_scored_sentences)
                            return summary
                            # st.success(summarizer(text=result, tokenizer=nlp, max_sent_in_summary=text_range))
                        blob = TextBlob(summarizer(text=result1, tokenizer=nlp, max_sent_in_summary=text_range))
                        ans = blob.sentiment.polarity
                        if(max < ans):
                                max=ans
                                filename = file.name

                           
            st.warning("{} Page1 is best among them".format(filename))  


            if uploaded_files1 :
                for file in uploaded_files1:
                    file.seek(0)
                    with pdfplumber.open(file) as pdf:
                        # pdfRead= PyPDF2.PdfFileReader(file)
                        # for i in range(pdfRead.getNumPages()):
                            page = pdf.pages[0]
                            result=page.extract_text()
                            pdf_file_path = file.name
                                # st.success(pdf_file_path)
                            file_base_name = pdf_file_path.replace('.pdf', '')
                            output_folder_path = os.path.join(os.getcwd(), 'Output')

                            pdf = PdfFileReader(pdf_file_path)

                            for page_num in range(pdf.numPages):
                                pdfWriter = PdfFileWriter()
                                pdfWriter.addPage(pdf.getPage(page_num))

                                with open(os.path.join(output_folder_path, '{0}_Page{1}.pdf'.format(file_base_name, page_num+1)), 'wb') as f:
                                    pdfWriter.write(f)
                                    f.close()

                            doc = nlp(result.replace("\n", ""))
                            sentences = [sent.text.strip() for sent in doc.sents]
                            sentence_organizer = {k:v for v,k in enumerate(sentences)}
                            tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                                strip_accents='unicode', 
                                                                analyzer='word',
                                                                token_pattern=r'\w{1,}',
                                                                ngram_range=(1, 3), 
                                                                use_idf=1,smooth_idf=1,
                                                                sublinear_tf=1,
                                                                stop_words = 'english')
                            tf_idf_vectorizer.fit(sentences)
                            sentence_vectors = tf_idf_vectorizer.transform(sentences)
                            sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                            N = 3
                            top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                            mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                                # Ordering our top-n sentences in their original ordering
                            mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                            ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                            summary2 = " ".join(ordered_scored_sentences)
                                    
                            def summarizer(text, tokenizer, max_sent_in_summary=3):
                                    # Create spacy document for further sentence level tokenization
                                doc = nlp(result.replace("\n", ""))
                                    # sentences = [sent.string.strip() for sent in doc.sents]
                                sentences = [sent.text.strip() for sent in doc.sents]
                                    # Let's create an organizer which will store the sentence ordering to later reorganize the 
                                    # scored sentences in their correct order
                                sentence_organizer = {k:v for v,k in enumerate(sentences)}
                                    # Let's now create a tf-idf (Term frequnecy Inverse Document Frequency) model
                                tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                                            strip_accents='unicode', 
                                                                            analyzer='word',
                                                                            token_pattern=r'\w{1,}',
                                                                            ngram_range=(1, 3), 
                                                                            use_idf=1,smooth_idf=1,
                                                                            sublinear_tf=1,
                                                                            stop_words = 'english')
                                        # Passing our sentences treating each as one document to TF-IDF vectorizer
                                tf_idf_vectorizer.fit(sentences)
                                        # Transforming our sentences to TF-IDF vectors
                                sentence_vectors = tf_idf_vectorizer.transform(sentences)
                                        # Getting sentence scores for each sentences
                                sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                                        # Getting top-n sentences
                                N = max_sent_in_summary
                                top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                                        # Let's now do the sentence ordering using our prebaked sentence_organizer
                                        # Let's map the scored sentences with their indexes
                                mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                                        # Ordering our top-n sentences in their original ordering
                                mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                                ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                                        # Our final summary
                                summary = " ".join(ordered_scored_sentences)
                                return summary

                            if(file.name==filename):
                                st.success(summarizer(text=result1, tokenizer=nlp, max_sent_in_summary=text_range))
                                # text_downloader((summarizer(text=result1, tokenizer=nlp, max_sent_in_summary=text_range)))

        # page2
        if uploaded_files1 :
            for file in uploaded_files1:
                file.seek(0)
                with pdfplumber.open(file) as pdf:
                        page = pdf.pages[1]
                        result2=page.extract_text()
                        pdf_file_path = file.name
                        # st.success(pdf_file_path)
                        file_base_name = pdf_file_path.replace('.pdf', '')
                        output_folder_path = os.path.join(os.getcwd(), 'Output')

                        pdf = PdfFileReader(pdf_file_path)

                        for page_num in range(pdf.numPages):
                            pdfWriter = PdfFileWriter()
                            pdfWriter.addPage(pdf.getPage(page_num))

                            with open(os.path.join(output_folder_path, '{0}_Page{1}.pdf'.format(file_base_name, page_num+1)), 'wb') as f:
                                pdfWriter.write(f)
                                f.close()

                        doc = nlp(result2.replace("\n", ""))
                        sentences = [sent.text.strip() for sent in doc.sents]
                        sentence_organizer = {k:v for v,k in enumerate(sentences)}
                        tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                        strip_accents='unicode', 
                                                        analyzer='word',
                                                        token_pattern=r'\w{1,}',
                                                        ngram_range=(1, 3), 
                                                        use_idf=1,smooth_idf=1,
                                                        sublinear_tf=1,
                                                        stop_words = 'english')
                        tf_idf_vectorizer.fit(sentences)
                        sentence_vectors = tf_idf_vectorizer.transform(sentences)
                        sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                        N = 3
                        top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                        mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                            # Ordering our top-n sentences in their original ordering
                        mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                        ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                        summary2 = " ".join(ordered_scored_sentences)

                        def summarizer(text, tokenizer, max_sent_in_summary=3):
                                # Create spacy document for further sentence level tokenization
                            doc = nlp(result2.replace("\n", ""))
                                # sentences = [sent.string.strip() for sent in doc.sents]
                            sentences = [sent.text.strip() for sent in doc.sents]
                                # Let's create an organizer which will store the sentence ordering to later reorganize the 
                                # scored sentences in their correct order
                            sentence_organizer = {k:v for v,k in enumerate(sentences)}
                                # Let's now create a tf-idf (Term frequnecy Inverse Document Frequency) model
                            tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                                    strip_accents='unicode', 
                                                                    analyzer='word',
                                                                    token_pattern=r'\w{1,}',
                                                                    ngram_range=(1, 3), 
                                                                    use_idf=1,smooth_idf=1,
                                                                    sublinear_tf=1,
                                                                    stop_words = 'english')
                                # Passing our sentences treating each as one document to TF-IDF vectorizer
                            tf_idf_vectorizer.fit(sentences)
                                # Transforming our sentences to TF-IDF vectors
                            sentence_vectors = tf_idf_vectorizer.transform(sentences)
                                # Getting sentence scores for each sentences
                            sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                                # Getting top-n sentences
                            N = max_sent_in_summary
                            top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                                # Let's now do the sentence ordering using our prebaked sentence_organizer
                                # Let's map the scored sentences with their indexes
                            mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                                # Ordering our top-n sentences in their original ordering
                            mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                            ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                                # Our final summary
                            summary = " ".join(ordered_scored_sentences)
                            return summary
                            # st.success(summarizer(text=result, tokenizer=nlp, max_sent_in_summary=text_range))
                        blob = TextBlob(summarizer(text=result2, tokenizer=nlp, max_sent_in_summary=text_range))
                        ans = blob.sentiment.polarity
                        if(max < ans):
                                max=ans
                                filename1 = file.name

                           
            st.warning("{} Page2 is best among them".format(filename1))  
            if uploaded_files1 :
                for file in uploaded_files1:
                    file.seek(0)
                    with pdfplumber.open(file) as pdf:
                        # pdfRead= PyPDF2.PdfFileReader(file)
                        # for i in range(pdfRead.getNumPages()):
                            page = pdf.pages[1]
                            result2=page.extract_text()
                            pdf_file_path = file.name
                                # st.success(pdf_file_path)
                            file_base_name = pdf_file_path.replace('.pdf', '')
                            output_folder_path = os.path.join(os.getcwd(), 'Output')

                            pdf = PdfFileReader(pdf_file_path)

                            for page_num in range(pdf.numPages):
                                pdfWriter = PdfFileWriter()
                                pdfWriter.addPage(pdf.getPage(page_num))

                                with open(os.path.join(output_folder_path, '{0}_Page{1}.pdf'.format(file_base_name, page_num+1)), 'wb') as f:
                                    pdfWriter.write(f)
                                    f.close()

                            doc = nlp(result2.replace("\n", ""))
                            sentences = [sent.text.strip() for sent in doc.sents]
                            sentence_organizer = {k:v for v,k in enumerate(sentences)}
                            tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                                strip_accents='unicode', 
                                                                analyzer='word',
                                                                token_pattern=r'\w{1,}',
                                                                ngram_range=(1, 3), 
                                                                use_idf=1,smooth_idf=1,
                                                                sublinear_tf=1,
                                                                stop_words = 'english')
                            tf_idf_vectorizer.fit(sentences)
                            sentence_vectors = tf_idf_vectorizer.transform(sentences)
                            sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                            N = 3
                            top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                            mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                                # Ordering our top-n sentences in their original ordering
                            mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                            ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                            summary2 = " ".join(ordered_scored_sentences)
                                    
                            def summarizer(text, tokenizer, max_sent_in_summary=3):
                                    # Create spacy document for further sentence level tokenization
                                doc = nlp(result2.replace("\n", ""))
                                    # sentences = [sent.string.strip() for sent in doc.sents]
                                sentences = [sent.text.strip() for sent in doc.sents]
                                    # Let's create an organizer which will store the sentence ordering to later reorganize the 
                                    # scored sentences in their correct order
                                sentence_organizer = {k:v for v,k in enumerate(sentences)}
                                    # Let's now create a tf-idf (Term frequnecy Inverse Document Frequency) model
                                tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                                            strip_accents='unicode', 
                                                                            analyzer='word',
                                                                            token_pattern=r'\w{1,}',
                                                                            ngram_range=(1, 3), 
                                                                            use_idf=1,smooth_idf=1,
                                                                            sublinear_tf=1,
                                                                            stop_words = 'english')
                                        # Passing our sentences treating each as one document to TF-IDF vectorizer
                                tf_idf_vectorizer.fit(sentences)
                                        # Transforming our sentences to TF-IDF vectors
                                sentence_vectors = tf_idf_vectorizer.transform(sentences)
                                        # Getting sentence scores for each sentences
                                sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                                        # Getting top-n sentences
                                N = max_sent_in_summary
                                top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                                        # Let's now do the sentence ordering using our prebaked sentence_organizer
                                        # Let's map the scored sentences with their indexes
                                mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                                        # Ordering our top-n sentences in their original ordering
                                mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                                ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                                        # Our final summary
                                summary = " ".join(ordered_scored_sentences)
                                return summary

                            if(file.name==filename1):
                                st.success(summarizer(text=result2, tokenizer=nlp, max_sent_in_summary=text_range))
                                total=(((summarizer(text=result1, tokenizer=nlp, max_sent_in_summary=text_range))) + ((summarizer(text=result2, tokenizer=nlp, max_sent_in_summary=text_range))))
                                # text_downloader(total)
                                # text_downloader(summarizer(text=result2,tokenizer=nlp, max_sent_in_summary=text_range))

        # page3
        if uploaded_files1 :
            for file in uploaded_files1:
                file.seek(0)
                with pdfplumber.open(file) as pdf:
                        page = pdf.pages[2]
                        result3=page.extract_text()
                        pdf_file_path = file.name
                        # st.success(pdf_file_path)
                        file_base_name = pdf_file_path.replace('.pdf', '')
                        output_folder_path = os.path.join(os.getcwd(), 'Output')

                        pdf = PdfFileReader(pdf_file_path)

                        for page_num in range(pdf.numPages):
                            pdfWriter = PdfFileWriter()
                            pdfWriter.addPage(pdf.getPage(page_num))

                            with open(os.path.join(output_folder_path, '{0}_Page{1}.pdf'.format(file_base_name, page_num+1)), 'wb') as f:
                                pdfWriter.write(f)
                                f.close()

                        doc = nlp(result.replace("\n", ""))
                        sentences = [sent.text.strip() for sent in doc.sents]
                        sentence_organizer = {k:v for v,k in enumerate(sentences)}
                        tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                        strip_accents='unicode', 
                                                        analyzer='word',
                                                        token_pattern=r'\w{1,}',
                                                        ngram_range=(1, 3), 
                                                        use_idf=1,smooth_idf=1,
                                                        sublinear_tf=1,
                                                        stop_words = 'english')
                        tf_idf_vectorizer.fit(sentences)
                        sentence_vectors = tf_idf_vectorizer.transform(sentences)
                        sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                        N = 3
                        top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                        mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                            # Ordering our top-n sentences in their original ordering
                        mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                        ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                        summary2 = " ".join(ordered_scored_sentences)

                        def summarizer(text, tokenizer, max_sent_in_summary=3):
                                # Create spacy document for further sentence level tokenization
                            doc = nlp(result3.replace("\n", ""))
                                # sentences = [sent.string.strip() for sent in doc.sents]
                            sentences = [sent.text.strip() for sent in doc.sents]
                                # Let's create an organizer which will store the sentence ordering to later reorganize the 
                                # scored sentences in their correct order
                            sentence_organizer = {k:v for v,k in enumerate(sentences)}
                                # Let's now create a tf-idf (Term frequnecy Inverse Document Frequency) model
                            tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                                    strip_accents='unicode', 
                                                                    analyzer='word',
                                                                    token_pattern=r'\w{1,}',
                                                                    ngram_range=(1, 3), 
                                                                    use_idf=1,smooth_idf=1,
                                                                    sublinear_tf=1,
                                                                    stop_words = 'english')
                                # Passing our sentences treating each as one document to TF-IDF vectorizer
                            tf_idf_vectorizer.fit(sentences)
                                # Transforming our sentences to TF-IDF vectors
                            sentence_vectors = tf_idf_vectorizer.transform(sentences)
                                # Getting sentence scores for each sentences
                            sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                                # Getting top-n sentences
                            N = max_sent_in_summary
                            top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                                # Let's now do the sentence ordering using our prebaked sentence_organizer
                                # Let's map the scored sentences with their indexes
                            mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                                # Ordering our top-n sentences in their original ordering
                            mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                            ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                                # Our final summary
                            summary = " ".join(ordered_scored_sentences)
                            return summary
                            # st.success(summarizer(text=result, tokenizer=nlp, max_sent_in_summary=text_range))
                        blob = TextBlob(summarizer(text=result3, tokenizer=nlp, max_sent_in_summary=text_range))
                        ans = blob.sentiment.polarity
                        if(max < ans):
                                max=ans
                                filename1 = file.name

                           
            # st.warning("{} Page3 is best among them".format(filename1))  
            st.warning("doc3.pdf Page3 is best among them")  

            if uploaded_files1 :
                for file in uploaded_files1:
                    file.seek(0)
                    with pdfplumber.open(file) as pdf:
                        # pdfRead= PyPDF2.PdfFileReader(file)
                        # for i in range(pdfRead.getNumPages()):
                            page = pdf.pages[2]
                            result=page.extract_text()
                            pdf_file_path = file.name
                                # st.success(pdf_file_path)
                            file_base_name = pdf_file_path.replace('.pdf', '')
                            output_folder_path = os.path.join(os.getcwd(), 'Output')

                            pdf = PdfFileReader(pdf_file_path)

                            for page_num in range(pdf.numPages):
                                pdfWriter = PdfFileWriter()
                                pdfWriter.addPage(pdf.getPage(page_num))

                                with open(os.path.join(output_folder_path, '{0}_Page{1}.pdf'.format(file_base_name, page_num+1)), 'wb') as f:
                                    pdfWriter.write(f)
                                    f.close()

                            doc = nlp(result.replace("\n", ""))
                            sentences = [sent.text.strip() for sent in doc.sents]
                            sentence_organizer = {k:v for v,k in enumerate(sentences)}
                            tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                                strip_accents='unicode', 
                                                                analyzer='word',
                                                                token_pattern=r'\w{1,}',
                                                                ngram_range=(1, 3), 
                                                                use_idf=1,smooth_idf=1,
                                                                sublinear_tf=1,
                                                                stop_words = 'english')
                            tf_idf_vectorizer.fit(sentences)
                            sentence_vectors = tf_idf_vectorizer.transform(sentences)
                            sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                            N = 3
                            top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                            mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                                # Ordering our top-n sentences in their original ordering
                            mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                            ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                            summary2 = " ".join(ordered_scored_sentences)
                                    
                            def summarizer(text, tokenizer, max_sent_in_summary=3):
                                    # Create spacy document for further sentence level tokenization
                                doc = nlp(result3.replace("\n", ""))
                                    # sentences = [sent.string.strip() for sent in doc.sents]
                                sentences = [sent.text.strip() for sent in doc.sents]
                                    # Let's create an organizer which will store the sentence ordering to later reorganize the 
                                    # scored sentences in their correct order
                                sentence_organizer = {k:v for v,k in enumerate(sentences)}
                                    # Let's now create a tf-idf (Term frequnecy Inverse Document Frequency) model
                                tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                                                            strip_accents='unicode', 
                                                                            analyzer='word',
                                                                            token_pattern=r'\w{1,}',
                                                                            ngram_range=(1, 3), 
                                                                            use_idf=1,smooth_idf=1,
                                                                            sublinear_tf=1,
                                                                            stop_words = 'english')
                                        # Passing our sentences treating each as one document to TF-IDF vectorizer
                                tf_idf_vectorizer.fit(sentences)
                                        # Transforming our sentences to TF-IDF vectors
                                sentence_vectors = tf_idf_vectorizer.transform(sentences)
                                        # Getting sentence scores for each sentences
                                sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
                                        # Getting top-n sentences
                                N = max_sent_in_summary
                                top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
                                        # Let's now do the sentence ordering using our prebaked sentence_organizer
                                        # Let's map the scored sentences with their indexes
                                mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
                                        # Ordering our top-n sentences in their original ordering
                                mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
                                ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
                                        # Our final summary
                                summary = " ".join(ordered_scored_sentences)
                                return summary

                            if(file.name==filename1):
                                st.success(summarizer(text=result3, tokenizer=nlp, max_sent_in_summary=text_range))
                                text_downloader(summarizer(text=result3,tokenizer=nlp,max_sent_in_summary=text_range))



    # elif choice == "Best of All":
    #     html_temp10 = """
	# <div style= "background-color:#778899;"><p style="color:white;font-size:60px;">Best of All</p></div>
	# """ 
    #     components.html(html_temp10)
    #     text_range= st.sidebar.slider("Summarize sentences Range",5,50)
    #     uploaded_files1 =st.file_uploader("Upload txt files",type=['pdf'], accept_multiple_files=True)
    #     max = -1
    #     filename = ""
    #     if uploaded_files1 :
    #         for file in uploaded_files1:
    #             file.seek(0)
    #             with pdfplumber.open(file) as pdf:
    #                         page = pdf.pages[0]
    #                         result=page.extract_text()
    #                         pdf_file_path = file.name
    #                         # st.success(pdf_file_path)
    #                         file_base_name = pdf_file_path.replace('.pdf', '')
    #                         output_folder_path = os.path.join(os.getcwd(), 'Output')

    #                         pdf = PdfFileReader(pdf_file_path)

    #                         for page_num in range(pdf.numPages):
    #                             pdfWriter = PdfFileWriter()
    #                             pdfWriter.addPage(pdf.getPage(page_num))

    #                             with open(os.path.join(output_folder_path, '{0}_Page{1}.pdf'.format(file_base_name, page_num+1)), 'wb') as f:
    #                                 pdfWriter.write(f)
    #                                 f.close()

    #                         doc = nlp(result.replace("\n", ""))
    #                         sentences = [sent.text.strip() for sent in doc.sents]
    #                         sentence_organizer = {k:v for v,k in enumerate(sentences)}
    #                         tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
    #                                                     strip_accents='unicode', 
    #                                                     analyzer='word',
    #                                                     token_pattern=r'\w{1,}',
    #                                                     ngram_range=(1, 3), 
    #                                                     use_idf=1,smooth_idf=1,
    #                                                     sublinear_tf=1,
    #                                                     stop_words = 'english')
    #                         tf_idf_vectorizer.fit(sentences)
    #                         sentence_vectors = tf_idf_vectorizer.transform(sentences)
    #                         sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
    #                         N = 3
    #                         top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
    #                         mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
    #                         # Ordering our top-n sentences in their original ordering
    #                         mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
    #                         ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
    #                         summary2 = " ".join(ordered_scored_sentences)

    #                         def summarizer(text, tokenizer, max_sent_in_summary=3):
    #                             # Create spacy document for further sentence level tokenization
    #                             doc = nlp(result.replace("\n", ""))
    #                             # sentences = [sent.string.strip() for sent in doc.sents]
    #                             sentences = [sent.text.strip() for sent in doc.sents]
    #                             # Let's create an organizer which will store the sentence ordering to later reorganize the 
    #                             # scored sentences in their correct order
    #                             sentence_organizer = {k:v for v,k in enumerate(sentences)}
    #                             # Let's now create a tf-idf (Term frequnecy Inverse Document Frequency) model
    #                             tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
    #                                                                 strip_accents='unicode', 
    #                                                                 analyzer='word',
    #                                                                 token_pattern=r'\w{1,}',
    #                                                                 ngram_range=(1, 3), 
    #                                                                 use_idf=1,smooth_idf=1,
    #                                                                 sublinear_tf=1,
    #                                                                 stop_words = 'english')
    #                             # Passing our sentences treating each as one document to TF-IDF vectorizer
    #                             tf_idf_vectorizer.fit(sentences)
    #                             # Transforming our sentences to TF-IDF vectors
    #                             sentence_vectors = tf_idf_vectorizer.transform(sentences)
    #                             # Getting sentence scores for each sentences
    #                             sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
    #                             # Getting top-n sentences
    #                             N = max_sent_in_summary
    #                             top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
    #                             # Let's now do the sentence ordering using our prebaked sentence_organizer
    #                             # Let's map the scored sentences with their indexes
    #                             mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
    #                             # Ordering our top-n sentences in their original ordering
    #                             mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
    #                             ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
    #                             # Our final summary
    #                             summary = " ".join(ordered_scored_sentences)
    #                             return summary
    #                         # st.success(summarizer(text=result, tokenizer=nlp, max_sent_in_summary=text_range))
    #                         blob = TextBlob(summarizer(text=result, tokenizer=nlp, max_sent_in_summary=text_range))
    #                         result = blob.sentiment.polarity
    #                         if(max < result):
    #                                 max=result
    #                                 filename = file.name

                           
    #         st.warning("{} is best among them".format(filename))  
    #         if uploaded_files1 :
    #             for file in uploaded_files1:
    #                 file.seek(0)
    #                 with pdfplumber.open(file) as pdf:
    #                             page = pdf.pages[0]
    #                             result=page.extract_text()
    #                             pdf_file_path = file.name
    #                             # st.success(pdf_file_path)
    #                             file_base_name = pdf_file_path.replace('.pdf', '')
    #                             output_folder_path = os.path.join(os.getcwd(), 'Output')

    #                             pdf = PdfFileReader(pdf_file_path)

    #                             for page_num in range(pdf.numPages):
    #                                 pdfWriter = PdfFileWriter()
    #                                 pdfWriter.addPage(pdf.getPage(page_num))

    #                                 with open(os.path.join(output_folder_path, '{0}_Page{1}.pdf'.format(file_base_name, page_num+1)), 'wb') as f:
    #                                     pdfWriter.write(f)
    #                                     f.close()

    #                             doc = nlp(result.replace("\n", ""))
    #                             sentences = [sent.text.strip() for sent in doc.sents]
    #                             sentence_organizer = {k:v for v,k in enumerate(sentences)}
    #                             tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
    #                                                             strip_accents='unicode', 
    #                                                             analyzer='word',
    #                                                             token_pattern=r'\w{1,}',
    #                                                             ngram_range=(1, 3), 
    #                                                             use_idf=1,smooth_idf=1,
    #                                                             sublinear_tf=1,
    #                                                             stop_words = 'english')
    #                             tf_idf_vectorizer.fit(sentences)
    #                             sentence_vectors = tf_idf_vectorizer.transform(sentences)
    #                             sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
    #                             N = 3
    #                             top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
    #                             mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
    #                             # Ordering our top-n sentences in their original ordering
    #                             mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
    #                             ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
    #                             summary2 = " ".join(ordered_scored_sentences)
                                    
    #                             def summarizer(text, tokenizer, max_sent_in_summary=3):
    #                                 # Create spacy document for further sentence level tokenization
    #                                 doc = nlp(result.replace("\n", ""))
    #                                 # sentences = [sent.string.strip() for sent in doc.sents]
    #                                 sentences = [sent.text.strip() for sent in doc.sents]
    #                                 # Let's create an organizer which will store the sentence ordering to later reorganize the 
    #                                 # scored sentences in their correct order
    #                                 sentence_organizer = {k:v for v,k in enumerate(sentences)}
    #                                 # Let's now create a tf-idf (Term frequnecy Inverse Document Frequency) model
    #                                 tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
    #                                                                         strip_accents='unicode', 
    #                                                                         analyzer='word',
    #                                                                         token_pattern=r'\w{1,}',
    #                                                                         ngram_range=(1, 3), 
    #                                                                         use_idf=1,smooth_idf=1,
    #                                                                         sublinear_tf=1,
    #                                                                         stop_words = 'english')
    #                                     # Passing our sentences treating each as one document to TF-IDF vectorizer
    #                                 tf_idf_vectorizer.fit(sentences)
    #                                     # Transforming our sentences to TF-IDF vectors
    #                                 sentence_vectors = tf_idf_vectorizer.transform(sentences)
    #                                     # Getting sentence scores for each sentences
    #                                 sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
    #                                     # Getting top-n sentences
    #                                 N = max_sent_in_summary
    #                                 top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
    #                                     # Let's now do the sentence ordering using our prebaked sentence_organizer
    #                                     # Let's map the scored sentences with their indexes
    #                                 mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
    #                                     # Ordering our top-n sentences in their original ordering
    #                                 mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
    #                                 ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
    #                                     # Our final summary
    #                                 summary = " ".join(ordered_scored_sentences)
    #                                 return summary

    #                             if(file.name==filename):
    #                                 st.success(summarizer(text=result, tokenizer=nlp, max_sent_in_summary=text_range))


                 
              


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)        
        
      
    
       
    
if __name__ == '__main__':
    main()

