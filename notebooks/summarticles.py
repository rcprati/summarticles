import os
import sys
import shutil
from distutils.dir_util import copy_tree
import re

from regex import F

sys.path.insert(0,os.path.dirname(os.getcwd()))
sys.path.insert(0,os.path.join(os.getcwd(),'grobid'))
sys.path.insert(0,os.getcwd())

import numpy as np
import pandas as pd

from grobid import grobid_client
import grobid_tei_xml
from grobid_to_dataframe import grobid_cli, xmltei_to_dataframe
from text import text_prep, text_mining, text_viz

import plotly.express as px
import tkinter as tk
from tkinter import filedialog

from customMsg import customMsg

import time
from datetime import datetime as dt

from joblib import dump, load

import streamlit as st
import streamlit.components.v1 as components

import networkx as nx
import matplotlib.pyplot as plt
from graph.pyvis.network import Network
import nltk

from annotated_text import annotated_text
import spacy
from spacy_streamlit import visualize_parser, visualize_ner

import random

# from keybert import KeyBERT
import yake

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

# docker run -t --rm --init -p 8080:8070 -p 8081:8071 --memory="9g" lfoppiano/grobid:0.7.0
# docker run -t --rm --init -p 8080:8070 -p 8081:8071 lfoppiano/grobid:0.6.2

path = os.path.dirname(os.getcwd())
global input_path

gcli = grobid_client.GrobidClient(config_path=os.path.join(path,"grobid/config.json"))


def get_path(path_input_path):
    """"""
    if os.path.exists(path_input_path):
        return path_input_path
    
    return os.getcwd()


def batch_process_path(path_input_path, n_workers=10, check_cache=True, cache_folder_name='summarticles_cache', config_path="./grobid/config.json"):
    
    """"""
    
    gcli = grobid_cli(config_path=config_path)
    result_batch = gcli.process_pdfs(input_path=path_input_path,
                                     check_cache=check_cache,
                                     cache_folder_name=cache_folder_name,
                                     n_workers=n_workers,
                                     service="processFulltextDocument",
                                     generateIDs=True,
                                     include_raw_citations=True,
                                     include_raw_affiliations=True,
                                     consolidate_header=False,
                                     consolidate_citations=False,
                                     tei_coordinates=False,
                                     segment_sentences=True,
                                     verbose=True)
    return result_batch


def get_dataframes(result_batch):
    
    """"""
    
    xml_to_df = xmltei_to_dataframe()
    dict_dfs, dict_errors = xml_to_df.get_dataframe_articles(result_batch)
    
    return dict_dfs, dict_errors


def files_path(path):
    
    """"""
    
    list_dir = os.listdir(path)
    files = []
    for file in list_dir:
        if os.path.isfile(os.path.join(path,file)):
            files.append(os.path.join(path,file))
    return files

def clean_error_results(result_batch):
    
    """"""
    
    new_result = []
    for result in result_batch:
        if "500" not in str(result[1]):
            new_result.append(result)
    return new_result


def check_input_path(st, path_input):

    # path_input = os.path.join(path,'artifacts','test_article')
    input_folder_path = get_path(path_input)
    
    print('[Path]:',input_folder_path)
    
    if not len(input_folder_path) or pd.isna(input_folder_path) or input_folder_path == "":
        st.error(f"❌ You need to specify a valid folder path, click on **'📁 Get folder path!'** Folder path: {str(input_folder_path)}!")
        return False
    elif not os.path.exists(input_folder_path):
        st.error(f"❌ The path doesn't exist! You need to specify a valid folder path! Folder path: {str(input_folder_path)}")
        return False
    elif not len(files_path(input_folder_path)):
        st.error(f"❌ There are no files in this folder path! You need to specify a valid folder path! Folder path: {str(input_folder_path)}")
        return False
    elif not gcli.check_typefile_inpath(files_path(input_folder_path)):
        st.error(f"❌ There are no files in this folder with APP required file type! Please make sure if that path is the correct path!")
        return False
    return True


def input_path_success_message(st, input_folder_path):
    st.success(f"✔️ **In this folder path we found: {str(len(files_path(input_folder_path)))} files!** Folder path: {str(input_folder_path)}")


def run_batch_process(st, path_input, cache_folder_name='summarticles_cache', n_workers=10, display_articles_data=True, save_xmltei=True):

    """"""

    # path_input = os.path.join(path,'artifacts','test_article')
    input_folder_path = get_path(path_input)
        
    if len(files_path(input_folder_path)) < 2:
        st.error(f"❌ You need to specify a path with at least two pdf files!")
        return None
    
    with st.spinner('⚡ [Extract Text from Articles] Running batch process...'):
        
        result_batch = batch_process_path(input_folder_path, n_workers= n_workers)
        result_batch = clean_error_results(result_batch)
        
        if not len(result_batch):
            st.error(f"⚠️ Something is wrong, I can't get any result! 😕 Please, look if you selected the correct file path or if files in the selected path have information to extract!")
            return None
        
        dict_dfs, dict_errors = get_dataframes(result_batch)

        if save_xmltei:
            gcli.save_xmltei_files(result_batch, input_folder_path, cache_folder_name=cache_folder_name)

        if display_articles_data and len(result_batch):
            with st.spinner('🧾 Showing articles information...'):
                show_articles_data(st, dict_dfs)

    print('[Process has been finished!!!]')
    #msg.empty()
    #msg = customMsg('⚡ Finish process!','warning')
    
    return dict_dfs


def chars_graph(dict_dfs):
    
    """"""
    
    list_chars = []
    for id,row in dict_dfs['df_doc_info'].iterrows():
        for c in row['raw_data']:
            list_chars.append(c)
            
    df_counts = pd.DataFrame({'chars':pd.value_counts(list_chars).index.tolist(),'counts':pd.value_counts(list_chars).tolist()})
    df_counts = df_counts.sort_values(by='counts',ascending=False)
    
    fig = px.bar(df_counts.head(20), x='chars', y='counts')
    
    return fig


def tk_configs():
    
    """"""
    
    # Set up tkinter
    tk_root = tk.Tk()
    tk_root.withdraw()
    
    # Make folder picker dialog appear on top of other windows
    tk_root.wm_attributes('-topmost', 1)
    
    return tk_root


def _max_width_(st):
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


def make_sidebar(st):
    
    """"""

    with st.sidebar:
        st.markdown("""Menu Sidebar""", unsafe_allow_html=False)
        
        clear_all = st.button("🛑 Clear all memory and execution!", key="clear_all")
        if clear_all:
            st.session_state = {}
            st.experimental_memo.clear()
            st.experimental_singleton.clear()
            st.experimental_rerun()
            
def make_head(st):

    """"""

    # Head
    st.set_page_config(
        page_title="Summarticles",
        page_icon="📑", # https://www.freecodecamp.org/news/all-emojis-emoji-list-for-copy-and-paste/, https://share.streamlit.io/streamlit/emoji-shortcodes
        layout="wide", # centered
        initial_sidebar_state="collapsed", #collapsed #auto #expanded
        menu_items={"About":"https://github.com/Vieirbat/PGC",
                    "Get help":"https://github.com/Vieirbat/PGC",
                    "Report a bug":"https://github.com/Vieirbat/PGC"}) 
    
    st.markdown("""<h1 style="text-align:center;">🧾➜📊 Summarticles</h1>""",
                unsafe_allow_html=True)
    
    st.markdown("""<h3 style="text-align:center;"><b>Summarticles is an application to summarize articles information, 
                    using IA and analytics.</b></h3>""", 
                unsafe_allow_html=True) # st.write("Application")
    
    st.markdown("""<h6 style="text-align:center;">Do you want to use it? So, you only need to specify a folder path (with PDF article files) clicking 
                on '📁 Select a folder path!' and let the magic happen!</h6>""",
                unsafe_allow_html=True)


def make_getpath_button(st):
    
    """"""
    
    _, btn_col1, _ = st.columns([3,3,1])
    
    with btn_col1:
        btn_getfolder = st.button('📁 Select a folder path!',key='btn_getfolder')
        st.markdown("""<p style="text-align:left;font-size:11px;">This application only works with PDF files.</p>""", unsafe_allow_html=True)
    
    return btn_getfolder


def make_run_button(st):
    
    """"""
    
    _, btn_col1, _ = st.columns([3,3,1])
    
    with btn_col1:
        btn_getfolder = st.button('⚡ Run process!', key='btn_run_app')
        st.markdown("""<p style="text-align:left;font-size:11px;">Only enabled if input path is correctly!</p>""", unsafe_allow_html=True)
    
    return btn_getfolder


def show_macro_numbers(st, dict_dfs):
    
    st.markdown("""<h3 style="text-align:left;"><b>Articles Macro Numbers</b></h3>""", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🧾 Read Articles", str(dict_dfs['df_doc_info'].shape[0]))
    col2.metric("👥 Total Authors", str(dict_dfs['df_doc_authors'].shape[0]))
    col3.metric("📄➞📄 Total Citations",str(dict_dfs['df_doc_citations'].shape[0]))
    col4.metric("👥➞👥 Total Authors from Citations",str(dict_dfs['df_doc_authors_citations'].shape[0]))

    # st.plotly_chart(chars_graph(dict_dfs),use_container_width=True)


def text_statistics(text):
    
    """"""
    
    tprep = text_prep()
    
    dictStats = {}
    dictStats['num_chars'] = len(text)
    list_tokens = tprep.text_tokenize(str(text))
    dictStats['num_words'] = len(list_tokens)
    dictStats['num_words_unique'] = len(set(list_tokens))
    len_words = pd.Series([len(w) for w in list_tokens]).describe().to_dict()
    dictStats['mean_lenght_word'] = len_words['mean']
    dictStats['min_lenght_word'] = len_words['min']
    dictStats['max_lenght_word'] = len_words['max']
    dictStats['std_lenght_word'] = len_words['std']
    dictStats['first_quartile_lenght_word'] = len_words['25%']
    dictStats['second_quartile_median_lenght_word'] = len_words['50%']
    dictStats['third_quartile_lenght_word'] = len_words['75%']
    
    return dictStats

    
def show_text_numbers(st, dict_dfs):
    
    """"""
    
    tprep = text_prep()
    
    if not checkey(dict_dfs,'show_text_numbers'):
    
        dict_dfs['show_text_numbers'] = {}
        
        if not checkey(dict_dfs['show_text_numbers'],'documents_abs'):
            documents_abs = dict_dfs['df_doc_info']['abstract_prep'].fillna(' ').tolist()
            dict_dfs['show_text_numbers']['documents_abs'] = documents_abs
        else:
            documents_abs = dict_dfs['show_text_numbers']['documents_abs']
        
        if not checkey(dict_dfs['show_text_numbers'],'documents_body'):
            documents_body = dict_dfs['df_doc_info']['body_prep'].fillna(' ').tolist()
            dict_dfs['show_text_numbers']['documents_body'] = documents_body
        else:
            documents_body = dict_dfs['show_text_numbers']['documents_body']
        
        if not checkey(dict_dfs['show_text_numbers'],'documents_all_text'):
            documents_all_text = [' '.join([abst, body]) for abst, body in zip(documents_abs, documents_body)]
            dict_dfs['show_text_numbers']['documents_all_text'] = documents_all_text
        else:
            documents_body = dict_dfs['show_text_numbers']['documents_body']
    
    # ------------------------------------------------------------------------
    # Making stats
    
    if not checkey(dict_dfs['show_text_numbers'],'dict_agg_stats'):
    
        df_articles_stats = pd.DataFrame(list(map(lambda e: text_statistics(e), documents_all_text)))
        df_articles_stats['file_name'] = [os.path.split(e)[-1] for e in dict_dfs['df_doc_info']['file'].tolist()]
        dict_agg_stats = {}

        # Chars
        dict_agg_stats['num_total_chars'] = df_articles_stats['num_chars'].sum()
        dict_agg_stats['num_mean_chars'] = df_articles_stats['num_chars'].mean()
        dict_agg_stats['num_min_chars'] = df_articles_stats['num_chars'].min()
        dict_agg_stats['num_max_chars'] = df_articles_stats['num_chars'].max()

        # num_words
        dict_agg_stats['num_total_words'] = df_articles_stats['num_words'].sum()
        dict_agg_stats['num_mean_words'] = df_articles_stats['num_words'].mean()
        dict_agg_stats['num_min_words'] = df_articles_stats['num_words'].min()
        dict_agg_stats['num_max_words'] = df_articles_stats['num_words'].max()

        # num_words_unique
        dict_agg_stats['num_total_words_unique'] = df_articles_stats['num_words_unique'].sum()
        dict_agg_stats['num_mean_words_unique'] = df_articles_stats['num_words_unique'].mean()
        dict_agg_stats['num_min_words_unique'] = df_articles_stats['num_words_unique'].min()
        dict_agg_stats['num_max_chars_unique'] = df_articles_stats['num_words_unique'].max()

        # mean_lenght_word
        dict_agg_stats['mean_length_words'] = df_articles_stats['mean_lenght_word'].mean()

        # mean_lenght_word
        dict_agg_stats['lexical_density'] = dict_agg_stats['num_mean_words']/dict_agg_stats['num_mean_words_unique']
        
        # Number articles at least 280 characters
        filtro = (df_articles_stats.num_chars <= 280)
        dict_agg_stats['twitter_articles'] = df_articles_stats.loc[(filtro)].shape[0]
        
        # Number articles at least 280 characters
        filtro = (df_articles_stats.num_words < 100)
        dict_agg_stats['articles_lower'] = df_articles_stats.loc[(filtro)].shape[0]
        
        dict_dfs['show_text_numbers']['dict_agg_stats'] = dict_agg_stats
        dict_dfs['show_text_numbers']['df_articles_stats'] = df_articles_stats
        
    else:
        dict_agg_stats = dict_dfs['show_text_numbers']['dict_agg_stats']
        df_articles_stats = dict_dfs['show_text_numbers']['df_articles_stats']
    
    
    # token all text
    if not checkey(dict_dfs['show_text_numbers'],'token_text'):
        token_text = tprep.text_tokenize(' '.join(documents_all_text))
        dict_dfs['show_text_numbers']['token_text'] = token_text
    else:
        token_text = dict_dfs['show_text_numbers']['token_text']
    
    # regex
    def f_reg(t):
        texto = re.sub(r'\W+',' ',str(t))
        texto = re.sub(r'\s+', ' ', texto)
        texto = texto.strip()
        return texto
    
    # Unigram
    if not checkey(dict_dfs['show_text_numbers'],'words_freq'):
        words_freq = pd.value_counts(token_text)
        words_freq = pd.DataFrame(words_freq,columns=['frequency'])
        words_freq.index.name = 'unigram'
        words_freq = words_freq.reset_index()
        dict_agg_stats['num_total_words_unique'] = len(list(set(words_freq.unigram.tolist())))
        words_freq.unigram = words_freq.unigram.apply(f_reg)
        dict_dfs['show_text_numbers']['words_freq'] = words_freq
    else:
        words_freq = dict_dfs['show_text_numbers']['words_freq']
    
    # Bigram - this code can be in the prep/mining class on the text.py
    if not checkey(dict_dfs['show_text_numbers'],'df_bigram'):
        list_bigrams = list(nltk.bigrams(token_text))
        bigram_freq = pd.value_counts(list_bigrams)
        df_bigram = pd.DataFrame(bigram_freq, columns=['frequency'])
        df_bigram.index.name = 'bigram'
        df_bigram = df_bigram.reset_index()
        df_bigram.bigram = df_bigram.bigram.apply(f_reg)
        dict_dfs['show_text_numbers']['df_bigram'] = df_bigram
    else:
        df_bigram = dict_dfs['show_text_numbers']['df_bigram']
    
    # Trigram - this code can be in the prep/mining class on the text.py
    if not checkey(dict_dfs['show_text_numbers'],'df_trigram'):
        list_trigam = list(nltk.trigrams(token_text))
        trigam_freq = pd.value_counts(list_trigam)
        df_trigram = pd.DataFrame(trigam_freq, columns=['frequency'])
        df_trigram.index.name = 'trigram'
        df_trigram = df_trigram.reset_index()
        df_trigram.trigram = df_trigram.trigram.apply(f_reg)
        dict_dfs['show_text_numbers']['df_trigram'] = df_trigram
    else:
        df_trigram = dict_dfs['show_text_numbers']['df_trigram']
    
    with st.container():
        _ , col1, col2, col3, _ = st.columns([1.5, 3, 3, 3, 0.25])
        with col1:
            col1.metric("🔠 Total Words", str(dict_agg_stats['num_total_words']))
            col1.metric("🔢 Mean Words per Article", str(round(dict_agg_stats['num_mean_words'],1)))
        with col2:
            col2.metric("🆕 Total Unique Words", str(dict_agg_stats['num_total_words_unique']))
            col2.metric("🔢 Mean Unique Words per Article", str(round(dict_agg_stats['num_mean_words_unique'],1)))
        with col3:
            col3.metric("🔣 Total Characters", str(dict_agg_stats['num_total_chars']))
            col3.metric("🔤 Mean Lenght Words", str(round(dict_agg_stats['mean_length_words'],1)))
            
        with st.expander(" ❕ Information!"):
            body = """Above:<br>These numbers represent informations from all article text together.<br><br>
                      Below:<br>This chart represent the number of unique words by article cross number of words by article.<br>
                      Each point represent an article, you can pass cursor over these points and get more information."""
            st.markdown(body, unsafe_allow_html=True)
        
        with st.container():
            fig = px.scatter(df_articles_stats, 
                            x="num_words_unique", 
                            y="num_words", 
                            size="mean_lenght_word", 
                            color="num_chars",
                            custom_data=['file_name','num_chars','mean_lenght_word'],
                            labels={"num_words_unique":"Number of Unique Words",
                                    "num_words":"Number of Words",
                                    "num_chars":"Number of<br>Characters"})
            
            labels = ["Article File Name: %{customdata[0]}",
                      "Number Words: %{y}",
                      "Number Unique Words: %{x}",
                      "Total Number of Characters: %{customdata[1]}",
                      "Mean Leangth of Words: %{customdata[2]}"]
                        
            fig.update_traces(hovertemplate="<br>".join(labels))
            
            st.plotly_chart(fig, use_container_width=True)
            
            
    with st.container():
        _ , col1, col2, col3, _ = st.columns([0.15,2.25,2.5,2.75,0.15])
        with col1:
            AgGrid(words_freq.head(100),
                   data_return_mode='AS_INPUT', 
                   # update_mode='MODEL_CHANGED', 
                   fit_columns_on_grid_load=False,
                   # theme='fresh',
                   enable_enterprise_modules=False,
                   height=250, 
                   width='100%',
                   reload_data=True)
        with col2:
            AgGrid(df_bigram.head(50),
                   data_return_mode='AS_INPUT', 
                   # update_mode='MODEL_CHANGED', 
                   fit_columns_on_grid_load=False,
                   # theme='fresh',
                   enable_enterprise_modules=False,
                   height=250, 
                   width='100%',
                   reload_data=True)
        with col3:
            AgGrid(df_trigram.head(50),
                   data_return_mode='AS_INPUT', 
                   # update_mode='MODEL_CHANGED', 
                   fit_columns_on_grid_load=False,
                   # theme='fresh',
                   enable_enterprise_modules=False,
                   height=250, 
                   width='100%',
                   reload_data=True)
            
        with st.expander(" ❕ Information!"):
            body = """Unigram: only one frequent word that happen with high frequency over the all articles text<br>
                      Bigram: two words that happen with high frequency together over the all articles text<br>
                      Trigram: three words that happen with high frequency together over the all articles text<br>"""
            st.markdown(body, unsafe_allow_html=True)
            
    return dict_dfs


def show_articles_data(st, dict_dfs):
    
    """"""
    
    st.write("**[Articles Data] df_doc_info (with 5 rows sample):**")
    st.dataframe(dict_dfs['df_doc_info'].head(5).astype(str), width=None, height=None)
    
    st.write("**[Head Articles Data] df_doc_head (with 5 rows sample):**")
    st.dataframe(dict_dfs['df_doc_head'].head(5).astype(str), width=None, height=None)
    
    st.write("**[Authors Articles Data] df_doc_authors (with 5 rows sample):**")
    st.dataframe(dict_dfs['df_doc_authors'].head(5).astype(str), width=None, height=None)
    
    st.write("**[Citations Articles Data] df_doc_citations (with 5 rows sample):**")
    st.dataframe(dict_dfs['df_doc_citations'].head(5).astype(str), width=None, height=None)
    
    st.write("**[Authors Citations Articles Data] df_doc_authors_citations (with 5 rows sample):**")
    st.dataframe(dict_dfs['df_doc_authors_citations'].head(5).astype(str), width=None, height=None)
    
    return dict_dfs


def show_word_cloud(st, dict_dfs, input_path, folder_images='app_images', wc_image_name='wc_image.png',
                    cache_folder_name='summarticles_cache'):
    
    """"""
    
    tviz = text_viz()
    tprep = text_prep()
    
    with st.expander(" ❕ Information!"):
        st.write("""This WordCloud contain information from all article text together! 
                    But, only uses the most frequent unigrams and bigrams.""")
    
    # st.warning("🛠🧾 Text in preparation for WordCloud!")
    if not checkey(dict_dfs,'show_word_cloud'):
        
        dict_dfs['show_word_cloud'] = {}
        
        # if "abstract_prep" not in dict_dfs['df_doc_info']:
        #     dict_dfs['df_doc_info']['abstract_prep'] = tprep.text_preparation_column(dict_dfs['df_doc_info']['abstract'])
        documents = dict_dfs['df_doc_info']['abstract_prep'].fillna(' ').tolist()
        
        path_images = os.path.join(input_path, cache_folder_name, folder_images)
        if not os.path.exists(path_images):
            os.mkdir(path_images)

        wc, ax, fig = tviz.word_cloud(documents, 
                                      path_image=os.path.join(path_images, wc_image_name), 
                                      show_wc=False, 
                                      width=1000, 
                                      height=500, 
                                      collocations=True, 
                                      background_color='white')
        
        # st.markdown("""<h5 style="text-align:left;"><b>WordCloud:</b></h5>""",unsafe_allow_html=True)
        dict_dfs['show_word_cloud']['word_cloud_fig'] = fig
    
    st.pyplot(dict_dfs['show_word_cloud']['word_cloud_fig'])
    
    return dict_dfs


def show_keyword_word_cloud(st, dict_dfs,
                            input_path,
                            folder_images='app_images',
                            wc_image_name='wc_image_keyword.png',
                            cache_folder_name='summarticles_cache'):
    
    """"""
    
    tviz = text_viz()
    tprep = text_prep()
    
    # st.warning("🛠🧾 Text in preparation for WordCloud!")
    if not 'show_keyword_word_cloud' in dict_dfs:
        
        dict_dfs['show_keyword_word_cloud'] = {}
        
        df_unigram = dict_dfs['keywords']['df_unigram']
        df_unigram = df_unigram.rename(columns={'keyword_unigram':'keyword',
                                                'value_unigram':'value'})
        df_bigram = dict_dfs['keywords']['df_bigram']
        df_bigram = df_bigram.rename(columns={'keyword_bigram':'keyword',
                                              'value_bigram':'value'})
        df_trigram = dict_dfs['keywords']['df_trigram'].rename(columns={'keyword_trigram':'keyword',
                                                                        'value_trigram':'value'})
        df_trigram = df_trigram.rename(columns={'keyword_trigram':'keyword',
                                                'value_trigram':'value'})
        
        df_keywords = pd.concat([df_unigram, df_bigram, df_trigram])
        
        dict_freq = {}
        for i, row in df_keywords.iterrows():
            dict_freq[row['keyword']] = 1/row['value']
        
        path_images = os.path.join(input_path, cache_folder_name, folder_images)
        if not os.path.exists(path_images):
            os.mkdir(path_images)

        wc, ax, fig = tviz.keyword_word_cloud(dict_freq, 
                                              path_image=os.path.join(path_images, wc_image_name), 
                                              show_wc=False, 
                                              width=1000, 
                                              height=300, 
                                              collocations=True, 
                                              background_color='white')
        
        # st.markdown("""<h5 style="text-align:left;"><b>WordCloud:</b></h5>""",unsafe_allow_html=True)
        dict_dfs['show_keyword_word_cloud']['word_cloud_fig'] = fig
    
    st.pyplot(dict_dfs['show_keyword_word_cloud']['word_cloud_fig'])
    
    return dict_dfs


def cossine_similarity_data(st, dict_dfs, column='abstract', n_sim=200,
                            percentil="99%", sim_value_min=0, sim_value_max=0.99):
    
    """"""

    tmining = text_mining()
    
    with st.spinner('📄✢📄 Making similarity relations...'):
        
        documents = dict_dfs['df_doc_info'][column + '_prep'].fillna(' ').tolist()
        df_tfidf_abstract = tmining.get_df_tfidf(documents)
        
        df_cos_tfidf_sim = tmining.get_cossine_similarity_matrix(df_tfidf_abstract,
                                                                 dict_dfs['df_doc_info'].index.tolist())
        
    with st.spinner('📑🔍 Filtering best similarities relations...'):
        
        df_cos_tfidf_sim_filter = tmining.filter_sim_matrix(df_cos_tfidf_sim,
                                                            df_cos_tfidf_sim.index.tolist(),
                                                            n_sim=n_sim,
                                                            percentil=percentil,
                                                            value_min=sim_value_min,
                                                            value_max=sim_value_max)
    
    return  df_cos_tfidf_sim_filter


def nodes_data(st, dict_dfs, df_cos_sim_filter):
    
    """"""
    
    with st.spinner('📄➞📄 Similarity Graph: extract nodes information...'):
        
        # Selecting head article data
        cols_head = ['title_head', 'doi_head', 'date_head',]
        head_data = dict_dfs['df_doc_head'].loc[:,cols_head].reset_index().copy()
        head_data['title_head'] = head_data['title_head'].apply(lambda e: str(e)[0:50] + "..." if len(str(e)) > 50 else str(e))

        # Selecting head article data
        cols_info = ['abstract','file']
        doc_info_data = dict_dfs['df_doc_info'].loc[:,cols_info].reset_index().copy()
        doc_info_data['file_name'] = doc_info_data['file'].apply(lambda e: os.path.split(e)[-1])
        doc_info_data['abstract_short'] = doc_info_data['abstract'].apply(lambda e: str(e)[0:20] + "..." if len(str(e)) > 20 else str(e))
        doc_info_data.drop(labels=['abstract'], axis=1, inplace=True)

        # Selecting authors information
        authors_data = dict_dfs['df_doc_authors'].reset_index()
        authors_data = authors_data.groupby(by=['article_id'], as_index=False)['full_name_author'].count()
        authors_data.rename(columns={'full_name_author':'author_count'}, inplace=True)

        # Selecting citations information
        citations_data = dict_dfs['df_doc_citations'].reset_index()
        citations_data = citations_data.groupby(by=['article_id'], as_index=False)['index_citation'].count()
        citations_data.rename(columns={'index_citation':'citation_count'}, inplace=True)

        nodes = list(set(df_cos_sim_filter.doc_a.tolist() + df_cos_sim_filter.doc_b.tolist()))
        df_nodes = pd.DataFrame(nodes, columns=['article_id'])

        df_nodes = df_nodes.merge(head_data, how='left', on='article_id')
        df_nodes = df_nodes.merge(doc_info_data, how='left', on='article_id')
        df_nodes = df_nodes.merge(authors_data, how='left', on='article_id')
        df_nodes = df_nodes.merge(citations_data, how='left', on='article_id')

    return df_nodes

def part_of_speech(st, dict_dfs):

    article_titles = dict_dfs['df_doc_head']['title_head'].tolist()
    file_names = dict_dfs['df_doc_info']['file'].apply(lambda e: os.path.split(e)[-1])

    list_articles = list(zip(article_titles,file_names))
    list_articles_select = [' - '.join([str(e[0]),str(e[1])]) for e in list_articles]

    abstracts = dict_dfs['df_doc_info']['abstract'].tolist()
    list_articles_abstracts = list(zip(list_articles_select,abstracts))
    dict_articles_abstracts = {e[0]:e[1] for e in list_articles_abstracts}
    
    choice_exec = st.selectbox("Please, choose one article: ", list_articles_select, key="select_box_pos")
    
    # Há mais modelos para testar aqui https://github.com/allenai/scispacy
    # python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_sci_sm")
    doc = nlp(dict_articles_abstracts[choice_exec])
    list_pos_tag = [(ent.text,ent.label_) for ent in doc.ents]
    
    # col1, col2, col3 = st.columns([0.1,2,0.1])
    # with col2:
    
    doc.user_data["title"] = f"File: {choice_exec}"
    
    # https://github.com/explosion/spacy-streamlit
    visualize_ner(doc, show_table=False, title=False)
    #visualize_parser(doc)
    
    # annotated_text(*list_pos_tag)


def similarity_graph(st, dict_dfs, input_folder_path, folder_graph='graphs', name_file="graph.html", cache_folder_name='summarticles_cache', 
                     column='abstract', n_sim=200, percentil="99%", sim_value_min=0, sim_value_max=0.99, buttons=False):
    
    """"""

    if  not ('similarity_graph' in dict_dfs):
        
        dict_dfs['similarity_graph'] = {}
    
        tmining = text_mining()
        
        df_cos_tfidf_sim_filter = cossine_similarity_data(st, dict_dfs, column, n_sim, percentil,
                                                          sim_value_min, sim_value_max)
        
        dict_dfs['similarity_graph']['df_cos_tfidf_sim_filter'] = df_cos_tfidf_sim_filter
    
        df_nodes = nodes_data(st, dict_dfs, df_cos_tfidf_sim_filter)
        dict_dfs['similarity_graph']['df_nodes'] = df_nodes
        
        path_write_graph = os.path.join(input_folder_path, cache_folder_name)
        dict_dfs['similarity_graph']['path_write_graph'] = path_write_graph

        print('SIM SHAPE: ', df_cos_tfidf_sim_filter.shape)

        sim_graph, path_graph, path_folder_graph = tmining.make_sim_graph(matrix=df_cos_tfidf_sim_filter, node_data=df_nodes,
                                                    source_column="doc_a", to_column="doc_b", value_column="value",
                                                    height="500px", width="100%", directed=False, notebook=False,
                                                    bgcolor="#ffffff", font_color=False, layout=None, 
                                                    heading="Similarity Graph: this graph shows you similarity across articles.",
                                                    path_graph=path_write_graph, folder_graph=folder_graph, buttons=buttons,
                                                    name_file=name_file)
        
        dict_dfs['similarity_graph']['sim_graph'] = sim_graph
        dict_dfs['similarity_graph']['path_graph'] = path_graph
        dict_dfs['similarity_graph']['path_folder_graph'] = path_folder_graph
        
    with st.container():
        
        df_sim = None
        df_sim = dict_dfs['similarity_graph']['df_cos_tfidf_sim_filter']
        df_sim['value'] = df_sim['value'].apply(lambda e: round(e,2))
        df_sim['doc_a'] = df_sim['doc_a'].apply(lambda e: e[0:4])
        df_sim['doc_b'] = df_sim['doc_b'].apply(lambda e: e[0:4])

        df_nodes_info = dict_dfs['similarity_graph']['df_nodes']
        cols = ['article_id','title_head','file_name']
        df_nodes_info = df_nodes_info.loc[:,cols].copy()
        df_sim_all = df_sim.merge(df_nodes_info,
                                  how='left', 
                                  left_on='doc_a', 
                                  right_on='article_id')
        df_sim_all.drop(labels=['article_id'], axis=1, inplace=True)
        df_sim_all.rename(columns={"doc_a":"ID Article Source",
                                   "value":"Similarity",
                                   "doc_b":"ID Article Target",
                                   "title_head":"Article Title Source", 
                                   "file_name":"Article File Name Source"}, inplace=True)
        df_sim_all = df_sim_all.merge(df_nodes_info,
                                      how='left', 
                                      left_on='ID Article Target', 
                                      right_on='article_id')
        df_sim_all.drop(labels=['article_id'], axis=1, inplace=True)
        df_sim_all.rename(columns={"title_head":"Article Title Target", 
                                   "file_name":"Article File Name Target"}, inplace=True)
                
        df_sim = df_sim.rename(columns={"value":"Similarity",
                                        "doc_a":"Article Source",
                                        "doc_b":"Article Target"})
        
        AgGrid(df_sim_all,
               data_return_mode='AS_INPUT', 
               fit_columns_on_grid_load=False,
               enable_enterprise_modules=False,
               height=350, 
               width='100%',
               reload_data=True)
            
        # col1, col2 = st.columns([1,2])
        # with col1:
        #     AgGrid(df_sim.head(n_sim),
        #            data_return_mode='AS_INPUT', 
        #            # update_mode='MODEL_CHANGED', 
        #            fit_columns_on_grid_load=False,
        #            # theme='fresh',
        #            enable_enterprise_modules=False,
        #            height=510, 
        #            width='100%',
        #            reload_data=True)
        # with col2:
        
        show_graph(dict_dfs['similarity_graph']['sim_graph'],
                    dict_dfs['similarity_graph']['path_graph'],
                    dict_dfs['similarity_graph']['path_folder_graph'],
                    text_spinner='👁‍🗨 Similarity Graph: drawing...')
            
        relations_size = dict_dfs['similarity_graph']['df_cos_tfidf_sim_filter'].shape[0]
        sim_max_val = dict_dfs['similarity_graph']['df_cos_tfidf_sim_filter'].value.max()
        
        return dict_dfs, relations_size, sim_max_val


def get_component_from_file(path_html):
    
    """"""

    GraphHtmlFile = open(path_html, 'r', encoding='utf-8')
    GraphHtml = GraphHtmlFile.read()
    GraphHtmlFile.close()
    
    return GraphHtml


def text_preparation(st, dict_dfs, input_folder_path):
    
    """"""
    
    tprep = text_prep()
    
    if not checkey(dict_dfs,'text_preparation'):
        
        dict_dfs['text_preparation'] = {}
        
        dict_dfs['df_doc_info']['abstract_prep'] = tprep.text_preparation_column(dict_dfs['df_doc_info']['abstract'])
        dict_dfs['text_preparation']['abstract_prep'] = dict_dfs['df_doc_info']['abstract_prep']
        
        # dict_dfs['df_doc_info']['acknowledgement_prep'] = tprep.text_prep_column(dict_dfs['df_doc_info']['acknowledgement'])
        
        dict_dfs['df_doc_info']['body_prep'] = tprep.text_preparation_column(dict_dfs['df_doc_info']['body'])
        dict_dfs['text_preparation']['body_prep'] = dict_dfs['df_doc_info']['body_prep']
    else:
        dict_dfs['df_doc_info']['abstract_prep'] = dict_dfs['text_preparation']['abstract_prep']
        dict_dfs['df_doc_info']['body_prep'] = dict_dfs['text_preparation']['body_prep']
        
    return dict_dfs


def generate_keywords(st, dict_dfs):
    
    """"""
    
    if 'keywords' not in dict_dfs:
        
        dict_dfs['keywords'] = {}
        
        # kw_model = KeyBERT()
        
        dict_keywords = {}
        id_column = 'article_id'
        text_column = 'abstract'
        col_select = [id_column,text_column]
        docs = dict_dfs['df_doc_info'].reset_index().loc[:, col_select]

        list_keywordsdf = []
        list_keywordsdf_article = []
        for i, row in docs.head(20).iterrows():
            
            doc = str(row[text_column])
            id = row[id_column]
            
            # keywords_unigram = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words='english', highlight=False, top_n=10)
            kw_extractor  = yake.KeywordExtractor(lan='en', n=1, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top='20', features=None)
            keywords_unigram = kw_extractor.extract_keywords(doc)
            if len(keywords_unigram):
                df_unigram = pd.DataFrame([{'keyword':v[0],'value':v[1]} for v in keywords_unigram])
            else:
                df_unigram = pd.DataFrame([], columns=['keyword','value'])

            # keywords_bigram = kw_model.extract_keywords(doc, keyphrase_ngram_range=(2, 2), stop_words='english', highlight=False, top_n=10)
            kw_extractor  = yake.KeywordExtractor(lan='en', n=2, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top='20', features=None)
            keywords_bigram = kw_extractor.extract_keywords(doc)
            if len(keywords_bigram):
                df_bigram = pd.DataFrame([{'keyword':v[0],'value':v[1]} for v in keywords_bigram])
            else:
                df_bigram = pd.DataFrame([], columns=['keyword','value'])

            # keywords_trigam = kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english', highlight=False, top_n=10)
            kw_extractor  = yake.KeywordExtractor(lan='en', n=3, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top='20', features=None)
            keywords_trigam = kw_extractor.extract_keywords(doc)
            if len(keywords_trigam):
                df_trigram = pd.DataFrame([{'keyword':v[0],'value':v[1]} for v in keywords_trigam])
            else:
                df_trigram = pd.DataFrame([], columns=['keyword','value'])
            
            dict_keywords[id] = {'unigram':df_unigram, 'bigram':df_bigram, 'trigram':df_trigram}
            
            df_article_keywords = pd.concat([df_unigram, df_bigram, df_trigram])
            df_article_keywords[id_column] = id
            df_article_keywords = df_article_keywords.loc[:,[id_column,'keyword', 'value']].copy()
            list_keywordsdf_article.append(df_article_keywords)
            
            df_unigram.rename(columns={'keyword':'keyword_unigram','value':'value_unigram'}, inplace=True)
            df_bigram.rename(columns={'keyword':'keyword_bigram','value':'value_bigram'}, inplace=True)
            df_trigram.rename(columns={'keyword':'keyword_trigram','value':'value_trigram'}, inplace=True)
            
            df_keywords_article = pd.concat([df_unigram, df_bigram, df_trigram], axis=1)
            dict_keywords[id]['df_keywords'] = df_keywords_article
            
            list_keywordsdf.append(df_keywords_article)
        
        dict_dfs['keywords']['list_keywordsdf'] = list_keywordsdf
        dict_dfs['keywords']['list_keywordsdf_article'] = list_keywordsdf_article
        dict_dfs['keywords']['df_unigram'] = df_unigram
        dict_dfs['keywords']['df_bigram'] = df_bigram
        dict_dfs['keywords']['df_trigram'] = df_trigram
        
        
    df_keywords_all = pd.concat(dict_dfs['keywords']['list_keywordsdf'])
    df_keywords_all.dropna(inplace=True)

    df_article_keywords_all = pd.concat(dict_dfs['keywords']['list_keywordsdf_article'])
    df_article_keywords_all.dropna(inplace=True)
    dict_dfs['keywords']['df_article_keywords_all'] = df_article_keywords_all

    df_keywords_unigram = df_keywords_all.groupby(by=['keyword_unigram'], as_index=False)['value_unigram'].sum()
    df_keywords_unigram.sort_values(by='value_unigram', ascending=True, inplace=True)

    df_keywords_bigram = df_keywords_all.groupby(by=['keyword_bigram'], as_index=False)['value_bigram'].sum()
    df_keywords_bigram.sort_values(by='value_bigram', ascending=True, inplace=True)

    df_keywords_trigram = df_keywords_all.groupby(by=['keyword_trigram'], as_index=False)['value_trigram'].sum()
    df_keywords_trigram.sort_values(by='value_trigram', ascending=True, inplace=True)

    df_keywords_all = pd.concat([df_keywords_unigram, df_keywords_bigram, df_keywords_trigram], axis=1)
    df_keywords_all = df_keywords_all.head(200)
    
    f = lambda e: np.round(e,3) if not pd.isna(e) else e
    df_keywords_all['value_unigram'] = df_keywords_all['value_unigram'].apply(f)
    df_keywords_all['value_bigram'] = df_keywords_all['value_bigram'].apply(f)
    df_keywords_all['value_trigram'] = df_keywords_all['value_trigram'].apply(f)
    dict_dfs['keywords']['df_keywords_all'] = df_keywords_all
    
    return dict_dfs


def show_keywords(st, dict_dfs):
    
    """"""
    
    # with st.container():
    #     _, col, _ = st.columns([0.1,8,0.1])
    #     with col:
    #         AgGrid(dict_dfs['keywords']['df_keywords_all'].head(50),
    #             data_return_mode='AS_INPUT', 
    #             # update_mode='MODEL_CHANGED', 
    #             fit_columns_on_grid_load=False,
    #             # theme='fresh',
    #             enable_enterprise_modules=False,
    #             height=510, 
    #             width='100%',
    #             reload_data=True)
            
    with st.container():
        _ , col1, col2, col3, _ = st.columns([0.01,3,3,3,0.01])
        with col1:
            df_show = dict_dfs['keywords']['df_unigram']
            df_show = df_show.rename(columns={"keyword_unigram":"Keyword Unigram",
                                              "value_unigram":"Relevance"})
            df_show['Relevance'] = df_show['Relevance'].apply(lambda x: np.round(float(x),3) if not pd.isna(x) else x)
            df_show = df_show.sort_values(by=['Relevance'], ascending=True)
            
            AgGrid(df_show.head(100),
                   data_return_mode='AS_INPUT', 
                   # update_mode='MODEL_CHANGED', 
                   fit_columns_on_grid_load=False,
                   # theme='fresh',
                   enable_enterprise_modules=False,
                   height=250, 
                   width='100%',
                   reload_data=True)
        with col2:
            df_show = dict_dfs['keywords']['df_bigram']
            df_show = df_show.rename(columns={"keyword_bigram":"Keyword Bigram",
                                              "value_bigram":"Relevance"})
            df_show['Relevance'] = df_show['Relevance'].apply(lambda x: np.round(float(x),3) if not pd.isna(x) else x)
            df_show = df_show.sort_values(by=['Relevance'], ascending=True)
            
            AgGrid(df_show.head(100),
                   data_return_mode='AS_INPUT', 
                   # update_mode='MODEL_CHANGED', 
                   fit_columns_on_grid_load=False,
                   # theme='fresh',
                   enable_enterprise_modules=False,
                   height=250, 
                   width='100%',
                   reload_data=True)
        with col3:
            df_show = dict_dfs['keywords']['df_trigram']
            df_show = df_show.rename(columns={"keyword_trigram":"Keyword Trigram",
                                              "value_trigram":"Relevance"})
            df_show['Relevance'] = df_show['Relevance'].apply(lambda x: np.round(float(x),3) if not pd.isna(x) else x)
            df_show = df_show.sort_values(by=['Relevance'], ascending=True)
            
            AgGrid(df_show.head(100),
                   data_return_mode='AS_INPUT', 
                   # update_mode='MODEL_CHANGED', 
                   fit_columns_on_grid_load=False,
                   # theme='fresh',
                   enable_enterprise_modules=False,
                   height=250, 
                   width='100%',
                   reload_data=True)


def agg_keys_node_data(grupo):
    """"""
    dictAgg = {}
    dictAgg['keyword'] = grupo['keyword'].iat[0]
    dictAgg['article_count'] = grupo['article_id'].shape[0]
    dictAgg['value_sum'] = grupo['value'].sum()
    dictAgg['value_mean'] = grupo['value'].mean()
    
    return pd.Series(dictAgg)


def show_keywords_graph(st, dict_dfs, df_article_keywords_all, input_folder_path, folder_graph='graphs', 
                        name_file="graph_keywords.html", cache_folder_name='summarticles_cache', top_keywords=10, buttons=False):
    
    """"""
    
    if 'graph' not in dict_dfs['keywords']:
    
        dict_dfs['keywords']['graph'] = {}
    
        tmining = text_mining()

        df_keyword_data = df_article_keywords_all.groupby(by=['keyword'], as_index=False).apply(agg_keys_node_data)
        df_keyword_data = df_keyword_data.sort_values(by=['article_count'], ascending=False).head(top_keywords)
        
        df_keyword_data['value_sum'] = df_keyword_data['value_sum'].apply(lambda e: np.round(e,2))
        df_keyword_data['value_mean'] = df_keyword_data['value_mean'].apply(lambda e: np.round(e,2))
        dict_dfs['keywords']['graph']['df_keyword_data'] = df_keyword_data
        
        # Selecting edges that contains top keywords
        filtro = (df_article_keywords_all.keyword.isin(df_keyword_data.keyword.tolist()))
        df_art_key_all = df_article_keywords_all.loc[(filtro)].copy()

        # Selecting nodes in the list of selected edges
        df_nodes = get_node_data(dict_dfs)

        filtro = (df_nodes['article_id'].isin(df_art_key_all['article_id'].tolist()))
        df_nodes = df_nodes.loc[(filtro)].copy()
        
        path_write_graph = os.path.join(input_folder_path, cache_folder_name)
            
        keywords_graph, path_graph, path_folder_graph = tmining.make_keywords_graph(edges_key_articles=df_art_key_all, node_data=df_nodes,
                                                        node_keywords_data=df_keyword_data, source_column="keyword", to_column="article_id", 
                                                        value_column="value", height="500px", width="100%", directed=False, notebook=False,
                                                        bgcolor="#ffffff", font_color=False, layout=None, heading="", path_graph=path_write_graph,
                                                        folder_graph=folder_graph, buttons=buttons, name_file=name_file)
        dict_dfs['keywords']['graph']['keywords_graph'] = keywords_graph
        dict_dfs['keywords']['graph']['path_graph'] = path_graph
        dict_dfs['keywords']['graph']['path_folder_graph'] = path_folder_graph
        
    col1, col2 = st.columns([2,1])
    with col1:
        show_graph(dict_dfs['keywords']['graph']['keywords_graph'],
                   dict_dfs['keywords']['graph']['path_graph'],
                   dict_dfs['keywords']['graph']['path_folder_graph'],
                   text_spinner='👁‍🗨 Similarity Graph: drawing...')
    with col2:
        df_show = dict_dfs['keywords']['graph']['df_keyword_data'].head(100)
        df_show = df_show.rename(columns={"keyword":"Keyword",
                                          "article_count":"Total Articles",
                                          "value_sum":"Total Relevance",
                                          "value_mean":"Mean Relevance"})
        AgGrid(df_show,
               data_return_mode='AS_INPUT', 
               # update_mode='MODEL_CHANGED', 
               fit_columns_on_grid_load=False,
               # theme='fresh',
               enable_enterprise_modules=False,
               height=510, 
               width='100%',
               reload_data=True)
        
    return dict_dfs
        
    
def show_graph(graph, path_graph, path_folder_graph, text_spinner='👁‍🗨 Keyword Graph: drawing...'):
    
    """"""
    
    path = os.path.dirname(os.getcwd())
    path_graph_depend = os.path.join(path,'notebooks','graph','pyvis','templates','dependencies')
    path_depend_dst = os.path.join(path_folder_graph,'dependencies')
    
    if not os.path.exists(path_depend_dst):
        os.mkdir(path_depend_dst)
    
    list_files = os.listdir(path_graph_depend)
    for file in list_files:
        f_source = os.path.join(path_graph_depend, file)
        f_destiny = os.path.join(path_depend_dst, file)
        shutil.copy(f_source, f_destiny)
        
    with st.spinner(text_spinner):
        
        GraphHtmlFile = open(path_graph, 'r', encoding='utf-8')
        GraphHtml = GraphHtmlFile.read()
        GraphHtmlFile.close()
        
        css_inject = open(os.path.join(path_graph_depend,'vis-network.css'), 'r', encoding='utf-8')
        css_inject_str = css_inject.read()
        css_inject.close()
        css_inject_str = f"""<style type="text/css">{css_inject_str}</style>"""
        
        script_inject = open(os.path.join(path_graph_depend,'vis-network.min.js'), 'r', encoding='utf-8')
        script_inject_str = script_inject.read()
        script_inject.close()
        script_inject_str = f"""<script type="text/javascript">{script_inject_str}</script>"""
        
        str_replace_css = """<link rel="stylesheet" href="./dependencies/vis.min.css" type="text/css" />"""
        str_replace_script = """<script type="text/javascript" src="./dependencies/vis-network.min.js"> </script>"""
        
        GraphHtml = GraphHtml.replace(str_replace_css, css_inject_str)
        GraphHtml = GraphHtml.replace(str_replace_script, script_inject_str)
        components.html(GraphHtml, height=600, width=None, scrolling=True)


def get_node_data(dict_dfs):
    
    """"""
    
    # Selecting head article data
    cols_head = ['title_head', 'doi_head', 'date_head',]
    head_data = dict_dfs['df_doc_head'].loc[:,cols_head].reset_index().copy()
    head_data['title_head'] = head_data['title_head'].apply(lambda e: str(e)[0:50] + "..." if len(str(e)) > 50 else str(e))

    # Selecting head article data
    cols_info = ['abstract','file']
    doc_info_data = dict_dfs['df_doc_info'].loc[:,cols_info].reset_index().copy()
    doc_info_data['file_name'] = doc_info_data['file'].apply(lambda e: os.path.split(e)[-1])
    doc_info_data['abstract_short'] = doc_info_data['abstract'].apply(lambda e: str(e)[0:20] + "..." if len(str(e)) > 20 else str(e))
    doc_info_data.drop(labels=['abstract'], axis=1, inplace=True)

    # Selecting authors information
    authors_data = dict_dfs['df_doc_authors'].reset_index()
    authors_data = authors_data.groupby(by=['article_id'], as_index=False)['full_name_author'].count()
    authors_data.rename(columns={'full_name_author':'author_count'}, inplace=True)

    # Selecting citations information
    citations_data = dict_dfs['df_doc_citations'].reset_index()
    citations_data = citations_data.groupby(by=['article_id'], as_index=False)['index_citation'].count()
    citations_data.rename(columns={'index_citation':'citation_count'}, inplace=True)

    nodes = dict_dfs['df_doc_info'].reset_index()['article_id'].tolist()
    df_nodes = pd.DataFrame(nodes, columns=['article_id'])

    df_nodes = df_nodes.merge(head_data, how='left', on='article_id')
    df_nodes = df_nodes.merge(doc_info_data, how='left', on='article_id')
    df_nodes = df_nodes.merge(authors_data, how='left', on='article_id')
    df_nodes = df_nodes.merge(citations_data, how='left', on='article_id')
    
    return df_nodes


def choose_filepath(st):
    
    """"""
    
    with st.spinner(' 📁 Choosing a file path...'):
        # Get articles files path
        st.warning("⚠️ Feature in development!")
        input_folder_path = input_path = ""
        input_folder_path = st.text_input('Selected folder:',filedialog.askdirectory(master=tk_root), key='txt_input_path')
    return input_folder_path


def get_last_executions(input_folder_path, cache_folder_name='summarticles_cache', folder_execs='summa_files', ext_file='summa'):
    
    path_execs = os.path.join(input_folder_path, cache_folder_name, folder_execs)
    list_files_execs = []
    if os.path.exists(path_execs):
        for f in os.listdir(path_execs):
            if str(f).endswith(ext_file):
                list_files_execs.append(f)
    return list_files_execs


def load_previous_execution(file, inputpath, cache_folder_name='summarticles_cache', folder_execs='summa_files', ext_file='summa'):
    """"""
    dict_dfs = None
    file_path = os.path.join(inputpath,cache_folder_name,folder_execs,file)
    if os.path.exists(file_path):
        dict_dfs = load(file_path)
    return dict_dfs


def write_previous_execution(object, input_path, cache_folder_name='summarticles_cache', folder_execs='summa_files', file_name="report_summarticles", ext_file='summa'):
    """"""
    path = os.path.join(input_path, cache_folder_name, folder_execs)
    if not os.path.exists(path):
        rpath = os.mkdir(path)
    
    dtnow = dt.now().strftime("%Y%m%d%H%M%S")
    file_name_full = f"{file_name}_{dtnow}.{ext_file}" 
    file_path = os.path.join(path, file_name_full)
    rdump = dump(object, file_path, compress=0)
    
    st.success("✅ Save execution complete!")
    

def make_select_box(list_files_execs, label, id):
    
    values = """"""
    for valor in list_files_execs:
        values += f"""<option value="{valor}">{valor}</option>"""
    
    html = f"""<label for="cars">{label}</label>
               <select name="{id}" id="{id}">
                   {values}
               </select>"""


def show_last_executions(st, list_files_execs, inputpath, cache_folder_name='summarticles_cache', 
                         folder_execs='summa_files', ext_file='summa'):
    """"""
    
    choice_exec = None
    dict_dfs = None
    st.session_state['previous_exec_check'] = False
    
    opt_default = "Select an option"
    opt_new_run = "No I don't want to use previously execution, continue with new execution"
    default_opt = [opt_default, opt_new_run]
    list_files_execs_final = default_opt + list_files_execs
    
    if len(list_files_execs_final) > len(default_opt):
        
        st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        st.info("❕ We found previously executions, do you want to recovery it?")
        
        choice_exec = st.selectbox("Please, choose one: ", list_files_execs_final, key="select_box_execs")
        # st.write('You selected:', choice_exec)
        if (choice_exec != opt_new_run) and (choice_exec != opt_default):
            try:
                dict_dfs = load_previous_execution(choice_exec, inputpath)
                st.session_state['previous_exec_check'] = True
                st.session_state['save_execution'] = False
                st.success("✅ Reload complete!")
            except Exception as error:
                st.error(f"❌ Error while reload previously execution: {str(error)}")
        elif choice_exec == opt_new_run:
            st.session_state['previous_exec_check'] = True
            st.session_state['save_execution'] = True
        else:
            st.session_state['previous_exec_check'] = False
            st.session_state['save_execution'] = True
            st.warning("⚠️ You need to choose an option!")
    else:
        st.session_state['previous_exec_check'] = True
        st.session_state['save_execution'] = True
        st.info("❕ There is no previously executions to recovery.")
        
    return dict_dfs
            

def checkey(dic, key):
    """"""
    return True if key in dic else False


def make_reset_button(st):
    """"""
    _, btn_reset, _ = st.columns([3,3,1])
    with btn_reset:
        clear_all = st.button("🔄 Reset execution!", key="reset_exec")
        if clear_all:
            st.session_state = {}
            st.experimental_memo.clear()
            st.experimental_singleton.clear()
            st.experimental_rerun()
            
            
def clustering_3d(dict_dfs, title_text="Group Articles", n_components=3, algorithm='UMAP'):
    
    """"""
    
    tmining = text_mining()
    
    if "clustering_data" not in dict_dfs:
        dict_dfs["clustering_data"] = {}
    
    if "clustering_data_3d" not in dict_dfs["clustering_data"]:
        dict_dfs["clustering_data"]["clustering_data_3d"] = {}
    
    if "documents_abs" not in dict_dfs["clustering_data"]["clustering_data_3d"]:
        dict_dfs["clustering_data"]["clustering_data_3d"]['documents_abs'] = dict_dfs['df_doc_info']['abstract_prep'].fillna(' ').tolist()
    # documents_body = dict_dfs['df_doc_info']['body_prep'].fillna(' ').tolist()
    
    if "df_tfidf_abstract_abs" not in dict_dfs["clustering_data"]["clustering_data_3d"]:
        dict_dfs["clustering_data"]["clustering_data_3d"]['df_tfidf_abstract_abs'] = tmining.get_df_tfidf(dict_dfs["clustering_data"]["clustering_data_3d"]['documents_abs'])
    # df_tfidf_abstract_body = tmining.get_df_tfidf(documents_body)
    
    # df_bow_abstract_abs = tmining.get_df_bow(documents_abs)
    # df_bow_abstract_body = tmining.get_df_bow(documents_body)
    if "cluster_label" not in dict_dfs["clustering_data"]:
        cluster_labels = tmining.make_clustering(dict_dfs["clustering_data_3d"]['df_tfidf_abstract_abs'].values,
                                                lim_sup=None, 
                                                init='k-means++', 
                                                n_init=10, 
                                                max_iter=30, 
                                                tol=1e-4, 
                                                random_state=0)
        dict_dfs["clustering_data"]["cluster_label"] = cluster_labels
    
    if "cluster_reduce_dim3" not in dict_dfs['clustering_data']:
        dictRedDim = tmining.reduce_dimensionality(dict_dfs['clustering_data']["clustering_data_3d"]['df_tfidf_abstract_abs'], 
                                                      y=dict_dfs['clustering_data']['cluster_label'],
                                                      n_components=n_components)
        dict_dfs['clustering_data']['cluster_reduce_dim3'] = dictRedDim

    dict_dfs['df_doc_info']['file_name'] = dict_dfs['df_doc_info']['file'].apply(lambda e: os.path.split(e)[-1])
        
    # Concatenate X and y arrays
    article_title = dict_dfs['df_doc_head']['title_head'].apply(lambda e: ''.join([str(e)[0:30],'...']) if len(str(e)) >= 30 else str(e)).values.reshape(dict_dfs['df_doc_info']['file'].shape[0],1)
    file_name = dict_dfs['df_doc_info']['file_name'].values.reshape(dict_dfs['df_doc_info']['file_name'].shape[0],1)

    arr_concat=np.concatenate((dict_dfs['clustering_data']['cluster_reduce_dim3'][algorithm],
                               dict_dfs["clustering_data"]["cluster_label"].reshape(dict_dfs["clustering_data"]["cluster_label"].shape[0],1),
                               file_name,
                               article_title), axis=1)

    # Create a Pandas dataframe using the above array
    df=pd.DataFrame(arr_concat, columns=['x', 'y', 'z', 'label', 'file_name', 'title_head'])
    
    dict_dfs['cluster_data_table'] = df.loc[:,['file_name', 'title_head','label']].copy()
    
    # Convert label data type from float to integer
    df['label'] = df['label'].astype(int)
    # Finally, sort the dataframe by label
    df.sort_values(by='label', axis=0, ascending=True, inplace=True)
    #--------------------------------------------------------------------------#
    # Create a 3D graph
    fig = px.scatter_3d(df, 
                        x='x',
                        y='y',
                        z='z',
                        color='label',
                        height=600,
                        width=750,
                        custom_data=['file_name','title_head','label','x','y','z'])

    # Update chart looks
    fig.update_layout(title_text=title_text,
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5))

    labels = ["Article File Name: %{customdata[0]}",
                "Article Title: %{customdata[1]}",
                "Grupo: %{customdata[2]}",
                "X: %{x}",
                "Y: %{y}",
                "Z: %{z}"]
                
    fig.update_traces(hovertemplate="<br>".join(labels))
    fig.update_coloraxes(showscale=False)

    # Update marker size
    # fig.update_traces(marker=dict(size=3, line=dict(color='black', width=0.1)))
    dict_dfs['clustering_data']["clustering_data_3d"]["fig"] = fig
        
    st.plotly_chart(dict_dfs['clustering_data']["clustering_data_3d"]["fig"], use_container_width=True)
    
    return dict_dfs
    
    
def clustering_2d(dict_dfs, title_text="Group Articles", n_components=2, algorithm='UMAP'):
    
    """"""
    tmining = text_mining()
    
    if "clustering_data" not in dict_dfs:
        dict_dfs["clustering_data"] = {}
    
    if "clustering_data_2d" not in dict_dfs["clustering_data"]:
        dict_dfs["clustering_data"]["clustering_data_2d"] = {}
    
    if "documents_abs" not in dict_dfs['clustering_data']["clustering_data_2d"]:
        dict_dfs['clustering_data']["clustering_data_2d"]['documents_abs'] = dict_dfs['df_doc_info']['abstract_prep'].fillna(' ').tolist()
    # documents_body = dict_dfs['df_doc_info']['body_prep'].fillna(' ').tolist()
    
    if "df_tfidf_abstract_abs" not in dict_dfs['clustering_data']["clustering_data_2d"]:
        dict_dfs['clustering_data']["clustering_data_2d"]['df_tfidf_abstract_abs'] = tmining.get_df_tfidf(dict_dfs['clustering_data']["clustering_data_2d"]['documents_abs'])
    # df_tfidf_abstract_body = tmining.get_df_tfidf(documents_body)
    
    # df_bow_abstract_abs = tmining.get_df_bow(documents_abs)
    # df_bow_abstract_body = tmining.get_df_bow(documents_body)
    
    if "cluster_label" not in dict_dfs["clustering_data"]:
        cluster_labels = tmining.make_clustering(dict_dfs['clustering_data']["clustering_data_2d"]['df_tfidf_abstract_abs'],
                                                lim_sup=None, 
                                                init='k-means++', 
                                                n_init=10, 
                                                max_iter=30, 
                                                tol=1e-4, 
                                                random_state=0)
        dict_dfs['clustering_data']['cluster_label'] = cluster_labels
    
    if "cluster_reduce_dim" not in dict_dfs['clustering_data']:
        dictRedDim, y = tmining.reduce_dimensionality(dict_dfs['clustering_data']["clustering_data_2d"]['df_tfidf_abstract_abs'], 
                                                    y=dict_dfs['clustering_data']['cluster_label'],
                                                    n_components=n_components
                                                    ), dict_dfs['clustering_data']['cluster_label']
        dict_dfs['clustering_data']['cluster_reduce_dim'] = dictRedDim

    dict_dfs['df_doc_info']['file_name'] = dict_dfs['df_doc_info']['file'].apply(lambda e: os.path.split(e)[-1])
        
    # Concatenate X and y arrays
    article_title = dict_dfs['df_doc_head']['title_head'].apply(lambda e: ''.join([str(e)[0:30],'...']) if len(str(e)) >= 30 else str(e)).values.reshape(dict_dfs['df_doc_info']['file'].shape[0],1)
    file_name = dict_dfs['df_doc_info']['file_name'].values.reshape(dict_dfs['df_doc_info']['file_name'].shape[0],1)

    arr_concat=np.concatenate((dict_dfs['clustering_data']['cluster_reduce_dim'][algorithm],
                               dict_dfs['clustering_data']['cluster_label'].reshape(dict_dfs['clustering_data']['cluster_label'].shape[0],1),
                               file_name,
                               article_title), axis=1)

    # Create a Pandas dataframe using the above array
    df=pd.DataFrame(arr_concat, columns=['x', 'y', 'label', 'file_name', 'title_head'])
    
    dict_dfs['clustering_data']['cluster_data_table'] = df.loc[:,['file_name', 'title_head','label']].copy()
    
    # Convert label data type from float to integer
    df['label'] = df['label'].astype(int)
    # Finally, sort the dataframe by label
    df.sort_values(by='label', axis=0, ascending=True, inplace=True)
    #--------------------------------------------------------------------------#

    # Create a 3D graph
    fig = px.scatter(df, 
                        x='x',
                        y='y',
                        color='label',
                        height=550,
                        width=750,
                        custom_data=['file_name','title_head','label','x','y'])

    # Update chart looks
    fig.update_layout(title_text=title_text,
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5))

    labels = ["Article File Name: %{customdata[0]}",
                "Article Title: %{customdata[1]}",
                "Group: %{customdata[2]}",
                "X: %{x}",
                "Y: %{y}"]
                
    fig.update_traces(hovertemplate="<br>".join(labels))
    fig.update_coloraxes(showscale=False)

    # Update marker size
    # fig.update_traces(marker=dict(size=3, line=dict(color='black', width=0.1)))
    dict_dfs['clustering_data']["clustering_data_2d"]["fig"] = fig
        
    st.plotly_chart(dict_dfs['clustering_data']["clustering_data_2d"]["fig"], use_container_width=True)
    
    return dict_dfs

###############################################################################################
# ---------------------------------------------------------------------------------------------
# ANALYTICS VISUALIZATION

def getColumnsWithData(df, return_percent=False, n_round=2):
    
    """"""
    
    list_col_with_data = []
    for col in df.columns.tolist():
        rows = df[col].shape
        n_null = df[col].isnull().sum()
        not_null_data_perc = (1-n_null/rows)
        if not_null_data_perc:
            if return_percent:
                list_col_with_data.append((col,np.round(not_null_data_perc, n_round)))
            list_col_with_data.append(col)
            
    return list_col_with_data     


def article_overview_information(st, dict_dfs):
    
    """"""
    
    import plotly.express as px
    
    df_doc_info = dict_dfs['df_doc_info'].loc[:,getColumnsWithData(dict_dfs['df_doc_info'])]
    df_doc_head = dict_dfs['df_doc_head'].loc[:,getColumnsWithData(dict_dfs['df_doc_head'])]
    df_doc_info_head = df_doc_info.join(df_doc_head, how='left')
    
    if 'date_head' in df_doc_info_head.columns.tolist():
        df_doc_info_head.date_head = df_doc_info_head.date_head.apply(lambda e: pd.to_datetime(e))
        df_doc_info_head['year'] = df_doc_info_head.date_head.apply(lambda e: e if pd.isna(e) else int(e.year))
        df_doc_info_head.year = df_doc_info_head.year.fillna('Null Value')
        fig = px.pie(df_doc_info_head, 
                    values='year', 
                    names='year',
                    title='Number of Articles by Year',
                    hover_data=['year'], 
                    labels={'values':'Percentage','year':'Year of Article'}, hole=.5)

        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)
        
    # _, col, _ = st.columns([0.1,3,0.1])
    # with col:
        # fig.show()
        
    return dict_dfs


def article_citation_information(st, dict_dfs):
    
    df_doc_authors_citations = dict_dfs['df_doc_authors_citations'].loc[:,getColumnsWithData(dict_dfs['df_doc_authors_citations'])]
    df_doc_authors = dict_dfs['df_doc_authors'].loc[:,getColumnsWithData(dict_dfs['df_doc_authors'])]
    
    import plotly.express as px
    
    with st.expander(" ❕ Information!"):
        body = """This section contains information about all of citations from load articles."""
        st.markdown(body, unsafe_allow_html=True)
    
    def agg_citations(grupo):
        dictR = {}
        dictR['citation_name'] = grupo['full_name_citation'].iat[0]
        dictR['citation_count'] = grupo.shape[0]
        dictR['number_authors'] = grupo['full_name_author'].nunique()
        dictR['number_countries'] = grupo['country_author'].nunique()
        return pd.Series(dictR)

    df_authors_citations = df_doc_authors_citations.loc[:,['full_name_citation']].reset_index()
    df_authors = df_doc_authors.loc[:,['full_name_author','country_author']].reset_index()

    df_authors_citations = df_authors_citations.drop_duplicates(subset=['article_id','full_name_citation'])
    df_authors = df_authors.drop_duplicates(subset=['article_id','full_name_author'])

    df_authors_and_citations = df_authors.merge(df_authors_citations, on='article_id')

    df_citation_plot = df_authors_and_citations.groupby(by=['full_name_citation'], as_index=False).apply(lambda g: agg_citations(g))

    df_citation_plot = df_citation_plot.sort_values(by=['citation_count'], ascending=False)
    df_citation_plot = df_citation_plot.head(100)

    import plotly.express as px

    fig = px.treemap(df_citation_plot, 
                     path=['full_name_citation'],
                     values='citation_count',
                     color='number_authors',
                     hover_data=['number_countries'],
                     color_continuous_scale='RdBu',
                     width=925,
                     height=400,
                     maxdepth=2,
                     labels={"number_authors":"Number of<br>Authors"})
    
    fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
    
    labels = ["Author Cited: %{id}",
               "Number of Citations: %{value}",
               "Number of Unique Authors Using This Citation: %{color}",
               "Number of Distinct Countries from Citation: %{customdata[0]}"]
                
    fig.update_traces(hovertemplate="<br>".join(labels))
    fig.update_traces(root_color="white")
    fig.update_traces(marker_line_width=4)
    
    st.plotly_chart(fig)
    
    with st.expander(" ❕ How can I read this chart?"):
        body = """Each rectangle represent an mentioned author, the size of space area mean the number of citations.<br>
                  The color represent the number of distinct authors that quote this author (cited).<br><br>
                  If you need more information, then you can pass cursor over each rectangle."""
        st.markdown(body, unsafe_allow_html=True)
    
    return dict_dfs
    

def plot_maps(st, dict_dfs):
    """Plot folium maps."""
    
    from streamlit_folium import st_folium, folium_static
    import folium
    from folium.plugins import HeatMap
    
    df_doc_authors = dict_dfs['df_doc_authors'].loc[:,getColumnsWithData(dict_dfs['df_doc_authors'])]
    
    path_geo = os.path.join(path,'data','external')
    country_latlong = pd.read_csv(os.path.join(path_geo,'countries_lat_long.csv'), sep=';', decimal='.')
    shapes_correct = pd.read_csv(os.path.join(path_geo,'shapes_correct.csv'), encoding='latin-1',sep=';', decimal='.')


    df_country_agg = df_doc_authors.groupby(by=['country_author'], as_index=False, dropna=True)['full_name_author'].count()
    dictCorrectShapes = {e[0]:e[1] for e in zip(shapes_correct.convert, shapes_correct.name)}
    df_country_agg.country_author = df_country_agg.country_author.apply(lambda e: dictCorrectShapes.get(e,e))
    df_country_agg = df_country_agg.groupby(by=['country_author'], as_index=False)['full_name_author'].sum()
    df_country_agg = df_country_agg.rename(columns={'country_author':'Country',
                                                'full_name_author':'count'})
    
    df_country_agg = df_country_agg.merge(country_latlong, how='left', on=['Country'])
    df_country_agg = df_country_agg.dropna()
    
    _, col1, _ = st.columns([0.75,3,0.1])
    with st.container():
        # with col1:
            # map = folium.Map(location=[25.552354,14.814465], zoom_start=1.5)
            # heat_points = []
            # for i,row in df_country_agg.iterrows():
            #     heat_points.append([row['Latitude'], row['Longitude'], row['count']])
            #     folium.Marker([row['Latitude'], row['Longitude']],
            #                 popup="<i>Number of Authors {0}<i>".format(row['count']),
            #                 tooltip=f"Number of Authors {row['count']}").add_to(map)
            #     st_point_map = st_folium(map)
        with col1:
            map = folium.Map(location=[25.552354,14.814465], zoom_start=1)
            heat_points = []
            for i,row in df_country_agg.iterrows():
                heat_points.append([row['Latitude'], row['Longitude'], row['count']])
            HeatMap(heat_points, radius=40, blur=20).add_to(map)
            st_heat_map = folium_static(map, width=600, height=400)
    
    return dict_dfs


def article_authors_information(st, dict_dfs):
    
    """"""
    
    import plotly.express as px
    
    with st.expander(" ❕ Information!"):
        body = """This section contains authors articles information."""
        st.markdown(body, unsafe_allow_html=True)
    
    # DataSets and Preparation
    df_doc_info = dict_dfs['df_doc_info'].loc[:,getColumnsWithData(dict_dfs['df_doc_info'])]
    df_doc_head = dict_dfs['df_doc_head'].loc[:,getColumnsWithData(dict_dfs['df_doc_head'])]
    df_doc_authors = dict_dfs['df_doc_authors'].loc[:,getColumnsWithData(dict_dfs['df_doc_authors'])]
    df_doc_info_head = df_doc_info.join(df_doc_head, how='left')
    
    list_delete_authors = ['A R T I C L E I N F O', np.nan, 'Null', 'NaN','nan', 'null', '', ' ']
    filter_delete_authors = ~(df_doc_authors.full_name_author.isin(list_delete_authors))
    df_doc_authors = df_doc_authors.loc[filter_delete_authors].copy()
    
    # ---------------------------------------------------------------------------
    # Top Authors
    top_authors = df_doc_authors.full_name_author.value_counts()
    df_top_authors = pd.DataFrame({'Full Name':top_authors.index,
                                'Number of Articles':top_authors.values.tolist()})
    
    top_authors = df_top_authors.nlargest(20,'Number of Articles')
    top_authors = top_authors.sort_values('Number of Articles',ascending=True)
    fig_authors = px.bar(top_authors,
                         title='Top 20 Number of Articles by Authors',
                         y='Full Name',
                         x='Number of Articles',
                         color='Number of Articles',
                         width=400, 
                         height=600,
                         text='Number of Articles')
    fig_authors.update(layout_coloraxis_showscale=False)
    # fig_authors.update_traces(showlegend=False)
    # fig_authors.update_traces(marker_showscale=False)
    fig_authors.update_xaxes(visible=False)
    fig_authors.update_layout(yaxis_title=None, xaxis_title=True)

    # ---------------------------------------------------------------------------
    # Authors by Location
    
    columns_select = ['country_author','settlement_author', 'institution_author']
    df_sun_agg = df_doc_authors.groupby(by=columns_select, as_index=False, dropna=False)['full_name_author'].count()
    df_sun_agg = df_sun_agg.fillna("Null Value")
    df_sun_agg.rename(columns={'country_author':'Author Country',
                               'settlement_author':'Author Settlement',
                               'institution_author':'Author Institution',
                               'full_name_author':'Number of Authors'},
                      inplace=True)
    
    df_sun_agg['Percentage'] = (df_sun_agg['Number of Authors']/df_sun_agg['Number of Authors'].sum())
    df_sun_agg['Percentage'] = df_sun_agg['Percentage'].apply(lambda e: int(100*np.round(float(e),2)))
    
    fig_authors_loc = px.sunburst(df_sun_agg,
                                  title='Number of Authors by Location',
                                  width=550, 
                                  height=600,
                                  path=['Author Country',
                                        'Author Settlement',
                                        'Author Institution'],
                                  hover_data=['Percentage'],
                                  values='Number of Authors')
                                  # values='Percentage')

    col0 , _,col1 = st.columns([0.25,2,3])
    with col0:
        st.plotly_chart(fig_authors)
    with col1:
        st.plotly_chart(fig_authors_loc)
        
        # fig.show()
        
    return dict_dfs
    
    

###############################################################################################
# ---------------------------------------------------------------------------------------------
# For process execution and streamlit app

if __name__ == '__main__':
    
    # ----------------------------------------------------------------------------
    # ATTENTION THIS CODE NEED SOME MODULARIZATION AND ORGANIZATION
    # BUT AT THIS TIME WE ONLY START APP DEVELOP
    
    # ----------------------------------------------------------------------------
    # State variables
    if 'input_path' not in st.session_state:
        st.session_state['input_path'] = ""
    if 'path_check' not in st.session_state:
        st.session_state['path_check'] = False
    if 'previous_exec_check' not in st.session_state:
        st.session_state['previous_exec_check'] = False
    if 'choice_exec' not in st.session_state:
        st.session_state['choice_exec'] = "Select one option"
    if 'dict_dfs' not in st.session_state:
        st.session_state['dict_dfs'] = None
    if 'save_execution' not in st.session_state:
        st.session_state['save_execution'] = True
    if 'param_n_sim' not in st.session_state:
        st.session_state['param_n_sim'] = 100
    if 'param_sim' not in st.session_state:
        st.session_state['param_sim'] = 0.0
    if 'rb_reddim' not in st.session_state:
        st.session_state['rb_reddim'] = 'PCA'
        
    # ----------------------------------------------------------------------------
    # Entrance of app
    make_head(st) # st.header("")

    # ----------------------------------------------------------------------------
    # TKINTER configs for get file path with filebox dialog
    tk_root = tk_configs()
    
    # ----------------------------------------------------------------------------
    # Sidebar Menu    
    make_sidebar(st)
    
    # ----------------------------------------------------------------------------
    # Reset execution
    if st.session_state['path_check']:
        make_reset_button(st)
        
    
    # ----------------------------------------------------------------------------
    # button getpath containers
    
    if not st.session_state['path_check']:
    
        btn_getfolder = make_getpath_button(st)
        
        # ----------------------------------------------------------------------------
        # Settings
        # This variables shall convert to checkbox 
        
        with st.expander("Settings and parameters"):
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                st.session_state['show_text_macro'] = show_text_macro = st.checkbox("Show Macro Text Information", True, key="chk_show_text_macro")
                st.session_state['show_wordcloud'] = show_wordcloud = st.checkbox("Show WordCloud Chart", True, key="chk_show_wordcloud")
            with c2:
                st.session_state['show_keywords_table'] = show_keywords_table = st.checkbox("Show KeyWords Information", True, key="chk_show_keywords_table")
                st.session_state['show_keywords_graph_cond'] = show_keywords_graph_cond = st.checkbox("Show KeyWords Chart", True, key="chk_show_keywords_graph_cond")
            with c3:
                st.session_state['show_similaritygraph'] = show_similaritygraph = st.checkbox("Show Similarity Graph", True, key="chk_show_similaritygraph")
                st.session_state['show_clustering'] = show_clustering = st.checkbox("Show Clustering Graph", True, key="chk_show_clustering")

        if btn_getfolder:
            
            input_folder_path = choose_filepath(st)
            
            if check_input_path(st, input_folder_path):
                st.session_state['input_path'] = input_folder_path
                st.session_state['path_check'] = True
            else:
                st.session_state['path_check'] = False
                
        if st.session_state['path_check']:
            make_reset_button(st)
    
    # ----------------------------------------------------------------------------
    # Check if there are another executions in the cache
    
    if st.session_state['path_check'] and not st.session_state['previous_exec_check']:
        
        input_path_success_message(st, st.session_state['input_path'])
    
        with st.spinner('⚙️ Check if there are another executions in the cache!'):
            
            list_last_executions = get_last_executions(st.session_state['input_path'],
                                                       cache_folder_name='summarticles_cache',
                                                       folder_execs='summa_files',
                                                       ext_file='summa')
            dict_dfs = show_last_executions(st, list_last_executions, st.session_state['input_path'],
                                            cache_folder_name='summarticles_cache',
                                            folder_execs='summa_files', ext_file='summa')
                   
            st.session_state['dict_dfs'] = dict_dfs
        
    # ----------------------------------------------------------------------------
    # button getpath containers
    
    if st.session_state['previous_exec_check']:
        
        input_folder_path = st.session_state['input_path']
        with st.spinner('💻⚙️ Process running... Leave the rest to us! Meanwhile maybe you can have some coffee. ☕'):

            # Run and display batch process, return a dictionary of dataframes with all data extract from articles
            # if not option and not checkey(dict_dfs, 'df_doc_info'):
            
            if not st.session_state['dict_dfs']:
                st.session_state['dict_dfs'] = run_batch_process(st, input_folder_path, n_workers=10, cache_folder_name='summarticles_cache',
                                                display_articles_data=False, save_xmltei=True)
            
            if not st.session_state['dict_dfs']:
                st.error("❓ There is no information to extract from articles in the specified path! Please, choose another file path.")
            else:
                if not len(st.session_state['dict_dfs'].keys()) or not st.session_state['dict_dfs']['df_doc_info'].shape[0]:
                    st.error("❓ There is no information to extract from articles in the specified path! Please, choose another file path.")
                else:
                    with st.container():
                        with st.spinner('🔢📊 Generating articles macro numbers...'):
                            st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
                            show_macro_numbers(st, st.session_state['dict_dfs'])
                        
                    with st.spinner('🛠️📄 Text prepatation...'):
                        st.session_state['dict_dfs'] = text_preparation(st, st.session_state['dict_dfs'], input_folder_path)
                    
                    # Container for wordcloud and text macro numbers
                    with st.container():
                        
                        st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                        st.markdown("""<h3 style="text-align:left;"><b>Text Macro Numbers</b></h3>""",unsafe_allow_html=True)
                    
                        if st.session_state['show_wordcloud']:
                            with st.spinner('📄➞☁️ Making WordCloud...'):
                                st.session_state['dict_dfs'] = show_word_cloud(st, st.session_state['dict_dfs'], input_folder_path, 
                                                                               cache_folder_name='summarticles_cache',
                                                                               folder_images='images', wc_image_name='wc_image.png')
                                
                        if st.session_state['show_text_macro']:
                            with st.spinner('🔢📊 Generating articles text numbers/stats...'):
                                st.markdown("""<hr style="height:0.1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                                st.session_state['dict_dfs'] = show_text_numbers(st, st.session_state['dict_dfs'])
                        
                        # with st.container():
                        #     with st.spinner('📄➞📄  Overview information...'):
                        #         st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                        #         st.markdown("""<h3 style="text-align:left;"><b>Overview Information</b></h3>""", unsafe_allow_html=True)

                        #         st.session_state['dict_dfs'] = article_overview_information(st, st.session_state['dict_dfs'])
                                
                        with st.container():
                            with st.spinner('📄➞📄  Authors Information...'):
                                st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                                st.markdown("""<h3 style="text-align:left;"><b>Authors Information</b></h3>""", unsafe_allow_html=True)
                                st.session_state['dict_dfs'] = article_authors_information(st, st.session_state['dict_dfs'])
                                st.session_state['dict_dfs'] = plot_maps(st, st.session_state['dict_dfs'])
                                
                        with st.container():
                            with st.spinner('📄➞📄  Citation Information...'):
                                st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                                st.markdown("""<h3 style="text-align:left;"><b>Citation Information</b></h3>""", unsafe_allow_html=True)
                                st.session_state['dict_dfs'] = article_citation_information(st, st.session_state['dict_dfs'])

                    with st.container():
                        if  st.session_state['show_keywords_table']:
                            
                            st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                            st.markdown("""<h3 style="text-align:left;"><b>KeyWords</b></h3>""",unsafe_allow_html=True)
                            
                            with st.expander("How it works?"):
                                st.write("""Below you can see all keywords extract from abstract text over all articles.
                                            The lower the score, the more relevant the keyword is.""")
                            
                            with st.spinner('📄➞🔤  Extracting KeyWords...'):
                                st.session_state['dict_dfs'] = generate_keywords(st, st.session_state['dict_dfs'])
                                
                            with st.spinner('📄➞🔤  Showing KeyWords...'):
                                show_keywords(st, st.session_state['dict_dfs'])
                                
                            with st.spinner('📄➞🔤  Showing KeyWords WordCloud...'):
                                st.session_state['dict_dfs'] = show_keyword_word_cloud(st, st.session_state['dict_dfs'], input_folder_path, 
                                                                                       cache_folder_name='summarticles_cache',
                                                                                       folder_images='images', wc_image_name='wc_image_keyword.png')
                                
                            if st.session_state['show_keywords_graph_cond']:
                                with st.spinner('📄➞📄  Making KeyWord Graph...'):
                                    with st.expander(" ❕ How can I read this graph?"):
                                        body = """In this graph Summarticles plot Keywords (from Abstract) and Articles.<br>
                                                  The blue dots are articles, and you can pass cursor over this points and get more information.<br>
                                                  The colors boxes are keywords, and you can pass cursor over this boxes and get more information.<br><br>
                                                  The edges conecting blue dots with keyword boxes represent relevance of the keyword for the abstract article
                                                  and the edge thickness represent the level/value of keyword relevance.<br><br>
                                                  For now, you only can see the top 10 keywords.<br><br>
                                                  You can interact with graph, moving blue dots, boxes and edges.<br>
                                                  You also can zoom-in and zoom-out the graph."""
                                        st.markdown(body, unsafe_allow_html=True)
                                    st.session_state['dict_dfs'] = show_keywords_graph(st, st.session_state['dict_dfs'], 
                                                                   st.session_state['dict_dfs']['keywords']['df_article_keywords_all'],
                                                                   input_folder_path)
                                    
                    with st.container():
                        if st.session_state['show_similaritygraph']:
                            with st.spinner('📄➞📄  Making Similarity Graph...'):
                                
                                st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                                st.markdown("""<h3 style="text-align:left;"><b>Similarity Graph: this graph shows you similarity across articles.</b></h3>""", unsafe_allow_html=True)

                                with st.expander("How it works?"):
                                    st.write("""Using abstract text from all articles, Summarticles create a TF-IDF matrix and 
                                                compute cossine similarity cross all articles. 
                                                In Summarticles cossine similarity range from 0 to 1, closer than 1 means high similarity
                                                0 is the opposite.""")
                                    
                                def del_similarity_graph():
                                    del st.session_state['dict_dfs']['similarity_graph']
                                    
                                st.session_state['dict_dfs'], rel_size, sim_size = similarity_graph(st, st.session_state['dict_dfs'],
                                                                                                    input_folder_path,
                                                                                                    percentil="75%",
                                                                                                    n_sim=st.session_state['param_n_sim'],
                                                                                                    cache_folder_name='summarticles_cache')
                                # _, sld1, _ , sld2, _ = st.columns([1,3,0.5,3,1])
                                # with sld1:
                                #     st.session_state['param_n_sim'] = st.slider(label='Select number of relations:', min_value=0, max_value=rel_size, value=st.session_state['param_n_sim'], on_change=del_similarity_graph)
                                # with sld2:
                                #     st.session_state['param_sim'] = st.slider(label='Select min similarity score:', min_value=0.0, max_value=float(sim_size), value=st.session_state['param_sim'], on_change=del_similarity_graph)
                                    
                    with st.container():
                        if st.session_state['show_clustering']:
                            with st.spinner('📄➞📄  Making Clustering...'):
                                
                                st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                                st.markdown("""<h3 style="text-align:left;"><b>Clustering Articles</b></h3>""", unsafe_allow_html=True)

                                with st.expander("How it works?"):
                                    st.write("""Using abstract text from all articles, Summarticles create a TF-IDF matrix and clustering the articles.
                                                For group the data, Summarticles use K-Means and the number of the groups is defined
                                                by committee vote using Silhouette-Score, Dabies-Bouldin Index and Calinsk-Harabaz Index.""")
                                
                                c1, c2 = st.columns([0.5,1])
                                
                                with st.container():
                                    with c2:
                                        with st.container():
                                            st.session_state['rb_reddim'] = st.radio("Select Data Projection Algorithm:",
                                                                                    ('PCA', 'UMAP', 'MDS', 'TSNE'),
                                                                                    horizontal=True,
                                                                                    help="Choose one of these algorithms for groups data projection!")
                                        st.session_state['dict_dfs'] = clustering_2d(st.session_state['dict_dfs'],
                                                                                     title_text="Group Articles 2D",
                                                                                     n_components=2,
                                                                                     algorithm=st.session_state['rb_reddim']) # UMAP, TSNE, PCA, MDS
                                    with c1:
                                        df_show = st.session_state['dict_dfs']['clustering_data']['cluster_data_table']
                                        df_show = df_show.rename(columns={"file_name":"File Name",
                                                                          "title_head":"Title",
                                                                          "label":"Group"})
                                        
                                        AgGrid(df_show,
                                               data_return_mode='AS_INPUT', 
                                               # update_mode='MODEL_CHANGED', 
                                               fit_columns_on_grid_load=False,
                                               # theme='fresh',
                                               enable_enterprise_modules=False,
                                               height=550, 
                                               width='100%',
                                               reload_data=True)
                                
                                with st.container():
                                    st.session_state['dict_dfs'] = clustering_3d(st.session_state['dict_dfs'],
                                                                                 title_text="Group Articles 3D",
                                                                                 n_components=3,
                                                                                 algorithm=st.session_state['rb_reddim']) # UMAP, TSNE, PCA, MDS
                     
                    with st.container():
                        with st.spinner('📄➞📄  Part-of-speech and Named Entities...'):
                            
                            st.markdown("""<hr style="height:1px;border:none;color:#F1F1F1;background-color:#F1F1F1;" /> """, unsafe_allow_html=True)
                            st.markdown("""<h3 style="text-align:left;"><b>Part-of-speech and Named Entities</b></h3>""", unsafe_allow_html=True)

                            part_of_speech(st, st.session_state['dict_dfs'])              
                                
    if st.session_state['dict_dfs'] and st.session_state['save_execution']:
                             
        write_previous_execution(st.session_state['dict_dfs'], 
                                 st.session_state['input_path'],
                                 file_name="report_summarticles",
                                 ext_file='summa',
                                 cache_folder_name='summarticles_cache', 
                                 folder_execs='summa_files')
        
        st.session_state['save_execution'] = False
        
