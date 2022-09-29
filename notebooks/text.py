import nltk
import re
import os
import sys
import pandas as pd
import numpy as np
#import spacy
#import corenlp
#import textblob
#import gensim
#import transformers

from wordcloud import WordCloud
from matplotlib import pyplot as plt

import random

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN, OPTICS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from graph.pyvis.network import Network

sys.path.insert(0,os.path.dirname(os.getcwd()))
sys.path.insert(0,os.path.join(os.getcwd(),'grobid'))
sys.path.insert(0,os.getcwd())

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

class text_prep(object):
    
    """"""
    
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words('english')
        pass
    
    
    def text_tokenize(self, text, language='english', preserve_line=False):
        
        """"""
        
        return nltk.tokenize.word_tokenize(text, language=language, preserve_line=preserve_line)


    def clean_text_regex(self, words_list, regex="[^a-zA-Z]+", replace='', min_word_len=1):
        
        """Testado em https://regex101.com/"""
        
        new_words = []
        for word in words_list:
            word = re.sub(regex, replace, word)
            if len(word) > min_word_len:
                new_words.append(word)
        return new_words


    def remove_stopwords(self, words_list, stopwords_list):
        
        """"""
        
        new_words = []
        for word in words_list:
            if word not in stopwords_list:
                new_words.append(word)
        return new_words


    def lemmatizer(self, words_list):
        
        """"""
        
        obj_lemmatizer = nltk.stem.WordNetLemmatizer()
        words_lemma = []
        for word in words_list:
            words_lemma.append(obj_lemmatizer.lemmatize(word,pos=nltk.corpus.wordnet.VERB))
        return words_lemma


    def stem_text(self, words_list):
        
        """"""
        
        p_stem = nltk.stem.PorterStemmer()
        words_stem = []
        for word in words_list:
            words_stem.append(p_stem.stem(word))
        return words_stem


    def preparation_text(self, text, clean_text=True, stopwords_remove=True, exec_lemmatizer=True, exec_stem=False, text_lower=False, stopwords_list=[], language='english',
                preserve_line=False, regex_chars_clean="[^a-zA-Z]+", replace_chars_clean='', min_word_len=1):
        
        """Text preparation."""
        
        text_preparation = self.text_tokenize(text, language=language, preserve_line=preserve_line)
        if clean_text:
            text_preparation = self.clean_text_regex(words_list=text_preparation,
                                                    regex=regex_chars_clean,
                                                    replace=replace_chars_clean,
                                                    min_word_len=min_word_len)
        if stopwords_remove:
            text_preparation = self.remove_stopwords(words_list=text_preparation,
                                                    stopwords_list=stopwords_list)
        if exec_lemmatizer:
            text_preparation = self.lemmatizer(words_list=text_preparation)
            
        if exec_stem:
            text_preparation = self.stem_text(words_list=text_preparation)
            
        text_preparation = ' '.join(text_preparation)
        if text_lower:
            text_preparation = text_preparation.lower()
            
        return text_preparation


    def text_preparation_column(self, colum_df):
        
        """"""
        
        f_prep_text = lambda text_data: self.preparation_text(text=text_data, clean_text=True, stopwords_remove=True, exec_lemmatizer=True, 
                                                  exec_stem=False,stopwords_list=self.stopwords, language='english', 
                                                  preserve_line=False, regex_chars_clean="[^a-zA-Z]+", replace_chars_clean='',
                                                  min_word_len=1, text_lower=True)
        
        colum_df = colum_df.apply(lambda e: e if pd.isna(e) else f_prep_text(e))
        
        return colum_df


class text_mining(object):
    
    def __init__(self):
        """"""
        pass


    def get_df_bow(self, documents_list, encoding="utf-8", stop_words="english", strip_accents="ascii", lowercase=True, 
                   preprocessor=None, tokenizer=None, token_pattern=r"""(?u)\b\w\w+\b""", ngram_range=(1,2), analyzer="word",
                   max_df=1.0, min_df=2, max_features=None, vocabulary= None, binary=False, dtype=np.int64):
        """"""
        obj_bow = CountVectorizer(encoding=encoding,
                                 stop_words=stop_words,
                                 strip_accents=strip_accents,
                                 lowercase=lowercase, 
                                 preprocessor=preprocessor,
                                 tokenizer=tokenizer,
                                 token_pattern=token_pattern,
                                 ngram_range=ngram_range, # Unigram and bigram
                                 analyzer=analyzer,
                                 max_df=max_df,
                                 min_df=min_df, # May have at least 2 frequency
                                 max_features=max_features, 
                                 vocabulary=vocabulary, 
                                 binary=binary, 
                                 dtype=dtype)
        
        obj_bow = obj_bow.fit(raw_documents=documents_list)
        bow_matrix = obj_bow.transform(documents_list)
        bow_matrix = bow_matrix.todense()
        df_bow = pd.DataFrame(bow_matrix, columns=obj_bow.get_feature_names())
        
        return df_bow


    def get_df_tfidf(self, documents_list, encoding="utf-8", stop_words="english", strip_accents="ascii", lowercase=True, 
                     preprocessor=None, tokenizer=None, token_pattern=r"""(?u)\b\w\w+\b""", ngram_range=(1,2), analyzer="word",
                     max_df=1.0, min_df=2, max_features=None, vocabulary= None, binary=False, dtype=np.float64, norm='l2', 
                     use_idf=True, smooth_idf=True, sublinear_tf=False):
        """"""
        obj_tfidf = TfidfVectorizer(encoding=encoding,
                                    stop_words=stop_words,
                                    strip_accents=strip_accents,
                                    lowercase=lowercase, 
                                    preprocessor=preprocessor,
                                    tokenizer=tokenizer,
                                    token_pattern=token_pattern,
                                    ngram_range=ngram_range, # Unigram and bigram
                                    analyzer=analyzer,
                                    max_df=max_df,
                                    min_df=min_df, # May have at least 2 frequency
                                    max_features=max_features, 
                                    vocabulary= vocabulary, 
                                    binary=binary, 
                                    dtype=dtype, 
                                    norm=norm, 
                                    use_idf=use_idf, 
                                    smooth_idf=smooth_idf, 
                                    sublinear_tf=sublinear_tf)

        obj_tfidf = obj_tfidf.fit(raw_documents=documents_list)
        tfidf_matrix = obj_tfidf.transform(documents_list)
        tfidf_matrix = tfidf_matrix.todense()
        df_tfidf = pd.DataFrame(tfidf_matrix, columns=obj_tfidf.get_feature_names())
        
        return df_tfidf
    
    
    def get_cossine_similarity_matrix(self, documents_vector_matrix, index_docs):
        
        """"""
        
        def isfloat(num):
            try:
                float(num)
                return True
            except ValueError:
                return False
        
        index_docs = [str(int(e)) if isfloat(e) else str(e) for e in index_docs]
        
        docs_sim = cosine_similarity(documents_vector_matrix, documents_vector_matrix)
        df_docs_sim = pd.DataFrame(docs_sim, columns=index_docs, index=index_docs)
        return df_docs_sim


    def filter_sim_matrix(self, matrix, columns, n_sim=200, percentil="50%", value_min=0, value_max=0.99, colum_value='value'):
        
        """"""
        
        list_elements = [] #
        for colum in columns: #
            list_elements += matrix[colum].tolist() #
        sim_describe = pd.Series(list_elements).describe(percentiles=np.arange(0, 1, 0.001)) #
        del list_elements #
        
        filter_matrix = sim_describe[percentil] #
        
        
        list_filter = []
        for num_line, irow in enumerate(matrix.iterrows()):
            i, row = irow
            for num_col, j in enumerate(row.index):
                if num_line >= num_col:
                    continue
                value = matrix.loc[i,j]
                logic_filter = value>=value_min and value<=value_max and value>=filter_matrix #
                if not pd.isna(value) and logic_filter:
                    dictCell = {"doc_a":i,"doc_b":j,colum_value:matrix.loc[i,j]}
                    list_filter.append(dictCell)
        df_maxtrix_filter = pd.DataFrame(list_filter)
        del list_filter
        
        df_maxtrix_filter = df_maxtrix_filter.nlargest(n_sim, colum_value)
        
        return df_maxtrix_filter
    
    def get_aleatory_color(self):
        
        '''Returns color in hex format'''
    
        red_int = random.randint(0,255)
        green_int = random.randint(0,255)
        blue_int = random.randint(0,255)
    
        return '#{:02X}{:02X}{:02X}'.format(red_int, green_int, blue_int)


    def make_sim_graph(self, matrix, node_data, source_column="doc_a", to_column="doc_b",
                       value_column="value", height="1000px",width="1000px", directed=False,
                       notebook=False, bgcolor="#ffffff", font_color=False, layout=None, heading="",
                       path_graph="./", folder_graph="graphs", name_file="graph.html", buttons=True):
        
        """"""
        
        graph = Network(height=height,
                        width=width,
                        directed=directed,
                        notebook=notebook,
                        bgcolor=bgcolor,
                        font_color=font_color,
                        layout=layout,
                        heading=heading)

        for i, row in node_data.iterrows():
            
            article_id = str(row['article_id'])
            article_title = str(row['title_head'])
            article_abstract_short = str(row['abstract_short'])
            article_date = str(row['date_head'])
            article_number_authors = str(row['author_count'])
            article_number_citations = str(row['citation_count'])
            article_doi = str(row['doi_head'])
            article_file_name = str(row['file_name'])
            article_file_path = str(row['file'])
            
            title_html = f"""ARTICLE INFORMATION:
                            TITLE:{article_title}
                            DATE:{article_date}
                            NUMBER AUTHORS:{article_number_authors}
                            NUMBER CITATIONS:{article_number_citations}
                            DOI:{article_doi}
                            FILE NAME:{article_file_name}"""
            
            graph.add_node(n_id=str(article_id), 
                           label=f"Node ID: {str(article_id)}\n{article_title}", 
                           borderWidth=1, 
                           borderWidthSelected=2, 
                           #brokenImage="url", 
                           #group="a", 
                           #hidden=False, 
                           #image="url", 
                           #labelHighlightBold=True, 
                           #level=1, 
                           #mass=1, 
                           #physics=True,
                           shape="dot", # image, circularImage, diamond, dot, star, triangle, triangleDown, square and icon
                           size=1, 
                           title=title_html,  
                           #x=0.5, 
                           #y=1.0)
                           value=1)
            
        for i,row in matrix.iterrows():
            graph.add_edge(source=str(row[source_column]),
                           to=str(row[to_column]),
                           value=np.round(row[value_column],3),
                           title="Similarity: " + str(np.round(row[value_column],3)))
                           #width=row['value'],
                           #arrowStrikethrough=False,
                           #physics=False,
                           #hidden=False)
        
        graph.force_atlas_2based(gravity=-50,
                                 central_gravity=0.01,
                                 spring_length=360,
                                 spring_strength=0.08,
                                 damping=0.4,
                                 overlap=0)
        
        if buttons:
            graph.show_buttons(filter_=['physics'])
            
        path_graph_final = os.path.join(path_graph, folder_graph)
        if not os.path.exists(path_graph_final):
            os.mkdir(path_graph_final)
        path_graph_final = os.path.join(path_graph_final, name_file)
        graph.save_graph(path_graph_final)
        # graph.show(name_file)
        
        return graph, path_graph_final, os.path.join(path_graph, folder_graph)
    
    
    def make_keywords_graph(self, edges_key_articles, node_data, node_keywords_data,
                            source_column="keyword",to_column="article_id", value_column="value",
                            height="1000px",width="1000px", directed=False, notebook=False,
                            bgcolor="#ffffff", font_color=False, layout=None, heading="", buttons=True,
                            path_graph="./", folder_graph="graphs", name_file="graph_keyword.html"):
        
        """"""
        
        graph = Network(height=height,
                        width=width,
                        directed=directed,
                        notebook=notebook,
                        bgcolor=bgcolor,
                        font_color=font_color,
                        layout=layout,
                        heading=heading)

        for i, row in node_data.iterrows():
            
            article_id = str(row['article_id'])
            article_title = str(row['title_head'])
            article_abstract_short = str(row['abstract_short'])
            article_date = str(row['date_head'])
            article_number_authors = str(row['author_count'])
            article_number_citations = str(row['citation_count'])
            article_doi = str(row['doi_head'])
            article_file_name = str(row['file_name'])
            article_file_path = str(row['file'])
            
            title_html = f"""Article Title:{article_title}
                             Article Date:{article_date}
                             Article Number Authors:{article_number_authors}
                             Article Number Citations:{article_number_citations}
                             Article DOI:{article_doi}
                             Article File Name:{article_file_name}"""
            
            graph.add_node(n_id=article_id, 
                        label=f"Node ID: {str(article_id)[0:4]}", 
                        borderWidth=1, 
                        borderWidthSelected=2, 
                        #brokenImage="url", 
                        #group="a", 
                        #hidden=False, 
                        #image="url", 
                        #labelHighlightBold=True, 
                        #level=1, 
                        #mass=1, 
                        #physics=True,
                        shape="dot", # image, circularImage, diamond, dot, star, triangle, triangleDown, square and icon
                        size=1, 
                        title=title_html,  
                        #x=0.5, 
                        #y=1.0)
                        value=1)
            
        for i, row in node_keywords_data.iterrows():
            
            keyword_id = str(row['keyword'])
            article_count = row['article_count']
            value_sum = row['value_sum']
            value_mean = row['value_mean']
            
            title_html = f"""KeyWord: {keyword_id}
                            Article Count: {article_count}
                            Value Sum: {value_sum}
                            Value Mean: {value_mean}
                        """
            
            graph.add_node(n_id=keyword_id, 
                           label=keyword_id, 
                           borderWidth=2, 
                           borderWidthSelected=4,
                           color=self.get_aleatory_color(),
                           #brokenImage="url", 
                           #group="a", 
                           #hidden=False, 
                           #image="url", 
                           #labelHighlightBold=True, 
                           #level=1, 
                           #mass=1, 
                           #physics=True,
                           shape="box", # image, circularImage, diamond, dot, star, triangle, triangleDown, square and icon, box, text
                           size=article_count, 
                           title=title_html,  
                           #x=0.5, 
                           #y=1.0)
                           value=article_count*1000)
        
        for i, row in edges_key_articles.iterrows():
            
            graph.add_edge(source=str(row[source_column]),
                           to=str(row[to_column]),
                           value=np.round(1/row[value_column],3),
                           title="Relevance: " + str(np.round(row[value_column],3)))
                           #width=row['value'],
                           #arrowStrikethrough=False,
                           #physics=False,
                           #hidden=False)
        
        graph.force_atlas_2based(gravity=-50,
                                 central_gravity=0.01,
                                 spring_length=360,
                                 spring_strength=0.08,
                                 damping=0.4,
                                 overlap=0)
        
        if buttons:
            graph.show_buttons(filter_=['physics'])
            
        path_graph_final = os.path.join(path_graph, folder_graph)
        if not os.path.exists(path_graph_final):
            os.mkdir(path_graph_final)
        path_graph_final = os.path.join(path_graph_final, name_file)
        graph.save_graph(path_graph_final)
        # graph.show(name_file)
        
        return graph, path_graph_final, os.path.join(path_graph, folder_graph)
    
    def make_clustering(self,
                        X,
                        metric_func=np.mean,
                        lim_sup=None, 
                        init='k-means++', 
                        n_init=10, 
                        max_iter=30, 
                        tol=1e-4, 
                        random_state=0):
        
        lim_sup = range(2,min(int(X.shape[0]**0.5),3)) if lim_sup == None else lim_sup
        list_result = []

        print(lim_sup)
        for c in lim_sup:
            
            objGroup = KMeans(n_clusters=c,
                            init=init,
                            n_init=n_init,
                            max_iter=max_iter,
                            tol=tol, 
                            random_state=random_state)
            
            objGroup = objGroup.fit(X)
            
            inertia = objGroup.inertia_
            s = silhouette_score(X, objGroup.labels_, metric='euclidean', random_state=random_state)
            ch = calinski_harabasz_score(X, objGroup.labels_)
            db = davies_bouldin_score(X, objGroup.labels_)
            
            list_result.append({'cluster':c,
                                'inertia':inertia,
                                'silhouette':s,
                                'calinski_harabasz':ch,
                                'davies_bouldin':db})

        df_metrics = pd.DataFrame(list_result)

        ss = df_metrics.nlargest(1,'silhouette')['cluster'].iat[0]
        ch = df_metrics.nlargest(1,'calinski_harabasz')['cluster'].iat[0]
        db = df_metrics.nsmallest(1,'davies_bouldin')['cluster'].iat[0]

        final_cluster_value = int(metric_func([ss, ch, db]))

        objGroup = KMeans(n_clusters=final_cluster_value,
                        init='k-means++',
                        n_init=10,
                        max_iter=30,
                        tol=1e-4, 
                        random_state=0)

        objGroup = objGroup.fit(X)
        
        return objGroup.labels_
    
    def reduce_dimensionality(self, X, y=None, n_components=3):
        
        """This function get the X data and reduce dimensionality to n_components.
        
        algorithm: UMAP, TSNE, PCA, MDS
        
        """

        dictReduceDim = {}
    
        # TSNE
        objTSNE = TSNE(n_components=n_components, init='random')
        X_reduce = objTSNE.fit_transform(X)
        dictReduceDim['TSNE'] = X_reduce
        
        # PCA
        objPCA = PCA(n_components=n_components, random_state =0)
        X_reduce = objPCA.fit_transform(X)
        dictReduceDim['PCA'] = X_reduce
        
        # MDS
        objMDS = MDS(n_components=n_components)
        X_reduce = objMDS.fit_transform(X)
        dictReduceDim['MDS'] = X_reduce
        
        # Configure UMAP hyperparameters
        reducer = UMAP(n_neighbors=100, # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
                    n_components=n_components, # default 2, The dimension of the space to embed into.
                    metric='euclidean', # default 'euclidean', The metric to use to compute distances in high dimensional space.
                    n_epochs=1000, # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings. 
                    learning_rate=1.0, # default 1.0, The initial learning rate for the embedding optimization.
                    init='spectral', # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
                    min_dist=0.1, # default 0.1, The effective minimum distance between embedded points.
                    spread=1.0, # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
                    low_memory=False, # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
                    set_op_mix_ratio=1.0, # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
                    local_connectivity=1, # default 1, The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local level.
                    repulsion_strength=1.0, # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
                    negative_sample_rate=5, # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
                    transform_queue_size=4.0, # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
                    a=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
                    b=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
                    random_state=42, # default: None, If int, random_state is the seed used by the random number generator;
                    metric_kwds=None, # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
                    angular_rp_forest=False, # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
                    target_n_neighbors=-1, # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
                    #target_metric='categorical', # default 'categorical', The metric used to measure distance for a target array is using supervised dimension reduction. By default this is 'categorical' which will measure distance in terms of whether categories match or are different. 
                    #target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target metric when performing supervised dimension reduction. If None then no arguments are passed on.
                    #target_weight=0.5, # default 0.5, weighting factor between data topology and target topology.
                    transform_seed=42, # default 42, Random seed used for the stochastic aspects of the transform operation.
                    verbose=False, # default False, Controls verbosity of logging.
                    unique=False # default False, Controls if the rows of your data should be uniqued before being embedded. 
                    )
        X_reduce = reducer.fit_transform(X, y)
        dictReduceDim['UMAP'] = X_reduce
        
        return dictReduceDim


class text_viz(object):
    
    """"""
    
    def __init__(self):
        pass

    def word_cloud(self, documents, path_image=None, show_wc=True, width=1000, height=200, collocations=True, background_color='white'):
        
        """Create and plot a wordcloud from documents list. Return: objWC, ax"""
        
        fig, ax = plt.subplots(1,1)
        objWC = WordCloud(collocations=collocations, background_color=background_color, width=width, height=height)
        text = ' '.join([' ' if pd.isna(t) else t for t in documents])
        objWC = objWC.generate_from_text(text)
        
        if path_image!=None:
            objWC.to_file(path_image)
            
        ax.imshow(objWC)
        ax.axis("off")
        
        if show_wc:
            ax.show()
            
        return objWC, ax, fig
    
    def keyword_word_cloud(self, frequencies, path_image=None, show_wc=True, width=1000, height=200, collocations=True, background_color='white'):
        
        """Create and plot a wordcloud from documents list. Return: objWC, ax"""
        
        fig, ax = plt.subplots(1,1)
        objWC = WordCloud(collocations=collocations,
                          background_color=background_color,
                          width=width,
                          height=height)
        objWC = objWC.generate_from_frequencies(frequencies)
        
        if path_image!=None:
            objWC.to_file(path_image)
            
        ax.imshow(objWC)
        ax.axis("off")
        
        if show_wc:
            ax.show()
            
        return objWC, ax, fig
