import os
import sys
import re
import numpy as np
import pandas as pd
from grobid import grobid_client
import grobid_tei_xml
import datetime
import json

sys.path.insert(0,os.path.dirname(os.getcwd()))
sys.path.insert(0,os.path.join(os.getcwd(),'grobid'))
sys.path.insert(0,os.getcwd())

path = os.path.dirname(os.getcwd())
path_input = os.path.join(path,'artifacts','test_article')
path_output = os.path.join(path,'output','xml')
path_article = os.path.join(path,'artifacts','test_article','b617684b.pdf')



class grobid_cli(object):
    
    """"""
    
    def __init__(self, config_path="./grobid/config.json"):
        
        """"""
        
        self.client = grobid_client.GrobidClient(config_path=config_path)
    
    
    def process_pdf(self, pdf_file, service="processFulltextDocument", generateIDs=True, include_raw_citations=True,
                    include_raw_affiliations=True, consolidate_header=False, consolidate_citations=False, tei_coordinates=False,
                    segment_sentences=True):
        
        """"""
        
        pdf_file, status, xml = self.client.process_pdf(service=service,
                                                        pdf_file=pdf_file,
                                                        generateIDs=generateIDs,
                                                        consolidate_header=consolidate_header, # Usa informações externas para consolidar informações de cabeçalho
                                                        consolidate_citations=consolidate_citations, # Usa informações externas para consolidar informações de bibliografia
                                                        include_raw_citations=include_raw_citations, # Citações
                                                        include_raw_affiliations=include_raw_affiliations, # Afiliações
                                                        tei_coordinates=tei_coordinates, # Gera coordenadas para gerar visualização de marcações no PDF
                                                        segment_sentences=segment_sentences) # Usa um motor externo para segmentar as sentenças
                
        return pdf_file, status, xml
    
    
    def process_pdfs(self, input_path, n_workers, check_cache=True, cache_folder_name='summarticles_cache', service="processFulltextDocument", generateIDs=True, include_raw_citations=True,
                    include_raw_affiliations=True, consolidate_header=False, consolidate_citations=False, tei_coordinates=False,
                    segment_sentences=True,verbose=True):
        
        """"""
        
        list_results = self.client.batch_process(service=service,
                                                 input_path=input_path,
                                                 n_workers=n_workers,
                                                 generateIDs=generateIDs,
                                                 consolidate_header=consolidate_header, # Usa informações externas para consolidar informações de cabeçalho
                                                 consolidate_citations=consolidate_citations, # Usa informações externas para consolidar informações de bibliografia
                                                 include_raw_citations=include_raw_citations, # Citações
                                                 include_raw_affiliations=include_raw_affiliations, # Afiliações
                                                 tei_coordinates=tei_coordinates, # Gera coordenadas para gerar visualização de marcações no PDF
                                                 segment_sentences=segment_sentences,
                                                 verbose=verbose) # Usa um motor externo para segmentar as sentenças
        
        return list_results


class xmltei_to_dataframe(object):

    """"""
    
    def __init__(self):
        pass

    def get_process_datetime(self,doc,varnull=np.nan,deltah=3):
        r = doc.get('grobid_timestamp',varnull)
        if not pd.isna(r):
            r = pd.to_datetime(doc['grobid_timestamp'],errors='coerce',format='%Y-%m-%dT%H:%M+0000')
            r = r-pd.Timedelta(deltah,unit='hour')
            r = r.strftime('%Y-%m-%d %H:%M:00')
        return r
    
    def get_raw_doc(self,xml):
        
        """Get raw article information from the xml/text"""
        
        doc = grobid_tei_xml.parse_document_xml(xml)
        doc = doc.to_dict()
        
        return doc

    def get_doc(self,doc):
        
        """Get article information from the article documment"""
        
        dict_article = {'grobid_version':doc.get('grobid_version',np.nan),
                        'grobid_timestamp':self.get_process_datetime(doc,np.nan,3),
                        'pdf_md5':doc.get('pdf_md5',np.nan),
                        'language_code':doc.get('language_code',np.nan),
                        'acknowledgement':doc.get('acknowledgement',np.nan),
                        'abstract':doc.get('abstract',np.nan),
                        'body':doc.get('body',np.nan),
                        'annex ':doc.get('annex ',np.nan)}
        
        return [dict_article]


    def get_head(self,doc, suffix='head'):
        
        """Get head information from the article documment"""

        default_dict = {'_'.join(['index',suffix]):np.nan,
                        '_'.join(['id',suffix]):np.nan,
                        '_'.join(['unstructured',suffix]):np.nan,
                        '_'.join(['date',suffix]):np.nan,
                        '_'.join(['title',suffix]):np.nan,
                        '_'.join(['book_title',suffix]):np.nan,
                        '_'.join(['series_title',suffix]):np.nan,
                        '_'.join(['journal',suffix]):np.nan,
                        '_'.join(['journal_abbrev',suffix]):np.nan,
                        '_'.join(['publisher',suffix]):np.nan,
                        '_'.join(['institution',suffix]):np.nan,
                        '_'.join(['issn',suffix]):np.nan,
                        '_'.join(['eissn',suffix]):np.nan,
                        '_'.join(['volume',suffix]):np.nan,
                        '_'.join(['issue',suffix]):np.nan,
                        '_'.join(['pages',suffix]):np.nan,
                        '_'.join(['first_page',suffix]):np.nan,
                        '_'.join(['last_page',suffix]):np.nan,
                        '_'.join(['note',suffix]):np.nan,
                        '_'.join(['doi',suffix]):np.nan,
                        '_'.join(['pmid',suffix]):np.nan,
                        '_'.join(['pmcid',suffix]):np.nan,
                        '_'.join(['arxiv_id',suffix]):np.nan,
                        '_'.join(['ark',suffix]):np.nan,
                        '_'.join(['istex_id',suffix]):np.nan,
                        '_'.join(['url',suffix]):np.nan}
        
        head = doc.get('header',np.nan)
        if not pd.isna(head):
            dict_head = {'_'.join(['index',suffix]):head.get('index',np.nan),
                        '_'.join(['id',suffix]):head.get('id',np.nan),
                        '_'.join(['unstructured',suffix]):head.get('unstructured',np.nan),
                        '_'.join(['date',suffix]):head.get('date',np.nan),
                        '_'.join(['title',suffix]):head.get('title',np.nan),
                        '_'.join(['book_title',suffix]):head.get('book_title',np.nan),
                        '_'.join(['series_title',suffix]):head.get('series_title',np.nan),
                        '_'.join(['journal',suffix]):head.get('journal',np.nan),
                        '_'.join(['journal_abbrev',suffix]):head.get('journal_abbrev',np.nan),
                        '_'.join(['publisher',suffix]):head.get('publisher',np.nan),
                        '_'.join(['institution',suffix]):head.get('institution',np.nan),
                        '_'.join(['issn',suffix]):head.get('issn',np.nan),
                        '_'.join(['eissn',suffix]):head.get('eissn',np.nan),
                        '_'.join(['volume',suffix]):head.get('volume',np.nan),
                        '_'.join(['issue',suffix]):head.get('issue',np.nan),
                        '_'.join(['pages',suffix]):head.get('pages',np.nan),
                        '_'.join(['first_page',suffix]):head.get('first_page',np.nan),
                        '_'.join(['last_page',suffix]):head.get('last_page',np.nan),
                        '_'.join(['note',suffix]):head.get('note',np.nan),
                        '_'.join(['doi',suffix]):head.get('doi',np.nan),
                        '_'.join(['pmid',suffix]):head.get('pmid',np.nan),
                        '_'.join(['pmcid',suffix]):head.get('pmcid',np.nan),
                        '_'.join(['arxiv_id',suffix]):head.get('arxiv_id',np.nan),
                        '_'.join(['ark',suffix]):head.get('ark',np.nan),
                        '_'.join(['istex_id',suffix]):head.get('istex_id',np.nan),
                        '_'.join(['url',suffix]):head.get('url',np.nan)}
            return [dict_head]
        return [default_dict]


    def get_authors(self,doc, key_doc='header',key_authors='authors',suffix='author'):
        
        """Get authors from the article documment"""
        
        default_fict = {'_'.join(['full_name',suffix]):np.nan,
                        '_'.join(['given_name',suffix]):np.nan,
                        '_'.join(['middle_name',suffix]):np.nan,
                        '_'.join(['surname',suffix]):np.nan,
                        '_'.join(['email',suffix]):np.nan,
                        '_'.join(['orcid',suffix]):np.nan,
                        '_'.join(['institution',suffix]):np.nan,
                        '_'.join(['department',suffix]):np.nan,
                        '_'.join(['laboratory',suffix]):np.nan,
                        '_'.join(['addr_line',suffix]):np.nan,
                        '_'.join(['post_code',suffix]):np.nan,
                        '_'.join(['settlement',suffix]):np.nan,
                        '_'.join(['country',suffix]):np.nan}
        
        head = doc.get(key_doc,np.nan)
        if not pd.isna(head):
            authors = head.get(key_authors,[])
            if len(authors):
                lista_authors = []
                for author in authors:
                    affiliation = author.get('affiliation',np.nan)
                    address = affiliation.get('address',np.nan) if not pd.isna(affiliation) else np.nan
                    dict_authors = {'_'.join(['full_name',suffix]):author.get('full_name',np.nan),
                                    '_'.join(['given_name',suffix]):author.get('given_name',np.nan),
                                    '_'.join(['middle_name',suffix]):author.get('middle_name',np.nan),
                                    '_'.join(['surname',suffix]):author.get('surname',np.nan),
                                    '_'.join(['email',suffix]):author.get('email',np.nan),
                                    '_'.join(['orcid',suffix]):author.get('orcid',np.nan),
                                    '_'.join(['institution',suffix]):affiliation.get('institution',np.nan) if not pd.isna(affiliation) else np.nan,
                                    '_'.join(['department',suffix]):affiliation.get('department',np.nan) if not pd.isna(affiliation) else np.nan,
                                    '_'.join(['laboratory',suffix]):affiliation.get('laboratory',np.nan) if not pd.isna(affiliation) else np.nan,
                                    '_'.join(['addr_line',suffix]):address.get('addr_line',np.nan) if not pd.isna(address) else np.nan,
                                    '_'.join(['post_code',suffix]):address.get('post_code',np.nan) if not pd.isna(address) else np.nan,
                                    '_'.join(['settlement',suffix]):address.get('settlement',np.nan) if not pd.isna(address) else np.nan,
                                    '_'.join(['country',suffix]):address.get('country',np.nan) if not pd.isna(address) else np.nan}
                    lista_authors.append(dict_authors)
                return lista_authors
            return [default_fict]
        return [default_fict]


    def get_citations(self,doc, suffix='citation'):
        
        """Get citation informations from the article documment"""
        
        try:

            default_dict = {'_'.join(['index',suffix]):np.nan,
                            '_'.join(['id',suffix]):np.nan,
                            '_'.join(['unstructured',suffix]):np.nan,
                            '_'.join(['date',suffix]):np.nan,
                            '_'.join(['title',suffix]):np.nan,
                            '_'.join(['book_title',suffix]):np.nan,
                            '_'.join(['series_title',suffix]):np.nan,
                            '_'.join(['journal',suffix]):np.nan,
                            '_'.join(['journal_abbrev',suffix]):np.nan,
                            '_'.join(['publisher',suffix]):np.nan,
                            '_'.join(['institution',suffix]):np.nan,
                            '_'.join(['issn',suffix]):np.nan,
                            '_'.join(['eissn',suffix]):np.nan,
                            '_'.join(['volume',suffix]):np.nan,
                            '_'.join(['issue',suffix]):np.nan,
                            '_'.join(['pages',suffix]):np.nan,
                            '_'.join(['first_page',suffix]):np.nan,
                            '_'.join(['last_page',suffix]):np.nan,
                            '_'.join(['note',suffix]):np.nan,
                            '_'.join(['doi',suffix]):np.nan,
                            '_'.join(['pmid',suffix]):np.nan,
                            '_'.join(['pmcid',suffix]):np.nan,
                            '_'.join(['arxiv_id',suffix]):np.nan,
                            '_'.join(['ark',suffix]):np.nan,
                            '_'.join(['istex_id',suffix]):np.nan,
                            '_'.join(['url',suffix]):np.nan}
            
            citations = doc.get('citations',[])
            if len(citations):
                lista_citations = []
                for citation in citations:
                    dict_cit = {'_'.join(['index',suffix]):citation.get('index',np.nan),
                                '_'.join(['id',suffix]):citation.get('id',np.nan),
                                '_'.join(['unstructured',suffix]):citation.get('unstructured',np.nan),
                                '_'.join(['date',suffix]):citation.get('date',np.nan),
                                '_'.join(['title',suffix]):citation.get('title',np.nan),
                                '_'.join(['book_title',suffix]):citation.get('book_title',np.nan),
                                '_'.join(['series_title',suffix]):citation.get('series_title',np.nan),
                                '_'.join(['journal',suffix]):citation.get('journal',np.nan),
                                '_'.join(['journal_abbrev',suffix]):citation.get('journal_abbrev',np.nan),
                                '_'.join(['publisher',suffix]):citation.get('publisher',np.nan),
                                '_'.join(['institution',suffix]):citation.get('institution',np.nan),
                                '_'.join(['issn',suffix]):citation.get('issn',np.nan),
                                '_'.join(['eissn',suffix]):citation.get('eissn',np.nan),
                                '_'.join(['volume',suffix]):citation.get('volume',np.nan),
                                '_'.join(['issue',suffix]):citation.get('issue',np.nan),
                                '_'.join(['pages',suffix]):citation.get('pages',np.nan),
                                '_'.join(['first_page',suffix]):citation.get('first_page',np.nan),
                                '_'.join(['last_page',suffix]):citation.get('last_page',np.nan),
                                '_'.join(['note',suffix]):citation.get('note',np.nan),
                                '_'.join(['doi',suffix]):citation.get('doi',np.nan),
                                '_'.join(['pmid',suffix]):citation.get('pmid',np.nan),
                                '_'.join(['pmcid',suffix]):citation.get('pmcid',np.nan),
                                '_'.join(['arxiv_id',suffix]):citation.get('arxiv_id',np.nan),
                                '_'.join(['ark',suffix]):citation.get('ark',np.nan),
                                '_'.join(['istex_id',suffix]):citation.get('istex_id',np.nan),
                                '_'.join(['url',suffix]):citation.get('url',np.nan)}
                    lista_citations.append(dict_cit)
                return lista_citations
            return default_dict
        except:
            print(default_dict)


    def get_citation_authors(self,citation, suffix='citation'):
        
        """Get authors information from one only citation in the article documment"""

        default_fict = {'id':citation.get('id',np.nan),
                        'index':citation.get('index',np.nan),
                        '_'.join(['full_name',suffix]):np.nan,
                        '_'.join(['given_name',suffix]):np.nan,
                        '_'.join(['middle_name',suffix]):np.nan,
                        '_'.join(['surname',suffix]):np.nan,
                        '_'.join(['email',suffix]):np.nan,
                        '_'.join(['orcid',suffix]):np.nan,
                        '_'.join(['institution',suffix]):np.nan,
                        '_'.join(['department',suffix]):np.nan,
                        '_'.join(['laboratory',suffix]):np.nan,
                        '_'.join(['addr_line',suffix]):np.nan,
                        '_'.join(['post_code',suffix]):np.nan,
                        '_'.join(['settlement',suffix]):np.nan,
                        '_'.join(['country',suffix]):np.nan}
        
        authors = citation.get('authors',[])
        if len(authors):
            lista_authors = []
            for author in authors:
                affiliation = author.get('affiliation',np.nan)
                address = affiliation.get('address',np.nan) if not pd.isna(affiliation) else np.nan
                dict_authors = {'id':citation.get('id',np.nan),
                                'index':citation.get('index',np.nan),
                                '_'.join(['full_name',suffix]):author.get('full_name',np.nan),
                                '_'.join(['given_name',suffix]):author.get('given_name',np.nan),
                                '_'.join(['middle_name',suffix]):author.get('middle_name',np.nan),
                                '_'.join(['surname',suffix]):author.get('surname',np.nan),
                                '_'.join(['email',suffix]):author.get('email',np.nan),
                                '_'.join(['orcid',suffix]):author.get('orcid',np.nan),
                                '_'.join(['institution',suffix]):affiliation.get('institution',np.nan) if not pd.isna(affiliation) else np.nan,
                                '_'.join(['department',suffix]):affiliation.get('department',np.nan) if not pd.isna(affiliation) else np.nan,
                                '_'.join(['laboratory',suffix]):affiliation.get('laboratory',np.nan) if not pd.isna(affiliation) else np.nan,
                                '_'.join(['addr_line',suffix]):address.get('addr_line',np.nan) if not pd.isna(address) else np.nan,
                                '_'.join(['post_code',suffix]):address.get('post_code',np.nan) if not pd.isna(address) else np.nan,
                                '_'.join(['settlement',suffix]):address.get('settlement',np.nan) if not pd.isna(address) else np.nan,
                                '_'.join(['country',suffix]):address.get('country',np.nan) if not pd.isna(address) else np.nan}
                lista_authors.append(dict_authors)
            return lista_authors
        return [default_fict]


    def get_citations_authors(self,doc, suffix='citation'):
        
        """Get authors information from all the citations in the article documment"""

        default_fict = {'id':np.nan,
                        'index':np.nan,
                        '_'.join(['full_name',suffix]):np.nan,
                        '_'.join(['given_name',suffix]):np.nan,
                        '_'.join(['middle_name',suffix]):np.nan,
                        '_'.join(['surname',suffix]):np.nan,
                        '_'.join(['email',suffix]):np.nan,
                        '_'.join(['orcid',suffix]):np.nan,
                        '_'.join(['institution',suffix]):np.nan,
                        '_'.join(['department',suffix]):np.nan,
                        '_'.join(['laboratory',suffix]):np.nan,
                        '_'.join(['addr_line',suffix]):np.nan,
                        '_'.join(['post_code',suffix]):np.nan,
                        '_'.join(['settlement',suffix]):np.nan,
                        '_'.join(['country',suffix]):np.nan}
        
        citations = doc.get('citations',[])
        if len(citations):
            lista_citations_authors = []
            for citation in citations:
                lista_citations_authors += self.get_citation_authors(citation)
            return lista_citations_authors
        return default_fict


    def get_dataframe_article(self,doc):

        """"""
        
        dict_dataframes = {'df_doc_info':np.nan,
                           'df_doc_head':np.nan,
                           'df_doc_authors':np.nan,
                           'df_doc_citations':np.nan,
                           'df_doc_authors_citations':np.nan}
        
        df_doc_info = pd.DataFrame(self.get_doc(doc))
        dict_dataframes['df_doc_info'] = df_doc_info
        
        # What collum ID do I set into DF? PDFMD5?
        
        df_doc_head = pd.DataFrame(self.get_head(doc))
        dict_dataframes['df_doc_head'] = df_doc_head
        
        df_doc_authors = pd.DataFrame(self.get_authors(doc))
        dict_dataframes['df_doc_authors'] = df_doc_authors
        
        df_doc_citations = pd.DataFrame(self.get_citations(doc))
        dict_dataframes['df_doc_citations'] = df_doc_citations
        
        df_doc_authors_citations = pd.DataFrame(self.get_citations_authors(doc))
        dict_dataframes['df_doc_authors_citations'] = df_doc_authors_citations
        
        return dict_dataframes
    

    def get_dataframe_articles(self, bath_process_result):
        
        """"""
        try:
            
            dict_dataframes = {'df_doc_info':[],
                            'df_doc_head':[],
                            'df_doc_authors':[],
                            'df_doc_citations':[],
                            'df_doc_authors_citations':[]}
            dict_erros = {}
            
            if bath_process_result or len(bath_process_result):
                
                dict_erros['number_article_error'] = 0
                dict_erros['list_article_error'] = []
                
                for i,result in enumerate(bath_process_result, start=1):
                    try:
                        
                        # Return of batch process for each file in the input path selected by the user
                        file = result[0]
                        status = result[1]
                        xml = result[2]
                    
                        doc = self.get_raw_doc(xml)
                        dict_dfs = self.get_dataframe_article(doc)
                        
                        # If there are records in DF, then create new columns
                        if dict_dfs['df_doc_info'].shape[0]:
                            
                            dict_dfs['df_doc_info']['file'] = file
                            dict_dfs['df_doc_info']['status'] = status
                            dict_dfs['df_doc_info']['raw_data'] = xml
                            
                            # Set article id
                            list_dframes = ['df_doc_info','df_doc_head','df_doc_authors',
                                            'df_doc_citations','df_doc_authors_citations']
                            for df_name in list_dframes:
                                dict_dfs[df_name]['article_id'] = str(i)
                                dict_dfs[df_name].set_index('article_id', inplace=True)
                                dict_dataframes[df_name].append(dict_dfs[df_name])
                        
                    except Exception as e:
                        dict_erros['number_article_error'] += 1
                        dict_erros['list_article_error'].append({'file':file,
                                                                 'error':e.__class__,
                                                                 'error_text':str(e)})
                        continue
                    
                del bath_process_result
                
                dict_dataframes['df_doc_info'] = pd.concat(dict_dataframes['df_doc_info'])
                dict_dataframes['df_doc_head'] = pd.concat(dict_dataframes['df_doc_head'])
                dict_dataframes['df_doc_authors'] = pd.concat(dict_dataframes['df_doc_authors'])
                dict_dataframes['df_doc_citations'] = pd.concat(dict_dataframes['df_doc_citations'])
                dict_dataframes['df_doc_authors_citations'] = pd.concat(dict_dataframes['df_doc_authors_citations'])
                
                print("Processed articles:", str(dict_dataframes['df_doc_info'].shape[0]))
                print("Number articles with errors:", str(dict_erros['number_article_error']))
                
            return dict_dataframes, dict_erros
        
        except Exception as e:
            print("Arquivo:",file)
            print("Oops!", e.__class__, "occurred.")
            print(str(e))
            