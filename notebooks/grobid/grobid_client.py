"""Grobid Client.

This version uses the standard ProcessPoolExecutor for parallelizing the
concurrent calls to the GROBID services.  Given the limits of
ThreadPoolExecutor (the legendary GIL, input stored in memory, blocking
Executor.map until the whole input is acquired), ProcessPoolExecutor works with
batches of PDF of a size indicated in the config.json file (default is 1000
entries). We are moving from first batch to the second one only when the first
is entirely processed - which means it is slightly sub-optimal, but should
scale better. Working without batch would mean acquiring a list of millions of
files in directories and would require something scalable too (e.g. done in a
separate thread), which is not implemented for the moment and possibly not
implementable in Python as long it uses the GIL.
"""
import os
import io
import json
import argparse
import time
import concurrent.futures
import ntpath
import requests
import pathlib
import sys

sys.path.insert(0,os.path.dirname(os.getcwd()))
sys.path.insert(0,os.path.join(os.getcwd(),'grobid'))
sys.path.insert(0,os.getcwd())

from .client import ApiClient


class ServerUnavailableException(Exception):
    pass

class GrobidClient(ApiClient):

    def __init__(self, grobid_server='localhost', 
                 grobid_port='8070',
                 batch_size=1000, 
                 coordinates=["persName", "figure", "ref", "biblStruct", "formula", "s" ], 
                 sleep_time=5,
                 timeout=60,
                 config_path=None, 
                 check_server=True):
        self.config = {
            'grobid_server': grobid_server,
            'grobid_port': grobid_port,
            'batch_size': batch_size,
            'coordinates': coordinates,
            'sleep_time': sleep_time,
            'timeout': timeout
        }
        if config_path:
            self._load_config(config_path)
        if check_server:
            self._test_server_connection()

    def _load_config(self, path="./config.json"):
        """Load the json configuration
        """
        config_json = open(path).read()
        self.config = json.loads(config_json)

    def _test_server_connection(self):
        """Test if the server is up and running."""
        the_url = "http://" + self.config["grobid_server"]
        if len(self.config["grobid_port"]) > 0:
            the_url += ":" + self.config["grobid_port"]
        the_url += "/api/isalive"
        try:
            r = requests.get(the_url)
        except:
            print("GROBID server does not appear up and running, the connection to the server failed")
            raise ServerUnavailableException

        status = r.status_code

        if status != 200:
            print("GROBID server does not appear up and running " + str(status))
        else:
            print("GROBID server is up and running")

    def _output_file_name(self, input_file, input_path, output):
        # we use ntpath here to be sure it will work on Windows too
        if output is not None:
            input_file_name = str(os.path.relpath(os.path.abspath(input_file), input_path))
            filename = os.path.join(
                output, os.path.splitext(input_file_name)[0] + ".tei.xml"
            )
        else:
            input_file_name = ntpath.basename(input_file)
            filename = os.path.join(
                ntpath.dirname(input_file),
                os.path.splitext(input_file_name)[0] + ".tei.xml",
            )

        return filename

    def process(
        self,
        service,
        input_path,
        output=None,
        n=10,
        generateIDs=False,
        consolidate_header=True,
        consolidate_citations=False,
        include_raw_citations=False,
        include_raw_affiliations=False,
        tei_coordinates=False,
        segment_sentences=False,
        force=True,
        verbose=False,
    ):
        batch_size_pdf = self.config["batch_size"]
        input_files = []

        for (dirpath, dirnames, filenames) in os.walk(input_path):
            for filename in filenames:
                if filename.endswith(".pdf") or filename.endswith(".PDF") or \
                    (service == 'processCitationList' and (filename.endswith(".txt") or filename.endswith(".TXT"))):
                    if verbose:
                        try:
                            print(filename)
                        except Exception:
                            # may happen on linux see https://stackoverflow.com/questions/27366479/python-3-os-walk-file-paths-unicodeencodeerror-utf-8-codec-cant-encode-s
                            pass
                    input_files.append(os.sep.join([dirpath, filename]))

                    if len(input_files) == batch_size_pdf:
                        self.process_batch(
                            service,
                            input_files,
                            input_path,
                            output,
                            n,
                            generateIDs,
                            consolidate_header,
                            consolidate_citations,
                            include_raw_citations,
                            include_raw_affiliations,
                            tei_coordinates,
                            segment_sentences,
                            force,
                            verbose,
                        )
                        input_files = []

        # last batch
        if len(input_files) > 0:
            self.process_batch(
                service,
                input_files,
                input_path,
                output,
                n,
                generateIDs,
                consolidate_header,
                consolidate_citations,
                include_raw_citations,
                include_raw_affiliations,
                tei_coordinates,
                segment_sentences,
                force,
                verbose,
            )

    def process_batch(
        self,
        service,
        input_files,
        input_path,
        output,
        n,
        generateIDs,
        consolidate_header,
        consolidate_citations,
        include_raw_citations,
        include_raw_affiliations,
        tei_coordinates,
        segment_sentences,
        force,
        verbose=False,
    ):
        if verbose:
            print(len(input_files), "files to process in current batch")

        # with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n) as executor:
            results = []
            for input_file in input_files:
                # check if TEI file is already produced
                filename = self._output_file_name(input_file, input_path, output)
                if not force and os.path.isfile(filename):
                    print(filename, "already exist, skipping... (use --force to reprocess pdf input files)")
                    continue

                selected_process = self.process_pdf
                if service == 'processCitationList':
                    selected_process = self.process_txt
                
                r = executor.submit(
                    selected_process,
                    service,
                    input_file,
                    generateIDs,
                    consolidate_header,
                    consolidate_citations,
                    include_raw_citations,
                    include_raw_affiliations,
                    tei_coordinates,
                    segment_sentences)

                results.append(r)

        for r in concurrent.futures.as_completed(results):
            input_file, status, text = r.result()
            filename = self._output_file_name(input_file, input_path, output)

            if text is None:
                print("Processing of", input_file, "failed with error", str(status))
            else:
                # writing TEI file
                try:
                    pathlib.Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
                    with open(filename,'w',encoding='utf8') as tei_file:
                        tei_file.write(text)
                except OSError:
                   print("Writing resulting TEI XML file", filename, "failed")


    def get_input_files(self,
                        input_path, type_file='.pdf'):
        
        """"""
        
        input_files = []
        for file in os.listdir(input_path):
            if os.path.isfile(os.path.join(input_path, file)):
                if file.endswith(type_file) or file.endswith(type_file.upper()):
                    input_files.append(os.path.join(input_path, file))    
        return input_files


    def check_typefile_inpath(self, files_list, type='pdf'):
        
        """check if there is at least one file in the path with a specific type, like PDF for example."""
        
        type = str(type).lower().strip()
        
        for file in files_list:
            file = str(file).lower().strip()
            if file.endswith(''.join(['.',type])):
                return True
        return False 


    def change_filetype(self, file_change, target_type='xml'):
        
        """"""
        
        file_change = str(file_change).strip()
        target_type = str(target_type).strip()
        
        file_split = file_change.split('.')
        if len(file_split) > 1:
            endstr_file = file_split[-1]
            file_change = ''.join([file_change[:len(file_change)-len(endstr_file)],target_type])
        else:
            file_change = '.'.join([file_change,target_type])
            
        return file_change


    def save_xmltei_files(self, result_batch, input_folder_path, cache_folder_name='summarticles_cache'):
    
        """"""
        
        # If summarticles path cache doesn't exist than we have to create it
        path_cache = os.path.join(input_folder_path, cache_folder_name)
        if not os.path.exists(path_cache):
            os.mkdir(path_cache)
        
        for article_batch in result_batch:
            
            file_article = article_batch[0]
            status_grobid = article_batch[1]
            text_xml = article_batch[2]
            file = self.change_filetype(file_article, target_type='xml')
            file = os.path.split(file)[-1]
            
            file_path_xml = os.path.join(input_folder_path, cache_folder_name, file)
            
            if not os.path.isfile(file_path_xml):
                f = open(file_path_xml, "w", encoding="utf-8")
                f.write(text_xml)
                f.close()


    def get_file_name(self, path):
        path_split = os.path.split(str(path))
        path_file = path_split[-1].split('.')
        path_file.pop()
        path_file = '.'.join(path_file)
        return str(path_file)


    def read_cache_files(self, input_path, cache_folder_name="summarticles_cache", status_return="status 200"):
        
        return_results = []
        input_files_dif = []
        end_process = False
        
        path_cache = os.path.join(input_path, cache_folder_name)
        if not os.path.exists(path_cache):
            os.mkdir(path_cache)
        
        input_files = self.get_input_files(input_path, type_file='.pdf')
        cache_files = self.get_input_files(os.path.join(input_path, cache_folder_name), type_file='.xml')
        
        if len(input_files) and len(cache_files):
            
            input_files_comp = {self.get_file_name(f_input):f_input for f_input in input_files}
            print("[Input Files]", len(input_files))
            
            cache_files_comp = {self.get_file_name(f_cache):f_cache for f_cache in cache_files}
            print("[Cache Files]", len(cache_files))
            
            input_files_set = set(list(input_files_comp.keys()))
            cache_files_set = set(list(cache_files_comp.keys()))
            input_files_dif = input_files_set.difference(cache_files_set)
            cache_files_int = input_files_set.intersection(cache_files_set)
            
            input_files = [input_files_comp[f_input] for f_input in input_files_dif]
            
            print("In the end, we have:", len(input_files)," new files to process!")
            print("And we have :", len(cache_files_int)," files to back from cache!")
            
            if len(cache_files_int):
                for f_cache in cache_files_int:
                    with open(cache_files_comp[f_cache], "r", encoding="utf-8") as f:
                        content_file = f.read()
                    return_results.append([input_files_comp[f_cache], status_return, str(content_file)])
                    
            if not len(input_files):
                end_process = True
            
        return return_results, input_files, end_process
    
    
    def batch_process(
        self,
        service,
        input_path,
        n_workers,
        check_cache=True,
        cache_folder_name='summarticles_cache',
        generateIDs=True,
        consolidate_header=False,
        consolidate_citations=False,
        include_raw_citations=True,
        include_raw_affiliations=True,
        tei_coordinates=False,
        segment_sentences=False,
        verbose=True,
    ):
        
        """"""
        
        
        input_files = self.get_input_files(input_path, type_file='.pdf')
        return_results = []
        
        if verbose:
            print(len(input_files), "files to process in current batch")
        
        # If there are no files, then return a empty list
        if not len(input_files):
            return return_results
        
        # Load cache files
        if check_cache:
            cache_results, input_files_dif, end_process = self.read_cache_files(input_path, cache_folder_name=cache_folder_name)
            # If all articles already in cache, then we dont need to process any article in grobid
            if end_process:
                return cache_results # return files in cache
            return_results += cache_results
            input_files = input_files_dif

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = []
            for input_file in input_files:
                
                selected_process = self.process_pdf
        
                r = executor.submit(selected_process,
                                    service,
                                    input_file,
                                    generateIDs,
                                    consolidate_header,
                                    consolidate_citations,
                                    include_raw_citations,
                                    include_raw_affiliations,
                                    tei_coordinates,
                                    segment_sentences)

                results.append(r)
                
        executor.shutdown(wait=True)
        
        
        for r in concurrent.futures.as_completed(results):
            
            input_file, status, text = r.result()
            
            if text is None and verbose:
                print("Processing of", input_file, "failed with error", str(status))
                
            return_results.append([input_file, status, text])
            
        return return_results
                

    def process_pdf(
        self,
        service,
        pdf_file,
        generateIDs,
        consolidate_header,
        consolidate_citations,
        include_raw_citations,
        include_raw_affiliations,
        tei_coordinates,
        segment_sentences
    ):
        files = {
            "input": (
                pdf_file,
                open(pdf_file, "rb"),
                "application/pdf",
                {"Expires": "0"},
            )
        }

        the_url = "http://" + self.config["grobid_server"]
        if len(self.config["grobid_port"]) > 0:
            the_url += ":" + self.config["grobid_port"]
        the_url += "/api/" + service

        # set the GROBID parameters
        the_data = {}
        if generateIDs:
            the_data["generateIDs"] = "1"
        if consolidate_header:
            the_data["consolidateHeader"] = "1"
        if consolidate_citations:
            the_data["consolidateCitations"] = "1"
        if include_raw_citations:
            the_data["includeRawCitations"] = "1"
        if include_raw_affiliations:
            the_data["includeRawAffiliations"] = "1"
        if tei_coordinates:
            the_data["teiCoordinates"] = self.config["coordinates"]
        if segment_sentences:
            the_data["segmentSentences"] = "1"

        try:
            res, status = self.post(
                url=the_url, files=files, data=the_data, headers={"Accept": "text/plain"}, timeout=self.config['timeout']
            )

            if status == 503:
                time.sleep(self.config["sleep_time"])
                return self.process_pdf(
                    service,
                    pdf_file,
                    generateIDs,
                    consolidate_header,
                    consolidate_citations,
                    include_raw_citations,
                    include_raw_affiliations,
                    tei_coordinates,
                    segment_sentences
                )
        except requests.exceptions.ReadTimeout:
            return (pdf_file, 408, None)

        return (pdf_file, status, res.text)

    def process_txt(
        self,
        service,
        txt_file,
        generateIDs,
        consolidate_header,
        consolidate_citations,
        include_raw_citations,
        include_raw_affiliations,
        tei_coordinates,
        segment_sentences
    ):
        # create request based on file content
        references = None
        with open(txt_file) as f:
            references = [line.rstrip() for line in f]

        the_url = "http://" + self.config["grobid_server"]
        if len(self.config["grobid_port"]) > 0:
            the_url += ":" + self.config["grobid_port"]
        the_url += "/api/" + service

        # set the GROBID parameters
        the_data = {}
        if consolidate_citations:
            the_data["consolidateCitations"] = "1"
        if include_raw_citations:
            the_data["includeRawCitations"] = "1"
        the_data["citations"] = references
        res, status = self.post(
            url=the_url, data=the_data, headers={"Accept": "application/xml"}
        )

        if status == 503:
            time.sleep(self.config["sleep_time"])
            return self.process_txt(
                service,
                txt_file,
                generateIDs,
                consolidate_header,
                consolidate_citations,
                include_raw_citations,
                include_raw_affiliations,
                tei_coordinates,
                segment_sentences
            )

        return (txt_file, status, res.text)

def main():
    valid_services = [
        "processFulltextDocument",
        "processHeaderDocument",
        "processReferences",
        "processCitationList"
    ]

    parser = argparse.ArgumentParser(description="Client for GROBID services")
    parser.add_argument(
        "service",
        help="one of " + str(valid_services),
    )
    parser.add_argument(
        "--input", default=None, help="path to the directory containing PDF files or .txt (for processCitationList only, one reference per line) to process"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="path to the directory where to put the results (optional)",
    )
    parser.add_argument(
        "--config",
        default="./config.json",
        help="path to the config file, default is ./config.json",
    )
    parser.add_argument("--n", default=10, help="concurrency for service usage")
    parser.add_argument(
        "--generateIDs",
        action="store_true",
        help="generate random xml:id to textual XML elements of the result files",
    )
    parser.add_argument(
        "--consolidate_header",
        action="store_true",
        help="call GROBID with consolidation of the metadata extracted from the header",
    )
    parser.add_argument(
        "--consolidate_citations",
        action="store_true",
        help="call GROBID with consolidation of the extracted bibliographical references",
    )
    parser.add_argument(
        "--include_raw_citations",
        action="store_true",
        help="call GROBID requesting the extraction of raw citations",
    )
    parser.add_argument(
        "--include_raw_affiliations",
        action="store_true",
        help="call GROBID requestiong the extraciton of raw affiliations",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="force re-processing pdf input files when tei output files already exist",
    )
    parser.add_argument(
        "--teiCoordinates",
        action="store_true",
        help="add the original PDF coordinates (bounding boxes) to the extracted elements",
    )
    parser.add_argument(
        "--segmentSentences",
        action="store_true",
        help="segment sentences in the text content of the document with additional <s> elements",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print information about processed files in the console",
    )

    args = parser.parse_args()

    input_path = args.input
    config_path = args.config
    output_path = args.output

    if args.n is not None:
        try:
            n = int(args.n)
        except ValueError:
            print("Invalid concurrency parameter n:", n, ", n = 10 will be used by default")
            pass

    # if output path does not exist, we create it
    if output_path is not None and not os.path.isdir(output_path):
        try:
            print("output directory does not exist but will be created:", output_path)
            os.makedirs(output_path)
        except OSError:
            print("Creation of the directory", output_path, "failed")
        else:
            print("Successfully created the directory", output_path)

    service = args.service
    generateIDs = args.generateIDs
    consolidate_header = args.consolidate_header
    consolidate_citations = args.consolidate_citations
    include_raw_citations = args.include_raw_citations
    include_raw_affiliations = args.include_raw_affiliations
    force = args.force
    tei_coordinates = args.teiCoordinates
    segment_sentences = args.segmentSentences
    verbose = args.verbose

    if service is None or not service in valid_services:
        print("Missing or invalid service, must be one of", valid_services)
        exit(1)

    try:
        client = GrobidClient(config_path=config_path)
    except ServerUnavailableException:
        exit(1)

    start_time = time.time()

    client.process(
        service,
        input_path,
        output=output_path,
        n=n,
        generateIDs=generateIDs,
        consolidate_header=consolidate_header,
        consolidate_citations=consolidate_citations,
        include_raw_citations=include_raw_citations,
        include_raw_affiliations=include_raw_affiliations,
        tei_coordinates=tei_coordinates,
        segment_sentences=segment_sentences,
        force=force,
        verbose=verbose,
    )

    runtime = round(time.time() - start_time, 3)
    print("runtime: %s seconds " % (runtime))

if __name__ == "__main__":
    main()
