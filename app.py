import torch
import re
import streamlit as st
import more_itertools
import numpy as np
import pandas as pd
import time
import io

import nltk

from txtai.embeddings import Embeddings
from txtai.pipeline import Similarity, Segmentation, Textractor

import re

import pdfkit
import pdfrw
from pdfrw import PdfReader, PdfWriter
from pyvirtualdisplay import Display
import hashlib

import uuid
import os
import json
import pathlib

import tokenizers
import sqlite3

from annotater import Annotate

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static' / 'static'

ASSETS_PATH = (STREAMLIT_STATIC_PATH / "assets")
if not ASSETS_PATH.is_dir():
    ASSETS_PATH.mkdir()

st.set_page_config(page_title="context-search", )

is_git_sync_button_clicked = st.button("Git repository remote sync")
if (is_git_sync_button_clicked):
    os.chdir("/content/context-search-ui")
    os.system('git fetch --all')
    os.system('git reset --hard origin')
    st.legacy_caching.clear_cache()


t0 = time.time()


@st.cache
def load_nltk():
    nltk.download('punkt')


load_nltk()

model_name = None

if (None not in [model_name]):
    bi_encoder_model_name = st.text_area(
        "Enter the name of the pre-trained bi-encoder model from sentence transformers that we are using for searching.",
        value="sentence-transformers/multi-qa-mpnet-base-cos-v1")
    cross_encoder_model_name = st.text_area(
        "Enter the name of the pre-trained cross-encoder model from sentence transformers that we are using for searching.",
        value="cross-encoder/ms-marco-MiniLM-L-6-v2")
    st.caption("This will download a new model, so it may take awhile or even break if the model is too large.")
    st.caption("See the list of pre-trained models that are available here: https://www.sbert.net/docs/pretrained_models.html.")

    model_name = {
        "bi-encoder": bi_encoder_model_name,
        "cross-encoder": cross_encoder_model_name
    }


def hash_tensor(x):
    bio = io.BytesIO()
    torch.save(x, bio)
    return bio.getvalue()


@st.cache(hash_funcs={torch.Tensor: hash_tensor, tokenizers.Tokenizer: lambda x: json.dumps(x.__dict__, sort_keys=True), sqlite3.Connection: lambda x: hash(x), sqlite3.Cursor: lambda x: hash(x), sqlite3.Row: lambda x: hash(x)})
def get_embeddings(model_name, method, data=None):
    embeddings = Embeddings(
        {"path": model_name, "content": True, "objects": True, "method": method})
    embeddings.index([(id, text, None) for id, text in enumerate(data)])
    embeddings.ann = embeddings.ann.index_cpu_to_all_gpus(embeddings.ann)
    return embeddings


corpus_source_type = st.radio(
    "What is corpus source type?", ('plain text', 'document', 'web'), index=0)


pdf_file_path = None  # string
corpus = None


@st.cache
def get_pdf_from_file_upload(file_upload):
    file_hash = hashlib.md5(file_upload.getbuffer()).hexdigest()
    file_base_name = "{}.pdf".format(str(file_hash))
    file_path = str(ASSETS_PATH / file_base_name)
    with open(file_path, "wb") as f:
        f.write(file_upload.getbuffer())

    pdf_file_path = file_path
    return pdf_file_path


if (corpus_source_type in ["plain text", "web"]):
    corpus = st.text_area('Enter a corpus.')

if (corpus_source_type in ['document']):
    file_upload = st.file_uploader(
        "Upload a document", type=['pdf'], accept_multiple_files=False)

    if None not in [file_upload]:
        pdf_file_path = get_pdf_from_file_upload(file_upload)
        st.success("File uploaded!")


@st.cache
def get_pdf_from_url(url):
    pdf_file_path = None
    with Display():
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        file_base_name = "{}.pdf".format(str(url_hash))
        file_path = str(ASSETS_PATH / file_base_name)
        pdfkit.from_url(url, file_path, options=options)
        pdf_file_path = file_path
    return pdf_file_path


if (corpus_source_type in ['web']):
    options = {
        'page-size': 'Letter',
        'margin-top': '0.25in',
        'margin-right': '1.00in',
        'margin-bottom': '0.25in',
        'margin-left': '1.00in',
    }

    if (None not in [corpus] and corpus != ""):
        pdf_file_path = get_pdf_from_url(corpus)


@st.cache(hash_funcs={pdfrw.objects.pdfstring.PdfString: lambda x: json.dumps(x.__dict__, sort_keys=True)})
def get_pdf_splitted_page_file(input_file_path, output_file_path):
    pdf_reader = PdfReader(input_file_path)
    pdf_writer = PdfWriter(output_file_path)

    for page_num in range(start_page - 1, end_page):
        pdf_writer.addpage(pdf_reader.pages[page_num])

    pdf_writer.write()

    return output_file_path


if (None not in [pdf_file_path]):
    if (corpus_source_type in ['document', 'web']):
        file_name = os.path.splitext(os.path.basename(pdf_file_path))[0]
        pdf_reader = PdfReader(pdf_file_path)
        pdf_max_page = len(pdf_reader.pages)

        start_page = st.number_input(
            f"Enter the start page of the pdf you want to be highlighted (1-{pdf_max_page}).", min_value=1, max_value=pdf_max_page, value=1)
        end_page = st.number_input(
            f"Enter the end page of the pdf you want to be highlighted (1-{pdf_max_page}).", min_value=1, max_value=pdf_max_page, value=1)

        splitted_file_base_name = f'{file_name}_split_{start_page}_to_{end_page}.pdf'
        splitted_file_path = str(ASSETS_PATH / splitted_file_base_name)

        if (not os.path.exists(splitted_file_path)):
            get_pdf_splitted_page_file(pdf_file_path, splitted_file_path)

        corpus = splitted_file_path


query = st.text_area('Enter a query.')
granularity = st.radio(
    "What level of granularity do you want to search at?", ('word', 'sentence', 'paragraph'), index=1)
window_sizes = st.text_area(
    'Enter a list of window sizes that seperated by a space.', value='1')
window_sizes = [int(i) for i in re.split("[^0-9]", window_sizes) if i != ""]
percentage = st.number_input(
    "Enter the percentage of the text you want to be highlighted.", min_value=0.0, max_value=1.0, value=0.3)
output_type = st.radio(
    "What output type do you want to view?", ('default', 'plain text'), index=0)


@st.cache
def get_shaped_corpus(corpus, corpus_source_type, granularity):
    raw_corpus = ""  # string
    granularized_corpus = []  # [string, ...]

    raw_corpus = corpus

    if (corpus_source_type in ["plain text"]):
        if granularity == "word":
            granularized_corpus += raw_corpus.split(" ")
        elif granularity == "sentence":
            segmentation = Segmentation(sentences=True)
            granularized_corpus = segmentation(raw_corpus)
        elif granularity == "paragraph":
            segmentation = Segmentation(paragraphs=True)
            granularized_corpus = segmentation(raw_corpus)
    elif (corpus_source_type in ["document", "web"]):
        if granularity == "word":
            granularized_corpus += raw_corpus.split(" ")
        elif granularity == "sentence":
            textractor = Textractor(sentences=True)
            granularized_corpus = textractor(corpus)
        elif granularity == "paragraph":
            textractor = Textractor(paragraphs=True)
            granularized_corpus = textractor(corpus)

    return {"raw": raw_corpus, "granularized": granularized_corpus}


shaped_corpus = None
if (None not in [corpus, corpus_source_type, granularity] and corpus != ""):
    shaped_corpus = get_shaped_corpus(
        corpus, corpus_source_type, granularity)


@st.cache
def get_windowed_granularized_corpus(shaped_corpus, granularity, window_sizes):
    windowed_granularized_corpus = {}  # {"window_size": [string, ...]}
    # {window_size: [({"corpus": string, "index": numeric}, ...), ...]}
    indexed_windowed_granularized_corpus = {}

    for window_size in window_sizes:
        windowed_granularized_corpus[window_size] = []
        indexed_windowed_granularized_corpus[window_size] = []
        for wgc in more_itertools.windowed(enumerate(shaped_corpus['granularized']), window_size):
            source_index = []
            windowed_corpus = []
            for index, item in wgc:
                source_index.append(index)
                windowed_corpus.append(item)

            if (granularity == 'sentence' or granularity == 'word'):
                windowed_corpus = " ".join(windowed_corpus)
            elif (granularity == 'paragraph'):
                windowed_corpus = "\n".join(windowed_corpus)

            windowed_granularized_corpus[window_size].append(windowed_corpus)
            indexed_windowed_granularized_corpus[window_size].append(
                source_index)

    return {"raw": windowed_granularized_corpus, "indexed": indexed_windowed_granularized_corpus}


windowed_granularized_corpus = None
if (None not in [shaped_corpus, granularity, window_sizes]):
    windowed_granularized_corpus = get_windowed_granularized_corpus(
        shaped_corpus, granularity, window_sizes)


# result = (id: string, score: numeric)
@st.cache(hash_funcs={torch.Tensor: hash_tensor, tokenizers.Tokenizer: lambda x: json.dumps(x.__dict__, sort_keys=True), sqlite3.Connection: lambda x: hash(x), sqlite3.Cursor: lambda x: hash(x), sqlite3.Row: lambda x: hash(x)})
def retrieval_search(queries, embeddings, data=None, limit=None):
    return [{"corpus_id": int(result["id"]), "score": result["score"]} for result in embeddings.search(queries, limit)]
    # return [{"corpus_id": id, "score": score} for id, score in embeddings.similarity(queries, data)]


@st.cache(hash_funcs={torch.Tensor: hash_tensor, tokenizers.Tokenizer: lambda x: json.dumps(x.__dict__, sort_keys=True), sqlite3.Connection: lambda x: hash(x), sqlite3.Cursor: lambda x: hash(x), sqlite3.Row: lambda x: hash(x)})
def rerank_search(queries, retrieved_documents, windowed_granularized_corpus_raw_sized, rerank_model):
    reranked_document = [windowed_granularized_corpus_raw_sized[result["corpus_id"]] for result in retrieved_documents]
    return [{"corpus_id": id, "score": score} for id, score in rerank_model(queries, reranked_document)]


@st.cache(hash_funcs={torch.Tensor: hash_tensor, tokenizers.Tokenizer: lambda x: json.dumps(x.__dict__, sort_keys=True), sqlite3.Connection: lambda x: hash(x), sqlite3.Cursor: lambda x: hash(x), sqlite3.Row: lambda x: hash(x)})
def semantic_search(model_name, query, window_sizes, windowed_granularized_corpus):
    # {window_size: [{"corpus_id": 0, "score": 0}]}
    semantic_search_result = {}
    # {corpus_id: {"score_mean": 0, count: 0}}
    final_semantic_search_result = {}

    for window_size in window_sizes:
        windowed_granularized_corpus_raw_sized = windowed_granularized_corpus["raw"][window_size]
        corpus_len = len(windowed_granularized_corpus_raw_sized)

        corpus_embeddings = get_embeddings(
            model_name["bi-encoder"], "sentence-transformers", windowed_granularized_corpus_raw_sized)

        retrieved_results = retrieval_search(query, corpus_embeddings, limit=corpus_len)

        # rerank_model = Similarity(model_name["cross-encoder"])
        # reranked_results = rerank_search(query, retrieved_document, windowed_granularized_corpus_raw_sized, rerank_model)

        semantic_search_result[window_size] = retrieved_results

        # averaging overlapping result
        for ssr in semantic_search_result[window_size]:
            for source_corpus_index in windowed_granularized_corpus["indexed"][window_size][ssr["corpus_id"]]:
                if (final_semantic_search_result.get(source_corpus_index, None) is None):
                    final_semantic_search_result[source_corpus_index] = {
                        "count": 1, "score_mean": ssr["score"]}
                    semantic_search_result[window_size] += []
                else:
                    old_count = final_semantic_search_result[source_corpus_index]["count"]
                    new_count = old_count + 1
                    final_semantic_search_result[source_corpus_index]["count"] = new_count
                    new_value = ssr["score"]
                    old_score_mean = final_semantic_search_result[source_corpus_index]["score_mean"]
                    new_score_mean = old_score_mean + \
                        ((new_value-old_score_mean)/new_count)
                    final_semantic_search_result[source_corpus_index]["score_mean"] = new_score_mean

        # add-up zero-score result
        windowed_granularized_corpus_ids = set(
            [id for id in range(0, corpus_len)])
        semantic_search_result_ids = set(
            [result["corpus_id"] for result in semantic_search_result[window_size]])
        zero_semantic_search_result_ids = list(
            windowed_granularized_corpus_ids-semantic_search_result_ids)

        for corpus_id in zero_semantic_search_result_ids:
            for source_corpus_index in windowed_granularized_corpus["indexed"][window_size][corpus_id]:
                if (final_semantic_search_result.get(source_corpus_index, None) is None):
                    final_semantic_search_result[source_corpus_index] = {
                        "count": 1, "score_mean": 0}

    return {"windowed": semantic_search_result, "aggregated": final_semantic_search_result}


search_result = None
if (None not in [model_name, query, window_sizes, windowed_granularized_corpus]):
    search_result = semantic_search(
        model_name, query, window_sizes, windowed_granularized_corpus)


@st.cache
def get_filtered_search_result(percentage, shaped_corpus, search_result, granularity):
    html_raw = shaped_corpus["granularized"][:]
    dict_raw = []
    top_k = int(np.ceil(len(shaped_corpus["granularized"])*percentage))
    score_mean = 0
    total_count = 0
    for key, val in sorted(search_result["aggregated"].items(), key=lambda x: x[1]["score_mean"], reverse=True)[:top_k]:
        old_count = total_count
        new_count = old_count + 1
        total_count = new_count
        new_value = val["score_mean"]
        old_score_mean = score_mean
        new_score_mean = old_score_mean + \
            ((new_value-old_score_mean)/new_count)
        score_mean = new_score_mean

        dict_raw.append(
            {"corpus_id": key, "score": val["score_mean"]})

        annotated = "<mark style='background-color: lightgreen'>{}</mark>".format(
            html_raw[key])
        html_raw[key] = annotated

    if (granularity == 'sentence' or granularity == 'word'):
        html_raw = " ".join(html_raw)
    elif (granularity == 'paragraph'):
        html_raw = "\n".join(html_raw)

    html_raw = '<br />'.join(html_raw.splitlines())

    return {"html_raw": html_raw, "dict_raw": dict_raw, "score_mean": score_mean}


filtered_search_result = None
if (None not in [percentage, shaped_corpus, search_result, granularity]):
    filtered_search_result = get_filtered_search_result(
        percentage, shaped_corpus, search_result, granularity)


def get_html_pdf(file_path):
    relative_url_path = "static/assets/{}".format(os.path.basename(file_path))
    # Embedding PDF in HTML
    pdf_display = F'<iframe src="{relative_url_path}" width="700" height="1000"></iframe>'

    return pdf_display


@st.cache
def get_annotated_pdf(search_result_raw, granularized_corpus_raw, input_path, output_path):
    Annotate().annotate(search_result_raw, granularized_corpus_raw, input_path, output_path)
    return path_highlighted


html_pdf = None
if (None not in [corpus, filtered_search_result, shaped_corpus, corpus_source_type, output_type]):
    if (corpus_source_type in ["document", "web"] and output_type in ["default"]):
        file_name = os.path.splitext(os.path.basename(corpus))[0]
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        percentage_hash = hashlib.md5(
            str(percentage).encode('utf-8')).hexdigest()
        highlighted_file_base_name = f'{file_name}_highlighted_{query_hash}_{percentage_hash}.pdf'
        highlighted_file_path = str(ASSETS_PATH / highlighted_file_base_name)

        path_raw = corpus
        path_highlighted = highlighted_file_path

        annotated_pdf_path = get_annotated_pdf(
            filtered_search_result['dict_raw'], shaped_corpus["granularized"], path_raw, path_highlighted)

        html_pdf = get_html_pdf(annotated_pdf_path)

        print(html_pdf)

    t1 = time.time()

    st.subheader("Output process duration")
    st.write("{} seconds".format(t1-t0))

    st.subheader("Output score overview")
    st.caption(
        "Metric to determine how sure the context of query is in the corpus (score to search result in descending order).")
    chart_df = pd.DataFrame(
        [result['score_mean'] for result in sorted(search_result["aggregated"].values(
        ), key=lambda x: x["score_mean"], reverse=True)],
        columns=['score']
    )

    st.line_chart(chart_df)

    st.subheader("Output score mean")
    st.caption(
        "Metric to determine how sure the context of query is in the highlighted corpus.")
    st.write(filtered_search_result["score_mean"])

    st.subheader("Output content")
    if (output_type in ["default"]):
        if (corpus_source_type in ["document", "web"]):
            st.markdown(html_pdf, unsafe_allow_html=True)
        else:
            st.write(
                filtered_search_result["html_raw"], unsafe_allow_html=True)
    else:
        st.write(filtered_search_result["html_raw"], unsafe_allow_html=True)

    st.subheader("Raw semantic search results")
    st.caption("The index of the word, sentence, or paragraph is 'corpus_id'. Mean of overlapped windowed corpus from raw scores by similarity scoring between the query and the corpus is 'score'.")
    # st.write(search_result["windowed"])
    # st.write(search_result["aggregated"])
    st.write(filtered_search_result["dict_raw"])

    st.subheader("Results of granularized corpus (segmentation/tokenization)")
    st.caption("This shows the representation that the webapp gets of the input corpus. Useful for debugging if you get strange output.")
    st.write(shaped_corpus["granularized"])
    # st.write(windowed_granularized_corpus["raw"])
    # st.write(windowed_granularized_corpus["indexed"])
