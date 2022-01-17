import torch
import re
import streamlit as st
import more_itertools
import numpy as np
import time
import io

import nltk

from txtai.embeddings import Embeddings
from txtai.pipeline import Similarity, Segmentation, Textractor

import re

import pdfkit
from pyvirtualdisplay import Display
from txtmarker.factory import Factory
import base64

import uuid


t0 = time.time()

st.set_page_config(page_title="context-search")

nltk.download('punkt')

model_name = st.text_area(
    "Enter the name of the pre-trained model from sentence transformers that we are using for summarization.",
    value="sentence-transformers/msmarco-distilbert-cos-v5")
st.caption("This will download a new model, so it may take awhile or even break if the model is too large.")
st.caption("See the list of pre-trained models that are available here! https://www.sbert.net/docs/pretrained_models.html.")


def hash_tensor(x):
    bio = io.BytesIO()
    torch.save(x, bio)
    return bio.getvalue()


@st.cache(allow_output_mutation=True)
def get_embeddings(model_name, data):
    embeddings = Embeddings(
        {"path": model_name, "content": True, "objects": True})
    embeddings.index([(id, text, None) for id, text in enumerate(data)])
    return embeddings


corpus = st.text_area('Enter a corpus.')
corpus_source_type = st.radio(
    "What is corpus source type?", ('text', 'document', 'web'), index=0)

if (corpus_source_type == 'web'):
    urls = [corpus]

    pdf_result = []  # [{"url": string, "file_name":numeric}]

    options = {
        'page-size': 'Letter',
        'margin-top': '0.25in',
        'margin-right': '1.00in',
        'margin-bottom': '0.25in',
        'margin-left': '1.00in',
    }

    with Display():
        for url in urls:
            file_name = "{}.pdf".format(str(uuid.uuid4()))
            file_path = file_name
            pdfkit.from_url(url, file_path, options=options)
            new_pdf = {"url": url, "file_name": file_name}
            pdf_result.append(new_pdf)

    pdf = pdf_result[0]['file_name']
    textractor = Textractor()
    corpus = textractor(pdf)

query = st.text_area('Enter a query.')
granularity = st.radio(
    "What level of granularity do you want to search at?", ('word', 'sentence', 'paragraph'), index=1)
window_sizes = st.text_area(
    'Enter a list of window sizes that seperated by a space.', value='1')
window_sizes = [int(i) for i in re.split("[^0-9]", window_sizes) if i != ""]
percentage = st.number_input(
    "Enter the percentage of the text you want highlighted.", max_value=1.0, min_value=0.0, value=0.3)


@st.cache()
def get_granularized_corpus(corpus, granularity, window_sizes):
    granularized_corpus = []  # [string, ...]
    granularized_corpus_windowed = {}  # {"window_size": [(string,...), ...]}
    # {window_size: [({"corpus": string, "index": numeric}, ...), ...]}
    granularized_corpus_windowed_indexed = {}

    if granularity == "sentence":
        segmentation = Segmentation(sentences=True)
        granularized_corpus = segmentation(corpus)
    elif granularity == "word":
        granularized_corpus += corpus.split(" ")
    elif granularity == "paragraph":
        segmentation = Segmentation(paragraphs=True)
        granularized_corpus = segmentation(corpus)

    for window_size in window_sizes:
        granularized_corpus_windowed[window_size] = []
        granularized_corpus_windowed_indexed[window_size] = []
        for wgc in more_itertools.windowed(enumerate(granularized_corpus), window_size):
            source_index = []
            windowed_corpus = []
            for index, item in wgc:
                source_index.append(index)
                windowed_corpus.append(item)

            if(granularity == 'sentence' or granularity == 'word'):
                windowed_corpus = " ".join(windowed_corpus)
            elif(granularity == 'paragraph'):
                windowed_corpus = "\n".join(windowed_corpus)

            granularized_corpus_windowed[window_size].append(windowed_corpus)
            granularized_corpus_windowed_indexed[window_size].append(
                source_index)

    return {"raw": granularized_corpus, "windowed": granularized_corpus_windowed, "windowed_indexed": granularized_corpus_windowed_indexed}


granularized_corpus = get_granularized_corpus(
    corpus, granularity, window_sizes)


# result = {"id": string, "text": string, "score": numeric}

def retrieval_search(queries, embeddings, limit):
    return [{"corpus_id": int(result["id"]), "score": result["score"]} for result in embeddings.search(queries, limit=limit)]


def rerank_search(queries, embeddings, similarity, limit):
    results = [result['text']
               for result in retrieval_search(queries, embeddings, limit)]
    return [{"corpus_id": id, "score": score} for id, score in similarity(queries, results)]


@st.cache(hash_funcs={torch.Tensor: hash_tensor})
def search(model_name, query, window_sizes, granularized_corpus):
    semantic_search_result = {}  # {window_size: {"corpus_id": 0, "score": 0}}
    final_semantic_search_result = {}  # {corpus_id: {"score_mean": 0, count: 0}}

    for window_size in window_sizes:
        corpus_len = len(granularized_corpus["windowed"][window_size])

        corpus_embeddings = get_embeddings(
            model_name, granularized_corpus["windowed"][window_size])

        # similarity = Similarity("cross-encoder/ms-marco-MiniLM-L-6-v2")
        # semantic_search_result[window_size] = rerank_search((query), corpus_embeddings, similarity, corpus_len)

        semantic_search_result[window_size] = retrieval_search(
            (query), corpus_embeddings, corpus_len)

        # averaging overlapping result
        for ssr in semantic_search_result[window_size]:
            for source_corpus_index in granularized_corpus["windowed_indexed"][window_size][ssr["corpus_id"]]:
                if(final_semantic_search_result.get(source_corpus_index, None) is None):
                    final_semantic_search_result[source_corpus_index] = {
                        "count": 1, "score_mean": ssr["score"]}
                else:
                    old_count = final_semantic_search_result[source_corpus_index]["count"]
                    new_count = old_count + 1
                    final_semantic_search_result[source_corpus_index]["count"] = new_count
                    new_value = ssr["score"]
                    old_score_mean = final_semantic_search_result[source_corpus_index]["score_mean"]
                    new_score_mean = old_score_mean + \
                        ((new_value-old_score_mean)/new_count)
                    final_semantic_search_result[source_corpus_index]["score_mean"] = new_score_mean

    return {"raw": semantic_search_result, "final": final_semantic_search_result}


search_result = search(model_name, query, window_sizes, granularized_corpus)


@st.cache()
def get_filtered_search_result(percentage, granularized_corpus, search_result, granularity):
    html_raw = granularized_corpus["raw"]
    dict_raw = []
    top_k = int(np.ceil(len(granularized_corpus["raw"])*percentage))
    score_mean = 0
    total_count = 0
    for key, val in sorted(search_result["final"].items(), key=lambda x: x[1]["score_mean"], reverse=True)[:top_k]:
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

    if(granularity == 'sentence' or granularity == 'word'):
        html_raw = " ".join(html_raw)
    elif(granularity == 'paragraph'):
        html_raw = "\n".join(html_raw)

    html_raw = '<br />'.join(html_raw.splitlines())

    return {"html_raw": html_raw, "dict_raw": dict_raw, "score_mean": score_mean}


filtered_search_result = get_filtered_search_result(
    percentage, granularized_corpus, search_result, granularity)


@st.cache()
def get_html_pdf(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    return pdf_display


if(corpus_source_type in ["document", "web"]):
    path_raw = pdf
    path_highlighted = "highlighted_{}".format(pdf)

    highlights = []
    for val in filtered_search_result['dict_raw']:
        name = "{:.4f}".format(val['score'])
        corpus_id = val['corpus_id']
        corpus_text = granularized_corpus["raw"][corpus_id]
        text = re.escape(corpus_text)
        highlight = (name, text)
        highlights.append(highlight)

    # Create annotated file
    highlighter = Factory.create("pdf")
    highlighter.highlight(path_raw, path_highlighted, highlights)

    html_pdf = get_html_pdf(path_highlighted)

t1 = time.time()

st.subheader("Output process duration")
st.write("{} s".format(t1-t0))

st.subheader("Output score mean")
st.caption(
    "Metric to determine how sure the context of query is in the highlighted corpus.")
st.write(filtered_search_result["score_mean"])

st.subheader("Output content")
if(corpus_source_type in ["document", "web"]):
    st.markdown(html_pdf, unsafe_allow_html=True)
else:
    st.write(filtered_search_result["html_raw"], unsafe_allow_html=True)

st.subheader("Raw semantic search results")
st.caption("corpus_id is the index of the word, sentence, or paragraph. score is mean of overlapped windowed corpus from raw scores by similarity scoring between the query and the corpus.")
st.write(filtered_search_result["dict_raw"])

st.subheader("Results of granularized corpus (segmentation/tokenization)")
st.caption("This shows the representation that the webapp gets of the input corpus. Useful for debugging if you get strange output.")
st.write(granularized_corpus["raw"])
