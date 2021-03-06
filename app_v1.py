import torch
import re
import streamlit as st
import more_itertools
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np
import time
import io

t0 = time.time()

st.set_page_config(page_title="context-search")

nlp = spacy.load('en_core_web_sm')

model_name = st.text_area(
    "Enter the name of the pre-trained model from sentence transformers that we are using for summarization.",
    value="sentence-transformers/msmarco-distilbert-cos-v5")
st.caption("This will download a new model, so it may take awhile or even break if the model is too large.")
st.caption("See the list of pre-trained models that are available here! https://www.sbert.net/docs/pretrained_models.html.")


scoring_technique = st.radio(
    "Enter the scoring technique based on suitable score function in selected pre-trained model from sentence transformers.", ("cos_sim", "dot_product", 'pairwise_cos_sim', 'pairwise_dot_product'), index=0)


def hash_tensor(x):
    bio = io.BytesIO()
    torch.save(x, bio)
    return bio.getvalue()


@st.cache(allow_output_mutation=True)
def get_model(model_name):
    model = SentenceTransformer(model_name, device='cuda')
    return model


@st.cache(hash_funcs={torch.Tensor: hash_tensor})
def get_embeddings(model_name, data):
    model = get_model(model_name)
    query_embedding = model.encode(
        data, convert_to_tensor=True).to('cuda')
    return query_embedding


corpus = st.text_area('Enter a document.')
query = st.text_area('Enter a query.')
granularity = st.radio(
    "What level of granularity do you want to search at?", ('word', 'sentence', 'paragraph'), index=1)
window_sizes = st.text_area(
    'Enter a list of window sizes that seperated by a space.', value='1')
window_sizes = [int(i) for i in re.split("[^0-9]", window_sizes) if i != ""]
percentage = st.number_input(
    "Enter the percentage of the text you want highlighted.", max_value=1.0, min_value=0.0, value=0.3)


@st.cache(hash_funcs={spacy.vocab.Vocab: lambda x: None})
def get_granularized_corpus(corpus, granularity, window_sizes, nlp):
    granularized_corpus = []  # ["", ...]
    granularized_corpus_windowed = {}  # {window_size: [("",...), ...]}
    # {window_size: [({"corpus": "", "index": 0}, ...), ...]}
    granularized_corpus_windowed_indexed = {}

    if granularity == "sentence":
        doc = nlp(corpus)
        granularized_corpus = [str(sent) for sent in doc.sents]
    elif granularity == "word":
        granularized_corpus += corpus.split(" ")
    elif granularity == "paragraph":
        granularized_corpus = corpus.splitlines()

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
    corpus, granularity, window_sizes, nlp)

query_embedding = get_embeddings(model_name, [query])


@st.cache
def set_scoring_technique(scoring_technique):
    if(scoring_technique == "dot_product"):
        score_function = util.dot_score
    elif (scoring_technique == "pairwise_dot_product"):
        score_function = util.pairwise_dot_score
    elif (scoring_technique == "pairwise_cos_sim"):
        score_function = util.pairwise_cos_sim
    else:
        score_function = util.cos_sim
    return score_function


@st.cache(hash_funcs={torch.Tensor: hash_tensor})
def search(model_name, scoring_technique, query, window_sizes, granularized_corpus):
    semantic_search_result = {}  # {window_size: {"corpus_id": 0, "score": 0}}
    final_semantic_search_result = {}  # {corpus_id: {"score_mean": 0, count: 0}}

    query_embeddings = get_embeddings(model_name, [query])

    for window_size in window_sizes:
        corpus_len = len(granularized_corpus["windowed"][window_size])

        corpus_embeddings = get_embeddings(
            model_name, granularized_corpus["windowed"][window_size])

        score_function = set_scoring_technique(scoring_technique)

        semantic_search_result[window_size] = util.semantic_search(
            query_embeddings, corpus_embeddings, top_k=corpus_len, score_function=score_function)

        # averaging overlapping result
        for ssr in semantic_search_result[window_size][0]:
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


query_maximum = corpus
window_sizes_maximum = [len(granularized_corpus["raw"])]
granularized_corpus_maximum = get_granularized_corpus(
    corpus, granularity, window_sizes_maximum, nlp)
search_result_maximum = search(model_name, scoring_technique,
                               query_maximum, window_sizes_maximum, granularized_corpus_maximum)

search_result = search(model_name, scoring_technique,
                       query, window_sizes, granularized_corpus)

query_minimum = ""
window_sizes_minimum = [0]
granularized_corpus_minimum = get_granularized_corpus(
    corpus, granularity, window_sizes_minimum, nlp)
search_result_minimum = search(model_name, scoring_technique,
                               query_minimum, window_sizes_minimum, granularized_corpus_minimum)


@st.cache
def get_filtered_search_result(percentage, granularized_corpus, search_result, granularity):
    print_corpus = granularized_corpus["raw"][:]
    shaped_raw_result = []
    top_k = int(np.ceil(len(print_corpus)*percentage))
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

        shaped_raw_result.append(
            {"corpus_id": key, "score": val["score_mean"]})

        # annotated = "\u0332".join(print_corpus[key])
        # annotated = "<font color='red'>{}</font>".format(print_corpus[key].splitlines())
        annotated = "<mark style='background-color: lightgreen'>{}</mark>".format(
            print_corpus[key])
        print_corpus[key] = annotated

    if(granularity == 'sentence' or granularity == 'word'):
        print_corpus = " ".join(print_corpus)
    elif(granularity == 'paragraph'):
        print_corpus = "\n".join(print_corpus)

    print_corpus = '<br />'.join(print_corpus.splitlines())

    return {"print_corpus": print_corpus, "shaped_raw": shaped_raw_result, "score_mean": score_mean}


filtered_search_result_minimum = get_filtered_search_result(
    percentage, granularized_corpus_minimum, search_result_minimum, granularity)
filtered_search_result = get_filtered_search_result(
    percentage, granularized_corpus, search_result, granularity)
filtered_search_result_maximum = get_filtered_search_result(
    percentage, granularized_corpus_maximum, search_result_maximum, granularity)

t1 = time.time()

st.subheader("Output process duration")
st.write("{} s".format(t1-t0))

st.subheader("Output score mean")
st.caption(
    "Metric to determine how sure the context of query is in the highlighted document.")
st.write("Raw: {}".format(filtered_search_result["score_mean"]))
st.write("Minimum: {}".format(filtered_search_result_minimum["score_mean"]))
st.write("Maximum: {}".format(filtered_search_result_maximum["score_mean"]))
st.write("Percentage of raw to delta minimum-maximum ratio: {}%".format(
    filtered_search_result["score_mean"]/(filtered_search_result_maximum["score_mean"]-filtered_search_result_minimum["score_mean"])*100))

st.subheader("Output content")
st.write(filtered_search_result["print_corpus"], unsafe_allow_html=True)

st.subheader("Raw semantic search results")
st.caption("corpus_id is the index of the word, sentence, or paragraph. score is mean of overlapped windowed corpus from raw scores by similarity scoring between the query and the document.")
st.write(filtered_search_result["shaped_raw"])

st.subheader("Results of granularized corpus (segmentation/tokenization)")
st.caption("This shows the representation that the webapp gets of the input document. Useful for debugging if you get strange output.")
st.write(granularized_corpus["raw"])
