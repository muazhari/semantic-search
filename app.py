from re import split
import streamlit as st
import more_itertools
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np

model_name = st.text_area(
    "Enter the name of the pre-trained model from sentence transformers that we are using for summarization", value="paraphrase-MiniLM-L3-v2")
st.caption("This will download a new model, so it may take awhile or even break if the model is too large")
st.caption("See the list of pre-trained models that are available here! https://www.sbert.net/docs/pretrained_models.html")

granularity = st.radio(
    "What level of granularity do you want to summarize at?", ('sentence', 'word', 'paragraph'))

query = st.text_area('Enter a query')
corpus = st.text_area('Enter a document')
percentage = st.number_input(
    "Enter the percentage of the text you want highlighted", max_value=1, min_value=0, value=0.3)
window_sizes = [int(x) for x in st.text_area(
    'Enter a list of window sizes that seperated by space').split(" ")]

nlp = spacy.load('en_core_web_sm')

embedder = SentenceTransformer(model_name, device='cuda')
query_embedding = embedder.encode(query, convert_to_tensor=True)
granularized_corpus = []  # ["", ...]
windowed_granularized_corpus = {}  # {window_size: [("",...), ...]}
# {window_size: [({"corpus": "", "index": 0}, ...), ...]}
windowed_granularized_corpus_indexed = {}
semantic_search_result = {}  # {window_size: {"corpus_id": 0, "score": 0}}

if granularity == "sentence":
    doc = nlp(corpus)
    granularized_corpus = [str(sent) for sent in doc.sents]
elif granularity == "word":
    doc = nlp(corpus)
    for sent in doc.sents:  # sentece break
        words_per_sentence = [word for word in str(sent).split()]
        granularized_corpus += words_per_sentence
elif granularity == "paragraph":
    granularized_corpus = corpus.splitlines()

for window_size in window_sizes:
    windowed_granularized_corpus[window_size] = []
    windowed_granularized_corpus_indexed[window_size] = []
    for wgc in more_itertools.windowed(enumerate(granularized_corpus), window_size):
        item_without_index = []
        item_with_index = []
        for item in wgc:
            item_without_index.append(item[1])
            item_with_index.append({"index": item[0], "corpus": item[1]})
        windowed_granularized_corpus[window_size].append(item_without_index)
        windowed_granularized_corpus_indexed[window_size].append(
            item_with_index)

for window_size in window_sizes:
    corpus_len = len(windowed_granularized_corpus[window_size])
    if(window_size == 1):
        corpus_embeddings = embedder.encode(
            [x[0] for x in windowed_granularized_corpus[window_size]], convert_to_tensor=True)
    else:
        corpus_embeddings = embedder.encode(
            windowed_granularized_corpus[window_size], convert_to_tensor=True)
    semantic_search_result[window_size] = util.semantic_search(
        query_embedding, corpus_embeddings, top_k=corpus_len)

final_semantic_search_result = {}  # {corpus_id: {"score_mean": 0, count: 0}}
for window_size in window_sizes:
    for ssr in semantic_search_result[window_size][0]:
        for ssrw in windowed_granularized_corpus_indexed[window_size][ssr["corpus_id"]]:
            source_corpus_index = ssrw["index"]
            if(final_semantic_search_result.get(source_corpus_index, None) is None):
                final_semantic_search_result[source_corpus_index] = {
                    "count": 1, "score_mean": ssr["score"]}
            else:
                old_count = final_semantic_search_result[source_corpus_index]["count"]
                new_count = old_count + 1
                final_semantic_search_result[source_corpus_index]["count"] = new_count
                old_score_mean = final_semantic_search_result[source_corpus_index]["score_mean"]
                new_score_mean = old_score_mean + \
                    ((ssr["score"]-old_score_mean)/new_count)
                final_semantic_search_result[source_corpus_index]["score_mean"] = new_score_mean

print_corpus = granularized_corpus[:]
cleaned_raw_result = []
top_k = int(np.ceil(len(print_corpus)*percentage))
total_score_mean = total_count = 0
for key, val in sorted(final_semantic_search_result.items(), key=lambda x: x[1]["score_mean"], reverse=True)[:top_k]:
    old_count = total_count
    new_count = old_count + 1
    total_count = new_count
    total_score = val["score_mean"]
    old_score_mean = total_score_mean
    new_score_mean = old_score_mean + ((total_score-old_score_mean)/new_count)
    total_score_mean = new_score_mean

    cleaned_raw_result.append(
        {"corpus_id": key, "score_mean": total_score_mean, "score": total_score})

    annotated = "\u0332".join(print_corpus[key])
    print_corpus[key] = annotated

print([total_score_mean, total_count])
" ".join(print_corpus)


st.subheader("Total score mean")
st.write(" ".join(total_score_mean))

st.subheader("Output summary")
st.write(" ".join(print_corpus))

st.subheader("Raw semantic search results")
st.caption("corpus_id is the number of the word, sentence, or paragraph. Score is the raw cosine similarty score between the document and the query")
st.write(cleaned_raw_result)

st.subheader("Results of granularized corpus (segmentation/tokenization)")
st.caption("This shows the representation that the webapp gets of the input document. Useful for debugging if you get strange output")
st.write(granularized_corpus)
