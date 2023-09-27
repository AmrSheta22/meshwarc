import pandas as pd
import numpy as np
import json
import html
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import spacy
from spacy.language import Language
import multiprocessing
from spacy_langdetect import LanguageDetector
import pickle
from transformers import AutoTokenizer, AutoModel, utils
from transformers import logging
import torch
from sentence_transformers import SentenceTransformer
import umap
import rgr40
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import ConvexHull
import os
from functools import partial
from ctfidfvectorizer import CTFIDFVectorizer
try:
    import cupy as cp
except ImportError:
    import numpy as cp
    print("cupy not installed, using numpy instead")


logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def filter_sentences(sentences):
    filtered = []
    for sentence in sentences:
        # to be transfered
        sentence = " ".join(html.unescape(sentence).split())
        sentence = sentence.replace("\n", "")
        # to be transfered
        if ("copyright" not in sentence.lower()) and (
            len(html.unescape(sentence).split(" ")) > 5
        ):
            filtered.append(sentence)
    return filtered


def cosine_similarity_matrix_gpu(X):
    # divide by the norm
    norm_X = cp.linalg.norm(X, axis=1, keepdims=True)
    normalized_X = X / norm_X
    # calc the sim
    similarity_matrix = cp.dot(normalized_X, normalized_X.T)
    return similarity_matrix


def cosine_similarity_matrix_cpu(X):
    # divide by the norm
    norm_X = np.linalg.norm(X, axis=1, keepdims=True)
    normalized_X = X / norm_X
    # calc the sim
    similarity_matrix = np.dot(normalized_X, normalized_X.T)
    return similarity_matrix


# get the similar indecies
def get_similar(cosine_sim_matrix, threshold=0.33, below_or_above="more"):
    out = {}
    iu1 = np.tril_indices(len(cosine_sim_matrix))
    if below_or_above == "more":
        cosine_sim_matrix[iu1] = -2
        cond = lambda x: x > threshold
    else:
        cosine_sim_matrix[iu1] = 2
        cond = lambda x: x < threshold
    for i, row in enumerate(cosine_sim_matrix):
        above_value = [x for x in row if cond(x)]
        above_index = np.where(cond(row))[0]
        out[i] = {"indcies": above_index, "values": above_value}
    return out


# get the dataframe
def get_similar_dataframe(out):
    records = []
    for i in list(out.keys()):
        for index, value in zip(
            out[list(out.keys())[i]]["indcies"], out[list(out.keys())[i]]["values"]
        ):
            records.append([i, index, value])
    similar_df = pd.DataFrame(
        records, columns=['Source', "Target", 'Weight']
    )
    similar_df = similar_df[similar_df["similarity_value"] < 0.98]
    similar_df.reset_index()
    return similar_df


def detect_lang(text, nlp):
    doc = nlp(text[:200])
    if doc._.language["language"] == "en":
        return True
    else:
        return False


def lemmatize_trans(translated_p, nlp):
    if len(translated_p) == 0:
        return translated_p
    if len(translated_p) >= 5000:
        translated_p = translated_p[:5000]
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    i = translated_p
    doc = nlp(i)
    x = " ".join([token.lemma_ for token in doc if not token.is_stop])
    return x


def detect_en_paragraphs(p_tags, nlp):
    partial_func = partial(detect_lang, nlp=nlp)
    with multiprocessing.Pool() as pool:
        en_only_text = np.array(pool.map(partial_func, p_tags))
        print("finished detecting")
    return en_only_text


def lemmatize_and_stopwords(to_lemma, nlp):
    partial_func = partial(lemmatize_trans, nlp=nlp)
    with multiprocessing.Pool() as pool:
        lemmatized_paragraphs = pool.map(partial_func, to_lemma)
    return lemmatized_paragraphs


def tfidf_data_prep(lemmatized_paragraphs, en_flag):
    tfidf_data = []
    count = 0
    for i in en_flag:
        if i:
            tfidf_data.append(lemmatized_paragraphs[count])
            count += 1
        else:
            tfidf_data.append("\n")
    return tfidf_data


def tfidf_top_dict(tfidf_kw, top_n, wanted_clusters):
    all_tfidf = {}
    for i, k in enumerate(tfidf_kw):
        # convert index to wanted cluster:
        all_tfidf[i] = k[:top_n]
    return all_tfidf


def load_model_tokenizer(model_path):
    model = AutoModel.from_pretrained(model_path, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def text_tokenization(input_text, model, tokenizer, device):
    batch_encoding = tokenizer.encode_plus(input_text, return_tensors="pt")
    tokenized_inputs = batch_encoding["input_ids"]
    tokenized_inputs = tokenized_inputs[:, :512].to(device)
    outputs = model(tokenized_inputs)  # Run model
    attention = outputs[-1]  # Retrieve attention from model outputs
    return attention, tokenized_inputs.cpu(), batch_encoding


def calculate_total_attention(attention):
    layer_sums = np.zeros((1, attention[0][0][0].shape[0]))
    for layer in attention:
        head_sums = np.zeros((1, layer[0][0].shape[1]))
        for head in layer[0].cpu():
            head = head.detach().numpy()
            head_sums += np.sum(head, axis=0)
        layer_sums += head_sums
    return layer_sums[0]


def filter_tokens(inputs, layer_sums):
    ids = inputs[0].detach().numpy()
    out = [
        101,
        102,
        1010,
        1011,
        1012,
        100,
        1005,
        1025,
        1026,
        1027,
        1006,
        1007,
        1031,
        1032,
        1000,
    ]
    mask1 = np.ones(ids.shape, dtype=bool)
    for i in range(len(mask1)):
        if ids[i] in out:
            mask1[i] = 0
    ids = ids[mask1]
    layer_sums = layer_sums[mask1]
    return ids, layer_sums, mask1


def arbitrary_threshold(layer_sums, ids, threshold=70):
    # get 90 percentile of layer sums
    if layer_sums.size > 0:
        threshold = np.percentile(layer_sums, threshold)
    mask2 = np.zeros(layer_sums.shape, dtype=bool)
    for i, k in enumerate(layer_sums):
        if k > threshold:
            mask2[i] = 1
    ids = ids[mask2]
    layer_sums = layer_sums[mask2]
    return ids, layer_sums, mask2


def get_word_indices(mask1, mask2):
    indices = np.arange(0, len(mask1))
    indices = indices[mask1]
    indices = indices[mask2]
    return indices


def get_corresponding_spans(batch_encoding, indices):
    all_spans = []
    for i in indices:
        lis = [batch_encoding.token_to_chars(i)[0], batch_encoding.token_to_chars(i)[1]]
        all_spans.append(lis)
    return all_spans


def spans_to_words(all_spans, input_text):
    words = []
    for i in all_spans:
        words.append(input_text[i[0] : i[1]])
    return words


def is_string_containing_digit(value: str) -> bool:
    return any(char.isdigit() for char in value)


# getting the words from all all spans
def get_ptags_attention(all_all_spans, p_tags):
    p_tags_attention = []
    for num, all_spans in enumerate(all_all_spans):
        p_tags_spanned = []
        end = 0
        for i in range(len(all_spans)):
            if all_spans[i][0] < end:
                continue
            first = p_tags[num].rfind(" ", 0, all_spans[i][0]) + 1
            end = p_tags[num].find(" ", all_spans[i][1] - 1)
            # p_tags_spanned.append(p_tags[num][all_spans[i][0]:all_spans[i][1]])
            imp_word = p_tags[num][first:end]
            if is_string_containing_digit(imp_word):
                continue
            p_tags_spanned.append(imp_word)

        p_tags_attention.append(" ".join(p_tags_spanned))
    return p_tags_attention


def reset_cluster_index(cluster):
    reseted = list(range(len(np.unique(cluster))))
    for i, k in zip(reseted, np.unique(cluster)):
        cluster[np.where(cluster == k)] = i
    return cluster


def approximate_convex_hull(points, k):
    # initialize the hull with the first point
    hull = [points[0]]
    # loop until k vertices are found or no more points are left
    while len(hull) < k and len(points) > 0:
        # compute the centroid of the current hull
        centroid = np.mean(hull, axis=0)
        # find the point that is farthest from the centroid
        distances = np.linalg.norm(points - centroid, axis=1)
        farthest = np.argmax(distances)
        hull.append(points[farthest])
        points = np.delete(points, farthest, axis=0)
    return np.array(hull)


########################### MAIN ##################################
def meshwarc(
    path,
    top_n=10,
    similarity_threshold=0.37,
    divisable_cluster_size=100,
    number_of_clusters=200,
    minimum_cluster_size=50,
    percentage_filtered=90,
    embedding=None,
    save_embeddings=True,
    gpu_cosine_sim=True,
    sim=True,
    all_spans_unknown=True,
    cluster_unknown=True,
    en_flag_unknown=True,
    lematized_unknown=True,
    dbscan_unknown=True,
):
    model = SentenceTransformer(
        "sentence-transformers/distiluse-base-multilingual-cased-v2", device="cuda"
    )
    print("started")
    data_ids = []
    p_tags = []
    heads = []
    with open(path) as f:
        data = json.load(f)
        for i in data:
            id = list(i.keys())[0]
            sentence = "\n".join(
                filter_sentences(html.unescape(i[list(i.keys())[0]]["p_tags"]))
            )
            head = i[list(i.keys())[0]]["head"].replace("\n", "")
            heads.append(head)
            p_tags.append(sentence)
            data_ids.append(id)

    if embedding == None:
        print("started embedding ... ")
        p_embeddings = model.encode(p_tags)
        df = pd.DataFrame(p_embeddings)
        if save_embeddings:
            df.to_csv("embeddings.csv")
        df.index = data_ids
        print("finished embedding ... ")
    else:
        df = pd.read_csv(embedding)
    sc = StandardScaler()
    df_scaled = sc.fit_transform(df)
    pca = PCA(n_components=400)
    df_scaled_pca = pca.fit_transform(df_scaled)

    # dbscan filtering
    print("dbscan filtering ...")
    if dbscan_unknown:
        clustering = DBSCAN(eps=20, min_samples=7).fit(df_scaled_pca)
        to_remove = np.where(clustering.labels_ != -1)[0]
        with open("to_remove.pkl", "wb") as f:
            pickle.dump(to_remove, f)
    else:
        with open("to_remove.pkl", "rb") as f:
            to_remove = pickle.load(f)
    df_scaled_filtered = []
    for i, k in enumerate(df_scaled_pca):
        if i not in to_remove:
            df_scaled_filtered.append(k)
    for i in reversed(to_remove):
        del p_tags[i]
        del data_ids[i]
        del heads[i]
    print(len(p_tags))
    print("finished dbscan filtering")
    if gpu_cosine_sim:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        similarity_matrix_gpu = cosine_similarity_matrix_gpu(
            cp.asarray(df_scaled_filtered)
        )
        cosine_sim_matrix = cp.asnumpy(similarity_matrix_gpu)
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    else:
        cosine_sim_matrix = cosine_similarity_matrix_cpu(np.array(df_scaled_filtered))
    # the similarity graph
    print("constructing the graph")
    if sim:
        out = get_similar(cosine_sim_matrix,threshold=similarity_threshold)
        similar_df = get_similar_dataframe(out)
    else:
        similar_df = pd.read_csv("similar_df.csv")
    print("finished outputing the final list")
    # get p_tags_attention
    print("started clustering")
    if cluster_unknown:
        embedding = umap.UMAP(n_components=4).fit_transform(df_scaled_filtered)
        cluster, cluster_sizes, cluster_centers = rgr40.the_rich_gets_richer(
            embedding, number_of_clusters, divisable_number = divisable_cluster_size
        )
        unique, counts = np.unique(cluster, return_counts=True)
        unique_bad = unique[np.where(counts < minimum_cluster_size)]
        nocluster_ind = np.where(np.isin(cluster, unique_bad))
        cluster[nocluster_ind] = np.negative(np.ones((len(nocluster_ind),)))
        cluster = reset_cluster_index(cluster)

        with open("cluster.pkl", "wb") as f:
            pickle.dump(cluster, f)
    else:
        with open("cluster.pkl", "rb") as f:
            cluster = pickle.load(f)
    print("finished clustering and started getting attention")

    # Load the model and tokenizer
    if all_spans_unknown:
        model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_attentions=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        all_all_spans = []
        for i in p_tags:
            attention, tokenized_inputs, batch_encoding = text_tokenization(
                i, model, tokenizer, device
            )
            layer_sums = calculate_total_attention(attention)
            del attention
            ids, layer_sums, mask1 = filter_tokens(tokenized_inputs, layer_sums)
            del tokenized_inputs
            ids, layer_sums, mask2 = arbitrary_threshold(layer_sums, ids, percentage_filtered)
            del layer_sums
            indices = get_word_indices(mask1, mask2)
            all_spans = get_corresponding_spans(batch_encoding, indices)
            all_all_spans.append(all_spans)

        with open("my_list.pkl", "wb") as f:
            pickle.dump(all_all_spans, f)
    else:
        with open("my_list.pkl", "rb") as f:
            all_all_spans = pickle.load(f)

    p_tags_attention = get_ptags_attention(all_all_spans, p_tags)

    print("finished attention and started lemmatization and tfidf data prep")
    ################

    """  God save our gracious King! 
          Long live our noble King! 
             God save the King!
            Send him victorious,
             Happy and glorious,
           Long to reign over us,
             God save the King.       """
    nlp = spacy.load("en_core_web_sm")  # 1
    nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)  # 2
    if en_flag_unknown:
        en_flag = detect_en_paragraphs(p_tags, nlp)
        with open("en_flag.pkl", "wb") as f:
            pickle.dump(en_flag, f)
    else:
        with open("en_flag.pkl", "rb") as f:
            en_flag = pickle.load(f)

    if lematized_unknown:
        to_lemma = []
        for i, k in zip(p_tags_attention, en_flag):
            if k:
                to_lemma.append(i)
        lemmatized_paragraphs = lemmatize_and_stopwords(to_lemma, nlp)
        with open("lemmatized.pkl", "wb") as f:
            pickle.dump(lemmatized_paragraphs, f)
    else:
        with open("lemmatized.pkl", "rb") as f:
            lemmatized_paragraphs = pickle.load(f)

    tfidf_data = tfidf_data_prep(lemmatized_paragraphs, en_flag)

    # once with tfidf and once with ctfidf

    # first  with ctfidf
    print("started tfidf and ctfidf")
    cluster = reset_cluster_index(cluster)
    docs = pd.DataFrame({"Document": tfidf_data, "Class": cluster})
    docs_per_class = docs.groupby(["Class"], as_index=False).agg({"Document": " ".join})
    # Create bag of words
    count_vectorizer = CountVectorizer().fit(docs_per_class.Document)
    count = count_vectorizer.transform(docs_per_class.Document)
    words = count_vectorizer.get_feature_names()
    # Extract top 10 words
    ctfidf = CTFIDFVectorizer().fit_transform(count, n_samples=len(docs)).toarray()
    # words_per_class_c = {label: [words[index] for index in ctfidf[label].argsort()[-top_n:]] for label in docs_per_class.Class}
    words_per_class_c = {}
    for label in docs_per_class.Class:
        top_words = []
        x = ctfidf[label].argsort()
        for index in ctfidf[label].argsort()[-top_n:]:
            top_words.append(words[index])
        words_per_class_c[label] = top_words
    # now with tfidf
    vectorizer = TfidfVectorizer(use_idf=True, max_df=0.5, min_df=1, ngram_range=(1, 1))
    vectors = vectorizer.fit_transform(docs_per_class["Document"])
    dict_of_tokens = {i[1]: i[0] for i in vectorizer.vocabulary_.items()}
    tfidf_vectors = []  # all deoc vectors by tfidf
    for row in vectors:
        tfidf_vectors.append(
            {
                dict_of_tokens[column]: value
                for (column, value) in zip(row.indices, row.data)
            }
        )
    doc_sorted_tfidfs = []  # list of doc features each with tfidf weight
    # sort each dict of a document
    for dn in tfidf_vectors:
        newD = sorted(dn.items(), key=lambda x: x[1], reverse=True)
        newD = dict(newD)
        doc_sorted_tfidfs.append(newD)
    tfidf_kw = []  # get the keyphrases as a list of names without tfidf values
    for doc_tfidf in doc_sorted_tfidfs:
        ll = list(doc_tfidf.keys())
        tfidf_kw.append(ll)
    words_per_class = tfidf_top_dict(tfidf_kw, top_n, cluster)

    # cluster df
    cluster_df = pd.DataFrame({"URL": data_ids, "Label": heads, "cluster": cluster})
    cluster_df["ID"] = cluster_df.index
    ctfidf_keywords = pd.DataFrame(
        words_per_class_c.items(), columns=["cluster", "keywords"]
    )
    tfidf_keywords = pd.DataFrame(
        words_per_class.items(), columns=["cluster", "keywords"]
    )
    merged_tfidf = tfidf_keywords.merge(ctfidf_keywords, on='cluster')

    cluster_df['cluster_x'] = cluster_df['cluster'].apply(lambda x: merged_tfidf['keywords_x'][x])
    cluster_df['cluster_y'] = cluster_df['cluster'].apply(lambda x: merged_tfidf['keywords_y'][x])

    return similar_df, cluster_df, ctfidf_keywords, tfidf_keywords, merged_tfidf