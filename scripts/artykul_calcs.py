#!/usr/bin/env python
"""Utilities for reproducing the analysis from 'Świadectwa ewangelikalne i katolickie w perspektywie korpusowej analizy dyskursu' (Socjolingwistyka, Tom 34, 2020) by Albert Leśniak and Zbigniew Pasek."""
# coding: utf-8

import sys
from pathlib import Path




DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_DIR = DATA_DIR / "corpus"
prot_path = CORPUS_DIR / "protestant_corpus.pickle"
cat_path = CORPUS_DIR / "catolic_corpus.pickle"
if not prot_path.exists() or not cat_path.exists():
    print("Expected corpus files at data/corpus/protestant_corpus.pickle and data/corpus/catolic_corpus.pickle.")
    print("Place the files in data/corpus/ to run the full analysis.")
    sys.exit(0)

import re
import os
import pickle
import math
import csv
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations



# ### Przekształcenie danych tekstowych w Bag of Words



tags_to_exclude = ['interp']
with open(DATA_DIR / "stopwords.txt", "r", encoding="utf-8") as f:
    stopwords = f.read().splitlines()



with open(prot_path, 'rb') as f:
    prot = pickle.load(f)

with open(cat_path, 'rb') as f:
    cats = pickle.load(f)



for id_, doc in cats.items():
    data = [item[0] for item in doc]
    if 'zaproszenie' in data:
        print(data)
        print('=='*25)



for doc, content in prot.items():
    lemmatized = []
    for segment in content:
        if segment[2] != 'interp' and segment[1] not in stopwords:
            lemmatized.append(segment[1])
    with open(CORPUS_DIR / 'lemmatized' / 'protestanckie' / f'{doc}.txt', 'w') as f:
        f.write(" ".join(lemmatized))



# klasyfikacja płci
def gender_classifier(doc):
    f, m = 0, 0
    for i, item in enumerate(doc):
        if 'aglt:sg' in item[2]:
            x = re.match(r'praet:sg:(\w{1,2}):', doc[i-1][2])
            if x and x.group(1) == 'f':
                f += 1
            elif x and x.group(1) == 'm1':
                m += 1
            else:
                continue
    if f > m:
        gender = 'K'
    elif m > f:
        gender = 'M'
    else:
        gender = 'UNKNOW'
    return gender



cats_gender = {}
for id_, doc in cats.items():
    cats_gender[id_] = gender_classifier(doc)



t = len(cats_gender.items())
m = list(cats_gender.values()).count("M")
k = list(cats_gender.values()).count("K")
u = list(cats_gender.values()).count("UNKNOW")
print(f'Total: {t}')
print(f'M: {m} ({round((m*100)/t, 2)}%)')
print(f'K: {k} ({round((k*100)/t, 2)}%)')
print(f'U: {u} ({round((u*100)/t, 2)}%)')



prot_gender = {}
for id_, doc in prot.items():
    prot_gender[id_] = gender_classifier(doc)



t = len(prot_gender.items())
m = list(prot_gender.values()).count("M")
k = list(prot_gender.values()).count("K")
u = list(prot_gender.values()).count("UNKNOW")
print(f'Total: {t}')
print(f'M: {m} ({round((m*100)/t, 2)}%)')
print(f'K: {k} ({round((k*100)/t, 2)}%)')
print(f'U: {u} ({round((u*100)/t, 2)}%)')



unknowns = []
for cats_id, gender in cats_gender.items():
    if gender == 'UNKNOW':
        unknowns.append(cats_id)



un = iter(unknowns)



next(un)



cats['196']



prot_corpus = {}
for prot_nr, prot_segments in prot.items():
    prot_corpus[prot_nr] = [t[1] for t in prot_segments if t[2] not in tags_to_exclude]



cats_corpus = {}
for cats_nr, cats_segments in cats.items():
    cats_corpus[cats_nr] = [t[1] for t in cats_segments if t[2] not in tags_to_exclude]



to_delete = []
for doc_id, doc in cats_corpus.items():
    if len(doc) < 1:
        to_delete.append(doc_id)
for doc_id in to_delete:
    del cats_corpus[doc_id]



prot_docs_nr = len(prot_corpus.keys())
prot_tokens = sum([len(doc) for doc in prot_corpus.values()])
prot_lemmas = []
for doc in prot_corpus.values():
    for lemma in doc:
        prot_lemmas.append(lemma)
prot_lemmas = set(prot_lemmas)
prot_avg = prot_tokens / prot_docs_nr
prot_min = min([len(doc) for doc in prot_corpus.values()])
prot_max = max([len(doc) for doc in prot_corpus.values()])



cats_docs_nr = len(cats_corpus.keys())
cats_tokens = sum([len(doc) for doc in cats_corpus.values()])
cats_lemmas = []
for doc in cats_corpus.values():
    for lemma in doc:
        cats_lemmas.append(lemma)
cats_lemmas = set(cats_lemmas)
cats_avg = cats_tokens / cats_docs_nr
cats_min = min([len(doc) for doc in cats_corpus.values()])
cats_max = max([len(doc) for doc in cats_corpus.values()])



prot_bow = defaultdict(int)



for doc in prot_corpus.values():
    for token in doc:
        prot_bow[token] += 1



prot_bow



cats_bow = defaultdict(int)



for doc in cats_corpus.values():
    for token in doc:
        cats_bow[token] += 1



cats_tokens

# ### Wygenerowanie słów kluczowych



def compute_log_likelihood(o_11, o_21, corpus_c, corpus_r):    
    e_11 = (corpus_c * (o_11 + o_21)) / (corpus_c + corpus_r)
    e_21 = (corpus_r * (o_11 + o_21)) / (corpus_c + corpus_r)
    try:
        log_likelihood = 2 * ((o_11 * math.log(o_11 / e_11)) + (o_21 * math.log(o_21 / e_21)))
    except:
        log_likelihood = 0
    return log_likelihood



keywords_loglikelihood = {}
for keyword in cats_bow.keys():
    if prot_bow[keyword] == 0:
        continue
    log = compute_log_likelihood(o_11= cats_bow[keyword], o_21=prot_bow[keyword], corpus_c=cats_tokens, corpus_r=prot_tokens)
    if log > 0.05 and keyword not in stopwords:
        keywords_loglikelihood[keyword] = log



keyness_sorted = sorted(keywords_loglikelihood.items(), key=lambda kv: kv[1], reverse=True)



keyness_sorted



with open(DATA_DIR / 'keywords_log-likelihood.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['keyword','Log-likelihood_score'])
    for row in keyness_sorted:
        csv_out.writerow(row)



cats_keywords = []
prot_keywords = []



for keyword, keyness in keyness_sorted:
    cat_score = (cats_bow[keyword] / cats_tokens) * 1000
    prot_score = (prot_bow[keyword] / prot_tokens) * 1000
    if keyword in stopwords:
        continue
    if cat_score > prot_score:
        cats_keywords.append((keyword, keyness, cat_score))
    else:
        prot_keywords.append((keyword, keyness, prot_score))



cats_keywords



with open(DATA_DIR / 'catolic_keywords_per1000.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['keyword','log-likelihood', 'score'])
    for row in cats_keywords:
        csv_out.writerow(row)



prot_keywords



with open(DATA_DIR / 'evangelical_keywords_per1000.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['keyword','log-likelihood', 'score'])
    for row in prot_keywords:
        csv_out.writerow(row)

# ### Wygenerowanie kolokacji dla pierwszych 50 słów kluczowych



def loglikelihood(co_occ, node_freq, colocat_freq, tokens):
    a = co_occ
    b = colocat_freq - co_occ
    c = node_freq - co_occ
    d = tokens - (a + b + c)
    try:
        result = 2 * (a * math.log(a) + b * math.log(b) + c * math.log(c)
                      + d * math.log(d)
                      - (a + b) * math.log(a + b) - (a + c) * math.log(a + c)
                      - (b + d) * math.log(b + d) - (c + d) * math.log(c + d)
                      + (a + b + c + d) * math.log(a + b + c + d))
    except:
        result = None
    return result



def collocates_counter(span, corpus, keyword):
    collocates = defaultdict(int)
    for text in corpus.values():
        keyword_indexes = [i for i, token in enumerate(text) if token == keyword]
        if len(keyword_indexes) > 0:
            for kw_index in keyword_indexes:
                if kw_index >= span:
                    for c in text[kw_index-span:kw_index]:
                        collocates[c] += 1
                    for c in text[kw_index+1:kw_index+1+span]:
                        collocates[c] += 1
                else:
                    for c in text[:kw_index]:
                        collocates[c] += 1
                    for c in text[kw_index+1:kw_index+1+span]:
                        collocates[c] += 1
    return collocates



def get_collocates_list(node_keyword, corpus, span):
    node_keyword_colocates = collocates_counter(span=span, corpus=corpus, keyword=node_keyword)
    weighted_collocates = []
    for colocate, co_occurance in node_keyword_colocates.items():
        col_score = loglikelihood(co_occ=co_occurance,
                            node_freq=cats_bow[node_keyword],
                            colocat_freq=cats_bow[colocate],
                            tokens=cats_tokens)
        if col_score != None:
            weighted_collocates.append((colocate, col_score))
        else:
            pass
    return sorted(weighted_collocates, key=lambda x: x[1], reverse=True)



top_keywords = list(zip(*keyness_sorted))[0][:100]



span = 2
with pd.ExcelWriter(DATA_DIR / 'collocations_loglikelihood.xlsx', engine='xlsxwriter') as writer:
    for node_keyword in top_keywords:
        df_prot = pd.DataFrame(get_collocates_list(node_keyword=node_keyword,
                                                   corpus=prot_corpus,
                                                   span=span),
                               columns=['prot_collcates', 'prot_loglikelihood'])
        df_cats = pd.DataFrame(get_collocates_list(node_keyword=node_keyword,
                                                   corpus=cats_corpus,
                                                   span=span),
                               columns=['cats_collcates', 'cats_loglikelihood'])
        df = pd.concat([df_prot, df_cats], axis=1)
        df.to_excel(writer, sheet_name=node_keyword, index=False)
writer.save()
writer.close()

# ### Wygenerowanie grafu kolokacji dla 15 najważniejszych słów kluczowych w poszczególnych korpusach



def collocates_without_stopwords(node_keyword, corpus, span, stopwords, min_ll):
    collocates = [(col[0], col[1]) for col in get_collocates_list(node_keyword=node_keyword, corpus=corpus, span=span)
                 if col[0] not in stopwords and col[1] >= min_ll]            
    return collocates

# ##### Korpus katolicki



def catolic_network(node_keywords, min_ll, span, firsts_cols, seconds_cols):
    G = nx.Graph()
    for node_keyword in node_keywords:
        G.add_node(node_keyword,
                   group='group_0',
                   score=cats_bow[node_keyword] / cats_tokens,
                   keyness=compute_log_likelihood(o_11= cats_bow[node_keyword], o_21=prot_bow[node_keyword], corpus_c=cats_tokens, corpus_r=prot_tokens))
    for node_keyword in node_keywords:
        collocates = collocates_without_stopwords(node_keyword=node_keyword, corpus=cats_corpus, stopwords=stopwords, span=span, min_ll=min_ll)[:firsts_cols]
        for coll, ll in collocates:
            if not G.has_node(coll):
                G.add_node(coll,
                           group='group_1',
                           score=cats_bow[coll] / cats_tokens,
                           keyness=compute_log_likelihood(o_11= cats_bow[coll], o_21=prot_bow[coll], corpus_c=cats_tokens, corpus_r=prot_tokens))
            G.add_edge(node_keyword, coll, weight=ll)
    second_edges= list(G.edges())
    for _, second_node in second_edges:
        second_colls = collocates_without_stopwords(node_keyword=second_node, corpus=cats_corpus, span=span, stopwords=stopwords, min_ll=min_ll)[:seconds_cols]
        for second_coll, ll in second_colls:
            if not G.has_node(second_coll):
                G.add_node(second_coll,
                           group='group_2',
                           score=cats_bow[second_coll] / cats_tokens,
                           keyness=compute_log_likelihood(o_11= cats_bow[second_coll], o_21=prot_bow[second_coll], corpus_c=cats_tokens, corpus_r=prot_tokens))
            G.add_edge(second_node, second_coll, weight=ll)
    return G



cats_keys = [x[0] for x in cats_keywords][:50]



G_cat = catolic_network(node_keywords=cats_keys, min_ll=15, span=5, firsts_cols=5, seconds_cols=0)



len(G_cat.nodes)



nx.draw_networkx(G_cat, with_labels=True)
G_cat.show()



nx.write_gexf(G_cat, "catholic_s5_ll15_f5_s0.gexf")















G_cat = catolic_network(node_keywords=cats_keys, min_ll=15, span=5)
nx.draw_networkx(G_cat, with_labels=True)
# graph_pos = nx.spring_layout(G_cat)
# nx.draw_networkx_nodes(G_cat, graph_pos, node_size=100, node_color='blue', alpha=0.3)
# nx.draw_networkx_edges(G_cat, graph_pos)
# nx.draw_networkx_labels(G_cat, graph_pos, font_size=10, font_family='sans-serif')
# plt.savefig("plot.png", format = "png", dpi = 1000)
plt.show()



nx.write_gexf(G_cat, "uzdrowienie_choroba_lekarz_ll30_s3.gexf")

# #### Korpus protestancki



def protestant_network(node_keywords, min_ll, span):
    G = nx.Graph()
    for node_keyword in node_keywords:
        G.add_node(node_keyword,
                   group='group_0',
                   score=prot_bow[node_keyword] / prot_tokens,
                   keyness=compute_log_likelihood(o_11= prot_bow[node_keyword], o_21=cats_bow[node_keyword], corpus_c=prot_tokens, corpus_r=cats_tokens))
    for node_keyword in node_keywords:
        collocates = collocates_without_stopwords(node_keyword=node_keyword, corpus=prot_corpus, stopwords=stopwords, span=span, min_ll=min_ll)
        for coll, ll in collocates:
            if not G.has_node(coll):
                G.add_node(coll,
                           group='group_1',
                           score=prot_bow[coll] / prot_tokens,
                           keyness=compute_log_likelihood(o_11= prot_bow[coll], o_21=cats_bow[coll], corpus_c=prot_tokens, corpus_r=cats_tokens))
            G.add_edge(node_keyword, coll, weight=ll)
    second_edges= list(G.edges())
    for _, second_node in second_edges:
        second_colls = collocates_without_stopwords(node_keyword=second_node, corpus=prot_corpus, span=span, stopwords=stopwords, min_ll=min_ll)
        for second_coll, ll in second_colls:
            if not G.has_node(second_coll):
                G.add_node(second_coll,
                           group='group_2',
                           score=prot_bow[second_coll] / prot_tokens,
                           keyness=compute_log_likelihood(o_11= prot_bow[second_coll], o_21=cats_bow[second_coll], corpus_c=prot_tokens, corpus_r=cats_tokens))
            G.add_edge(second_node, second_coll, weight=ll)
    return G



G_prot = protestant_network(node_keywords=['bóg', 'Jezus', 'Biblia', 'życie', 'zbór', 'alkohol', 'Chrystus', 'czytać', 'chrzest', 'społeczność', 'pastor', 'kościół', 'nabożeństwo', 'papieros'], min_ll=80, span=3)
nx.draw_networkx(G_prot, with_labels=True)
plt.show()



nx.write_gexf(G_prot, "pierwsze15_ll80_s3.gexf")



compute_log_likelihood(o_11=cats_bow['czystość'] , o_21=prot_bow['czystość'], corpus_c=cats_tokens, corpus_r=prot_tokens)



def specify_prov(k):
    cat_score = (cats_bow[k] / cats_tokens) * 1000
    prot_score = (prot_bow[k] / prot_tokens) * 1000
    if cat_score > prot_score:
        return 'catholic'
    else:
        return 'protestant'



min_ll=30
span=3
top_keywords = list(zip(*keyness_sorted))[0][:100]



G = nx.Graph()
for node_keyword in top_keywords:
    keyness=compute_log_likelihood(o_11= prot_bow[node_keyword],
                                   o_21=cats_bow[node_keyword],
                                   corpus_c=prot_tokens,
                                   corpus_r=cats_tokens)
    G.add_node(node_keyword,
               group='group_0',
               prov=specify_prov(node_keyword),
               keyness=keyness
               )
for node_keyword in top_keywords:
    collocates = collocates_without_stopwords(node_keyword=node_keyword, corpus=prot_corpus, stopwords=stopwords, span=span, min_ll=min_ll)
    for coll, ll in collocates:
        if not G.has_node(coll):
            G.add_node(coll,
                       group='group_1',
                       prov=specify_prov(coll),
                       keyness=compute_log_likelihood(o_11= prot_bow[coll], o_21=cats_bow[coll], corpus_c=prot_tokens, corpus_r=cats_tokens))
        G.add_edge(node_keyword, coll, weight=ll)
second_edges= list(G.edges())
for _, second_node in second_edges:
    second_colls = collocates_without_stopwords(node_keyword=second_node, corpus=prot_corpus, span=span, stopwords=stopwords, min_ll=min_ll)
    for second_coll, ll in second_colls:
        if not G.has_node(second_coll):
            G.add_node(second_coll,
                       group='group_2',
                       prov=specify_prov(second_coll),
                       keyness=compute_log_likelihood(o_11= prot_bow[second_coll], o_21=cats_bow[second_coll], corpus_c=prot_tokens, corpus_r=cats_tokens))
        G.add_edge(second_node, second_coll, weight=ll)



G.edges



nx.write_gexf(G, "pierwsze50_s3_ll30.gexf")









# ### Notatki



collocates_without_stopwords(node_keyword="niebo", corpus=prot_corpus, stopwords=stopwords, span=5, min_ll=20)



collocates_without_stopwords(node_keyword="czytać", corpus=cats_corpus, stopwords=stopwords, span=5, min_ll=20)



collocates_without_stopwords(node_keyword='czystość', corpus=cats_corpus, stopwords=stopwords, span=5, min_ll=20)

# ## Współczynnik częstości



cats_words = ['chłopak', 'związek', 'czystość', 'ciąża', 'małżeństwo', 'poród', 'dziewczyna', 'seks', 'małżeński', 'ukochana', 'łóżko', 'przytulić', 'zakochać', 'rodzinny', 'kobieta']



df_cats = pd.DataFrame(columns=['Słowo kluczowe', 'Ilość wystąpień w korpusie katolickim', 'Ilość wystąpień w korpusie katolickim w przeliczeniu na 1000 słów', 'Ilość wystąpień w korpusie protestanckim', 'Ilość wystąpień w korpusie protestanckim w przeliczeniu na 1000 słów', 'różnica'])



df_cats.set_index('Słowo kluczowe')



for word in cats_words:
    df_cats = df_cats.append({
        'Słowo kluczowe' : word,
        'Ilość wystąpień w korpusie katolickim': cats_bow[word],
        'Ilość wystąpień w korpusie katolickim w przeliczeniu na 1000 słów' : round(cats_bow[word]/cats_tokens * 1000, 2),
        'Ilość wystąpień w korpusie protestanckim' : prot_bow[word],
        'Ilość wystąpień w korpusie protestanckim w przeliczeniu na 1000 słów' : round(prot_bow[word]/prot_tokens * 1000, 2),
        'różnica' :  round((cats_bow[word]/cats_tokens * 1000) - (prot_bow[word]/prot_tokens * 1000), 2)      
    }, ignore_index=True)



df_cats.sort_values(by=['różnica'], ascending=False, inplace=True)



df_cats
df.to_excel('seksualnosc_tabelka.xlsx')


prot_words = ['alkohol', 'papieros', 'pić', 'palenie', 'picie', 'alkoholik', 'kac', 'narkotyk', 'więzienie', 'wyrok', 'karny']



df_prot = pd.DataFrame(columns=['Słowo kluczowe', 'Ilość wystąpień w korpusie katolickim', 'Ilość wystąpień w korpusie katolickim w przeliczeniu na 1000 słów', 'Ilość wystąpień w korpusie protestanckim', 'Ilość wystąpień w korpusie protestanckim w przeliczeniu na 1000 słów', 'różnica'])



df_prot.set_index('Słowo kluczowe')



for word in prot_words:
    df_prot = df_prot.append({
        'Słowo kluczowe' : word,
        'Ilość wystąpień w korpusie katolickim': cats_bow[word],
        'Ilość wystąpień w korpusie katolickim w przeliczeniu na 1000 słów' : round(cats_bow[word]/cats_tokens * 1000, 2),
        'Ilość wystąpień w korpusie protestanckim' : prot_bow[word],
        'Ilość wystąpień w korpusie protestanckim w przeliczeniu na 1000 słów' : round(prot_bow[word]/prot_tokens * 1000, 2),
        'różnica' :  round((prot_bow[word]/prot_tokens * 1000) - (cats_bow[word]/cats_tokens * 1000), 2)      
    }, ignore_index=True)



df_prot.sort_values(by=['różnica'], ascending=False, inplace=True)



df_prot



prot_bow['narkotyk']



df_prot.to_excel('grzech_tabelka.xlsx')

# ## Konkordancje



def concordance(keyword, span, corpus, corpus_path=CORPUS_DIR / 'swiadectwa_protestanckie'):
    df = pd.DataFrame(columns=['Lewy kontekst', 'Słowo kluczowe', 'Prawy kontekst', 'Plik'])
    for file in [f for f in os.listdir(corpus_path) if f.endswith('.txt')]:
        document = corpus[file.split('.')[0]]
        lemmas = [lemma for token, lemma, tag in document]
        tokens = [token for token, lemma, tag in document]
        indices = [i for i, x in enumerate(lemmas) if x == keyword]
        if len(indices) > 0:
            pattern = ''
            for w in set([tokens[i] for i in indices]):
                pattern += f"\\b{w}.*?\\b|"
            pattern = pattern[:-1]
            with open(corpus_path / file, 'r') as f:
                text = f.read()
            for m in re.finditer(pattern, text):
                if m.start() <= span:
                    left_range = text[:m.start()]
                else:
                    left_range = text[m.start()-span:m.start()]
                if len(text) - m.end() <= span:
                    right_range = text[m.end():]
                else:
                    right_range = text[m.end():m.end()+span+1]
                df = df.append({
                    'Lewy kontekst' : left_range,
                    'Słowo kluczowe' : m.group(0),
                    'Prawy kontekst' : right_range,
                    'Plik' : file
                }, ignore_index=True)
        else:
            continue
    return df



cat_words = ['chłopak', 'związek', 'czystość', 'ciąża', 'małżeństwo', 'poród', 'dziewczyna', 'seks', 'małżeński', 'ukochana', 'łóżko', 'przytulić', 'zakochać', 'rodzinny', 'kobieta']



with open(CORPUS_DIR / 'catolic_corpus.pickle', 'rb') as f:
    catolic_corpus = pickle.load(f)

with open(CORPUS_DIR / 'protestant_corpus.pickle', 'rb') as f:
    protestant_corpus = pickle.load(f)



writer = pd.ExcelWriter(DATA_DIR / 'konkordancje_katolickie.xlsx', engine='xlsxwriter')
for word in cat_words:
    df = concordance(keyword=word, span=150, corpus=catolic_corpus, corpus_path=CORPUS_DIR / 'swiadectwa_katolickie')
    df.to_excel(writer, sheet_name=word)
writer.save()
writer.close()



prot_words = ['alkohol', 'papieros', 'pić', 'palenie', 'picie', 'alkoholik', 'kac', 'narkotyki', 'więzienie', 'wyrok', 'karny', 'miłosierdzie']



writer = pd.ExcelWriter(DATA_DIR / 'konkordancje_protestanckie.xlsx', engine='xlsxwriter')
for word in prot_words:
    df = concordance(keyword=word, span=150, corpus=protestant_corpus, corpus_path=CORPUS_DIR / 'swiadectwa_protestanckie')
    df.to_excel(writer, sheet_name=word)
writer.save()
writer.close()



drugs.to_excel('narkotyk.xlsx')



concordance(keyword='narkotyk', span=80, corpus=protestant_corpus, corpus_path=CORPUS_DIR / 'swiadectwa_protestanckie')



# klasyfikacja płci
def gender_classifier(doc):
    f, m = 0, 0
    for i, item in enumerate(doc):
        if 'aglt:sg' in item[2]:
            x = re.match(r'praet:sg:(\w{1,2}):', doc[i-1][2])
            if x and x.group(1) == 'f':
                f += 1
            elif x and x.group(1) == 'm1':
                m += 1
            else:
                continue
    if f > m:
        gender = 'K'
    elif m > f:
        gender = 'M'
    else:
        gender = 'UNKNOW'
    return gender



cats_gender = {}
for id_, doc in cats.items():
    cats_gender[id_] = gender_classifier(doc)



t = len(cats_gender.items())
m = list(cats_gender.values()).count("M")
k = list(cats_gender.values()).count("K")
u = list(cats_gender.values()).count("UNKNOW")
print(f'Total: {t}')
print(f'M: {m} ({round((m*100)/t, 2)}%)')
print(f'K: {k} ({round((k*100)/t, 2)}%)')
print(f'U: {u} ({round((u*100)/t, 2)}%)')



prot_gender = {}
for id_, doc in prot.items():
    prot_gender[id_] = gender_classifier(doc)



t = len(prot_gender.items())
m = list(prot_gender.values()).count("M")
k = list(prot_gender.values()).count("K")
u = list(prot_gender.values()).count("UNKNOW")
print(f'Total: {t}')
print(f'M: {m} ({round((m*100)/t, 2)}%)')
print(f'K: {k} ({round((k*100)/t, 2)}%)')
print(f'U: {u} ({round((u*100)/t, 2)}%)')







for token, lemma, tag in cats['008']:
    print(token)



gender_classifier(cats['008'])
























