from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import FlatIPFaissSearch    
import json, os, pathlib
import sys
retriever = sys.argv[1]
llm = sys.argv[2]
dataset = sys.argv[3]
q = {}
q_cot = {}
ctx = {}
qid2answers = {}
with open(f"{dataset}_questions_cot_{llm}.json", "r") as f:
    for i, line in enumerate(f):
        line = json.loads(line)
        q[str(i)] = line["question"]
        q_cot[str(i)] = "\n".join(line["augmented_questions"].split("\n")[1:])
        qid2answers[str(i)] = line["answers"]

with open("passages.json", "r") as f:
    for i, line in enumerate(f):
        line = json.loads(line)
        ctx[str(i)] = {"text": line["text"], "title": ""}

########################################################
#### FLATIP: Flat Inner Product (Exhaustive Search) ####
########################################################

# model_name="msmarco-distilbert-base-tas-b"
# model = models.SentenceBERT(model_name)
# faiss_search = FlatIPFaissSearch(model, 
#                                  batch_size=128)

if retriever == "dpr":
    faiss_search = FlatIPFaissSearch(models.SentenceBERT((
        "facebook-dpr-question_encoder-multiset-base",
        "facebook-dpr-ctx_encoder-multiset-base",
        " [SEP] "), batch_size=128))
elif retriever == "ance":
    faiss_search = FlatIPFaissSearch(models.SentenceBERT("msmarco-roberta-base-ance-firstp"), batch_size=32)
elif retriever == "sbert":
    faiss_search = FlatIPFaissSearch(models.SentenceBERT(("msmarco-distilbert-base-tas-b"), batch_size=128))
elif retriever == "contriever":
    faiss_search = FlatIPFaissSearch(models.SentenceBERT("facebook/contriever"), batch_size=32)
elif retriever == "e5":
    faiss_search = FlatIPFaissSearch(models.SentenceBERT("intfloat/e5-base-v2"), batch_size=32)

######################################################
#### PQ: Product Quantization (Exhaustive Search) ####
######################################################

# faiss_search = PQFaissSearch(model, 
#                              batch_size=128, 
#                              num_of_centroids=96, 
#                              code_size=8)

#####################################################
#### HNSW: Approximate Nearest Neighbours Search ####
#####################################################

# faiss_search = HNSWFaissSearch(model, 
#                                batch_size=128, 
#                                hnsw_store_n=512, 
#                                hnsw_ef_search=128,
#                                hnsw_ef_construction=200)

###############################################################
#### HNSWSQ: Approximate Nearest Neighbours Search with SQ ####
###############################################################

# faiss_search = HNSWSQFaissSearch(model, 
#                                 batch_size=128, 
#                                 hnsw_store_n=128, 
#                                 hnsw_ef_search=128,
#                                 hnsw_ef_construction=200)

#### Load faiss index from file or disk ####
# We need two files to be present within the input_dir!
# 1. input_dir/{prefix}.{ext}.faiss => which loads the faiss index.
# 2. input_dir/{prefix}.{ext}.faiss => which loads mapping of ids i.e. (beir-doc-id \t faiss-doc-id).

prefix = retriever       # (default value)
ext = "flat"              # or "pq", "hnsw", "hnsw-sq"
path = os.path.join(pathlib.Path(".").parent.absolute(), "faiss-index")

if os.path.exists(os.path.join(path, "{}.{}.faiss".format(prefix, ext))):
    faiss_search.load(input_dir=path, prefix=prefix, ext=ext)
else:
    faiss_search.index(ctx)
    faiss_search.save(output_dir=path, prefix=prefix, ext=ext)

#### Retrieve dense results (format of results is identical to qrels)
retriever = EvaluateRetrieval(faiss_search, score_function="dot") # or "cos_sim"

print (retriever, dataset)
# q
import time
# q + cot of gpt-4
start = time.time()

results = retriever.retrieve(ctx, q_cot)

top_k = 100
with open(f"retriever_results/{dataset}_{retriever}_res_top100_qaug.json", "w") as wf:
    for query_id, ranking_scores in results.items():
        scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
        docs = []
        query = q_cot[query_id]
        for rank in range(top_k):
            doc_id = scores_sorted[rank][0]
            docs.append(ctx[doc_id].get("title") + ctx[doc_id].get("text"))
        wf.write(json.dumps({"query": query, "retrieved": docs, "answer": qid2answers[query_id]})+"\n")