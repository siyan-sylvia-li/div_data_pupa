from sklearn.metrics.pairwise import cosine_similarity
from dcscore_function import DCScore
import dspy
import numpy as np
import pandas
import random
import json

import dotenv

dotenv.load_dotenv(".env")

import os
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

dspy.configure_cache(
    disk_cache_dir="/ocean/projects/cis250134p/shared"
)

from sentence_transformers import SentenceTransformer

# Load an extremely efficient local model for retrieval
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
model = SentenceTransformer("AnnaWegmann/Style-Embedding", device="cuda")

style_embedder = dspy.Embedder(model.encode)

embedder = dspy.Embedder("openai/text-embedding-3-small")
def negative_cosine_sim(sents):
    embeddings = embedder(sents)
    cos_sim = cosine_similarity(embeddings, embeddings)
    cos_sim = cos_sim - np.eye(len(cos_sim))
    denom = ((cos_sim.size) - len(cos_sim)) // 2
    cos_sim = (np.sum(cos_sim) / 2) / denom
    return -cos_sim

def style_cosine_sim(reference, preds):
    ref_embeddings = style_embedder(reference)
    pred_embeddings = style_embedder(preds)
    cos_sim = cosine_similarity(ref_embeddings, pred_embeddings)
    cos_sim = np.mean(cos_sim)
    return cos_sim
    
def cosine_sim(reference, preds):
    ref_embeddings = embedder(reference)
    pred_embeddings = embedder(preds)
    cos_sim = cosine_similarity(ref_embeddings, pred_embeddings)
    cos_sim = np.mean(cos_sim)
    return cos_sim

def dc_score(sents):
    dcscore_evaluator = DCScore(embedder)
    batch_size = 20
    kernel_type = "cs"
    tau = 1
    # get embedding
    embeddings, n, d = dcscore_evaluator.get_embedding(sents, batch_size=batch_size)

    # calculate dcscore based on embedding
    dataset_dcscore = dcscore_evaluator.calculate_dcscore_by_embedding(embeddings, kernel_type=kernel_type, tau=tau)
    return dataset_dcscore

def brevity_penalty(
    candidate_length: int,
    reference_length: int
) -> float:
    """
    Calculate BLEU-style brevity penalty.
    
    The brevity penalty (BP) is defined as:
    - BP = 1 if candidate_length > reference_length
    - BP = exp(1 - reference_length/candidate_length) if candidate_length <= reference_length
    
    Args:
        candidate_length: Length of the candidate/generated text
        reference_length: Length of the reference text
    
    Returns:
        Brevity penalty value between 0 and 1
    
    Examples:
        >>> brevity_penalty(12, 12)  # Same length
        1.0
        >>> brevity_penalty(15, 12)  # Candidate longer
        1.0
        >>> brevity_penalty(8, 12)   # Candidate shorter
        0.6065306597126334
    """
    if candidate_length >= reference_length:
        return 1.0
    
    if candidate_length == 0:
        return 0.0
    
    return math.exp(1 - reference_length / candidate_length)




if __name__ == "__main__":
    random.seed(42)
    pupa_tnb_data = pandas.read_csv("PUPA_TNB.csv")
    random_sample = pupa_tnb_data.sample(n=15)
    
    all_examples = []
    
    for i, row in random_sample.iterrows():
        if not pandas.isna(row["user_query"]) and not pandas.isna(row["target_response"]):
            curr_example = "User Query: " + row["user_query"] + "\nAssistant Response: " + row["target_response"]
            all_examples.append(curr_example)
    print(dc_score(all_examples))
    print(negative_cosine_sim(all_examples))
    print(style_cosine_sim(all_examples, all_examples))
    
    # all_gen_data = json.load(open("generated_data.json"))[:15]
    # print(dc_score(all_gen_data))
    # print(negative_cosine_sim(all_gen_data))
    
    
    # print(dc_score(all_examples + all_gen_data))
    # print(negative_cosine_sim(all_examples + all_gen_data))
    