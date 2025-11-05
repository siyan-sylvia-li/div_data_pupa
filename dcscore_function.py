
"""
DCScore function
"""
import torch
from sklearn import preprocessing
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, chi2_kernel, polynomial_kernel, laplacian_kernel
import dspy
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# # Add preprocessing to dataset transforms
class TransformedIndexDataset(torch.utils.data.Dataset):
    """Wrapper class for dataset to add preprocess transform"""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, labels = self.dataset[idx]
        img = self.transform(img)
        return img, labels, idx

    def __len__(self):
        return len(self.dataset)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def to_batches(lst, batch_size):
    batches = []
    i = 0
    while i < len(lst):
        batches.append(lst[i : i + batch_size])
        i += batch_size
    return batches

class DCScore:
    def __init__(self, embedder: dspy.Embedder):
        # load model
        self.embedder = embedder
        

    def get_embedding(self, sents_list, batch_size=10):
        embeddings_all = self.embedder(sents_list, batch_size=batch_size)
        n, d = embeddings_all.shape
        return embeddings_all, n, d
        
    def calculate_dcscore_by_texts(self, texts_list, batch_size=10, tau=1):
        embeddings_all, n, d = self.get_embedding(texts_list, batch_size)
        embeddings_all = preprocessing.normalize(embeddings_all, axis=1)
        sim_product = torch.from_numpy((embeddings_all @ embeddings_all.T) / tau)
        sim_probs = sim_product.softmax(dim=-1)
        diversity = torch.sum(torch.diag(sim_probs))
        return diversity.item()
        
    def calculate_dcscore_by_embedding(self, embeddings_arr, kernel_type='cs', tau=1):
        if kernel_type == 'cs':
            # cosine similarity as teh kernel function
            # embeddings_arr = preprocessing.normalize(embeddings_arr, axis=1)
            sim_product = torch.from_numpy((embeddings_arr @ embeddings_arr.T) / tau)
            sim_probs = sim_product.softmax(dim=-1)
            diversity = torch.sum(torch.diag(sim_probs))
        elif kernel_type == 'rbf':
            sim_mat = rbf_kernel(embeddings_arr, embeddings_arr, tau)
            sim_probs = torch.nn.functional.softmax(torch.from_numpy(sim_mat), dim=-1)
            diversity = torch.sum(torch.diag(sim_probs))
        elif kernel_type == 'lap':
            sim_mat = laplacian_kernel(embeddings_arr, embeddings_arr, tau)
            sim_probs = torch.nn.functional.softmax(torch.from_numpy(sim_mat), dim=-1)
            diversity = torch.sum(torch.diag(sim_probs))
        elif kernel_type == 'poly':
            sim_mat = polynomial_kernel(embeddings_arr, embeddings_arr, tau)
            sim_probs = torch.nn.functional.softmax(torch.from_numpy(sim_mat), dim=-1)
            diversity = torch.sum(torch.diag(sim_probs))
        
        return diversity.item()