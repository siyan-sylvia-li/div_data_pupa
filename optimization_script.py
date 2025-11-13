import dspy
from dspy import SIMBA, GEPA

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch

print(torch.cuda.is_available())

from diversity_gen import OptDiverseDataGenerator, set_singleton
import pandas
from diversity_metrics import dc_score, negative_cosine_sim, cosine_sim, style_cosine_sim
import random
import json
import copy

from argparse import ArgumentParser

from dotenv import load_dotenv
load_dotenv(".env")
import os

from datetime import datetime

def final_metric(reference, gens):
    computed_dc_score = dc_score(reference + gens)
    computed_cos_score = cosine_sim(reference, gens)
    computed_style_cos_score = cosine_sim(reference, gens)
    computed_neg_cos_sim = negative_cosine_sim(reference + gens)
    return {
        "dc_score": computed_dc_score,
        "cosine_sim": computed_cos_score,
        "style_sim": computed_style_cos_score,
        "neg_sim": computed_neg_cos_sim
    }
    


def metric(gold, pred, trace=None):
    computed_dc_score = dc_score(pred.seen_data + pred.generated_data)
    computed_cos_score = cosine_sim(gold.gold_examples, pred.curr_gens)
    computed_style_cos_score = style_cosine_sim(gold.gold_examples, pred.curr_gens)
    if computed_cos_score > 0.6:
        computed_cos_score = 1
    elif computed_cos_score < 0.4:
        computed_cos_score = 1
    computed_neg_cos_sim = negative_cosine_sim(pred.seen_data + pred.generated_data)
    overall_score = computed_dc_score - computed_cos_score + computed_neg_cos_sim + computed_style_cos_score
    return overall_score

def metric_separate(gold, pred):
    computed_dc_score = dc_score(pred.seen_data + pred.generated_data)
    computed_cos_score = cosine_sim(gold.gold_examples, pred.curr_gens)
    computed_neg_cos_sim = negative_cosine_sim(pred.seen_data + pred.generated_data)
    computed_style_cos_score = style_cosine_sim(gold.gold_examples, pred.curr_gens)
    
    return dspy.Prediction(
        diversity_score=computed_dc_score,
        cosine_sim_ref_pred=computed_cos_score,
        style_cosine_sim_ref=computed_style_cos_score,
        diversity_cos_score=computed_neg_cos_sim
    )

def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    metric_score = metric_separate(gold, pred)
    overall_score = metric(gold, pred, trace)
    
    feedback_text = f"The overall score is {overall_score:.2f}, which computed as the cosine similarity between the in-context gold examples and generations ({metric_score.cosine_sim_ref_pred: .2f}) subtracted from the sum of two diversity scores (DC Score = {metric_score.diversity_score: .2f}, Negative Cosine Similarity = {metric_score.diversity_cos_score: .2f}) and Stylistic Cosine Similarity = {metric_score.style_cosine_sim_ref: .2f}. Try to improve the diversity of your response. The generations should be sufficiently similar to the in-context gold examples without being too similar."
    if metric_score.cosine_sim_ref_pred > 0.6:
        feedback_text += " The current cosine similarity between the in-context gold examples and the generations is too high. Aim to be more creative in the generations while adhering to the hard requirements."
        metric_score.cosine_sim_ref_pred = -10
    elif metric_score.cosine_sim_ref_pred < 0.4:
        feedback_text += " The current cosine similarity between the in-context gold examples and the generations is too low. Adhere to the hard requirements and still have generations to be sufficiently similar to the gold examples."
        metric_score.cosine_sim_ref_pred = -1
    if metric_score.style_cosine_sim_ref < 0.3:
        feedback_text += " The gold examples and the generations are not sufficiently stylistically similar."
    return dspy.Prediction(
        score=overall_score,
        feedback=feedback_text,
    )
    
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_iters", type=int)
    parser.add_argument("--optimizer", type=str, default="gepa")
    parser.add_argument("--bsize", type=int, default=16)
    args = parser.parse_args()
    
    pupa_tnb_data = pandas.read_csv("PUPA_TNB.csv")
    random.seed(42)
    random_sample = pupa_tnb_data.sample(n=20)

    all_examples = []

    for i, row in random_sample.iterrows():
        if not pandas.isna(row["user_query"]) and not pandas.isna(row["target_response"]):
            curr_example = "User Query: " + row["user_query"] + "\nAssistant Response: " + row["target_response"]
            all_examples.append(curr_example)    

    lm = dspy.LM("gpt-4.1", cache=True)
    dspy.configure(lm=lm)

    from constants import PUPA_REQUIREMENT
    data_gen = OptDiverseDataGenerator()

    # Start creating actual data for opt
    train_set = []

    for _ in range(250):
        train_set.append(dspy.Example({"gold_examples": random.choices(all_examples, k=3),
                                            "hard_requirement": PUPA_REQUIREMENT}).with_inputs("gold_examples", "hard_requirement"))
    dev_set = copy.deepcopy(train_set)
    
    if args.optimizer == "gepa":
        optimizer = GEPA(metric=gepa_metric, track_stats=True, 
            reflection_lm=dspy.LM(model='gpt-4.1', temperature=1.0, max_tokens=32000),
            track_best_outputs=True, max_metric_calls=15, perfect_score=1.5, component_selector="all")
    else:
        optimizer = SIMBA(metric=metric, max_steps=1, bsize=args.bsize)
    for iter in range(args.num_iters):
        data_gen = optimizer.compile(data_gen, trainset=train_set[args.bsize * iter:args.bsize * (iter + 1)], valset=dev_set[args.bsize * iter:args.bsize * (iter + 1)])
        if args.optimizer == "gepa":
            gen_data_max_len, gen_data, seen_data = 0, [], []
            data_summary = None
            for k in data_gen.detailed_results.best_outputs_valset:
                curr_gen_len = len(data_gen.detailed_results.best_outputs_valset[k][0][1].generated_data)
                if curr_gen_len > gen_data_max_len:
                    gen_data = data_gen.detailed_results.best_outputs_valset[k][0][1].generated_data + data_gen.detailed_results.best_outputs_valset[k][0][1].curr_gens
                    seen_data = data_gen.detailed_results.best_outputs_valset[k][0][1].seen_data
                    data_summary = data_gen.detailed_results.best_outputs_valset[k][0][1].data_summary
                    gen_data_max_len = curr_gen_len
            data_gen.generated_data = gen_data
            data_gen.seen_data = seen_data
            data_gen.data_summary = data_summary
    
    final_score = final_metric(
        train_set, data_gen.generated_data
    )
    final_score.update({
        "data_size": len(data_gen.generated_data)
    })
    fn = f"{args.optimizer}-{args.bsize}-{datetime.now().isoformat()}.json"
    json.dump(final_score, open(fn, "w+"))