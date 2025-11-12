import dspy

from diversity_gen import OptDiverseDataGenerator
import pandas
from diversity_metrics import dc_score, negative_cosine_sim, cosine_sim
import random
import json

def metric(gold, pred, trace=None):
    computed_dc_score = dc_score(pred.seen_data + pred.generated_data)
    computed_cos_score = cosine_sim(gold.gold_examples, pred.curr_gens)
    if computed_cos_score > 0.6:
        computed_cos_score = -1
    computed_neg_cos_sim = negative_cosine_sim(pred.seen_data + pred.generated_data)
    overall_score = computed_dc_score + computed_cos_score + computed_neg_cos_sim
    return overall_score > 2 if trace is not None else overall_score

if __name__ == "__main__":
    pupa_tnb_data = pandas.read_csv("PUPA_TNB.csv")
    random.seed(42)
    random_sample = pupa_tnb_data.sample(n=15)
    
    all_examples = []
    
    for i, row in random_sample.iterrows():
        curr_example = "User Query: " + row["user_query"] + "\nAssistant Response: " + row["target_response"]
        all_examples.append(curr_example)    
    
    lm = dspy.LM("gpt-4.1-nano")
    dspy.configure(lm=lm)
    
    from constants import PUPA_REQUIREMENT
    task_gen = OptDiverseDataGenerator()
    
    # Start creating actual data for opt
    dspy_examples = []
    
    for _ in range(250):
        dspy_examples.append(dspy.Example({"gold_examples": random.choices(all_examples, k=3),
                                           "hard_requirement": PUPA_REQUIREMENT}).with_inputs("gold_examples", "hard_requirement"))
        
    train_set = dspy_examples[:200]
    dev_set = dspy_examples[200:]
    tiny_dev = dspy_examples[240:]
    
    eval = dspy.Evaluate(metric=metric, devset=tiny_dev, return_all_scores=True, return_outputs=True)
    
    scores = eval(task_gen)
    print(scores)
    json.dump(scores, open("tiny_dev_score.json", "w+"))

