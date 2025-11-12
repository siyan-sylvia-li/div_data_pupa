# %%
import dspy


from diversity_gen import OptDiverseDataGenerator
import pandas
from diversity_metrics import dc_score, negative_cosine_sim, cosine_sim, style_cosine_sim
import random
import json

from dotenv import load_dotenv
load_dotenv(".env")


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


# %%
pupa_tnb_data = pandas.read_csv("PUPA_TNB.csv")
random.seed(42)
random_sample = pupa_tnb_data.sample(n=100)

all_examples = []

for i, row in random_sample.iterrows():
    curr_example = "User Query: " + row["user_query"] + "\nAssistant Response: " + row["target_response"]
    all_examples.append(curr_example)    

lm = dspy.LM("gpt-4.1-nano", cache=True)
dspy.configure(lm=lm)

from constants import PUPA_REQUIREMENT

task_gen = OptDiverseDataGenerator()

# Start creating actual data for opt
dspy_examples = []

for _ in range(250):
    dspy_examples.append(dspy.Example({"gold_examples": random.choices(all_examples, k=5),
                                        "hard_requirement": PUPA_REQUIREMENT}).with_inputs("gold_examples", "hard_requirement"))
    
train_set = dspy_examples[:200]
dev_set = dspy_examples[200:]
tiny_dev = dspy_examples[240:]


# %%

eval = dspy.Evaluate(metric=metric, devset=dev_set, return_all_scores=True)


# %%
# scores = eval(task_gen)

# %%
# scores

# %%
# from dspy import GEPA

# gepa = GEPA(metric=gepa_metric, track_stats=True, 
#             reflection_lm=dspy.LM(model='gpt-4.1', temperature=1.0, max_tokens=32000),
#             track_best_outputs=True, max_full_evals=2)
# new_prog = gepa.compile(task_gen, trainset=train_set[:30], valset=dev_set[:15])
# pareto_frontier = new_prog.detailed_results.val_aggregate_scores

# %%
optimizer = dspy.SIMBA(metric=metric, max_steps=3)
optimized_program = optimizer.compile(task_gen, trainset=train_set)

# Save optimize program for future use
optimized_program.save(f"optimized.json")

# %%
# new_prog.generated_data

# %%
import json

json.dump(optimized_program.generated_data, open("gen_data.json", "w+"))

# %%



