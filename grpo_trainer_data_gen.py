from datasets import load_dataset
from trl import GRPOTrainer
from trl.rewards import accuracy_rewards
from typing import Optional
from diversity_metrics import *
from thefuzz import fuzz
from transformers import TrainerCallback



class DataShareSingleton():
    def __init__(self):
        self.generated_data = []
        self.seen_examples = []
        self.data_summary = "No data has been generated yet."
    
data_storage = DataShareSingleton()

def set_singleton(generated_data, seen_examples, data_summary):
    global data_storage
    data_storage.generated_data = generated_data
    data_storage.seen_examples = seen_examples
    data_storage.data_summary = data_summary

def diversity_metric(gold_examples: list[str], pred: list[str]):
    computed_dc_score = dc_score(data_storage.seen_data + data_storage.generated_data)
    computed_style_cos_score = style_cosine_sim(gold_examples, pred)
    all_brevity_penalty = []
    for g in gold_examples:
        for p in pred:
            brevity = brevity_penalty(len(p.split()), len(g.split()))
            all_brevity_penalty.append(brevity)
    brevity_score = sum(all_brevity_penalty) / len(all_brevity_penalty)
    # overall_score = computed_dc_score - computed_cos_score + computed_neg_cos_sim + computed_style_cos_score + brevity_score
    overall_score = computed_dc_score + computed_style_cos_score + brevity_score
    return overall_score

def diversity_reward(completions: list[list[dict[str, str]]], golden_examples: list[str], pii_integration: Optional[list[str]], **kwargs) -> list[float | None]:
    """
    Sum between the diversity-driven metric score and the pii integration (computed by fuzzy match)
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, gold, pii in zip(contents, golden_examples, pii_integration):
        gold = gold.split("|||")
        # content = content.split("|||")
        reward = diversity_metric(gold, content.split("\n\n")) + fuzz.partial_token_set_ratio(pii, content) / 100.0
        rewards.append(reward)

    return rewards

if __name__ == "__main__":
    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
    )
    trainer.add_callback()
    trainer.train()