from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from typing import Optional
from diversity_metrics import *
from thefuzz import fuzz
from transformers import TrainerCallback
import torch
import random
from diversity_gen import ExampleSummarizer, DataCreator
from dspy.adapters import ChatAdapter
from dspy.utils.exceptions import AdapterParseError


class DynamicPromptDataGenCallback(TrainerCallback):
    def __init__(self, trainer: GRPOTrainer):
        super().__init__()
        self.trainer = trainer
        self.data_summaries = {}
        
    def on_step_start(self, args, state, control, **kwargs):
        dataset = self.trainer.train_dataset
        # Update prompts in dataset to include summary
        if state.global_step % 10 == 0:  # Every 50 steps
            if len(data_storage.generated_data):
                summary = self._generate_data_summary(
                    data_storage.generated_data,
                    data_storage.seen_examples
                )
                # Store for next round
                data_storage.data_summary = summary
                self.data_summaries[state.global_step] = summary
            else:
                self.data_summaries[state.global_step] = data_storage.data_summary
        self._update_prompts_with_summary(data_storage.data_summary)
        print(data_storage.data_summary)
        
        return control
    
    def _generate_data_summary(self, generated: set, seen: set) -> str:
        """Generate a summary of data collected so far"""
        model = self.trainer.model
        tokenizer = self.trainer.processing_class

        # Create summary prompt
        all_data = list(generated) + list(seen)
        random.shuffle(all_data)

        # Use chat message format for instruct model
        summary_messages = chat_adapter.format(
            ExampleSummarizer,
            demos=[],
            inputs={
                "example_list": all_data[:20],
                "curr_summary": data_storage.data_summary
            }
        )
        #         summary_messages = [
        #             {
        #                 "role": "user",
        #                 "content": f"""Given a list of example data points for a dataset, provide a brief summary of these examples. If there are no examples, your summary should be "No data has been generated yet". Be comprehensive in your summary but additionally concise. The summary should be at most 3 sentences.

        # Data points:
        # {chr(10).join(all_data[:50])}

        # Please provide your summary:"""
        #             }
        #         ]

        # Generate summary using chat template
        inputs = tokenizer.apply_chat_template(
            summary_messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=256,
                do_sample=False,  # Use greedy decoding for summary
                pad_token_id=tokenizer.eos_token_id
            )

        summary = tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        )
        
        try:
            summary = chat_adapter.parse(ExampleSummarizer, summary)["summary"]
        except AdapterParseError:
            summary = data_storage.data_summary

        return summary.strip()
    
    def _update_prompts_with_summary(self, summary: str):
        """Update dataset prompts to include current data summary"""
        dataset = self.trainer.train_dataset

        def add_summary_to_prompt(example):
            # Handle chat message format
            if isinstance(example["prompt"], list):
                # Already in chat format - update the last user message
                prompt_messages = example["prompt"].copy()
                if "<SUMMARY>" in prompt_messages[-1]["content"]:
                    prompt_messages[-1]["content"] = prompt_messages[-1]["content"].replace(
                        "<SUMMARY>", summary
                    )
                example["prompt"] = prompt_messages
            else:
                # String format - replace directly
                original_prompt = example["prompt"]
                enhanced_prompt = original_prompt.replace("<SUMMARY>", summary)
                example["prompt"] = enhanced_prompt

            return example

        # Update dataset
        self.trainer.train_dataset = dataset.map(add_summary_to_prompt)


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
    computed_dc_score = dc_score(data_storage.seen_examples + data_storage.generated_data)
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
        print(content)
        gold = gold.split("|||")
        data_storage.seen_examples.extend(gold)
        data_storage.seen_examples = list(set(data_storage.seen_examples))
        try:
            gen_data = chat_adapter.parse(DataCreator, content)["generated_instances"]
        except AdapterParseError:
            rewards.append(0)
            continue
        data_storage.generated_data.extend(gen_data)
        # content = content.split("|||")
        if pii:
            reward = diversity_metric(gold, gen_data) + fuzz.partial_token_set_ratio(pii, content) / 100.0
        else:
            reward = diversity_metric(gold, gen_data)
        rewards.append(reward)

    return rewards

if __name__ == "__main__":
    train_json = json.load(open("training_data.json"))
    test_json = json.load(open("test_data.json"))
    train_dataset = Dataset.from_dict(train_json)
    test_dataset = Dataset.from_dict(test_json)
    
    # grpo_config = GRPOConfig(
    #     use_vllm=True,
    #     vllm_mode="server",
    #     vllm_server_port=3000
    # )
    
    chat_adapter = ChatAdapter()

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        reward_funcs=diversity_reward,
        train_dataset=train_dataset
    )
    
    dynamic_callback = DynamicPromptDataGenCallback(trainer=trainer)
    trainer.add_callback(dynamic_callback)
    trainer.train()
    trainer.save_model("trained_qwen_05b/")