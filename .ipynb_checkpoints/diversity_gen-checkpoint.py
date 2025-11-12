import dspy
from pathlib import Path
import os
from typing import List
import dotenv
import json
from dspy.teleprompt import MIPROv2

import random
import copy

import pandas
from data_quality_judges import DataQualityJudge
import litellm

from constants import PUPA_REQUIREMENT

import copy
from copy import deepcopy

class DataShareSingleton():
    def __init__(self):
        self.generated_data = []
        self.seen_examples = []
        self.data_summary = "No data has been generated yet."
    
singleton = DataShareSingleton()

def set_singleton(generated_data, seen_examples, data_summary):
    global singleton
    singleton.generated_data = generated_data
    singleton.seen_examples = seen_examples
    singleton.data_summary = data_summary

random.seed(42)

dotenv.load_dotenv(".env")

class ExampleSummarizer(dspy.Signature):
    """Given a list of example data points for a dataset, provide a brief summary of these examples. If there are no examples, your summary should be \"No data has been generated yet\". Be comprehensive in your summary but additionally concise. The summary should be at most 3 sentences."""
    example_list: List[str] = dspy.InputField(desc="The list of examples")
    curr_summary: str = dspy.InputField(desc="The current summary of existing examples. Revise the current summary based on the new examples, and form your output accordingly")
    summary: str = dspy.OutputField(desc="The summary of existing examples")
    
class DataCreator(dspy.Signature):
    """Create a wide variety of new instances from a dataset given a set of examples. You should avoid duplicating examples already created, as provided to you as a summary. Avoid duplicating the examples provided."""
    examples: List[str] = dspy.InputField(desc="Example data instances")
    data_summary: str = dspy.InputField(desc="A summary of existing data points. You should not generate duplicate tasks.")
    requirement: str = dspy.InputField(desc="Hard requirements for the new generated instances")
    generated_instances: List[str] = dspy.OutputField(desc="A list of generated data instances that are sufficiently different from existing data")

class Rewriter(dspy.Signature):
    """You are provided example data points, generated data points meant to be similar to example data points but not substantial copies, as well as hard requirements the generated data points must follow. Rewrite the generated data points in accordance with the hard requirements."""
    examples: List[str] = dspy.InputField(desc="Example data instances")
    requirement: str = dspy.InputField(desc="Hard requirements for the new generated instances")
    generated_instances: List[str] = dspy.InputField(desc="A list of generated data instances that are sufficiently different from existing data")
    rewritten_instances: List[str] = dspy.OutputField(desc="The rewritten generated data points")

class DiverseDataGenerator(dspy.Module):
    def __init__(self, examples, hard_requirement, callbacks=None):
        super().__init__(callbacks)
        self.example_list = copy.deepcopy(examples)
        self.gold_examples = copy.deepcopy(examples)
        self.requirement = hard_requirement
        
        self.data_summary = None
        self.summarizer = dspy.ChainOfThought(ExampleSummarizer)
        self.proposer = dspy.ChainOfThought(DataCreator)
        

    def forward(self):
        if len(self.example_list):
            self.data_summary = self.summarizer(example_list=self.example_list).summary
        
        chosen_examples = random.choices(self.gold_examples, k=5) if len(self.gold_examples) else []
        
        generations = self.proposer(
            examples=chosen_examples,
            data_summary=self.data_summary,
            requirement=self.requirement
        ).generated_instances
        
        self.example_list.extend(generations)
        
    
class OptDiverseDataGenerator(dspy.Module):
    def __init__(self, callbacks=None):
        super().__init__(callbacks)
        self.seen_examples = singleton.seen_examples
        self.generated_data = singleton.generated_data
        
        self.data_summary = singleton.data_summary
        self.summarizer = dspy.ChainOfThought(ExampleSummarizer)
        self.proposer = dspy.ChainOfThought(DataCreator)
        self.rewriter = dspy.ChainOfThought(Rewriter)
        

    def forward(self, gold_examples, hard_requirement):
        
        self.seen_examples.extend(gold_examples)
        
        generations = self.proposer(
            examples=gold_examples,
            data_summary=self.data_summary,
            requirement=hard_requirement
        ).generated_instances
        
        generations = self.rewriter(
            examples=gold_examples,
            requirement=hard_requirement,
            generated_instances=generations
        ).rewritten_instances

        try:
            prev_summary = self.data_summary
            self.data_summary = self.summarizer(example_list=generations, curr_summary=self.data_summary).summary
        except litellm.exceptions.ContextWindowExceededError:
            self.data_summary = prev_summary
        
        self.generated_data.extend(generations)
        self.generated_data = list(set(self.generated_data))
        self.seen_examples = list(set(self.seen_examples))
        print(self.generated_data)
        return dspy.Prediction(
            generated_data=self.generated_data,
            seen_data=self.seen_examples,
            curr_gens=generations,
            data_summary=self.data_summary
        )
    




if __name__ == "__main__":
    pupa_tnb_data = pandas.read_csv("PUPA_TNB.csv")
    random_sample = pupa_tnb_data.sample(n=15)
    
    all_examples = []
    
    for i, row in random_sample.iterrows():
        curr_example = "User Query: " + row["user_query"] + "\nAssistant Response: " + row["target_response"]
        all_examples.append(curr_example)    
    
    lm = dspy.LM("gpt-4.1")
    dspy.configure(lm=lm)
    
    task_gen = DiverseDataGenerator(examples=all_examples, hard_requirement=PUPA_REQUIREMENT)
    data_judge = DataQualityJudge(hard_requirement=PUPA_REQUIREMENT)
    
    # mipro = MIPROv2(metric=metric_diversity, prompt_model=lm, num_candidates=10)
    # instructions = mipro._propose_instructions(program=task_gen, trainset=[], demo_candidates=[], view_data_batch_size=10, tip_aware_proposer=True, program_aware_proposer=True, data_aware_proposer=False, fewshot_aware_proposer=False)
    # print(instructions, len(instructions[0]))
    
    
    while len(task_gen.example_list) < 20:
        task_gen.forward()