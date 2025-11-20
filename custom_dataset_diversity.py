import random
import pandas
import dspy
from dspy.adapters import ChatAdapter
from diversity_gen import DataCreator
import json

if __name__ == "__main__":
    pupa_tnb_data = pandas.read_csv("PUPA_TNB.csv")
    random.seed(42)
    random_sample = pupa_tnb_data.sample(n=100)

    all_examples = []

    for i, row in random_sample.iterrows():
        if not pandas.isna(row["user_query"]) and not pandas.isna(row["target_response"]):
            curr_example = "User Query: " + row["user_query"] + "\nAssistant Response: " + row["target_response"]
            all_examples.append(curr_example)    

    from constants import PUPA_REQUIREMENT

    # Start creating actual data for opt
    train_set = []
    test_set = []

    for _ in range(1000):
        train_set.append({"gold_examples": random.choices(all_examples, k=3), 
                          "curr_summary": "<SUMMARY>",
                          "hard_requirement": PUPA_REQUIREMENT})
    
    for _ in range(200):
        test_set.append({"gold_examples": random.choices(all_examples, k=3), 
                         "curr_summary": "<SUMMARY>",
                         "hard_requirement": PUPA_REQUIREMENT})
    
    chat_adapter = ChatAdapter()
    
    custom_train, custom_test = {"prompt": [], "golden_examples": [], "pii_integration": []}, {"prompt": [], "golden_examples": [], "pii_integration": []}
    
    for t in train_set:
        pr = chat_adapter.format(DataCreator, demos=[], inputs=t)
        custom_train["prompt"].append(pr)
        custom_train["golden_examples"].append("|||".join(t["gold_examples"]))
        custom_train["pii_integration"].append(None)
    
    for t in test_set:
        pr = chat_adapter.format(DataCreator, demos=[], inputs=t)
        custom_test["prompt"].append(pr)
        custom_test["golden_examples"].append("|||".join(t["gold_examples"]))
        custom_test["pii_integration"].append(None)
    
    json.dump(custom_train, open("training_data.json", "w+"))
    json.dump(custom_test, open("test_data.json", "w+"))
    
    
    
    
        