from datasets import load_dataset
import pandas as pd

# Load both datasets
print("Loading datasets...")
ds1 = load_dataset("deepset/prompt-injections")
ds2 = load_dataset("jackhhao/jailbreak-classification")

# Peek at structure
print("\n--- Dataset 1: deepset/prompt-injections ---")
print(ds1)
print(ds1['train'][0])

print("\n--- Dataset 2: jackhhao/jailbreak-classification ---")
print(ds2)
print(ds2['train'][0])