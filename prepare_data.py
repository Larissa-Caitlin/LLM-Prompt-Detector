from datasets import load_dataset
import pandas as pd

print("Loading...")
ds1 = load_dataset("deepset/prompt-injections")
ds2 = load_dataset("jackhhao/jailbreak-classification")

# Convert to pandas
df1_train = pd.DataFrame(ds1['train'])
df1_test  = pd.DataFrame(ds1['test'])

df2_train = pd.DataFrame(ds2['train'])
df2_test  = pd.DataFrame(ds2['test'])

# Normalise dataset 2 to match dataset 1 format
# "benign" -> 0, "jailbreak" -> 1
df2_train = df2_train.rename(columns={"prompt": "text", "type": "label"})
df2_test  = df2_test.rename(columns={"prompt": "text", "type": "label"})
df2_train['label'] = df2_train['label'].map({"benign": 0, "jailbreak": 1})
df2_test['label']  = df2_test['label'].map({"benign": 0, "jailbreak": 1})

# Merge both
train_df = pd.concat([df1_train, df2_train], ignore_index=True)
test_df  = pd.concat([df1_test,  df2_test],  ignore_index=True)

# Shuffle
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df  = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv",  index=False)

# Summary
print(f"\nTraining samples : {len(train_df)}")
print(f"Test samples     : {len(test_df)}")
print(f"\nLabel distribution (train):")
print(train_df['label'].value_counts())
print(f"\nSample row:")
print(train_df.head(3))
print("\nSaved to data/train.csv and data/test.csv")