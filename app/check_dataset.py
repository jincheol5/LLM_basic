from datasets import load_dataset
ds=load_dataset(path="beomi/KoAlpaca-v1.1a")
train=ds["train"]
for i,row in enumerate(train):
    print(f"<<{i+1} row>>")
    print(f"instruction: {row['instruction']}")
    print(f"input: {row.get('input', '')}")
    print(f"output: {row['output']}")
    print()
    if i==10:
        break