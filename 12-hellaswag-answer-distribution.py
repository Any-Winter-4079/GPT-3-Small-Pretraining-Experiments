from collections import Counter
from datasets import load_dataset

split = "validation"

# HellaSwag train split answer label distribution:
#         A: 9986 (25.02%)
#         B: 10031 (25.14%)
#         C: 9867 (24.73%)
#         D: 10021 (25.11%)

# HellaSwag validation split answer label distribution:
#         A: 2515 (25.04%)
#         B: 2485 (24.75%)
#         C: 2584 (25.73%)
#         D: 2458 (24.48%)

dataset = load_dataset("hellaswag", split=split)

label_counts = Counter(int(example['label']) for example in dataset)
print(label_counts)
# Counter({2: 2584, 0: 2515, 1: 2485, 3: 2458})

total = sum(label_counts.values())
print(f"HellaSwag {split} split answer label distribution:")
for i in range(4):
    count = label_counts[i]
    percent = 100 * count / total
    print(f"\t{chr(65 + i)}: {count} ({percent:.2f}%)")