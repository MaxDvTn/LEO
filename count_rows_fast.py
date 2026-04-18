import os
for root, dirs, files in os.walk('data/synthetic'):
    for f in files:
        if f.endswith('.csv'):
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8') as file:
                count = sum(1 for _ in file)
            print(f"{path}: {count} rows")
for root, dirs, files in os.walk('data/gold'):
    for f in files:
        if f.endswith('.csv'):
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8') as file:
                count = sum(1 for _ in file)
            print(f"{path}: {count} rows")
