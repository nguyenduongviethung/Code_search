import os
import json
from datasets import load_dataset, disable_progress_bars, logging
from tqdm import tqdm

disable_progress_bars()
logging.set_verbosity_warning()

import logging
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

DATA_ROOT = "data"
LANGUAGES = ["javascript", "go", "python", "java", "php"]
os.makedirs(DATA_ROOT, exist_ok=True)

for lang in LANGUAGES:
    save_dir = os.path.join(DATA_ROOT, lang)
    os.makedirs(save_dir, exist_ok=True)
    docs = load_dataset('code-search-net/code_search_net', lang, trust_remote_code=True)
    queries = []
    codes = []

    for split in docs.keys():
        for doc in tqdm(docs[split], desc=f"Processing {lang} - {split}"):
            assert isinstance(doc, dict), f"Expected dict, got {type(doc)}"
            queries.append({
                "id": doc["func_code_url"],
                "split": doc["split_name"],
                "nl": doc["func_documentation_string"],
            })
            codes.append({
                "id": doc["func_code_url"],
                "split": doc["split_name"],
                "code": doc["func_code_string"]
            })

    with open(os.path.join(save_dir, f"{lang}_query.json"), "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=4)

    with open(os.path.join(save_dir, f"{lang}_code.json"), "w", encoding="utf-8") as f:
        json.dump(codes, f, indent=4)

print("Done.")