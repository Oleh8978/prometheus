import json
from pathlib import Path
import pandas as pd

results = json.loads((Path("data/eval_results.json")).read_text())
df = pd.DataFrame(results)
print(df.groupby("name")["score"].agg(["mean","min","max"]).round(3))