# Teammate setup — run the Streamlit app with ClimateBERT

Use this after cloning the repo from GitHub. You **do not** need to retrain unless you are updating the model.

---

## 1. Clone and install Python dependencies

```bash
git clone <REPO_URL>
cd <repo-folder>

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires **Python 3.10+** and **internet** (first run downloads ClimateBERT from Hugging Face).

---

## 2. Install the pretrained model (choose one)

### Option A — From the repo (recommended)

The zip is committed under `models/`:

```bash
mkdir -p outputs/climatebert_v2
unzip models/climatebert_v2_pretrained_demo.zip -d outputs/climatebert_v2/
```

You should have:

```
outputs/climatebert_v2/embedding_lr_multiclass.joblib
outputs/climatebert_v2/embedding_lr_binary_nihilism.joblib
outputs/climatebert_v2/climatebert_metrics.json
```

### Option B — GitHub Release

If your team uses a Release asset instead, download `climatebert_v2_pretrained_demo.zip` and unzip into `outputs/climatebert_v2/`.

### Option C — Train yourself (slow; needs data)

Only if you have `data/processed/cleaned_data-2.csv` (ask Jinxi / shared Drive):

```bash
python src/climatebert/prepare_v2_dataset.py
python src/climatebert/train.py --dataset-version v2
```

---

## 3. Test the model (command line)

```bash
export PYTHONPATH=src    # Windows: set PYTHONPATH=src

python -m demo.inference \
  --text "What is the point of recycling if corporations will never change." \
  --version v2
```

You should see `Predicted label`, `Nihilism probability`, and `Top 3 labels`.

---

## 4. Run the app

```bash
streamlit run app/streamlit_app.py
```

1. Open the URL Streamlit prints (usually http://localhost:8501).  
2. Sidebar → **Checkpoint: v2**.  
3. Paste a comment under **Micro View** → **Analyze Text**.  
4. Read results in the **ClimateBERT** column (TF-IDF may still be a placeholder).

**First prediction** can take 1–2 minutes while Hugging Face weights download; later runs are faster.

---

## Troubleshooting

| Problem | What to do |
|---------|------------|
| `No pretrained model at ...` | Repeat step 2; check the three files exist under `outputs/climatebert_v2/`. |
| `ModuleNotFoundError` | Activate `.venv` and `pip install -r requirements.txt`. |
| Hugging Face / network errors | Check internet; retry; optional `export HF_TOKEN=...`. |
| Slow training | Use Option A or B instead of training locally. |

---

## What is *not* required for the demo

- Large CSVs in `data/processed/` (only for retraining)  
- Running `train.py` if you used the zip  
- GPU (CPU is fine for inference)
