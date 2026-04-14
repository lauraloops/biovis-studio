# 🧬 BioVis Studio — Interactive Bioinformatics Data Visualization Platform

## 🏷️ Overview
BioVis Studio is a modular platform for building **interactive visualizations for biological data**
(genomics / transcriptomics / omics) using a **Streamlit frontend** and a clean, scalable codebase.

It’s structured so that:
- 🖥️ the UI lives in `app/`
- 🧠 the reusable analysis logic lives in `core/`
- 🌐 optional APIs live in `api/`
- ⚙️ long-running tasks can be moved to `workers/`

✅ Perfect for: demos, coursework, portfolio projects, and evolving into a production-style system.

---

## ✨ Key Features
- 📊 Interactive data exploration (filtering, summaries, plots)
- 🧩 Modular architecture (UI separated from logic)
- ♻️ Reusable analysis functions in `core/`
- 🌐 Optional API layer for programmatic access
- 🧵 Background workers for heavy computations (optional)
- 🧼 Clean repo organization for collaboration and growth

---

## 📁 Project Structure

```text
biovis-studio/
├─ 🖥️ app/                 # Streamlit frontend (pages + UI)
│  └─ 🏠 Home.py            # Main entrypoint for Streamlit
├─ 🧠 core/                # Analysis + business logic (reusable modules)
├─ 🌐 api/                 # Optional backend API layer
├─ ⚙️ workers/             # Background jobs / pipelines
├─ 🧪 data/                # Local datasets / sample inputs (often gitignored if large)
├─ 🧰 infra/               # Infrastructure configs (deploy, containers, etc.)
├─ 📚 instructions/        # Docs, guidelines, project notes
├─ 📦 requirements.txt     # Python dependencies
└─ 📝 README.md
```

**Guiding idea:**
- `app/` is for **UI**
- `core/` is for **logic**
- `api/` and `workers/` are **optional extensions** when you need them

---

## 🚀 Quickstart (Local Run)

### 1) ⬇️ Clone the repository
```bash
git clone https://github.com/lauraloops/biovis-studio.git
cd biovis-studio
```

### 2) 🐍 Create & activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) 📦 Install dependencies
```bash
pip install -r requirements.txt
```

### 4) ▶️ Run the Streamlit app
```bash
streamlit run app/Home.py
```

🔗 Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser.

---

## ⚙️ Configuration

### 🪟 Streamlit watcher fix (Linux)
If Streamlit hot-reload behaves weirdly (file watching issues), set:

```bash
export STREAMLIT_WATCHER_TYPE=poll
```

To persist it across terminal sessions:

```bash
echo 'export STREAMLIT_WATCHER_TYPE=poll' >> ~/.bashrc
source ~/.bashrc
```

---

## 🧑‍💻 Development Workflow

### 🌿 Create your own feature branch
```bash
git checkout main
git pull
git checkout -b laura/<feature-name>
git push -u origin laura/<feature-name>
```

✅ Example:
```bash
git checkout -b laura/preprocess-data
git push -u origin laura/preprocess-data
```

### 📝 Commit & push changes
```bash
git status
git add .
git commit -m "Describe your change"
git push
```

---

## 🧭 Where to put new code

### 🖥️ Add a new Streamlit page (UI)
- Put Streamlit code in `app/`
- Keep logic out of the UI as much as possible

### 🧠 Add reusable logic (analysis / processing)
Put all computation in `core/`, for example:
- data loading
- preprocessing
- statistics
- clustering / PCA / t-SNE / UMAP
- model training
- plotting helpers

### 🌐 Add an API endpoint (optional)
- Put API code in `api/`
- Useful for programmatic access & future integrations

### ⚙️ Add background jobs (optional)
- Put heavy tasks in `workers/`
- Typical use cases:
  - long analyses
  - batch processing
  - scheduled jobs
  - async tasks

---

## 🧪 Data Guidance
📌 Small demo datasets:
- Keep them inside `data/` so others can run your demo quickly.

📦 Large datasets:
- Prefer **not** committing to Git.
- Options:
  - keep them local and ignore via `.gitignore`
  - download via setup scripts
  - store in external storage later (bucket, drive, etc.)

---

## 🧯 Troubleshooting

### ❗ “ModuleNotFoundError” when running Streamlit
✅ Fix:
```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/Home.py
```

### ❗ Git shows: `M .gitignore`
Meaning: `.gitignore` is modified locally.

✅ You can:
- Commit it on your feature branch:
```bash
git add .gitignore
git commit -m "Update gitignore"
```
- Or discard if accidental:
```bash
git checkout -- .gitignore
```

### ❗ You pressed Ctrl+Z during a command and it says “Stopped”
That suspends the command as a job.

✅ If needed:
```bash
jobs
kill %1
```

---

## 🗺️ Roadmap (Suggested)
- ✅ Modular structure (app/core/api/workers)
- ⏳ Add sample datasets + “Demo” walkthrough pages
- ⏳ Add tests for `core/`
- ⏳ Add CI (GitHub Actions) for lint + tests
- ⏳ Optional: Docker + deployment scripts in `infra/`
- ⏳ Optional: API + workers integration for heavy workloads

---

## 🤝 Contributing (Simple Rules)
- 🌿 One branch per feature (`laura/<feature>`)
- 🧱 Small commits with clear messages
- 🧪 Keep logic in `core/`, UI in `app/`
- ✅ Merge to `main` when stable

---

## 📜 License
TBD (add MIT/Apache-2.0 if you want this to be open-source).

---

## 🙌 Credits
Built with:
- 🐍 Python
- 🧪 Streamlit
- 📦 Scientific Python stack (defined in `requirements.txt`)
