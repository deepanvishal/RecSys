# Deploy to Hugging Face Spaces

The app ships as a single Docker container that runs FastAPI on `127.0.0.1:8000`
(internal) and Streamlit on `0.0.0.0:7860` (public). The Space URL will be:

    https://huggingface.co/spaces/<your-username>/serko-recsys

## One-time setup

1. **Create a Hugging Face account** — https://huggingface.co/join
2. **Create an access token** — https://huggingface.co/settings/tokens → *New token* → type **Write**
3. **Install Git LFS** (required for files > 10 MB)
   ```bash
   # Windows (winget)
   winget install GitHub.GitLFS
   # or download: https://git-lfs.com
   git lfs install
   ```

## Create the Space

1. Go to https://huggingface.co/new-space
2. **Owner**: your username  **Space name**: `serko-recsys`
3. **License**: MIT  **SDK**: **Docker** → *Blank*
4. **Hardware**: CPU basic (free)  **Visibility**: Public
5. Click *Create Space*

## Push the code

HF gives you a git URL like `https://huggingface.co/spaces/<user>/serko-recsys`.
From the project root:

```bash
# first time only — add HF as a remote
git remote add hf https://huggingface.co/spaces/<your-username>/serko-recsys

# mark big files for LFS (already done by .gitattributes, but run once to be sure)
git lfs track "*.pkl" "*.pt" "*.npy" "*.parquet" "*.index"
git add .gitattributes

# stage the deployment files
git add Dockerfile start.sh requirements.docker.txt .dockerignore README.md

# stage the code + artifacts
git add api/ inference/ models/ config/ utils/ ui/
git add artifacts/models/ artifacts/two_tower/ artifacts/reranker/ artifacts/bias/
git add data/processed/train.parquet data/processed/val.parquet
git add data/processed/item_metadata.parquet data/processed/user_metadata.parquet
git add data/processed/dataset_stats.json

git commit -m "Deploy to HF Spaces"

# push — use your HF username and the write token as password when prompted
git push hf main
```

## First build

Go to the Space page. The first build takes ~5–8 minutes (installing torch CPU,
sentence-transformers, etc.). When it flips from *Building* to *Running*,
the app is live.

## Common issues

- **"File is larger than 10 MiB"** when pushing → you forgot `git lfs install`
  or did not `git add .gitattributes` before adding the big file. Remove it
  from history (`git rm --cached <file>`) and re-add after LFS tracking.
- **Build fails on `faiss-gpu`** → not possible here; `requirements.docker.txt`
  uses `faiss-cpu`. If you ever regenerate from `requirements.txt`, swap it back.
- **App stuck on "Starting"** → click *Logs* on the Space page. `start.sh`
  waits up to 2 min for `/health` before launching Streamlit.
- **Space sleeps after ~48 h idle** → first visit wakes it in ~20 s. Pin the
  Space or upgrade to CPU Upgrade ($9/mo) if you need always-on.

## Making changes

```bash
# edit files locally, then:
git add <changed files>
git commit -m "Fix X"
git push hf main
```

HF auto-rebuilds on every push.
