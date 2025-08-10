# Quiz App (Curated Subset)

This commit contains the minimal, production-relevant subset of the Knowledge App / Quiz generation platform prepared for the `seducedevata/quiz-app` repository.

## Included

- Next.js app in `knowledge-app-next/` (source, public assets, scripts, config, API bridge server)
- Core Python backend bridge: `python_bridge_server.py` + `src/knowledge_app` package (only essential code retained per .gitignore whitelists)
- Configuration: selected files under `config/`
- Requirements: `requirements.txt`, pinned variants

## Excluded

Large binaries, model weights, logs, caches, temp/test/debug scripts, auto-generated markdown summaries, executables, training datasets, screenshots, and experimental scripts.

See `.gitignore` for precise patterns.

## Local Development

1. Install Node dependencies:
   cd knowledge-app-next && npm install
2. (Optional) Python virtual environment & deps:
   python -m venv .venv
   .venv/Scripts/Activate.ps1  # Windows PowerShell
   pip install -r requirements.txt
3. Start bridge server (Python):
   python python_bridge_server.py
4. Start Next.js dev server (in another terminal):
   cd knowledge-app-next
   npm run dev

## Production Build

cd knowledge-app-next
npm run build
npm start

## Notes

If anything required is missing due to aggressive ignoring, adjust `.gitignore` then recommit.
