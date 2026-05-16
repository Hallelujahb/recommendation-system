# Production Cascade Hybrid Recommendation Engine

This repository contains `ProductionCascadeHybridEngine`, a single-file Python implementation of a 3-stage cascade hybrid recommender built for the university assignment described.

Quick start:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the engine script to print the cleaning report, train models, and evaluate:

```bash
python production_cascade_hybrid.py
```

The script prints outputs in the order required by the assignment:
1. Data cleaning report
2. Training progress logs
3. Recommendations for user_id=42
4. ILD score for that recommendation list
5. Full evaluation report across 50 users
6. Alpha sensitivity table

Files:
- `production_cascade_hybrid.py` — main engine implementing all phases
- `requirements.txt` — Python dependencies

If you want me to run this locally (where I can execute code), tell me and I will run the script and return full outputs.
