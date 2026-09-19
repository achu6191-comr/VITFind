# 🔎 VITFind — AI-Assisted Campus Lost & Found

VITFind is a beginner-friendly campus Lost & Found web application.

## Problem
Students often lose items around campus and have no simple way to compare a lost-item description with items reported as found.

## Solution
VITFind stores lost/found reports and uses **TF-IDF + cosine similarity** to compare a user's lost-item description with found-item reports. It ranks potential matches.

## Tech Stack
- Python
- Streamlit
- SQLite
- Scikit-learn
- TF-IDF
- Cosine Similarity

## Features
- Report a lost item
- Report a found item
- Store reports in SQLite
- Search for potential matches
- Rank matches by text similarity
- View all campus reports

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Demo flow
1. Go to **Report Item**.
2. Add a Found item:
   - Item: Black JBL headphones
   - Description: Black wireless JBL headphones, small scratch
   - Location: Library
3. Add another report or go to **Find Matches**.
4. Search:
   - "Black JBL bluetooth headphones"
   - Location: Library
5. VITFind ranks the found report as a possible match.

## Important limitation
This is an educational MVP. TF-IDF is keyword/statistical text similarity, not a modern semantic AI model. A future version could use sentence embeddings, image matching, authentication, notifications, and a real campus database.
