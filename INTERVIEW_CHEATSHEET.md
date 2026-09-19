# VITFind Interview Cheat Sheet

## 30-second explanation
"I built VITFind, an AI-assisted campus Lost & Found application. Students can report lost or found items, and the system compares a lost-item description against found reports using TF-IDF and cosine similarity. It ranks potential matches, while SQLite stores the reports."

## Why did you build it?
"Lost-and-found information is usually scattered. I wanted a simple campus-focused system that could connect lost and found reports."

## Why TF-IDF?
"TF-IDF converts text into numerical vectors based on the importance of words in the documents. It is simple, interpretable, and suitable for a beginner MVP."

## What is cosine similarity?
"It measures the angle between two vectors. A value closer to 1 means the text vectors are more similar."

## What is SQLite?
"SQLite is a lightweight relational database stored locally in a file. It is convenient for a small prototype."

## Is this really AI?
"It uses a machine-learning/NLP technique for text representation and similarity. It is an introductory NLP approach rather than a large language model."

## What would you improve?
- Sentence-transformer embeddings for semantic matching
- Image upload and image similarity
- Student authentication
- Email/push notifications
- Admin moderation
- Campus-specific locations
- Cloud deployment
- Better privacy controls

## If asked about limitations
"TF-IDF mainly captures word overlap. Two descriptions with different words but the same meaning may not match strongly. The current prototype also uses a local SQLite database and does not implement authentication."

## Key terms to learn
TF-IDF, vectorization, cosine similarity, NLP, database, CRUD, Streamlit, SQLite, train/test data, precision/recall.
