# 🔎 VITFind — Campus Lost & Found Platform

VITFind is an intelligent campus Lost & Found web application designed to help university students report lost and found items, protect student privacy, and surface high-confidence matches using text similarity.

---

## 🚀 Features

- **📢 Report Lost / Found Items**: Submit reports with title, category, detailed descriptions, campus locations, and contacts.
- **🔍 Smart Matching Engine**: 
  - Bidirectional search (*Lost $\leftrightarrow$ Found*).
  - N-gram TF-IDF text similarity with brand/title weighting.
  - Gated location bonuses to eliminate unrelated location-only false positives.
  - Clear match confidence tiers (`🟢 High`, `🟡 Moderate`, `🔵 Potential`).
- **🛡️ Privacy & Security (Anti-Scraping)**:
  - Automatic student PII masking for emails and phone numbers.
  - Click-to-reveal expanders to safeguard contact information from automated bots.
  - Strict input sanitization and character length bounds.
- **⚡ Performance & Stability**:
  - Streamlit caching (`@st.cache_data`) for instant responses and minimal database load.
  - Accurate Indian Standard Time (IST) timestamps.
  - Persistent cloud storage powered by Supabase (PostgreSQL).

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Database**: [Supabase](https://supabase.com/) (PostgreSQL)
- **Matching & Similarity**: [Scikit-learn](https://scikit-learn.org/) (TF-IDF Vectorizer + Cosine Similarity)
- **Language**: Python 3.10+

---

## ⚙️ Setup & Local Installation

### 1. Clone the repository
```bash
git clone https://github.com/achu6191-comr/VITFind.git
cd VITFind
```

### 2. Create virtual environment & install dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Supabase Database
In your Supabase project's **SQL Editor**, run the following table setup query:

```sql
CREATE TABLE IF NOT EXISTS reports (
    id BIGSERIAL PRIMARY KEY,
    report_type TEXT NOT NULL CHECK (report_type IN ('Lost', 'Found')),
    item_name TEXT NOT NULL,
    category TEXT DEFAULT 'General',
    description TEXT NOT NULL,
    location TEXT NOT NULL,
    contact TEXT NOT NULL,
    status TEXT DEFAULT 'Open' CHECK (status IN ('Open', 'Claimed', 'Resolved')),
    created_at TEXT NOT NULL
);

-- Enable Row Level Security (RLS)
ALTER TABLE reports ENABLE ROW LEVEL SECURITY;

-- Allow public read & insert policies
CREATE POLICY "Allow public select" ON reports FOR SELECT USING (true);
CREATE POLICY "Allow public insert" ON reports FOR INSERT WITH CHECK (true);
```

### 4. Configure Streamlit Secrets
Create a `.streamlit/secrets.toml` file in the project root:

```toml
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_KEY = "your-supabase-anon-key"
```

### 5. Run the application
```bash
streamlit run app.py
```

---

## 🌐 Deploying on Streamlit Community Cloud

1. Push your changes to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and link your repository.
3. In **App Settings** $\rightarrow$ **Secrets**, paste:
   ```toml
   SUPABASE_URL = "https://your-project-id.supabase.co"
   SUPABASE_KEY = "your-supabase-anon-key"
   ```
4. In **App Settings** $\rightarrow$ **Sharing**, ensure access is set to **Public** so campus students can access the app without a login barrier.

---

## 🗺️ Roadmap & Future Enhancements

- [ ] Semantic Vector Search with `pgvector` and Sentence-Transformers (for multi-language and true synonym understanding).
- [ ] Photo upload support via Supabase Storage.
- [ ] University SSO authentication (`@vitstudent.ac.in` domain restriction).
