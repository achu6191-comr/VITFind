import streamlit as st
import sqlite3
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB = "data/vitfind.db"

def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_type TEXT NOT NULL,
            item_name TEXT NOT NULL,
            description TEXT NOT NULL,
            location TEXT NOT NULL,
            contact TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def add_report(report_type, item_name, description, location, contact):
    conn = sqlite3.connect(DB)
    conn.execute(
        """INSERT INTO reports
        (report_type, item_name, description, location, contact, created_at)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (report_type, item_name, description, location, contact,
         datetime.now().strftime("%Y-%m-%d %H:%M"))
    )
    conn.commit()
    conn.close()

def get_reports(report_type=None):
    conn = sqlite3.connect(DB)
    if report_type:
        rows = conn.execute(
            """SELECT id, report_type, item_name, description, location,
               contact, created_at FROM reports WHERE report_type=?""",
            (report_type,)
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT id, report_type, item_name, description, location,
               contact, created_at FROM reports ORDER BY id DESC"""
        ).fetchall()
    conn.close()
    return rows

def find_matches(lost_description, lost_location):
    found = get_reports("Found")
    if not found:
        return []

    documents = [lost_description] + [
        f"{row[2]} {row[3]} {row[4]}" for row in found
    ]

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    results = []
    for row, score in zip(found, scores):
        location_bonus = 0.12 if lost_location.lower() in row[4].lower() else 0
        final_score = min(score + location_bonus, 1.0)
        results.append((final_score, row))

    return sorted(results, key=lambda x: x[0], reverse=True)

init_db()

st.set_page_config(page_title="VITFind", page_icon="🔎", layout="wide")

st.title("🔎 VITFind")
st.subheader("AI-assisted campus Lost & Found")
st.write("Report lost/found items and discover possible matches using text similarity.")

page = st.sidebar.radio("Navigate", ["🏠 Home", "📢 Report Item", "🔍 Find Matches", "📋 All Reports"])

if page == "🏠 Home":
    st.markdown("""
    ### How VITFind works

    1. Report a lost or found item.
    2. Describe the item and where it was seen.
    3. VITFind compares descriptions using **TF-IDF + cosine similarity**.
    4. Potential matches are ranked by similarity.
    """)
    reports = get_reports()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Reports", len(reports))
    c2.metric("Lost", sum(r[1] == "Lost" for r in reports))
    c3.metric("Found", sum(r[1] == "Found" for r in reports))

elif page == "📢 Report Item":
    st.header("Report an Item")
    with st.form("report_form"):
        report_type = st.selectbox("Report type", ["Lost", "Found"])
        item_name = st.text_input("Item name", placeholder="e.g. Black JBL headphones")
        description = st.text_area(
            "Detailed description",
            placeholder="Colour, brand, unique marks, approximate time, etc."
        )
        location = st.text_input("Location", placeholder="e.g. Library Block")
        contact = st.text_input("Contact", placeholder="College email / phone")
        submitted = st.form_submit_button("Submit Report")

    if submitted:
        if all([item_name.strip(), description.strip(), location.strip(), contact.strip()]):
            add_report(report_type, item_name, description, location, contact)
            st.success("Report submitted successfully!")
        else:
            st.error("Please fill in all fields.")

elif page == "🔍 Find Matches":
    st.header("Find Possible Matches")
    lost_description = st.text_area(
        "Describe your lost item",
        placeholder="Black JBL wireless headphones with a small scratch..."
    )
    lost_location = st.text_input(
        "Where did you lose it?",
        placeholder="Library"
    )

    if st.button("🔎 Find Matches"):
        if not lost_description.strip():
            st.warning("Enter a description first.")
        else:
            matches = find_matches(lost_description, lost_location)
            if not matches:
                st.info("No found-item reports are available yet.")
            else:
                shown = 0
                for score, row in matches:
                    if score >= 0.10:
                        shown += 1
                        st.markdown(f"### Possible Match #{shown}")
                        st.progress(float(score))
                        st.write(f"**Similarity:** {score*100:.1f}%")
                        st.write(f"**Item:** {row[2]}")
                        st.write(f"**Description:** {row[3]}")
                        st.write(f"**Location:** {row[4]}")
                        st.write(f"**Reported:** {row[6]}")
                        st.info(f"Contact: {row[5]}")
                        st.divider()
                if shown == 0:
                    st.info("No strong text matches found.")

elif page == "📋 All Reports":
    st.header("Campus Reports")
    reports = get_reports()
    if not reports:
        st.info("No reports yet. Add one from 'Report Item'.")
    else:
        for row in reports:
            label = "🔴 LOST" if row[1] == "Lost" else "🟢 FOUND"
            with st.expander(f"{label} — {row[2]}"):
                st.write(f"**Description:** {row[3]}")
                st.write(f"**Location:** {row[4]}")
                st.write(f"**Contact:** {row[5]}")
                st.caption(f"Reported: {row[6]}")
