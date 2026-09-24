import streamlit as st
from datetime import datetime
from supabase import create_client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="VITFind",
    page_icon="🔎",
    layout="wide"
)

# Connect to Supabase
supabase = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)


def add_report(report_type, item_name, description, location, contact):
    supabase.table("reports").insert({
        "report_type": report_type,
        "item_name": item_name,
        "description": description,
        "location": location,
        "contact": contact,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }).execute()


def get_reports(report_type=None):
    query = (
        supabase
        .table("reports")
        .select("*")
        .order("id", desc=True)
    )

    if report_type:
        query = query.eq("report_type", report_type)

    response = query.execute()

    return [
        (
            row["id"],
            row["report_type"],
            row["item_name"],
            row["description"],
            row["location"],
            row["contact"],
            row["created_at"]
        )
        for row in response.data
    ]


def find_matches(lost_description, lost_location):
    found = get_reports("Found")

    if not found:
        return []

    documents = [lost_description] + [
        f"{row[2]} {row[3]} {row[4]}"
        for row in found
    ]

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents)

    scores = cosine_similarity(
        matrix[0:1],
        matrix[1:]
    ).flatten()

    results = []

    for row, score in zip(found, scores):
        location_bonus = (
            0.12
            if lost_location.strip().lower() in row[4].lower()
            else 0
        )

        final_score = min(score + location_bonus, 1.0)

        results.append((final_score, row))

    return sorted(
        results,
        key=lambda x: x[0],
        reverse=True
    )


# ---------------- UI ----------------

st.title("🔎 VITFind")
st.subheader("AI-assisted campus Lost & Found")

st.write(
    "Report lost/found items and discover possible matches "
    "using text similarity."
)

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📢 Report Item",
        "🔍 Find Matches",
        "📋 All Reports"
    ]
)


if page == "🏠 Home":

    st.markdown("""
    ### How VITFind works

    1. Report a lost or found item.
    2. Describe the item and where it was seen.
    3. VITFind compares descriptions using
       **TF-IDF + cosine similarity**.
    4. Potential matches are ranked by similarity.
    """)

    reports = get_reports()

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Reports", len(reports))
    c2.metric(
        "Lost",
        sum(r[1] == "Lost" for r in reports)
    )
    c3.metric(
        "Found",
        sum(r[1] == "Found" for r in reports)
    )


elif page == "📢 Report Item":

    st.header("Report an Item")

    with st.form("report_form"):

        report_type = st.selectbox(
            "Report type",
            ["Lost", "Found"]
        )

        item_name = st.text_input(
            "Item name",
            placeholder="e.g. Black JBL headphones"
        )

        description = st.text_area(
            "Detailed description",
            placeholder="Colour, brand, unique marks, approximate time, etc."
        )

        location = st.text_input(
            "Location",
            placeholder="e.g. Library"
        )

        contact = st.text_input(
            "Contact",
            placeholder="College email / phone"
        )

        submitted = st.form_submit_button(
            "Submit Report"
        )

    if submitted:

        if all([
            item_name.strip(),
            description.strip(),
            location.strip(),
            contact.strip()
        ]):

            add_report(
                report_type,
                item_name,
                description,
                location,
                contact
            )

            st.success(
                "Report submitted successfully!"
            )

        else:

            st.error(
                "Please fill in all fields."
            )


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

            st.warning(
                "Enter a description first."
            )

        else:

            matches = find_matches(
                lost_description,
                lost_location
            )

            if not matches:

                st.info(
                    "No found-item reports are available yet."
                )

            else:

                shown = 0

                for score, row in matches:

                    if score >= 0.05:

                        shown += 1

                        st.markdown(
                            f"### Possible Match #{shown}"
                        )

                        st.progress(
                            float(score)
                        )

                        st.write(
                            f"**Similarity:** {score * 100:.1f}%"
                        )

                        st.write(
                            f"**Item:** {row[2]}"
                        )

                        st.write(
                            f"**Description:** {row[3]}"
                        )

                        st.write(
                            f"**Location:** {row[4]}"
                        )

                        st.write(
                            f"**Reported:** {row[6]}"
                        )

                        st.info(
                            f"Contact: {row[5]}"
                        )

                        st.divider()

                if shown == 0:

                    st.info(
                        "No possible matches found."
                    )


elif page == "📋 All Reports":

    st.header("Campus Reports")

    reports = get_reports()

    if not reports:

        st.info(
            "No reports yet. Add one from 'Report Item'."
        )

    else:

        for row in reports:

            label = (
                "🔴 LOST"
                if row[1] == "Lost"
                else "🟢 FOUND"
            )

            with st.expander(
                f"{label} — {row[2]}"
            ):

                st.write(
                    f"**Description:** {row[3]}"
                )

                st.write(
                    f"**Location:** {row[4]}"
                )

                st.write(
                    f"**Contact:** {row[5]}"
                )

                st.caption(
                    f"Reported: {row[6]}"
                )
