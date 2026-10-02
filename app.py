import re
import streamlit as st
from datetime import datetime, timezone, timedelta
from supabase import create_client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Indian Standard Time (IST) for VIT campuses
IST = timezone(timedelta(hours=5, minutes=30))

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


def mask_contact(contact_str: str) -> str:
    """
    Mask email or phone number to mitigate automated scraping and protect student PII.
    """
    contact_str = (contact_str or "").strip()
    if not contact_str:
        return "Not provided"

    if "@" in contact_str:
        parts = contact_str.split("@", 1)
        user, domain = parts[0], parts[1]
        if len(user) <= 2:
            masked_user = user[0] + "•••"
        else:
            masked_user = user[:2] + "•••" + user[-1]
        return f"{masked_user}@{domain}"

    digits = [c for c in contact_str if c.isdigit()]
    if len(digits) >= 10:
        return contact_str[:3] + " •••• " + contact_str[-4:]

    return contact_str[:2] + "••••" if len(contact_str) > 2 else "••••"


CATEGORIES = [
    "📱 Electronics & Gadgets",
    "💳 ID Cards & Wallets",
    "🔑 Keys",
    "🎒 Bags & Backpacks",
    "👕 Clothing & Accessories",
    "📚 Books & Stationery",
    "📦 Other"
]


def add_report(report_type, item_name, description, location, contact, category="📦 Other"):
    """
    Validate, sanitize, and insert report into Supabase with defensive fallback for schema changes.
    Returns (success: bool, message: str).
    """
    item_name = (item_name or "").strip()
    description = (description or "").strip()
    location = (location or "").strip()
    contact = (contact or "").strip()
    category = (category or "📦 Other").strip()

    # Input length and sanitization checks
    if len(item_name) < 2 or len(item_name) > 100:
        return False, "Item name must be between 2 and 100 characters."
    if len(description) < 5 or len(description) > 1500:
        return False, "Description must be between 5 and 1500 characters."
    if len(location) < 2 or len(location) > 100:
        return False, "Location must be between 2 and 100 characters."
    if len(contact) < 5 or len(contact) > 100:
        return False, "Contact must be between 5 and 100 characters (e.g. college email or phone)."

    # Check whether the exact same report already exists
    existing = (
        supabase
        .table("reports")
        .select("id")
        .eq("report_type", report_type)
        .eq("item_name", item_name)
        .eq("description", description)
        .eq("location", location)
        .eq("contact", contact)
        .limit(1)
        .execute()
    )

    if existing.data:
        return False, "A report with these exact details already exists."

    payload = {
        "report_type": report_type,
        "item_name": item_name,
        "category": category,
        "description": description,
        "location": location,
        "contact": contact,
        "status": "Open",
        "created_at": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    }

    try:
        supabase.table("reports").insert(payload).execute()
    except Exception as e:
        err_msg = str(e).lower()
        if "category" in err_msg or "status" in err_msg:
            # Fallback if user's existing Supabase table does not yet have 'category' or 'status'
            payload.pop("category", None)
            payload.pop("status", None)
            supabase.table("reports").insert(payload).execute()
        else:
            return False, f"Database error: {str(e)}"

    return True, "Report submitted successfully!"


def update_report_status(report_id, new_status):
    """Update status of a report (Open, Claimed, Resolved)."""
    try:
        supabase.table("reports").update({"status": new_status}).eq("id", report_id).execute()
        st.cache_data.clear()
        return True
    except Exception:
        return False


@st.cache_data(ttl=60)
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

    # Return list of dictionaries directly for safe key-based access
    return response.data or []


def find_matches(query_description, query_name="", query_location="", search_target="Found", category_filter="All Categories"):
    """
    Search for matches in candidate reports ('Found' or 'Lost').
    Features:
    - Filters to only active/Open reports
    - Optional category filtering
    - Title & brand keyword matching
    - N-gram TF-IDF cosine similarity
    - Location bonus only awarded when content relevance exists
    """
    candidates = get_reports(search_target)
    if not candidates:
        return []

    # Prioritize active/open reports
    candidates = [r for r in candidates if r.get("status", "Open") == "Open"]

    if category_filter and category_filter != "All Categories":
        candidates = [r for r in candidates if r.get("category") == category_filter]
        if not candidates:
            return []

    full_query = f"{query_name} {query_description}".strip()
    if not full_query:
        return []

    # 1. Clean and tokenize query keywords
    clean_query = re.sub(r"[^\w\s]", " ", full_query.lower())
    query_words = {w for w in clean_query.split() if len(w) > 2}

    clean_title = re.sub(r"[^\w\s]", " ", (query_name or "").lower())
    title_words = {w for w in clean_title.split() if len(w) > 2}

    query_loc = (query_location or "").strip().lower()

    # 2. Vectorize query + all candidate items at once
    corpus = [full_query] + [
        f"{r.get('item_name', '')} {r.get('description', '')}" for r in candidates
    ]

    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform(corpus)
        text_scores = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
    except ValueError:
        text_scores = [0.0] * len(candidates)

    results = []

    for i, row in enumerate(candidates):
        item_name = str(row.get("item_name") or "").lower()
        cand_location = str(row.get("location") or "").strip().lower()
        cand_desc = str(row.get("description") or "").lower()

        text_score = float(text_scores[i])

        # 3. Item title & brand match boost
        cand_item_words = set(re.sub(r"[^\w\s]", " ", item_name).split())
        matched_title_words = title_words & cand_item_words if title_words else set()
        title_bonus = min(len(matched_title_words) * 0.20, 0.35)

        # Keyword overlap across full item
        cand_all_words = cand_item_words | set(re.sub(r"[^\w\s]", " ", cand_desc).split())
        matched_query_words = query_words & cand_all_words
        keyword_bonus = min(len(matched_query_words) * 0.05, 0.15)

        # Baseline content relevance check (prevents unrelated items matching solely on location)
        has_content_relevance = (text_score > 0.08) or (len(matched_title_words) > 0) or (len(matched_query_words) >= 2)

        # 4. Location match (gated by content relevance)
        location_bonus = 0.0
        if has_content_relevance and query_loc and cand_location:
            if query_loc in cand_location or cand_location in query_loc:
                location_bonus = 0.15
            elif set(query_loc.split()) & set(cand_location.split()):
                location_bonus = 0.08

        # 5. Composite score calculation
        if not has_content_relevance:
            final_score = 0.0
        else:
            raw_score = (text_score * 0.50) + title_bonus + keyword_bonus + location_bonus
            final_score = min(raw_score, 1.0)

        results.append((round(final_score, 3), row))

    return sorted(results, key=lambda x: x[0], reverse=True)


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

    1. **Report**: Submit details of an item you lost or found on campus.
    2. **Smart Match**: VITFind runs an enhanced multi-factor similarity check (N-gram TF-IDF + brand & location weighting).
    3. **Protect**: Student emails and phone numbers are scrape-protected and masked by default.
    4. **Resolve**: Mark items as Claimed or Resolved once returned to keep the database fresh.
    """)

    reports = get_reports()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Reports", len(reports))
    c2.metric(
        "Active Lost",
        sum(r.get("report_type") == "Lost" and r.get("status", "Open") == "Open" for r in reports)
    )
    c3.metric(
        "Active Found",
        sum(r.get("report_type") == "Found" and r.get("status", "Open") == "Open" for r in reports)
    )
    c4.metric(
        "Claimed / Resolved",
        sum(r.get("status", "Open") in ("Claimed", "Resolved") for r in reports)
    )


elif page == "📢 Report Item":

    st.header("Report an Item")

    with st.form("report_form"):

        col_type, col_cat = st.columns(2)

        report_type = col_type.selectbox(
            "Report type",
            ["Lost", "Found"]
        )

        category = col_cat.selectbox(
            "Category",
            CATEGORIES
        )

        item_name = st.text_input(
            "Item name / Brand",
            placeholder="e.g. Black JBL Headphones, Titan Watch, Blue Milton Bottle"
        )

        description = st.text_area(
            "Detailed description",
            placeholder="Colour, brand, unique marks, approximate time seen, etc."
        )

        location = st.text_input(
            "Location",
            placeholder="e.g. Library 2nd Floor, SJT 412, TT Foodys"
        )

        contact = st.text_input(
            "Contact",
            placeholder="College email or phone number"
        )

        submitted = st.form_submit_button(
            "Submit Report"
        )

    if submitted:

        saved, msg = add_report(
            report_type=report_type,
            item_name=item_name,
            description=description,
            location=location,
            contact=contact,
            category=category
        )

        if saved:
            st.success(msg)
            st.cache_data.clear()
        else:
            st.error(msg)


elif page == "🔍 Find Matches":

    st.header("Find Possible Matches")

    search_direction = st.radio(
        "Search Direction",
        ["I lost an item (search Found reports)", "I found an item (search Lost reports)"],
        horizontal=True
    )
    search_target = "Found" if "Found reports" in search_direction else "Lost"

    col_name, col_cat, col_loc = st.columns([2, 2, 2])
    query_name = col_name.text_input(
        "Item Name / Brand",
        placeholder="e.g. Black JBL Headphones, Casio Watch"
    )
    category_filter = col_cat.selectbox(
        "Category Filter",
        ["All Categories"] + CATEGORIES
    )
    query_location = col_loc.text_input(
        "Location",
        placeholder="e.g. Library, SJT, TT Foodys"
    )

    query_description = st.text_area(
        "Detailed Description",
        placeholder="Mention colour, marks, model, key identifiers..."
    )

    if st.button("🔎 Find Matches"):

        if not query_name.strip() and not query_description.strip():

            st.warning(
                "Please enter at least an item name or description to search."
            )

        else:

            matches = find_matches(
                query_description=query_description,
                query_name=query_name,
                query_location=query_location,
                search_target=search_target,
                category_filter=category_filter
            )

            if not matches:

                st.info(
                    f"No active {search_target.lower()} item reports found matching criteria."
                )

            else:

                shown = 0

                for score, row in matches:

                    # Suppress low-relevance noise below 20%
                    if score >= 0.20:

                        shown += 1

                        if score >= 0.70:
                            badge = "🟢 High Match"
                        elif score >= 0.40:
                            badge = "🟡 Moderate Match"
                        else:
                            badge = "🔵 Potential Match"

                        st.markdown(
                            f"### {badge} #{shown} — {score * 100:.1f}% Match"
                        )

                        st.progress(
                            float(score)
                        )

                        st.write(
                            f"**Item:** {row.get('item_name', 'Unknown')} &nbsp;|&nbsp; **Category:** {row.get('category', '📦 Other')}"
                        )

                        st.write(
                            f"**Description:** {row.get('description', '')}"
                        )

                        st.write(
                            f"**Location:** {row.get('location', '')}"
                        )

                        st.write(
                            f"**Reported:** {row.get('created_at', '')}"
                        )

                        contact_val = row.get("contact", "").strip()
                        st.markdown(f"**Contact:** `{mask_contact(contact_val)}`")
                        with st.expander("👁️ View Contact Details"):
                            st.code(contact_val, language="text")

                        st.divider()

                if shown == 0:

                    st.info(
                        "No confident matches found. Try broadening your description or removing filters."
                    )


elif page == "📋 All Reports":

    st.header("Campus Reports")

    reports = get_reports()

    if not reports:

        st.info(
            "No reports yet. Add one from 'Report Item'."
        )

    else:

        col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
        search_kw = col_f1.text_input("🔎 Search reports", placeholder="Filter by item, location, description...")
        filter_type = col_f2.selectbox("Filter Type", ["All Types", "Lost", "Found"])
        filter_status = col_f3.selectbox("Filter Status", ["All Statuses", "Open", "Claimed", "Resolved"])

        filtered = reports

        if filter_type != "All Types":
            filtered = [r for r in filtered if r.get("report_type") == filter_type]

        if filter_status != "All Statuses":
            filtered = [r for r in filtered if r.get("status", "Open") == filter_status]

        if search_kw.strip():
            kw = search_kw.strip().lower()
            filtered = [
                r for r in filtered
                if kw in str(r.get("item_name", "")).lower()
                or kw in str(r.get("description", "")).lower()
                or kw in str(r.get("location", "")).lower()
            ]

        st.caption(f"Showing {len(filtered)} of {len(reports)} reports")

        for row in filtered:

            type_label = "🔴 LOST" if row.get("report_type") == "Lost" else "🟢 FOUND"
            status_val = row.get("status", "Open")

            if status_val == "Claimed":
                status_badge = "🤝 CLAIMED"
            elif status_val == "Resolved":
                status_badge = "✅ RESOLVED"
            else:
                status_badge = "🟢 OPEN"

            category_val = row.get("category", "📦 Other")
            title_text = f"{type_label} [{status_badge}] — {row.get('item_name', 'Unnamed Item')} ({category_val})"

            with st.expander(title_text):

                st.write(
                    f"**Description:** {row.get('description', '')}"
                )

                st.write(
                    f"**Location:** {row.get('location', '')}"
                )

                contact_val = row.get("contact", "").strip()
                st.markdown(f"**Contact:** `{mask_contact(contact_val)}`")
                with st.expander("👁️ View Contact Details"):
                    st.code(contact_val, language="text")

                st.caption(
                    f"Reported: {row.get('created_at', '')}"
                )

                # Status lifecycle action
                report_id = row.get("id")
                if status_val == "Open" and report_id:
                    st.markdown("---")
                    st.caption("Has this item been reunited with its owner?")
                    btn_col1, btn_col2 = st.columns(2)
                    if btn_col1.button("🤝 Mark as Claimed", key=f"claim_{report_id}"):
                        update_report_status(report_id, "Claimed")
                        st.success("Item marked as Claimed!")
                        st.rerun()
                    if btn_col2.button("✅ Mark as Resolved", key=f"resolve_{report_id}"):
                        update_report_status(report_id, "Resolved")
                        st.success("Item marked as Resolved!")
                        st.rerun()
