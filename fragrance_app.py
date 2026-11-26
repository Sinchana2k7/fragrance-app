import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from html import escape

st.set_page_config(page_title="ScentScout", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;1,500&family=DM+Sans:wght@400;500;700&display=swap');

.stApp {
    --forest: #030D07;
    --plum: #301224;
    --button-plum: #6E214A;
    --rose: #C47A99;
    --ivory: #EDEBE4;
    --cream: #EDDEC5;
    --line: #C47A9960;
    background:
        radial-gradient(ellipse at 50% 0%, rgba(236, 187, 120, 0.13), transparent 700px),
        #100E0B;
    color: var(--ivory);
}

.stApp, .stApp p, .stApp input, .stApp button {
    font-family: 'DM Sans', sans-serif;
}

.stApp h1, .stApp h2, .stApp h3 {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-weight: 600;
    color: var(--ivory);
}

.stApp p { color: var(--cream); }
.stApp [data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1080px; padding: 4.5rem 2rem; }

.hero-frame {
    position: relative;
    isolation: isolate;
    overflow: hidden;
    text-align: center;
    background:
        radial-gradient(ellipse at 50% 110%, var(--plum), transparent 75%),
        radial-gradient(ellipse at 50% 0%, rgba(236, 187, 120, 0.16), transparent 72%),
        #15120E;
    border: 1px solid var(--line);
    border-radius: 1.25rem;
    padding: clamp(3rem, 8vw, 6rem) 1.25rem;
}

.hero-frame::before, .hero-frame::after {
    content: "";
    position: absolute;
    z-index: -1;
    width: 44%;
    aspect-ratio: 1 / 1.4;
    bottom: -65%;
    border: 1px solid var(--line);
    border-radius: 50% 50% 0 0;
}

.hero-frame::before { left: -15%; transform: rotate(25deg); }
.hero-frame::after { right: -15%; transform: rotate(-25deg); }

.hero-kicker, .brand-name, .notes-label {
    color: var(--rose);
    text-transform: uppercase;
    letter-spacing: .14em;
    font-size: .75rem;
    font-weight: 700;
}

.hero-frame h1 {
    font-size: clamp(3rem, 8vw, 6.25rem);
    line-height: .94;
    letter-spacing: -.04em;
    margin: 1.25rem 0;
    padding: 0;
}

.hero-frame em { display: block; color: #e6a9c0; font-weight: 500; }
.hero-frame p { max-width: 27rem; margin: 0 auto; }
.st-key-discovery { margin-top: .5rem; }

.st-key-perfume_choice, .st-key-notes_choice, .perfume-card {
    border: 1px solid var(--line);
    border-radius: 1rem;
    background: linear-gradient(145deg, var(--plum), #1a151a);
    padding: clamp(1rem, 4vw, 1.75rem);
}

.scent-choice h2 { font-size: clamp(1.85rem, 4vw, 2.5rem); padding: 0; }
.scent-choice p { margin-bottom: .25rem; }

.stApp .stButton button {
    background: var(--button-plum);
    color: var(--ivory);
    border: 1px solid var(--rose);
    border-radius: .65rem;
    min-height: 2.85rem;
    padding: .5rem 1rem;
}

.stApp .stButton button p { color: inherit; }
.stApp .stButton button:hover { background: var(--rose); color: var(--forest); }

.stApp .stButton button:focus-visible,
.stApp [data-baseweb="select"]:focus-within {
    outline: 2px solid var(--cream);
    outline-offset: 3px;
}

.stApp .st-key-back_perfume button, .stApp .st-key-back_notes button {
    background: transparent;
    border-color: var(--line);
}

.stApp [data-baseweb="select"] > div {
    background: #151c17;
    border-color: var(--line);
    border-radius: .65rem;
    color: var(--ivory);
}

.stApp [data-baseweb="select"] input {
    color: var(--ivory);
    -webkit-text-fill-color: var(--ivory);
}

.stApp [data-baseweb="select"] svg { fill: var(--ivory); }

.perfume-card { overflow-wrap: anywhere; }

.card-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: .5rem;
}

.top-badge {
    background: var(--rose);
    color: var(--forest);
    border-radius: 999px;
    padding: .2rem .65rem;
    font-size: .7rem;
    font-weight: 700;
}

.perfume-card h2 { font-size: 2.1rem; line-height: 1; padding: .55rem 0 .9rem; }
.perfume-card .card-stats p { margin: .15rem 0; font-size: .88rem; }
.shared-notes { border-top: 1px solid var(--line); }
.note-tags { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .4rem; font-size: .84rem; }
.other-notes-label { margin-top: .9rem; }

.fragrance-family-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: .4rem 0 0;
}

.fragrance-family-tags:empty { display: none; }

.fragrance-family-tag {
    display: inline-block;
    padding: 5px 11px;
    border: 1px solid var(--rose);
    border-radius: 20px;
    background: #4A1C35;
    color: var(--cream);
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 500;
}

.note-tag, .note-more {
    background: #6E214A52;
    color: var(--ivory);
    border-radius: 999px;
    padding: .3rem .65rem;
}

.results-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    grid-auto-rows: 1fr;
    gap: 1.25rem 2rem;
    margin-top: .5rem;
}

.results-grid .perfume-card {
    box-sizing: border-box;
    height: auto;
    margin: 0;
    padding: 1.5rem;
    overflow: visible;
}

.results-grid .card-stats { margin-top: .75rem; }

.results-grid .shared-notes {
    margin-top: 1.25rem;
    padding-top: 1rem;
}

@media (max-width: 640px) {
    .block-container { padding: 3.5rem 1rem 2rem; }

    .st-key-discovery [data-testid="stHorizontalBlock"] {
        flex-direction: column;
    }

    .st-key-discovery [data-testid="stColumn"] {
        width: 100%;
        flex: auto;
    }

    .results-grid {
        grid-template-columns: minmax(0, 1fr);
        grid-auto-rows: auto;
        gap: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_fragrance_data():
    df = pd.read_csv(
        "data/fragrances_active_cleaned_v5.csv",
        encoding="utf-8",
    ).reset_index(drop=True)
    df["All_Notes"] = df["All_Notes"].fillna("").str.lower().apply(
        lambda x: [note.strip() for note in x.split(",") if note.strip()]
    )
    mlb = MultiLabelBinarizer()
    note_matrix = mlb.fit_transform(df["All_Notes"])
    return df, mlb, note_matrix


df, mlb, note_matrix = load_fragrance_data()

PRICE_DISCLAIMER = (
    "Prices are reference estimates in USD and may vary by retailer, "
    "bottle size, and promotions."
)

if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    st.markdown("""
    <section class="hero-frame">
        <div class="hero-kicker">SCENTSCOUT</div>
        <h1>Find your <em>perfect scent.</em></h1>
        <p>Discover a new favorite through the perfumes and notes you already love.</p>
    </section>
    """, unsafe_allow_html=True)


    with st.container(key="discovery"):
        perfume_col, notes_col = st.columns(2, gap="medium")

        with perfume_col, st.container(key="perfume_choice"):
            st.markdown(
                '<div class="scent-choice"><h2>Your Fragrance Twin</h2>'
                '<p>Start with a perfume you love and find fragrances with similar notes.</p></div>',
                unsafe_allow_html=True,
            )
            if st.button(
                "Explore by Perfume →", key="home_perfume", use_container_width=True
            ):
                st.session_state.page = "perfume"
                st.rerun()

        with notes_col, st.container(key="notes_choice"):
            st.markdown(
                '<div class="scent-choice"><h2>Discover by Notes</h2>'
                '<p>Choose up to five notes and explore fragrances that feature them.</p></div>',
                unsafe_allow_html=True,
            )
            if st.button(
                "Explore by Notes →", key="home_notes", use_container_width=True
            ):
                st.session_state.page = "notes"
                st.rerun()

    st.stop()


if st.session_state.page == "perfume":
    if st.button("← Back to Home", key="back_perfume"):
        st.session_state.page = "home"
        st.rerun()

    st.subheader("Find Your Fragrance Twin")
    st.caption(
        "Find fragrances with notes similar to your favorite perfume. "
        "Higher matches mean more notes in common."
    )

    selected_index = st.selectbox(
        "🔎 Search for a perfume you love",
        options=df.index[df["Perfume"].notna()].tolist(),
        index=None,
        placeholder="Type a perfume or brand name...",
        format_func=lambda i: (
            f'{str(df.loc[i, "Brand"]).replace("-", " ").title()}'
            f' — {str(df.loc[i, "Perfume"]).replace("-", " ").title()}'
        ),
    )
    st.caption(PRICE_DISCLAIMER)

    if st.button("Find My Matches"):
        if selected_index is None:
            st.warning("Please search for and select a perfume first.")
        else:
            selected_perfume_notes = set(df.loc[selected_index, "All_Notes"])
            scores = cosine_similarity(
                note_matrix[selected_index:selected_index + 1],
                note_matrix,
            )[0]
            ranked_indices = np.argsort(scores)[::-1]
            ranked_indices = ranked_indices[ranked_indices != selected_index]
            top_four = ranked_indices[:4]

            recommendations = df.iloc[top_four][
                ["Perfume", "Brand", "Rating Value", "Price_USD", "Fragrance_Families"]
            ].copy()
            recommendations["Similarity Score"] = scores[top_four].round(2)

            st.subheader("Your Fragrance Matches")
            cards = []

            for i, (perfume_index, perfume) in enumerate(recommendations.iterrows()):
                recommended_notes = set(df.loc[perfume_index, "All_Notes"])
                shared_notes = sorted(selected_perfume_notes & recommended_notes)
                other_notes = sorted(recommended_notes - selected_perfume_notes)

                visible_notes = shared_notes[:3]
                notes_html = " ".join(
                    f'<span class="note-tag">{escape(note.replace("-", " ").title())}</span>'
                    for note in visible_notes
                )
                remaining = len(shared_notes) - len(visible_notes)
                if remaining > 0:
                    notes_html += f' <span class="note-more">+{remaining} more</span>'

                visible_other = other_notes[:3]
                other_html = " ".join(
                    f'<span class="note-tag">{escape(note.replace("-", " ").title())}</span>'
                    for note in visible_other
                )
                other_remaining = len(other_notes) - len(visible_other)
                if other_remaining > 0:
                    other_html += f' <span class="note-more">+{other_remaining} more</span>'

                badge = '<div class="top-badge">#1 MATCH</div>' if i == 0 else ''
                brand = escape(str(perfume["Brand"]).replace("-", " ").title())
                name = escape(str(perfume["Perfume"]).replace("-", " ").title())
                family_value = perfume.get("Fragrance_Families", "")
                family_html = "".join(
                    f'<span class="fragrance-family-tag">{escape(family.strip())}</span>'
                    for family in str(family_value).split(",")
                    if family.strip() and family.strip().lower() != "nan"
                )

                similarity = f'{float(perfume["Similarity Score"]) * 100:.0f}%'
                rating_value = pd.to_numeric(perfume["Rating Value"], errors="coerce")
                rating = f"{rating_value:.2f} / 5" if pd.notna(rating_value) else "Not rated"

                card_html = (
                    '<div class="perfume-card has-notes">'
                    '<div class="card-header"><div>'
                    f'<span class="brand-name">{brand}</span>'
                    f'<div class="fragrance-family-tags">{family_html}</div>'
                    f'</div>{badge}</div>'
                    f'<h2>{name}</h2>'
                    '<div class="card-stats">'
                    f'<p>Similarity: {similarity}</p>'
                    f'<p>Rating: {rating}</p>'
                    f'<p>Reference price: ${float(perfume["Price_USD"]):,.2f}</p>'
                    '</div>'
                    '<div class="shared-notes">'
                    '<div class="notes-label">SHARED NOTES</div>'
                    f'<div class="note-tags">{notes_html}</div>'
                    '<div class="notes-label other-notes-label">OTHER NOTES TO EXPLORE</div>'
                    f'<div class="note-tags">{other_html or "No additional notes"}</div>'
                    '</div></div>'
                )
                cards.append(card_html)

            st.markdown(
                '<div class="results-grid">' + "".join(cards) + "</div>",
                unsafe_allow_html=True,
            )


if st.session_state.page == "notes":
    if st.button("← Back to Home", key="back_notes"):
        st.session_state.page = "home"
        st.rerun()

    st.subheader("Discover by Notes")
    st.caption(
        "Choose your favorite fragrance notes to discover "
        "perfumes that match your preferences."
    )
    selected_notes = st.multiselect(
        "Choose up to 5 fragrance notes",
        options=list(mlb.classes_),
        max_selections=5,
    )
    st.caption(PRICE_DISCLAIMER)

    if st.button("Find My Scent"):
        if not selected_notes:
            st.warning("Please select at least one fragrance note.")
        else:
            user_vector = mlb.transform([selected_notes])
            note_scores = cosine_similarity(user_vector, note_matrix)[0]
            ranked_notes = np.argsort(-note_scores, kind="stable")
            top_notes = ranked_notes[note_scores[ranked_notes] > 0][:4]

            note_recommendations = df.iloc[top_notes][
                ["Perfume", "Brand", "Rating Value", "Price_USD", "Fragrance_Families"]
            ].copy()
            note_recommendations["Similarity Score"] = note_scores[top_notes].round(2)

            st.subheader("Your Personalized Fragrance Matches")
            st.caption(
                "Discover fragrances featuring your favorite notes. "
                "Higher matches mean a closer match to your selections."
            )

            cards = []
            for i, (perfume_index, perfume) in enumerate(note_recommendations.iterrows()):
                perfume_notes = set(df.loc[perfume_index, "All_Notes"])
                matching_notes = sorted(set(selected_notes) & perfume_notes)
                other_notes = sorted(perfume_notes - set(selected_notes))

                matching_html = " ".join(
                    f'<span class="note-tag">{escape(note.replace("-", " ").title())}</span>'
                    for note in matching_notes
                )

                visible_other = other_notes[:3]
                other_html = " ".join(
                    f'<span class="note-tag">{escape(note.replace("-", " ").title())}</span>'
                    for note in visible_other
                )
                remaining = len(other_notes) - len(visible_other)
                if remaining > 0:
                    other_html += f' <span class="note-more">+{remaining} more</span>'

                badge = '<div class="top-badge">#1 MATCH</div>' if i == 0 else ''
                brand = escape(str(perfume["Brand"]).replace("-", " ").title())
                name = escape(str(perfume["Perfume"]).replace("-", " ").title())
                family_value = perfume.get("Fragrance_Families", "")
                family_html = "".join(
                    f'<span class="fragrance-family-tag">{escape(family.strip())}</span>'
                    for family in str(family_value).split(",")
                    if family.strip() and family.strip().lower() != "nan"
                )

                similarity = f'{float(perfume["Similarity Score"]) * 100:.0f}%'
                rating_value = str(perfume["Rating Value"]).replace(",", ".")
                try:
                    rating = (
                        f"{float(rating_value):.2f} / 5"
                        if pd.notna(perfume["Rating Value"])
                        else "Not rated"
                    )
                except ValueError:
                    rating = "Not rated"

                card_html = (
                    '<div class="perfume-card notes-discovery-card">'
                    '<div class="card-header"><div>'
                    f'<span class="brand-name">{brand}</span>'
                    f'<div class="fragrance-family-tags">{family_html}</div>'
                    f'</div>{badge}</div>'
                    f'<h2>{name}</h2>'
                    '<div class="card-stats">'
                    f'<p>Similarity: {similarity}</p>'
                    f'<p>Rating: {rating}</p>'
                    f'<p>Reference price: ${float(perfume["Price_USD"]):,.2f}</p>'
                    '</div>'
                    '<div class="shared-notes">'
                    '<div class="notes-label">MATCHES YOUR NOTES</div>'
                    f'<div class="note-tags">{matching_html}</div>'
                    '<div class="notes-label other-notes-label">OTHER NOTES TO EXPLORE</div>'
                    f'<div class="note-tags">{other_html}</div>'
                    '</div></div>'
                )
                cards.append(card_html)

            st.markdown(
                '<div class="results-grid">' + "".join(cards) + "</div>",
                unsafe_allow_html=True,
            )