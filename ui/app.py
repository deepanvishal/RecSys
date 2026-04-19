from pathlib import Path
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components


API_URL = 'http://localhost:8000'
BIAS_DIR = Path('artifacts/bias')

ALL_GENRES = ['Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'FilmNoir', 'Horror', 'Musical',
              'Mystery', 'Romance', 'SciFi', 'Thriller', 'War', 'Western']

GENDER_OPTIONS = {'Male': 1, 'Female': 0}
AGE_OPTIONS = {'<18': 0, '18-24': 1, '25-34': 2, '35-44': 3,
               '45-49': 4, '50-55': 5, '56+': 6}
AGE_LABELS = {v: k for k, v in AGE_OPTIONS.items()}


# ── Data loaders ──────────────────────────────────────────────────────────────


@st.cache_data
def load_movies():
    meta = pd.read_parquet('data/processed/item_metadata.parquet')
    return meta.set_index('item_id')[['title', 'genres', 'year']]


@st.cache_data
def load_title_map():
    """title string -> item_id, sorted alphabetically."""
    meta = pd.read_parquet('data/processed/item_metadata.parquet')
    mapping = dict(zip(meta['title'], meta['item_id']))
    return dict(sorted(mapping.items()))


@st.cache_data
def load_interaction_demo():
    train = pd.read_parquet('data/processed/train.parquet')[['user_id', 'item_id']]
    umeta = pd.read_parquet('data/processed/user_metadata.parquet')
    umeta = umeta[['user_id', 'gender', 'age_enc']]
    return train.merge(umeta, on='user_id')


@st.cache_data
def load_bias_data():
    return {
        'summary':    pd.read_csv(BIAS_DIR / 'bias_summary.csv'),
        'gender':     pd.read_csv(BIAS_DIR / 'sasrec_gender_bias.csv'),
        'age':        pd.read_csv(BIAS_DIR / 'sasrec_age_bias.csv'),
        'genre':      pd.read_csv(BIAS_DIR / 'sasrec_genre_concentration.csv'),
        'pop_sasrec': pd.read_csv(BIAS_DIR / 'sasrec_popularity_bias.csv'),
        'pop_svd':    pd.read_csv(BIAS_DIR / 'svd_popularity_bias.csv'),
    }


@st.cache_data
def load_user_profiles():
    train = pd.read_parquet('data/processed/train.parquet')
    item_meta = pd.read_parquet('data/processed/item_metadata.parquet')

    user_hist = (
        train.sort_values('timestamp')
        .groupby('user_id')['item_id'].apply(list).to_dict()
    )
    hist_len = {uid: len(h) for uid, h in user_hist.items()}
    genre_map = item_meta.set_index('item_id')['genres']

    def top_genre(uid):
        counts = {}
        for iid in user_hist.get(uid, []):
            try:
                gs = genre_map.loc[iid].split('|')
                for g in gs:
                    counts[g] = counts.get(g, 0) + 1
            except Exception:
                pass
        return max(counts, key=counts.get) if counts else 'Unknown'

    lens = pd.Series(hist_len)
    alex_id = int(lens.idxmax())

    sarah_candidates = lens[(lens >= 40) & (lens <= 60)].index.tolist()
    sarah_match = next(
        (u for u in sarah_candidates if top_genre(u) in ('Drama', 'Romance')),
        sarah_candidates[0] if sarah_candidates else alex_id,
    )
    sarah_id = int(sarah_match)

    marcus_candidates = lens[(lens >= 15) & (lens <= 25)].index.tolist()
    marcus_id = int(marcus_candidates[0]) if marcus_candidates else alex_id

    priya_candidates = lens[lens >= 20].index.tolist()
    priya_id = int(priya_candidates[len(priya_candidates) // 2]) if priya_candidates else alex_id

    return {
        'Alex — Power User': {
            'user_id': alex_id, 'history': user_hist[alex_id],
            'description': f'{len(user_hist[alex_id])} movies watched',
        },
        'Sarah — Movie Enthusiast': {
            'user_id': sarah_id, 'history': user_hist[sarah_id],
            'description': f'{len(user_hist[sarah_id])} movies watched',
        },
        'Marcus — Casual Viewer': {
            'user_id': marcus_id, 'history': user_hist[marcus_id],
            'description': f'{len(user_hist[marcus_id])} movies watched',
        },
        'Priya — Light Viewer': {
            'user_id': priya_id, 'history': user_hist[priya_id][:4],
            'description': '4 movies watched (sparse)',
        },
        'New User': {
            'user_id': None, 'history': [],
            'description': 'No watch history',
        },
    }


# ── API helpers ───────────────────────────────────────────────────────────────


def api_post(endpoint, payload, timeout=30):
    try:
        r = requests.post(f'{API_URL}{endpoint}', json=payload, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_homepage(history, gender_enc=None, age_enc=None, n=10):
    return api_post('/homepage', {
        'history': history, 'gender_enc': gender_enc,
        'age_enc': age_enc, 'n': n,
    }, timeout=30)


@st.cache_data(ttl=300, show_spinner=False)
def cached_homepage(history_tuple, gender_enc, age_enc, n):
    return api_homepage(list(history_tuple), gender_enc, age_enc, n)


@st.cache_data(ttl=300, show_spinner=False)
def cached_item_similarity(item_id, n):
    return api_post('/item_similarity', {'item_id': int(item_id), 'n': int(n)})


# ── Display helpers ───────────────────────────────────────────────────────────


def item_label(item_id, movies):
    try:
        row = movies.loc[int(item_id)]
        return f"{row['title']}"
    except Exception:
        return f'Item {item_id}'


def item_genres(item_id, movies):
    try:
        return movies.loc[int(item_id)]['genres']
    except Exception:
        return ''


def recs_to_table(items, movies):
    rows = []
    for i, iid in enumerate(items):
        rows.append({
            'Rank': i + 1,
            'Title': item_label(iid, movies),
            'Genres': item_genres(iid, movies),
        })
    return pd.DataFrame(rows)


def sim_to_table(items, scores, movies):
    rows = []
    for i, (iid, sc) in enumerate(zip(items, scores)):
        rows.append({
            'Rank': i + 1,
            'Title': item_label(iid, movies),
            'Genres': item_genres(iid, movies),
            'Score': round(float(sc), 4),
        })
    return pd.DataFrame(rows)


def _render_movie_card(rank, item_id, score, movies):
    title = item_label(item_id, movies)
    genres = item_genres(item_id, movies)
    st.markdown(
        f'''
        <div style="padding:12px; border:1px solid #333; border-radius:8px;
                    min-height:140px; background:rgba(255,255,255,0.03);
                    margin-bottom:8px;">
          <div style="font-size:13px; color:#888;">#{rank}</div>
          <div style="font-size:15px; font-weight:600; margin-top:4px;
                      line-height:1.3;">{title}</div>
          <div style="font-size:12px; color:#999; margin-top:6px;
                      line-height:1.3;">{genres}</div>
          <div style="font-size:12px; color:#6cf; margin-top:8px;">
                score: {score:.3f}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_sim_grid(label, items, scores, movies):
    st.markdown(f'### {label}')
    n = len(items)
    per_row = 5
    rows = (n + per_row - 1) // per_row
    for r in range(rows):
        cols = st.columns(per_row)
        for c in range(per_row):
            idx = r * per_row + c
            if idx < n:
                with cols[c]:
                    _render_movie_card(idx + 1, items[idx], scores[idx], movies)


def interaction_dist(item_id, idf):
    sub = idf[idf['item_id'] == int(item_id)]
    gender = sub['gender'].value_counts().rename('Count').reset_index()
    gender.columns = ['Gender', 'Interactions']
    age = sub['age_enc'].map(AGE_LABELS).value_counts().rename('Count').reset_index()
    age.columns = ['Age Group', 'Interactions']
    age = age.sort_values('Age Group')
    return gender, age, len(sub)


# ── App ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title='Serko RecSys', layout='wide')


# Global CSS — bump fonts, wrap table cells, larger headings.
st.markdown(
    """
    <style>
    html, body, .stApp, [class*="st-"] {
        font-size: 17px;
    }
    .stMarkdown p, .stMarkdown li {
        font-size: 17px;
        line-height: 1.55;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        font-size: 15px !important;
        color: #aaa !important;
    }
    h1 { font-size: 36px !important; }
    h2 { font-size: 28px !important; }
    h3 { font-size: 22px !important; }
    h4 { font-size: 19px !important; }

    /* st.table rendered tables — wrap text, larger font */
    [data-testid="stTable"] table {
        font-size: 16px;
        table-layout: fixed;
        width: 100%;
    }
    [data-testid="stTable"] table th,
    [data-testid="stTable"] table td {
        white-space: normal !important;
        word-wrap: break-word !important;
        word-break: break-word !important;
        vertical-align: top;
        padding: 8px 10px;
    }

    /* st.dataframe (interactive grid) — bump font where possible */
    [data-testid="stDataFrame"] {
        font-size: 16px;
    }

    /* Tabs — bold, larger labels */
    [data-baseweb="tab-list"] {
        gap: 8px;
    }
    [data-baseweb="tab"] {
        font-size: 20px !important;
        font-weight: 700 !important;
        padding: 12px 18px !important;
        letter-spacing: 0.2px;
    }
    [data-baseweb="tab"] [data-testid="stMarkdownContainer"] p {
        font-size: 20px !important;
        font-weight: 700 !important;
    }
    [data-baseweb="tab"][aria-selected="true"] {
        color: #4f8bff !important;
    }

    /* Metric labels & values */
    [data-testid="stMetricLabel"] { font-size: 15px !important; }
    [data-testid="stMetricValue"] { font-size: 26px !important; }
    [data-testid="stMetricDelta"] { font-size: 15px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title('Serko RecSys — MovieLens 1M')
st.caption(
    'A tiered recommendation engine on MovieLens 1M. '
    'Five models — Collaborative Filtering, SVD, Two-Tower, SASRec, BERT4Rec — '
    'each with a different approach to the recommendation problem.'
)

movies = load_movies()
title_map = load_title_map()
idf = load_interaction_demo()


tab_for_you, tab_sim, tab_bias, tab_cold, tab_wandb = st.tabs([
    'For You', 'Item Similarity', 'Model Bias', 'Cold Start', 'Training Runs',
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FOR YOU (Netflix-style homepage)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_homepage_row(row, movies_df):
    items = row.get('items', [])
    if not items:
        return
    label = row['label']
    model = row['model']
    st.markdown(f'### {label}')
    model_note = {
        'sasrec':             'SASRec sequential model',
        'popularity':         'Popularity-based',
        'two_tower_filtered': f"Two-Tower → filtered to {row.get('genre', '')}",
        'demographic_genre':  f"Popular in {row.get('genre', '')} for your demographic",
        'simulated':          'Simulated new arrivals',
    }.get(model, model)
    st.caption(model_note)

    per_row = 5
    n = len(items)
    n_rows = (n + per_row - 1) // per_row
    for r_idx in range(n_rows):
        cols = st.columns(per_row)
        for c_idx in range(per_row):
            i = r_idx * per_row + c_idx
            if i >= n:
                continue
            iid = items[i]
            with cols[c_idx]:
                try:
                    mr = movies_df.loc[int(iid)]
                    title = mr['title']
                    genres = mr['genres']
                    year = int(mr['year']) if pd.notna(mr['year']) else ''
                except Exception:
                    title, genres, year = f'Item {iid}', '', ''
                genres_short = (genres or '')[:60]
                card_html = (
                    "<div style=\"background:#1e1e2e;padding:12px;border-radius:8px;"
                    "min-height:140px;font-size:13px;color:#eee;margin-bottom:8px;\">"
                    f"<div style=\"font-size:13px;color:#888;\">#{i + 1}</div>"
                    f"<div style=\"font-size:15px;font-weight:600;margin-top:4px;"
                    "line-height:1.3;\">"
                    f"{title}</div>"
                    f"<div style=\"font-size:12px;color:#aaa;margin-top:6px;\">{year}</div>"
                    f"<div style=\"font-size:12px;color:#999;margin-top:4px;"
                    "line-height:1.3;\">"
                    f"{genres_short}</div>"
                    "</div>"
                )
                st.markdown(card_html, unsafe_allow_html=True)
    st.markdown('')


with tab_for_you:
    profiles = load_user_profiles()

    st.subheader('For You — Personalised Homepage')
    st.markdown(
        'A Netflix-style homepage built from your watch history. '
        'Each row is generated by a different model: SASRec for sequential next-item predictions, '
        'Two-Tower for genre-filtered recommendations, and popularity for trending fallbacks.'
    )
    st.divider()

    selected_profile = st.selectbox('Select a user profile', list(profiles.keys()))
    profile = profiles[selected_profile]
    st.caption(profile['description'])

    gender_enc_hp = None
    age_enc_hp = None
    if selected_profile == 'New User':
        c1, c2 = st.columns(2)
        with c1:
            g_sel = st.selectbox('Gender', list(GENDER_OPTIONS.keys()), key='hp_gender')
        with c2:
            a_sel = st.selectbox('Age group', list(AGE_OPTIONS.keys()), key='hp_age')
        gender_enc_hp = GENDER_OPTIONS[g_sel]
        age_enc_hp = AGE_OPTIONS[a_sel]

    with st.spinner('Building your homepage...'):
        hp_result = cached_homepage(
            tuple(profile['history']),
            gender_enc_hp,
            age_enc_hp,
            10,
        )

    if hp_result is None:
        st.error('API not reachable.')
    else:
        st.divider()
        for row in hp_result['rows']:
            _render_homepage_row(row, movies)
            st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ITEM SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.subheader('Item Similarity — Find Similar Movies')
    st.markdown(
        'Pick a movie. Three different models find the most similar movies in the catalog. '
        'Each model defines "similar" differently:'
    )
    st.markdown(
        '- **CF (Item-Item Cosine)** — two movies are similar if the same users tend to watch both. '
        'Pure co-occurrence signal from the user-item matrix.\n'
        '- **SVD (Latent Factors)** — two movies are similar if they share latent taste dimensions '
        '(e.g. "dark serious drama", "popcorn action"). Learned by matrix factorisation.\n'
        '- **Two-Tower (Content + ID)** — two movies are similar if their title text + genres + '
        'learned ID embedding produce close 128-dim vectors. Works for new movies with no ratings.'
    )
    st.divider()

    titles = list(title_map.keys())
    default_idx = titles.index('Star Wars (1977)') if 'Star Wars (1977)' in title_map else 0
    selected_title = st.selectbox('Select a movie', options=titles, index=default_idx, key='sim_title')
    selected_id = title_map[selected_title]

    n_recs = st.slider('Similar movies per model', 5, 15, 10, key='sim_n')

    st.divider()

    row = movies.loc[selected_id]
    info_c1, info_c2, info_c3 = st.columns(3)
    info_c1.metric('Title', row['title'])
    info_c2.metric('Genres', row['genres'])
    info_c3.metric('Year', int(row['year']) if pd.notna(row['year']) else 'N/A')

    gender_dist, age_dist, total_interactions = interaction_dist(selected_id, idf)
    st.markdown(f'**{total_interactions:,} interactions** in training data')

    dist_c1, dist_c2 = st.columns(2)
    with dist_c1:
        st.caption('Interactions by gender')
        st.bar_chart(gender_dist.set_index('Gender')['Interactions'])
    with dist_c2:
        st.caption('Interactions by age group')
        st.bar_chart(age_dist.set_index('Age Group')['Interactions'])

    with st.spinner('Computing similarities...'):
        sim_result = cached_item_similarity(int(selected_id), int(n_recs))

    if sim_result is None:
        st.error('API not reachable on localhost:8000')
    else:
        st.divider()
        for key, label in [
            ('cf', 'CF — Cosine Similarity'),
            ('svd', 'SVD — Latent Factors'),
            ('two_tower', 'Two-Tower — Content + ID'),
        ]:
            _render_sim_grid(
                label,
                sim_result[key]['items'],
                sim_result[key]['scores'],
                movies,
            )
            st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL BIAS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_bias:
    bias = load_bias_data()
    s = bias['summary'].iloc[0]

    st.subheader('Model Bias Audit')
    st.markdown(
        '**Goal.** A recommender can be accurate overall but unfair to subgroups, '
        'over-recommend popular items, or concentrate on a few genres. '
        'This tab quantifies three kinds of bias on the trained models so they can be compared honestly.'
    )

    with st.expander('How is bias measured?', expanded=False):
        st.markdown(
            '**Setup.** Take 1,000 sampled validation users. For each, ask the model for its top-10. '
            'Aggregate across users and compare against the dataset baseline.\n\n'
            '**Popularity ratio** = mean popularity (interaction count) of recommended items '
            '÷ mean popularity of catalog items. **1.0 = perfectly fair.** Higher means the model '
            'is biased toward already-popular items (echo chamber risk).\n\n'
            '**Long-tail fraction** = share of recommendations that are below median popularity. '
            'A perfectly diverse system would be 50%. **<10% = popularity-heavy.**\n\n'
            '**HR@10 by group** = hit rate at rank 10 split by user demographic. '
            'A fair model has equal HR@10 across genders, ages, etc. The **gap** (max − min) measures unfairness.\n\n'
            '**Genre concentration** = fraction of top-10 recs containing each genre. '
            'A perfectly uniform model would show ≈5.6% per genre (1/18). '
            'A genre at >25% indicates the model funnels users into that lane.'
        )

    st.divider()
    st.markdown('### 1. Popularity Bias')
    st.caption('How much does each model amplify already-popular items?')

    c1, c2, c3 = st.columns(3)
    c1.metric('SASRec popularity ratio', f"{s['sasrec_popularity_ratio']:.2f}×",
              help='Mean recommendation popularity ÷ mean catalog popularity')
    c2.metric('SVD popularity ratio', f"{s['svd_popularity_ratio']:.2f}×")
    c3.metric('SASRec long-tail %', f"{s['sasrec_longtail_fraction']:.1%}",
              help='Fraction of recommendations below median item popularity')

    st.markdown(
        '**How to read these numbers:**\n'
        '- `1.0×` = perfectly fair distribution. `2×` = recs are 2× more popular than catalog avg.\n'
        f'- **SASRec at {s["sasrec_popularity_ratio"]:.2f}×** is moderate popularity bias. Typical for sequential models on dense data.\n'
        f'- **SVD at {s["svd_popularity_ratio"]:.2f}×** is higher — matrix factorisation tends to amplify popular items more than self-attention.\n'
        f'- **Long-tail {s["sasrec_longtail_fraction"]:.1%}** is low (50% would be perfectly diverse). The model rarely recommends below-median items.'
    )
    st.caption('CF, Two-Tower, BERT4Rec popularity bias not yet computed (next iteration).')

    st.divider()
    st.markdown('### 2. Demographic Bias — SASRec')
    st.caption('Is the model equally accurate for all demographic groups?')

    b1, b2 = st.columns(2)
    with b1:
        st.markdown('**HR@10 by Gender**')
        st.table(bias['gender'][['group', 'n_users', 'HR@10']])
        gap = float(s['gender_gap_HR10'])
        verdict = 'mild' if abs(gap) < 0.05 else 'concerning'
        st.markdown(f'**Gap (Male − Female) = `{gap:+.4f}` HR@10** — {verdict}.')
        st.caption('A fair model would show 0.0. Anything under 0.01 is OK; 0.01–0.05 mild; over 0.05 problematic.')

    with b2:
        st.markdown('**HR@10 by Age Group**')
        st.table(bias['age'][['group', 'n_users', 'HR@10']])
        rng = float(s['age_hr10_range'])
        verdict = 'concerning' if rng > 0.10 else ('mild' if rng > 0.05 else 'OK')
        st.markdown(f'**Range (max − min) = `{rng:.4f}` HR@10** — {verdict}.')
        st.caption('A fair model would show 0.0 across age groups. Under-18 users typically underperform — small training population.')

    st.divider()
    st.markdown('### 3. Genre Concentration — SASRec')
    st.caption('What share of recommendations falls into each genre?')

    st.bar_chart(bias['genre'].head(10).set_index('genre')['fraction_in_recs'])
    top_genre = bias['genre'].iloc[0]
    st.markdown(
        f'**Top genre: `{top_genre["genre"]}` at `{top_genre["fraction_in_recs"]:.1%}` of recommendations.** '
        'A perfectly uniform model would distribute ~5.6% per genre across 18 genres. '
        f'The dominant share reflects ML-1M\'s catalog skew (Comedy/Drama account for ~50% of movies) '
        'plus some model-driven amplification.'
    )

    st.divider()
    with st.expander('Future bias audits to add', expanded=False):
        st.markdown(
            '- **Occupation bias** — HR@10 split by user occupation. We have the data; just not analyzed.\n'
            '- **Cross-demographic genre fairness** — does the model recommend romance more to women than men, beyond what their stated preferences justify?\n'
            '- **Calibration** — for users who click rec at rank K, does the model\'s predicted score correlate with click-through?\n'
            '- **Intra-list diversity** — average pairwise distance between items in a single top-10 list. '
            'A bundle of 10 nearly-identical action movies is less useful than 10 varied ones.\n'
            '- **Temporal bias** — older catalog items vs newer. Does the model recommend mostly pre-2000 movies even to younger users?\n'
            '- **Bias amplification through reranker** — compare bias before vs after the LightGBM reranker. The reranker has '
            '`item_popularity` as its top feature, so it likely amplifies popularity bias further.'
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COLD START
# ═══════════════════════════════════════════════════════════════════════════════
with tab_cold:
    st.subheader('Cold Start — No Interaction Data')
    st.markdown(
        'What happens when the system has no interaction signal? '
        'Two scenarios: a brand-new movie just added to the catalog, and a brand-new user who just signed up.'
    )

    cs_section = st.radio('Scenario', ['New Item', 'New User'], horizontal=True)
    st.divider()

    if cs_section == 'New Item':
        st.markdown('**Scenario: a new movie arrives with no ratings.**')
        st.markdown(
            'Most models (CF, SVD, SASRec, BERT4Rec) have no embedding for this item — '
            'they\'ve never seen it during training and would silently skip it. '
            'Only **Two-Tower** can handle this: its item tower encodes any movie from its '
            'title text (via a pretrained sentence transformer) and genre vector. '
            'No retraining, no waiting for ratings to accumulate.'
        )
        st.divider()

        ni_c1, ni_c2 = st.columns([1, 2])
        with ni_c1:
            ni_title = st.text_input('Movie title', value='The Dark Knight (2008)')
            ni_genres = st.multiselect('Genres', ALL_GENRES,
                                       default=['Action', 'Crime', 'Thriller'])
            ni_n = st.slider('Items to show', 5, 15, 10, key='ni_n')
            ni_btn = st.button('Encode + Recommend', type='primary', key='ni_run')

        with ni_c2:
            if ni_btn:
                if not ni_title.strip() or not ni_genres:
                    st.error('Enter a title and select at least one genre.')
                else:
                    with st.spinner('Encoding through Two-Tower item tower...'):
                        ni_result = api_post('/new_item_cold_start', {
                            'title': ni_title, 'genres': ni_genres,
                            'n_similar': ni_n, 'n_users': 30,
                        })

                    if ni_result is None:
                        st.error('API not reachable.')
                    else:
                        trend_items = api_post('/recommend', {'history': [], 'n': ni_n})

                        st.markdown('#### Trending vs Two-Tower Output')
                        t_col, tt_col = st.columns(2)
                        with t_col:
                            st.markdown('**Trending (popularity baseline)**')
                            st.caption('What every model defaults to when there is no signal.')
                            if trend_items:
                                st.table(recs_to_table(trend_items['items'][:ni_n], movies))
                        with tt_col:
                            st.markdown('**Two-Tower (content-based)**')
                            st.caption('Similar movies via title + genre encoding. No retraining needed.')
                            st.table(recs_to_table(
                                ni_result['similar_items']['items'][:ni_n], movies,
                            ))

                        st.divider()
                        st.markdown('#### Who Would Receive This Item?')

                        with st.expander('How is this calculated?', expanded=True):
                            st.markdown(
                                '**Step 1.** The new movie\'s title and genres are encoded '
                                'through the Two-Tower item tower → a 128-dim vector.\n\n'
                                '**Step 2.** We have a precomputed 128-dim vector for every '
                                'real user in the dataset (built once during model training, '
                                'from each user\'s watch history + demographics).\n\n'
                                '**Step 3.** Compute the dot product between the new movie\'s vector '
                                'and every user\'s vector. Higher = closer match.\n\n'
                                '**Step 4.** Take the top 30 users with the highest scores. '
                                'These are the users the model would most strongly recommend this movie to.\n\n'
                                '**Step 5.** Look at the demographic profile of those top 30 users '
                                'vs the overall dataset (Male/Female split, age distribution). '
                                'If the top users are 80% male while the dataset is 70% male, '
                                'the new movie is **over-represented to men** by 10 percentage points.\n\n'
                                '**Why this matters.** It exposes who the model thinks would like a movie '
                                '*before* anyone has rated it. Useful as a pre-launch fairness check: '
                                'does the system funnel certain content to certain demographics in a way '
                                'that mirrors stereotypes rather than actual preferences?'
                            )

                        gb = ni_result['gender_bias']
                        ab = ni_result['age_bias']
                        g_df = pd.DataFrame([
                            {'Gender': g,
                             'This movie %': round(gb[g]['top_pct'] * 100, 1),
                             'Dataset %':   round(gb[g]['base_pct'] * 100, 1),
                             'Delta':       round((gb[g]['top_pct'] - gb[g]['base_pct']) * 100, 1)}
                            for g in ['Male', 'Female']
                        ])
                        st.markdown('**Gender breakdown**')
                        st.table(g_df)

                        a_df = pd.DataFrame([
                            {'Age': label,
                             'This movie %': round(ab[label]['top_pct'] * 100, 1),
                             'Dataset %':   round(ab[label]['base_pct'] * 100, 1),
                             'Delta':       round((ab[label]['top_pct'] - ab[label]['base_pct']) * 100, 1)}
                            for label in ab
                        ])
                        st.markdown('**Age breakdown**')
                        st.table(a_df)

                        delta_chart = pd.DataFrame([
                            {'Age Group': label,
                             'Over/Under %': round((ab[label]['top_pct'] - ab[label]['base_pct']) * 100, 1)}
                            for label in ab
                        ]).set_index('Age Group')
                        st.caption('Positive bar = age group is over-represented in the top 30 users vs the overall dataset.')
                        st.bar_chart(delta_chart)

    else:
        st.markdown('**Scenario: a new user signs up. The system knows only their demographic profile.**')
        st.markdown(
            '**Trending for this demographic** counts the most-watched movies among existing users '
            'who share the same gender + age group. Pure popularity within the cohort.\n\n'
            '**Two-Tower (demographics only)** runs the user tower with no history, '
            'just the gender + age signals. The tower returns a 128-dim user embedding which we then '
            'use to score every movie. The result is what the model thinks a typical user with that '
            'profile would prefer — independent of what others in the cohort actually watched.'
        )
        st.divider()

        nu_c1, nu_c2 = st.columns([1, 2])
        with nu_c1:
            nu_gender = st.selectbox('Gender', list(GENDER_OPTIONS.keys()), key='nu_gender')
            nu_age = st.selectbox('Age group', list(AGE_OPTIONS.keys()), key='nu_age')
            nu_n = st.slider('Items to show', 5, 15, 10, key='nu_n')
            nu_btn = st.button('Show recommendations', type='primary', key='nu_run')

        with nu_c2:
            if nu_btn:
                with st.spinner('Computing...'):
                    nu_result = api_post('/new_user_cold_start', {
                        'gender_enc': GENDER_OPTIONS[nu_gender],
                        'age_enc':    AGE_OPTIONS[nu_age],
                        'n':          nu_n,
                    })

                if nu_result is None:
                    st.error('API not reachable.')
                else:
                    t_col, tt_col = st.columns(2)
                    with t_col:
                        st.markdown(f'**Trending for {nu_gender}, {nu_age}**')
                        st.caption('Most-watched movies among users with this demographic profile.')
                        st.table(recs_to_table(nu_result['trending'][:nu_n], movies))
                    with tt_col:
                        st.markdown('**Two-Tower (demographics only)**')
                        st.caption('Two-Tower user tower with demographic features, zero interaction history.')
                        st.table(recs_to_table(nu_result['two_tower'][:nu_n], movies))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TRAINING RUNS (Weights & Biases report)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_wandb:
    st.subheader('Training runs (Weights & Biases)')
    st.caption('Live report from the W&B project. Loss curves, eval metrics and config for every training run.')
    st.markdown(
        '[Open full report in a new tab ↗]'
        '(https://wandb.ai/deepanvishal-cvs-health/serko-recsys-movielens/reports/Untitled-Report--VmlldzoxNjU4OTgyMw)'
    )
    components.iframe(
        'https://wandb.ai/deepanvishal-cvs-health/serko-recsys-movielens/reports/Untitled-Report--VmlldzoxNjU4OTgyMw',
        height=1400,
        scrolling=True,
    )
