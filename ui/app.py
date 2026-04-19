from pathlib import Path
import pandas as pd
import requests
import streamlit as st


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
    """title string -> item_id, sorted alphabetically for dropdown."""
    meta = pd.read_parquet('data/processed/item_metadata.parquet')
    mapping = dict(zip(meta['title'], meta['item_id']))
    return dict(sorted(mapping.items()))


@st.cache_data
def load_interaction_demo():
    """Train interactions joined with user demographics."""
    train = pd.read_parquet('data/processed/train.parquet')[['user_id', 'item_id']]
    umeta = pd.read_parquet('data/processed/user_metadata.parquet')
    umeta = umeta[['user_id', 'gender', 'age_enc']]
    return train.merge(umeta, on='user_id')


@st.cache_data
def load_user_profiles():
    """
    Find representative user IDs from the dataset at startup.
    Returns dict: profile_label -> {user_id, history, description}.
    """
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

    # Alex — most interactions
    alex_id = int(lens.idxmax())

    # Sarah — 40-60 interactions, top genre Drama or Romance (fall back to any 40-60 user)
    sarah_candidates = lens[(lens >= 40) & (lens <= 60)].index.tolist()
    sarah_match = next(
        (u for u in sarah_candidates if top_genre(u) in ('Drama', 'Romance')),
        sarah_candidates[0] if sarah_candidates else alex_id,
    )
    sarah_id = int(sarah_match)

    # Marcus — 15-25 interactions
    marcus_candidates = lens[(lens >= 15) & (lens <= 25)].index.tolist()
    marcus_id = int(marcus_candidates[0]) if marcus_candidates else alex_id

    # Priya — 20+ interactions, capped to 4 items
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


# ── Shared helpers ────────────────────────────────────────────────────────────


def item_label(item_id, movies):
    try:
        row = movies.loc[int(item_id)]
        return f"{row['title']} | {row['genres']}"
    except Exception:
        return f'Item {item_id}'


def recs_to_df(items, movies):
    rows = []
    for i, iid in enumerate(items):
        title = item_label(iid, movies)
        rows.append({'Rank': i + 1, 'Title': title})
    return pd.DataFrame(rows)


def interaction_dist(item_id, idf):
    """Gender + age distribution for a given item from train interactions."""
    sub = idf[idf['item_id'] == int(item_id)]
    gender = sub['gender'].value_counts().rename('Count').reset_index()
    gender.columns = ['Gender', 'Interactions']
    age = sub['age_enc'].map(AGE_LABELS).value_counts().rename('Count').reset_index()
    age.columns = ['Age Group', 'Interactions']
    age = age.sort_values('Age Group')
    return gender, age, len(sub)


# ── App ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title='Serko RecSys', layout='wide')
st.title('Serko RecSys — MovieLens 1M')
st.caption(
    'A tiered recommendation engine built on MovieLens 1M. '
    'Five models — Collaborative Filtering, SVD, Two-Tower, SASRec, BERT4Rec — '
    'each with a different approach to the recommendation problem. '
    'Select a movie to see what each model recommends next.'
)

movies = load_movies()
title_map = load_title_map()
idf = load_interaction_demo()

tab1, tab2, tab3, tab5 = st.tabs([
    'Recommendations', 'Model Bias', 'Cold Start', 'For You',
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader('Select a movie to see recommendations')
    st.caption(
        'Each model recommends based on this movie as your only watched item. '
        'Two-Tower additionally supports demographic filtering — select gender and age below to personalise its output.'
    )
    st.divider()

    titles = list(title_map.keys())
    default_idx = titles.index('Star Wars (1977)') if 'Star Wars (1977)' in title_map else 0
    selected_title = st.selectbox('Select a movie', options=titles, index=default_idx)
    selected_id = title_map[selected_title]

    with st.expander('Demographic filter (Two-Tower only)', expanded=False):
        st.caption('Two-Tower is the only model that uses demographic signal. Other models ignore these selections.')
        demo_col1, demo_col2 = st.columns(2)
        with demo_col1:
            gender_sel = st.selectbox('Gender', ['None'] + list(GENDER_OPTIONS.keys()), key='t1_gender')
        with demo_col2:
            age_sel = st.selectbox('Age group', ['None'] + list(AGE_OPTIONS.keys()), key='t1_age')
    use_demo = gender_sel != 'None' and age_sel != 'None'

    n_recs = st.slider('Recommendations per model', 5, 15, 10, key='t1_n')
    run_btn = st.button('Get Recommendations', type='primary', key='t1_run')

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

    if run_btn:
        history = [int(selected_id)]
        with st.spinner('Running all models...'):
            result = api_post('/recommend_all', {'history': history, 'n': n_recs})
            tt_demo_items = None
            if use_demo and result is not None:
                tt_resp = api_post('/two_tower_with_demo', {
                    'history':    history,
                    'gender_enc': GENDER_OPTIONS[gender_sel],
                    'age_enc':    AGE_OPTIONS[age_sel],
                    'n':          n_recs,
                })
                if tt_resp:
                    tt_demo_items = tt_resp['items']

        if result is None:
            st.error('API not reachable on localhost:8000')
        else:
            tier_label = {
                1: 'Tier 1 — Popularity (no history)',
                2: 'Tier 2 — CF + SVD (sparse history)',
                3: 'Tier 3 — SASRec (warm user)',
            }
            st.success(f"{tier_label.get(result['tier'], '')} | 1 movie selected")
            if use_demo:
                st.info(f'Two-Tower demographic filter: {gender_sel}, {age_sel}')

            model_order = ['cf', 'svd', 'two_tower', 'sasrec', 'bert4rec']
            model_labels = ['CF', 'SVD', 'Two-Tower', 'SASRec', 'BERT4Rec']
            cols = st.columns(5)

            for col, key, label in zip(cols, model_order, model_labels):
                with col:
                    if key == 'two_tower' and tt_demo_items is not None:
                        st.markdown(f'**{label}** (+ demographics)')
                        items = tt_demo_items
                    else:
                        st.markdown(f'**{label}**')
                        items = result[key]['items']
                    df = recs_to_df(items[:n_recs], movies)
                    st.dataframe(df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL BIAS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    bias = load_bias_data()
    s = bias['summary'].iloc[0]

    st.subheader('Popularity Bias')
    st.caption('How much do models over-recommend popular items? 1.0x = perfectly fair. Higher = more popular-item bias.')

    c1, c2, c3 = st.columns(3)
    c1.metric('SASRec', f"{s['sasrec_popularity_ratio']:.2f}x",
              help='Ratio of mean recommendation popularity vs mean catalog popularity')
    c2.metric('SVD', f"{s['svd_popularity_ratio']:.2f}x")
    c3.metric('SASRec long-tail %', f"{s['sasrec_longtail_fraction']:.1%}",
              help='Fraction of recommendations below median item popularity')

    st.caption('CF, Two-Tower, BERT4Rec: full bias analysis not yet computed.')
    st.divider()

    st.subheader('Demographic Bias — SASRec')
    st.caption('HR@10 (hit rate at rank 10) broken down by user demographics. '
               'A fair model would show equal HR@10 across groups.')

    b1, b2 = st.columns(2)
    with b1:
        st.markdown('**HR@10 by Gender**')
        st.dataframe(bias['gender'][['group', 'n_users', 'HR@10']],
                     use_container_width=True, hide_index=True)
        st.caption(f'Gender gap (Male − Female): {s["gender_gap_HR10"]:+.4f} HR@10')
    with b2:
        st.markdown('**HR@10 by Age Group**')
        st.dataframe(bias['age'][['group', 'n_users', 'HR@10']],
                     use_container_width=True, hide_index=True)
        st.caption(f'Age range: {s["age_hr10_range"]:.4f} HR@10 (max − min)')

    st.divider()
    st.subheader('Genre Concentration — SASRec')
    st.caption('What fraction of top-10 recommendations contain each genre? '
               'A value near 1/18 ≈ 5.6% would be perfectly uniform.')
    st.bar_chart(bias['genre'].head(10).set_index('genre')['fraction_in_recs'])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COLD START
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader('Cold Start')
    st.caption(
        'What happens when the system has no interaction data? '
        'Two scenarios: a new movie (no ratings yet) and a new user (no history yet).'
    )

    cs_section = st.radio('Scenario', ['New Item', 'New User'], horizontal=True)
    st.divider()

    if cs_section == 'New Item':
        st.markdown('**A new movie arrives with no ratings. How does the system handle it?**')
        st.caption(
            'Popularity-based models show trending items as a baseline. '
            'Two-Tower encodes the new movie through its content tower (title + genres) '
            'and finds similar existing movies immediately — no retraining needed.'
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
                        t_col, tt_col = st.columns(2)
                        with t_col:
                            st.markdown('**Trending (popularity baseline)**')
                            st.caption('What the system shows before any content signal.')
                            if trend_items:
                                st.dataframe(
                                    recs_to_df(trend_items['items'][:ni_n], movies),
                                    use_container_width=True, hide_index=True,
                                )
                        with tt_col:
                            st.markdown('**Two-Tower (content-based)**')
                            st.caption('Similar movies via title + genre encoding. No retraining needed.')
                            st.dataframe(
                                recs_to_df(ni_result['similar_items']['items'][:ni_n], movies),
                                use_container_width=True, hide_index=True,
                            )

                        st.divider()
                        st.markdown('**Who would receive this item?**')
                        st.caption('Demographic profile of users whose Two-Tower embeddings '
                                   'are closest to this new movie vs the overall dataset.')

                        gb = ni_result['gender_bias']
                        ab = ni_result['age_bias']
                        g_df = pd.DataFrame([
                            {'Gender': g,
                             'This movie %': round(gb[g]['top_pct'] * 100, 1),
                             'Dataset %':   round(gb[g]['base_pct'] * 100, 1),
                             'Delta':       round((gb[g]['top_pct'] - gb[g]['base_pct']) * 100, 1)}
                            for g in ['Male', 'Female']
                        ])
                        st.dataframe(g_df, use_container_width=True, hide_index=True)

                        a_df = pd.DataFrame([
                            {'Age': label,
                             'This movie %': round(ab[label]['top_pct'] * 100, 1),
                             'Dataset %':   round(ab[label]['base_pct'] * 100, 1),
                             'Delta':       round((ab[label]['top_pct'] - ab[label]['base_pct']) * 100, 1)}
                            for label in ab
                        ])
                        delta_chart = pd.DataFrame([
                            {'Age Group': label,
                             'Over/Under %': round((ab[label]['top_pct'] - ab[label]['base_pct']) * 100, 1)}
                            for label in ab
                        ]).set_index('Age Group')
                        st.dataframe(a_df, use_container_width=True, hide_index=True)
                        st.caption('Over/under-representation vs dataset baseline:')
                        st.bar_chart(delta_chart)

    else:
        st.markdown('**A new user signs up. The system knows only their demographic profile.**')
        st.caption(
            'Popularity shows the most watched movies among users with the same demographics. '
            'Two-Tower encodes the demographic profile through its user tower '
            '(no history, demographics only) and finds movies that match that profile.'
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
                        st.caption('Most watched movies among users with this demographic profile.')
                        st.dataframe(
                            recs_to_df(nu_result['trending'][:nu_n], movies),
                            use_container_width=True, hide_index=True,
                        )
                    with tt_col:
                        st.markdown('**Two-Tower (demographics only)**')
                        st.caption('Two-Tower user tower with demographic features, zero interaction history.')
                        st.dataframe(
                            recs_to_df(nu_result['two_tower'][:nu_n], movies),
                            use_container_width=True, hide_index=True,
                        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FOR YOU (Netflix-style homepage)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_homepage_row(row, movies_df):
    """Render one horizontal row of movie cards."""
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

    cols = st.columns(len(items))
    for col, iid in zip(cols, items):
        with col:
            try:
                r = movies_df.loc[int(iid)]
                title = r['title']
                genres = r['genres']
                year = int(r['year']) if pd.notna(r['year']) else ''
            except Exception:
                title, genres, year = f'Item {iid}', '', ''
            # Truncate genres to avoid overflow
            genres_short = (genres or '')[:40]
            card_html = (
                "<div style=\"background:#1e1e2e;padding:8px;border-radius:6px;"
                "min-height:90px;font-size:12px;color:#eee;\">"
                f"<b>{title}</b><br/>"
                f"<span style=\"color:#aaa;font-size:11px;\">{year}</span><br/>"
                f"<span style=\"color:#888;font-size:10px;\">{genres_short}</span>"
                "</div>"
            )
            st.markdown(card_html, unsafe_allow_html=True)
    st.markdown('')


with tab5:
    profiles = load_user_profiles()

    st.subheader('For You')
    st.caption('A personalised homepage built from your watch history. '
               'Powered by SASRec, Two-Tower, and popularity signals.')
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

    hp_btn = st.button('Load my recommendations', type='primary', key='hp_run')

    if hp_btn:
        with st.spinner('Building your homepage...'):
            hp_result = api_homepage(
                history=profile['history'],
                gender_enc=gender_enc_hp,
                age_enc=age_enc_hp,
                n=10,
            )

        if hp_result is None:
            st.error('API not reachable.')
        else:
            st.divider()
            for row in hp_result['rows']:
                _render_homepage_row(row, movies)
                st.divider()
