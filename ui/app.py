from pathlib import Path
import pandas as pd
import requests
import streamlit as st


API_URL = 'http://localhost:8000'
BIAS_DIR = Path('artifacts/bias')


@st.cache_data
def load_movies():
    meta = pd.read_parquet('data/processed/item_metadata.parquet')
    return meta.set_index('item_id')[['title', 'genres']]


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


def get_recommendations(history, n=10):
    try:
        r = requests.post(
            f'{API_URL}/recommend',
            json={'history': history, 'n': n}, timeout=10,
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def get_similar(item_id, n=10):
    try:
        r = requests.post(
            f'{API_URL}/similar_items',
            json={'item_id': item_id, 'n': n}, timeout=10,
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def item_label(item_id, movies):
    try:
        row = movies.loc[item_id]
        return f"{row['title']} [{row['genres']}]"
    except Exception:
        return f'Item {item_id}'


# ---- App layout ----
st.set_page_config(page_title='Serko RecSys', layout='wide')
st.title('Serko RecSys — MovieLens 1M')
st.caption('Tiered recommendation engine: Popularity / SVD fold-in / SASRec')


movies = load_movies()
tab1, tab2 = st.tabs(['Recommender', 'Bias Audit'])


# ========== TAB 1: Recommender ==========
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader('Build User History')
        history_input = st.text_input(
            'Enter item IDs (comma-separated)',
            placeholder='e.g. 50, 296, 527, 858',
        )
        n_recs = st.slider('Number of recommendations', 5, 20, 10)
        get_recs_btn = st.button('Get Recommendations', type='primary')

        st.divider()
        st.subheader('Similar Items')
        similar_input = st.number_input(
            'Item ID', min_value=0, max_value=3951, value=50,
        )
        get_similar_btn = st.button('Find Similar')

    with col2:
        if get_recs_btn:
            history = []
            if history_input.strip():
                try:
                    history = [int(x.strip()) for x in history_input.split(',')]
                except ValueError:
                    st.error('Invalid item IDs — use integers separated by commas')
                    st.stop()

            result = get_recommendations(history, n=n_recs)
            if result is None:
                st.error('API not reachable. Is the server running on localhost:8000?')
            else:
                tier_labels = {
                    1: 'Tier 1 — Popularity',
                    2: 'Tier 2 — SVD fold-in',
                    3: 'Tier 3 — SASRec',
                }
                tier = result['tier']
                st.success(
                    f"{tier_labels.get(tier, 'Unknown tier')} | "
                    f"Model: {result['model']} | "
                    f"Latency: {result['latency_ms']:.1f}ms"
                )

                rows = []
                for rank, iid in enumerate(result['items'], 1):
                    rows.append({
                        'Rank': rank,
                        'Item ID': iid,
                        'Title [Genres]': item_label(iid, movies),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                if history:
                    with st.expander('History context'):
                        for iid in history:
                            st.write(f'• {item_label(iid, movies)}')

        if get_similar_btn:
            result = get_similar(int(similar_input), n=10)
            if result is None:
                st.error('API not reachable.')
            else:
                st.subheader(f'Similar to: {item_label(int(similar_input), movies)}')
                rows = [
                    {'Item ID': iid, 'Title [Genres]': item_label(iid, movies)}
                    for iid in result['items']
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ========== TAB 2: Bias Audit ==========
with tab2:
    bias = load_bias_data()
    summary = bias['summary'].iloc[0]

    st.subheader('Popularity Bias')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        'SASRec popularity ratio', f"{summary['sasrec_popularity_ratio']:.2f}×",
        help='Mean popularity of recs vs catalog avg. 1.0 = fair',
    )
    col2.metric(
        'SASRec long-tail fraction', f"{summary['sasrec_longtail_fraction']:.1%}",
        help='Fraction of recs below median popularity',
    )
    col3.metric('SVD popularity ratio', f"{summary['svd_popularity_ratio']:.2f}×")
    col4.metric('SVD long-tail fraction', f"{summary['svd_longtail_fraction']:.1%}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('HR@10 by Gender')
        st.dataframe(
            bias['gender'][['group', 'n_users', 'HR@10']],
            use_container_width=True, hide_index=True,
        )
        gap = summary['gender_gap_HR10']
        st.caption(f'Gender gap (M−F): {gap:+.4f} HR@10')

    with col2:
        st.subheader('HR@10 by Age Group')
        st.dataframe(
            bias['age'][['group', 'n_users', 'HR@10']],
            use_container_width=True, hide_index=True,
        )
        rng = summary['age_hr10_range']
        st.caption(f'Age range: {rng:.4f} HR@10 (max−min)')

    st.divider()
    st.subheader('Genre Concentration in Recommendations (SASRec)')
    genre_df = bias['genre'].head(10)
    st.bar_chart(genre_df.set_index('genre')['fraction_in_recs'])
    st.caption('Fraction of top-10 recommendations containing each genre.')
