"""
Streamlit dashboard for ProductionCascadeHybridEngine

Run: streamlit run app.py
"""
import streamlit as st
from engine import ProductionCascadeHybridEngine
import pandas as pd

@st.cache_resource
def load_engine():
    eng = ProductionCascadeHybridEngine()
    eng.load_and_preprocess()
    eng.fit_offline_models()
    return eng

engine = load_engine()

if 'user_idx' not in st.session_state:
    st.session_state['user_idx'] = 0

users = sorted(engine.ratings['user_id'].unique())

st.sidebar.title('User Selector')

# Search box
search = st.sidebar.text_input('Search user ID')
if search:
    try:
        uid = int(search)
        if uid in users:
            st.session_state['user_idx'] = users.index(uid)
            st.experimental_rerun()
        else:
            st.sidebar.error('User not found')
    except ValueError:
        st.sidebar.error('Enter numeric user id')

# Selectbox
idx = st.sidebar.selectbox('Select user', users, index=st.session_state['user_idx'])
st.session_state['user_idx'] = users.index(idx)

# Prev/Next
col1, col2 = st.sidebar.columns(2)
if col1.button('← Prev'):
    st.session_state['user_idx'] = (st.session_state['user_idx'] - 1) % len(users)
    st.experimental_rerun()
if col2.button('Next →'):
    st.session_state['user_idx'] = (st.session_state['user_idx'] + 1) % len(users)
    st.experimental_rerun()

current_user = users[st.session_state['user_idx']]
user_count = engine.train_df[engine.train_df['user_id'] == current_user].shape[0]
liked, disliked, weighted_liked = engine.build_preference_profile(current_user)
top_genres = engine.movies[engine.movies['item_id'].isin(liked)]['genres'].str.split().explode().value_counts().head(3).index.tolist()

st.sidebar.markdown(f'**User {current_user}**')
st.sidebar.markdown(f'Total ratings (train): {user_count}')
st.sidebar.markdown('Top genres: ' + ', '.join(top_genres) if top_genres else 'Top genres: N/A')

# Strategy badge
if user_count == 0:
    st.sidebar.error('❄️ Cold start — demographic fallback')
elif user_count < 5:
    st.sidebar.error('❄️ Cold start — demographic fallback')
elif user_count < 20:
    st.sidebar.warning('⚠️ Sparse — content-dominant hybrid')
elif user_count < 100:
    st.sidebar.info('🔵 Warm — weighted hybrid active')
else:
    st.sidebar.success('✅ Rich history — full hybrid active')

st.sidebar.markdown('---')
alpha = st.sidebar.slider('α — CF weight (1.0 = pure collaborative)', 0.0, 1.0, 0.5, 0.05)
lam = st.sidebar.slider('λ — MMR diversity (0.0 = maximum diversity)', 0.0, 1.0, 0.6, 0.1)

st.title(f'Recommendations — User {current_user}')
st.caption('Top genres: ' + ', '.join(top_genres) if top_genres else '')

recs = engine.execute_pipeline(current_user, top_n=10, alpha=alpha, lam=lam)
ild = engine.evaluate_intra_list_diversity(recs)

col1, col2, col3, col4 = st.columns(4)
col1.metric('Precision@10', '—')
col2.metric('NDCG@10', '—')
col3.metric('Recall@10', '—')
col4.metric('ILD', f'{ild:.4f}')

st.subheader('Top 10 Recommendations')
rows = []
for i, r in recs.iterrows():
    rows.append({'Rank': i+1, 'Title': r['title'], 'Genres': engine.movies[engine.movies['item_id'] == r['item_id']]['genres'].values[0], 'Score': r['hybrid_score']})
st.dataframe(pd.DataFrame(rows))

st.subheader('Rating history — top 10 highest rated')
hist = engine.train_df[engine.train_df['user_id'] == current_user].merge(engine.movies, on='item_id')
hist = hist.sort_values(['rating', 'timestamp'], ascending=[False, False]).head(10)[['title', 'genres', 'rating', 'timestamp']]
st.dataframe(hist)

with st.expander('Full evaluation report'):
    st.write('Alpha sensitivity table (Precision@10 vs ILD)')
    # compute small sensitivity table
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    table = []
    sample_users = engine.test_df['user_id'].unique()[:20]
    for a in alphas:
        ps, il = [], []
        for u in sample_users:
            r = engine.execute_pipeline(u, alpha=a)
            p, _, _ = engine.evaluate_single_user(u, r)
            ps.append(p)
            il.append(engine.evaluate_intra_list_diversity(r))
        table.append({'alpha': a, 'Precision@10': float(np.mean(ps)), 'ILD': float(np.mean(il))})
    st.dataframe(pd.DataFrame(table))

with st.expander('Inspect SVD latent factors'):
    movie_titles = engine.movies['title'].tolist()
    sel = st.selectbox('Select movie', movie_titles)
    mid = int(engine.movies[engine.movies['title'] == sel]['item_id'].iloc[0])
    st.write('Latent factors (bar chart)')
    try:
        trainset = engine.svd.trainset
        inner = trainset.to_inner_iid(str(mid))
        vec = engine.svd.qi[inner]
        st.bar_chart(vec)
        st.write('Top 5 similar movies in latent space')
        sims = engine.explain_latent_factors(mid)
    except Exception:
        st.write('Cannot show latent factors for this movie')
