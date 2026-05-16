"""
ProductionCascadeHybridEngine: full pipeline implementation

Implements data cleaning, recency weights, preference profiling,
demographic fallback, SVD retrieval, hybrid scoring with negative
feedback suppression, MMR reranking, evaluation and explainability.
"""
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from surprise import SVD, Dataset, Reader
except Exception:
    raise ImportError('Please install scikit-surprise')


class ProductionCascadeHybridEngine:
    """Single class implementing the 3-stage cascade and helpers."""

    # Slide 1: Data Loading
    # Formula: Load files into pandas DataFrames with given separators and encodings
    def __init__(self):
        self.ratings_raw = None
        self.movies_raw = None
        self.users_raw = None
        self.ratings = None
        self.movies = None
        self.users = None
        self.train_df = None
        self.test_df = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.item_id_to_idx = {}
        self.svd = None

    # Slide 2: Load and preprocess all files
    # Formula: Applies Steps 1-11 from assignment to produce cleaned frames
    def load_and_preprocess(self,
                            ratings_url='https://files.grouplens.org/datasets/movielens/ml-100k/u.data',
                            movies_url='https://files.grouplens.org/datasets/movielens/ml-100k/u.item',
                            users_url='https://files.grouplens.org/datasets/movielens/ml-100k/u.user'):
        # STEP 1 — Load Raw Data
        ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
        movies_cols = [
            'item_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
            'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
            'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]

        users_cols = ['user_id', 'age', 'gender', 'occupation', 'zip']

        self.ratings_raw = pd.read_csv(ratings_url, sep='\t', names=ratings_cols, encoding='latin-1')
        self.movies_raw = pd.read_csv(movies_url, sep='|', names=movies_cols, encoding='latin-1')
        self.users_raw = pd.read_csv(users_url, sep='|', names=users_cols, encoding='latin-1')

        print('=== BEFORE CLEANING SNAPSHOT ===')
        print('Ratings shape:', self.ratings_raw.shape)
        print(self.ratings_raw.dtypes)
        print(self.ratings_raw.isnull().sum())
        print('\nRatings describe:\n', self.ratings_raw['rating'].describe())
        print('\nMovies shape:', self.movies_raw.shape)
        print(self.movies_raw.dtypes)
        print(self.movies_raw.isnull().sum())

        # STEP 3 — Drop irrelevant columns
        # Reducing dimensionality, removing noise columns
        movies = self.movies_raw.copy()
        if 'video_release_date' in movies.columns:
            movies.drop(columns=['video_release_date'], inplace=True)
        if 'imdb_url' in movies.columns:
            movies.drop(columns=['imdb_url'], inplace=True)

        ratings = self.ratings_raw.copy()

        # STEP 4 — Handle missing values
        # Fill release_date with 'Unknown' and drop truly invalid rows
        movies['release_date'] = movies['release_date'].fillna('Unknown')
        ratings.dropna(subset=['user_id', 'item_id', 'rating'], inplace=True)
        movies.dropna(subset=['item_id', 'title'], inplace=True)

        # STEP 5 — Fix data types
        # Convert ids to int, rating to float, and timestamp to datetime
        ratings['user_id'] = ratings['user_id'].astype(int)
        ratings['item_id'] = ratings['item_id'].astype(int)
        ratings['rating'] = ratings['rating'].astype(float)
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
        movies['item_id'] = movies['item_id'].astype(int)

        # STEP 6 — Remove duplicates
        before = ratings.shape[0]
        ratings.drop_duplicates(subset=['user_id', 'item_id'], keep='last', inplace=True)
        duplicates_removed = before - ratings.shape[0]
        print(f'Duplicates removed: {duplicates_removed}')

        # STEP 7 — Filter sparse users and items
        users_before = ratings['user_id'].nunique()
        items_before = ratings['item_id'].nunique()
        user_counts = ratings['user_id'].value_counts()
        item_counts = ratings['item_id'].value_counts()
        ratings = ratings[ratings['user_id'].isin(user_counts[user_counts >= 20].index) &
                          ratings['item_id'].isin(item_counts[item_counts >= 10].index)].copy()
        users_after = ratings['user_id'].nunique()
        items_after = ratings['item_id'].nunique()
        print(f'Ratings before filter: {before}, after filter: {ratings.shape[0]}')

        # STEP 8 — Validate rating range
        if not ratings['rating'].between(1.0, 5.0).all():
            print('Warning: clipping ratings to [1,5]')
            ratings['rating'] = ratings['rating'].clip(1.0, 5.0)

        # STEP 9 — Build genres column
        genre_cols = [
            'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
            'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]

        def build_genres(row):
            tags = [g for g in genre_cols if g in row.index and row[g] == 1]
            return ' '.join(tags) if tags else 'Unknown'

        movies['genres'] = movies.apply(build_genres, axis=1)
        movies.drop(columns=[c for c in genre_cols if c in movies.columns], inplace=True)

        # STEP 10 — Align movies to filtered ratings
        movies = movies[movies['item_id'].isin(ratings['item_id'].unique())].copy()

        # STEP 11 — Final report
        n_ratings_before = self.ratings_raw.shape[0]
        n_ratings_after = ratings.shape[0]
        sparsity = 1.0 - (n_ratings_after / (users_after * items_after))
        print('\n=== DATA CLEANING COMPLETE ===')
        print(f'Ratings before: {n_ratings_before} after: {n_ratings_after}')
        print(f'Users before: {users_before} after: {users_after}')
        print(f'Items before: {items_before} after: {items_after}')
        print(f'Duplicates removed: {duplicates_removed}')
        print(f'Sparse users removed: {users_before - users_after}')
        print(f'Sparse items removed: {items_before - items_after}')
        print(f'Final sparsity: {sparsity:.4f}')
        print('\nSample cleaned ratings:')
        print(ratings.head(5))
        print('\nSample movies with genres:')
        print(movies[['item_id', 'title', 'genres']].head(5))

        # save
        self.ratings = ratings.reset_index(drop=True)
        self.movies = movies.reset_index(drop=True)
        self.users = self.users_raw.copy()

        # build item index for TF-IDF
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies['genres'].fillna('Unknown'))
        self.item_id_to_idx = {int(r.item_id): idx for idx, r in self.movies.reset_index().iterrows()}

        # temporal split
        self.train_df, self.test_df = self._temporal_train_test_split(self.ratings)
        return self.ratings, self.movies, self.users

    # Slide 3: Temporal train/test split
    # Formula: Sort by timestamp and split first 80% as train, last 20% as test
    def _temporal_train_test_split(self, df, train_frac=0.8):
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        cut = int(len(df_sorted) * train_frac)
        return df_sorted.iloc[:cut].copy(), df_sorted.iloc[cut:].copy()

    # Slide 4: Fit offline models
    # Formula: Train TF-IDF already done; train SVD on temporal training interactions
    def fit_offline_models(self, n_factors=50, lr_all=0.005, reg_all=0.02):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.train_df[['user_id', 'item_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        algo = SVD(n_factors=n_factors, lr_all=lr_all, reg_all=reg_all)
        algo.fit(trainset)
        self.svd = algo

    # Slide 5: Compute recency weights
    # Formula: w(t) = exp(-lambda * delta_days), lambda = ln(2)/half_life
    def compute_recency_weights(self, timestamps, half_life_days=180):
        now = timestamps.max()
        delta_days = (now - timestamps).dt.days
        decay_lambda = np.log(2) / half_life_days
        weights = np.exp(-decay_lambda * delta_days)
        return weights

    # Slide 6: Build preference profile
    # Formula: separate liked (>=4), disliked (<=2), weighted_liked (weighted rating>=3.5)
    def build_preference_profile(self, user_id):
        user_r = self.train_df[self.train_df['user_id'] == user_id].copy()
        if user_r.empty:
            return np.array([]), np.array([]), np.array([])
        user_r['weight'] = self.compute_recency_weights(user_r['timestamp'])
        user_r['weighted_rating'] = user_r['rating'] * user_r['weight']
        liked = user_r[user_r['rating'] >= 4]['item_id'].values
        disliked = user_r[user_r['rating'] <= 2]['item_id'].values
        weighted_liked = user_r[user_r['weighted_rating'] >= 3.5]['item_id'].values
        return liked, disliked, weighted_liked

    # Slide 7: Demographic fallback
    # Formula: recommend items popular among similar demographic neighbors
    def demographic_fallback(self, user_id, top_n=10):
        user = self.users[self.users['user_id'] == user_id]
        if user.empty:
            return []
        user = user.iloc[0]
        similar = self.users[(self.users['occupation'] == user['occupation']) &
                             (abs(self.users['age'] - user['age']) <= 10) &
                             (self.users['gender'] == user['gender']) &
                             (self.users['user_id'] != user_id)]['user_id'].values
        demo_recs = (self.train_df[self.train_df['user_id'].isin(similar)]
                     .groupby('item_id')['rating'].mean()
                     .sort_values(ascending=False)
                     .head(top_n))
        return demo_recs.index.astype(int).tolist()

    # Slide 8: Apply negative feedback suppression
    # Formula: subtract gamma * average similarity to disliked items from hybrid score
    def apply_negative_feedback(self, candidates_df, disliked_items, gamma=0.3):
        if len(disliked_items) == 0:
            return candidates_df
        for idx, row in candidates_df.iterrows():
            penalties = []
            for bad in disliked_items:
                if bad in self.item_id_to_idx and row['item_id'] in self.item_id_to_idx:
                    penalties.append(self._item_similarity(row['item_id'], bad))
            avg_pen = np.mean(penalties) if penalties else 0.0
            candidates_df.at[idx, 'hybrid_score'] = candidates_df.at[idx, 'hybrid_score'] - gamma * avg_pen
        return candidates_df

    def _item_similarity(self, a, b):
        ia = self.item_id_to_idx.get(int(a))
        ib = self.item_id_to_idx.get(int(b))
        if ia is None or ib is None:
            return 0.0
        return float(cosine_similarity(self.tfidf_matrix[ia], self.tfidf_matrix[ib])[0, 0])

    # Slide 9: Candidate retrieval using SVD
    # Formula: score unseen items with SVD predictions, take top 50
    def retrieve_candidates(self, user_id, k=50):
        all_items = set(self.movies['item_id'].astype(int).unique())
        seen = set(self.train_df[self.train_df['user_id'] == user_id]['item_id'].astype(int).unique())
        unseen = list(all_items - seen)
        scores = []
        for iid in unseen:
            try:
                est = self.svd.predict(user_id, iid).est
            except Exception:
                est = 3.0
            scores.append((int(iid), float(est)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    # Slide 10: Hybrid scoring with switching strategy and negative suppression
    # Formula: hybrid = alpha*CF + (1-alpha)*CB where CB is recency-weighted similarity
    def hybrid_score(self, user_id, candidates, alpha=0.5, gamma=0.3):
        # decide strategy
        user_count = self.train_df[self.train_df['user_id'] == user_id].shape[0]
        if user_count == 0:
            strategy = 'demographic'
        elif user_count < 5:
            strategy = 'cold_start'
        elif user_count < 20:
            strategy = 'sparse_hybrid'
        else:
            strategy = 'full_hybrid'
        print(f'Strategy for user {user_id}: {strategy}')

        if strategy == 'demographic':
            demo = self.demographic_fallback(user_id, top_n=10)
            return pd.DataFrame([{'item_id': iid, 'hybrid_score': 0.0} for iid in demo])

        liked, disliked, weighted_liked = self.build_preference_profile(user_id)

        rows = []
        for iid, cf_pred in candidates:
            cf_score = (cf_pred - 1.0) / 4.0
            # compute cb_score: average recency-weighted cosine similarity to liked items
            sims = []
            for lid in weighted_liked if len(weighted_liked) > 0 else liked:
                if lid in self.item_id_to_idx and iid in self.item_id_to_idx:
                    sims.append(self._item_similarity(iid, lid))
            cb_score = float(np.mean(sims)) if sims else 0.0

            # switch alpha by strategy
            if strategy == 'cold_start':
                alpha_final = 0.0
            elif strategy == 'sparse_hybrid':
                alpha_final = 0.2
            else:
                alpha_final = alpha

            hybrid = alpha_final * cf_score + (1.0 - alpha_final) * cb_score
            rows.append({'item_id': int(iid), 'hybrid_score': hybrid})

        import pandas as _pd
        cand_df = _pd.DataFrame(rows)
        cand_df = self.apply_negative_feedback(cand_df, disliked, gamma=gamma)
        cand_df.sort_values('hybrid_score', ascending=False, inplace=True)
        return cand_df

    # Slide 11: MMR re-ranking
    # Formula: select items maximizing lambda*relevance - (1-lambda)*max_sim_to_selected
    def mmr_rerank(self, cand_df, top_n=10, lam=0.6):
        candidates = cand_df['item_id'].tolist()
        scores = dict(zip(cand_df['item_id'], cand_df['hybrid_score']))
        selected = []
        while len(selected) < top_n and candidates:
            if not selected:
                pick = max(candidates, key=lambda x: scores.get(x, 0.0))
                selected.append(pick)
                candidates.remove(pick)
                continue
            mmr_scores = []
            for c in candidates:
                rel = scores.get(c, 0.0)
                sim_to_sel = max([self._item_similarity(c, s) for s in selected]) if selected else 0.0
                mmr_scores.append((c, lam * rel - (1 - lam) * sim_to_sel))
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            pick = mmr_scores[0][0]
            selected.append(pick)
            candidates.remove(pick)
        return selected

    # Slide 12: Execute full pipeline per user
    # Formula: retrieve -> hybrid score -> MMR rerank or demographic fallback
    def execute_pipeline(self, user_id, top_n=10, alpha=0.5, lam=0.6, gamma=0.3):
        candidates = self.retrieve_candidates(user_id, k=50)
        cand_df = self.hybrid_score(user_id, candidates, alpha=alpha, gamma=gamma)
        # if demographic fallback returned list
        if 'hybrid_score' not in cand_df.columns:
            cand_df = pd.DataFrame(cand_df)
        if cand_df.empty:
            return pd.DataFrame(columns=['item_id', 'title', 'hybrid_score'])
        selected = self.mmr_rerank(cand_df, top_n=top_n, lam=lam)
        rows = []
        for iid in selected:
            title = self.movies[self.movies['item_id'] == iid]['title'].values
            rows.append({'item_id': iid, 'title': title[0] if len(title) else 'Unknown', 'hybrid_score': float(cand_df[cand_df['item_id'] == iid]['hybrid_score'].iloc[0])})
        return pd.DataFrame(rows)

    # Slide 13: ILD metric
    # Formula: 1 - average pairwise cosine similarity among recommended items
    def evaluate_intra_list_diversity(self, recs_df):
        ids = recs_df['item_id'].tolist()
        vecs = [self.tfidf_matrix[self.item_id_to_idx[iid]].toarray()[0] for iid in ids if iid in self.item_id_to_idx]
        if len(vecs) <= 1:
            return 0.0
        sims = cosine_similarity(np.vstack(vecs))
        n = sims.shape[0]
        pairs = n * (n - 1) / 2
        avg_sim = np.sum(np.triu(sims, 1)) / pairs if pairs > 0 else 0.0
        return 1.0 - avg_sim

    # Slide 14: Single user evaluation (Precision, Recall, NDCG)
    # Formula: compute for given user's test interactions
    def evaluate_single_user(self, user_id, recs_df, k=10):
        relevant = set(self.test_df[self.test_df['user_id'] == user_id]['item_id'].tolist())
        if not relevant:
            return 0.0, 0.0, 0.0
        tp = len([i for i in recs_df['item_id'][:k] if i in relevant])
        precision = tp / k
        recall = tp / len(relevant)
        dcg = 0.0
        for i, iid in enumerate(recs_df['item_id'][:k]):
            rel = 1 if iid in relevant else 0
            dcg += (2 ** rel - 1) / math.log2(i + 2)
        ideal_rels = [1] * min(len(relevant), k)
        idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return precision, recall, ndcg

    # Slide 15: Full evaluation across N users and alpha sensitivity
    # Formula: aggregate metrics and compute alpha sensitivity table
    def run_full_evaluation(self, n_test_users=50):
        users = list(self.test_df['user_id'].unique())
        users = [u for u in users if u in self.train_df['user_id'].unique()][:n_test_users]
        precisions, recalls, ndcgs, ilds = [], [], [], []
        recommended_items = set()
        for u in users:
            recs = self.execute_pipeline(u)
            recommended_items.update(recs['item_id'].tolist())
            p, r, n = self.evaluate_single_user(u, recs)
            precisions.append(p)
            recalls.append(r)
            ndcgs.append(n)
            ilds.append(self.evaluate_intra_list_diversity(recs))
        coverage = len(recommended_items) / self.movies.shape[0]
        print('Precision@10: {:.4f} ± {:.4f}'.format(np.mean(precisions), np.std(precisions)))
        print('Recall@10: {:.4f} ± {:.4f}'.format(np.mean(recalls), np.std(recalls)))
        print('NDCG@10: {:.4f} ± {:.4f}'.format(np.mean(ndcgs), np.std(ndcgs)))
        print('ILD@10: {:.4f} ± {:.4f}'.format(np.mean(ilds), np.std(ilds)))
        print('Catalog coverage: {:.4f}'.format(coverage))

        # alpha sensitivity on 20 users
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        print('\nAlpha sensitivity table (Precision@10, ILD):')
        for a in alphas:
            ps, il = [], []
            for u in users[:20]:
                recs = self.execute_pipeline(u, alpha=a)
                p, _, _ = self.evaluate_single_user(u, recs)
                ps.append(p)
                il.append(self.evaluate_intra_list_diversity(recs))
            print(f'alpha={a}: Precision@10={np.mean(ps):.4f}, ILD={np.mean(il):.4f}')

    # Slide 16: Explain latent factors
    # Formula: print SVD item vector and top-5 similar items by cosine similarity
    def explain_latent_factors(self, item_id, top_k=5):
        trainset = self.svd.trainset
        try:
            inner = trainset.to_inner_iid(str(item_id))
        except Exception:
            print('Item not in trainset')
            return
        vec = self.svd.qi[inner]
        print('Discovered latent factors (Andrew Ng formulation):')
        print(vec)
        sims = cosine_similarity([vec], self.svd.qi)[0]
        idxs = np.argsort(sims)[::-1][1:top_k+1]
        for ii in idxs:
            print(trainset.to_raw_iid(ii), sims[ii])
