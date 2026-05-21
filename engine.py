"""
ProductionCascadeHybridEngine: full pipeline implementation
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

    def __init__(self):
        self.ratings_url = "ratings.csv"
        self.movies_url  = "movies.csv"
        self.users       = None
        self.ratings     = None
        self.movies      = None
        self.train_df    = None
        self.test_df     = None
        self.tfidf_matrix = None
        self.item_content_sim = None
        self.item_id_to_idx   = {}
        self.svd = None

    def load_and_preprocess(self):
        print("📥 Step 1: Loading ml-latest-small files...")

        self.ratings_raw = pd.read_csv(
            self.ratings_url,
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            skiprows=1
        )

        self.movies_raw = pd.read_csv(
            self.movies_url,
            names=['item_id', 'title', 'genres_raw'],
            skiprows=1
        )

        print(f"✅ Loaded {len(self.ratings_raw)} ratings, {len(self.movies_raw)} movies")

        print("\n=== BEFORE CLEANING SNAPSHOT ===")
        print("Ratings shape:", self.ratings_raw.shape)
        print(self.ratings_raw.dtypes)
        print(self.ratings_raw.isnull().sum())
        print(self.ratings_raw['rating'].describe())
        print("Movies shape:", self.movies_raw.shape)

        self.movies_raw['genres_raw'] = self.movies_raw['genres_raw'].fillna('Unknown')
        self.ratings_raw.dropna(subset=['user_id', 'item_id', 'rating'], inplace=True)
        self.movies_raw.dropna(subset=['item_id', 'title'], inplace=True)

        self.ratings_raw['user_id']   = self.ratings_raw['user_id'].astype(int)
        self.ratings_raw['item_id']   = self.ratings_raw['item_id'].astype(int)
        self.ratings_raw['rating']    = self.ratings_raw['rating'].astype(float)
        self.ratings_raw['timestamp'] = pd.to_datetime(
            self.ratings_raw['timestamp'], unit='s'
        )
        self.movies_raw['item_id'] = self.movies_raw['item_id'].astype(int)

        before = len(self.ratings_raw)
        self.ratings_raw.drop_duplicates(
            subset=['user_id', 'item_id'], keep='last', inplace=True
        )
        print(f"Duplicates removed: {before - len(self.ratings_raw)}")

        print(f"Ratings before filter: {len(self.ratings_raw)}")
        user_counts = self.ratings_raw['user_id'].value_counts()
        item_counts = self.ratings_raw['item_id'].value_counts()
        self.ratings_raw = self.ratings_raw[
            self.ratings_raw['user_id'].isin(user_counts[user_counts >= 20].index) &
            self.ratings_raw['item_id'].isin(item_counts[item_counts >= 10].index)
        ]
        print(f"Ratings after filter: {len(self.ratings_raw)}")

        self.ratings_raw['rating'] = self.ratings_raw['rating'].clip(0.5, 5.0)

        self.movies_raw['genres'] = (
            self.movies_raw['genres_raw']
            .str.replace('|', ' ', regex=False)
            .str.replace('(no genres listed)', 'Unknown', regex=False)
        )
        self.movies_raw.drop(columns=['genres_raw'], inplace=True)

        valid_items = set(self.ratings_raw['item_id'].unique())
        self.movies_raw = self.movies_raw[
            self.movies_raw['item_id'].isin(valid_items)
        ].reset_index(drop=True)

        n_users = self.ratings_raw['user_id'].nunique()
        n_items = self.ratings_raw['item_id'].nunique()
        sparsity = 1 - len(self.ratings_raw) / (n_users * n_items)

        print("\n=== DATA CLEANING COMPLETE ===")
        print(f"Ratings: {len(self.ratings_raw)}")
        print(f"Users:   {n_users}")
        print(f"Items:   {n_items}")
        print(f"Movies:  {len(self.movies_raw)}")
        print(f"Sparsity: {sparsity:.4f}")
        print(self.ratings_raw.head())
        print(self.movies_raw[['item_id', 'title', 'genres']].head())

        df_sorted = self.ratings_raw.sort_values('timestamp')
        cut = int(len(df_sorted) * 0.8)
        self.train_df = df_sorted.iloc[:cut].reset_index(drop=True)
        self.test_df  = df_sorted.iloc[cut:].reset_index(drop=True)
        print(f"Train: {len(self.train_df)}, Test: {len(self.test_df)}")

        self.ratings = self.ratings_raw
        self.movies  = self.movies_raw
        self.item_id_to_idx = {
            row['item_id']: idx
            for idx, row in self.movies.iterrows()
        }

        print("  |-- Computing TF-IDF content matrix...")
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies['genres'])
        print("✅ Preprocessing complete.")

    def fit_offline_models(self, n_factors=50, lr_all=0.005, reg_all=0.02):
        print("  |-- Training SVD matrix factorization...")
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            self.train_df[['user_id', 'item_id', 'rating']], reader
        )
        trainset = data.build_full_trainset()
        algo = SVD(n_factors=n_factors, lr_all=lr_all, reg_all=reg_all)
        algo.fit(trainset)
        self.svd = algo
        print("✅ Offline models trained.")

    def compute_recency_weights(self, timestamps, half_life_days=180):
        now = timestamps.max()
        delta_days = (now - timestamps).dt.days
        decay_lambda = np.log(2) / half_life_days
        weights = np.exp(-decay_lambda * delta_days)
        return weights

    def build_preference_profile(self, user_id):
        user_r = self.train_df[self.train_df['user_id'] == user_id].copy()
        if user_r.empty:
            return np.array([]), np.array([]), np.array([])
        user_r['weight'] = self.compute_recency_weights(user_r['timestamp'])
        user_r['weighted_rating'] = user_r['rating'] * user_r['weight']
        liked          = user_r[user_r['rating'] >= 4]['item_id'].values
        disliked       = user_r[user_r['rating'] <= 2]['item_id'].values
        weighted_liked = user_r[user_r['weighted_rating'] >= 3.5]['item_id'].values
        return liked, disliked, weighted_liked

    def demographic_fallback(self, user_id, top_n=10):
        popular = (
            self.train_df.groupby('item_id')
            .agg(avg_rating=('rating', 'mean'), count=('rating', 'count'))
            .query('count >= 50')
            .sort_values('avg_rating', ascending=False)
            .head(top_n)
        )
        return popular.index.astype(int).tolist()

    def apply_negative_feedback(self, candidates_df, disliked_items, gamma=0.3):
        if len(disliked_items) == 0:
            return candidates_df
        for idx, row in candidates_df.iterrows():
            penalties = []
            for bad in disliked_items:
                if bad in self.item_id_to_idx and row['item_id'] in self.item_id_to_idx:
                    penalties.append(self._item_similarity(row['item_id'], bad))
            avg_pen = np.mean(penalties) if penalties else 0.0
            candidates_df.at[idx, 'hybrid_score'] = (
                candidates_df.at[idx, 'hybrid_score'] - gamma * avg_pen
            )
        return candidates_df

    def _item_similarity(self, a, b):
        ia = self.item_id_to_idx.get(int(a))
        ib = self.item_id_to_idx.get(int(b))
        if ia is None or ib is None:
            return 0.0
        return float(
            cosine_similarity(self.tfidf_matrix[ia], self.tfidf_matrix[ib])[0, 0]
        )

    def retrieve_candidates(self, user_id, k=50):
        all_items = set(self.movies['item_id'].astype(int).unique())
        seen = set(
            self.train_df[self.train_df['user_id'] == user_id]['item_id']
            .astype(int).unique()
        )
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

    def hybrid_score(self, user_id, candidates, alpha=0.5, gamma=0.3):
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
            return pd.DataFrame([
                {'item_id': iid, 'hybrid_score': 0.0} for iid in demo
            ])

        liked, disliked, weighted_liked = self.build_preference_profile(user_id)

        rows = []
        for iid, cf_pred in candidates:
            # normalise SVD prediction [0.5, 5.0] -> [0, 1]
            cf_score = (cf_pred - 0.5) / 4.5

            ref_items = weighted_liked if len(weighted_liked) > 0 else liked
            sims = []
            for lid in ref_items:
                if lid in self.item_id_to_idx and iid in self.item_id_to_idx:
                    sims.append(self._item_similarity(iid, lid))
            cb_score = float(np.mean(sims)) if sims else 0.0

            if strategy == 'cold_start':
                alpha_final = 0.0
            elif strategy == 'sparse_hybrid':
                alpha_final = 0.2
            else:
                alpha_final = alpha

            hybrid = alpha_final * cf_score + (1.0 - alpha_final) * cb_score
            rows.append({'item_id': int(iid), 'hybrid_score': hybrid})

        cand_df = pd.DataFrame(rows)
        cand_df = self.apply_negative_feedback(cand_df, disliked, gamma=gamma)
        cand_df.sort_values('hybrid_score', ascending=False, inplace=True)
        return cand_df

    def mmr_rerank(self, cand_df, top_n=10, lam=0.6):
        candidates = cand_df['item_id'].tolist()
        scores = dict(zip(cand_df['item_id'], cand_df['hybrid_score']))
        selected = []
        while len(selected) < top_n and candidates:
            if not selected:
                pick = max(candidates, key=lambda x: scores.get(x, 0.0))
            else:
                mmr_scores = []
                for c in candidates:
                    rel = scores.get(c, 0.0)
                    sim_to_sel = max(
                        [self._item_similarity(c, s) for s in selected]
                    )
                    mmr_scores.append((c, lam * rel - (1 - lam) * sim_to_sel))
                mmr_scores.sort(key=lambda x: x[1], reverse=True)
                pick = mmr_scores[0][0]
            selected.append(pick)
            candidates.remove(pick)
        return selected

    def execute_pipeline(self, user_id, top_n=10, alpha=0.5, lam=0.6, gamma=0.3):
        candidates = self.retrieve_candidates(user_id, k=50)
        cand_df = self.hybrid_score(user_id, candidates, alpha=alpha, gamma=gamma)
        if cand_df.empty:
            return pd.DataFrame(columns=['item_id', 'title', 'hybrid_score'])
        selected = self.mmr_rerank(cand_df, top_n=top_n, lam=lam)
        rows = []
        for iid in selected:
            title = self.movies[self.movies['item_id'] == iid]['title'].values
            score = cand_df[cand_df['item_id'] == iid]['hybrid_score']
            rows.append({
                'item_id': iid,
                'title': title[0] if len(title) else 'Unknown',
                'hybrid_score': float(score.iloc[0]) if len(score) else 0.0
            })
        return pd.DataFrame(rows)

    def evaluate_intra_list_diversity(self, recs_df):
        ids = recs_df['item_id'].tolist()
        vecs = [
            self.tfidf_matrix[self.item_id_to_idx[iid]].toarray()[0]
            for iid in ids if iid in self.item_id_to_idx
        ]
        if len(vecs) <= 1:
            return 0.0
        sims = cosine_similarity(np.vstack(vecs))
        n = sims.shape[0]
        pairs = n * (n - 1) / 2
        avg_sim = np.sum(np.triu(sims, 1)) / pairs if pairs > 0 else 0.0
        return 1.0 - avg_sim

    def evaluate_single_user(self, user_id, recs_df, k=10):
        relevant = set(
            self.test_df[self.test_df['user_id'] == user_id]['item_id'].tolist()
        )
        if not relevant:
            return 0.0, 0.0, 0.0
        tp = len([i for i in recs_df['item_id'][:k] if i in relevant])
        precision = tp / k
        recall    = tp / len(relevant)
        dcg = 0.0
        for i, iid in enumerate(recs_df['item_id'][:k]):
            rel = 1 if iid in relevant else 0
            dcg += (2 ** rel - 1) / math.log2(i + 2)
        ideal_rels = [1] * min(len(relevant), k)
        idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return precision, recall, ndcg

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
        print('Precision@10: {:.4f} +/- {:.4f}'.format(np.mean(precisions), np.std(precisions)))
        print('Recall@10:    {:.4f} +/- {:.4f}'.format(np.mean(recalls),    np.std(recalls)))
        print('NDCG@10:      {:.4f} +/- {:.4f}'.format(np.mean(ndcgs),      np.std(ndcgs)))
        print('ILD@10:       {:.4f} +/- {:.4f}'.format(np.mean(ilds),        np.std(ilds)))
        print('Coverage:     {:.4f}'.format(coverage))

        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        print('\nAlpha sensitivity:')
        for a in alphas:
            ps, il = [], []
            for u in users[:20]:
                recs = self.execute_pipeline(u, alpha=a)
                p, _, _ = self.evaluate_single_user(u, recs)
                ps.append(p)
                il.append(self.evaluate_intra_list_diversity(recs))
            print(f'  alpha={a}: P@10={np.mean(ps):.4f}  ILD={np.mean(il):.4f}')

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
        idxs = np.argsort(sims)[::-1][1:top_k + 1]
        for ii in idxs:
            print(trainset.to_raw_iid(ii), sims[ii])


if __name__ == '__main__':
    engine = ProductionCascadeHybridEngine()
    engine.load_and_preprocess()
    engine.fit_offline_models()
    recs = engine.execute_pipeline(user_id=1)
    print(recs)
    engine.run_full_evaluation(n_test_users=50)