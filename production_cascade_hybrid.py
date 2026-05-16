"""
Production-Grade 3-Stage Cascade Hybrid Recommendation Engine

Implements the university assignment specification: data cleaning pipeline,
SVD candidate retrieval, TF-IDF content layer, hybrid scoring, MMR re-ranking,
and evaluation metrics including alpha sensitivity.

Dependencies: pandas, numpy, scikit-learn, surprise
"""
import warnings
warnings.filterwarnings('ignore')

import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

try:
    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import train_test_split
except Exception as e:
    raise ImportError('Please install scikit-surprise: pip install scikit-surprise')

from collections import defaultdict
from datetime import datetime


class ProductionCascadeHybridEngine:
    """ProductionCascadeHybridEngine implements the full 3-stage cascade.

    Methods:
    - load_and_preprocess(): runs all cleaning steps and prints report
    - fit_offline_models(): trains TF-IDF and SVD on the temporal train split
    - execute_pipeline(user_id, top_n=10, alpha=0.5, lam=0.6): returns top-N
    - evaluate_intra_list_diversity(recommended_df)
    - run_full_evaluation(n_test_users=50)
    - explain_latent_factors(item_id)
    - print_recommendation_report(user_id)
    """

    def __init__(self):
        # Data placeholders
        self.ratings_raw = None
        self.movies_raw = None
        self.ratings = None
        self.movies = None
        self.tv = None
        self.tfidf_matrix = None
        self.item_index = None
        self.svd_algo = None
        self.train_df = None
        self.test_df = None
        self.duplicates_removed = 0
        self.users_before = 0
        self.items_before = 0
        self.users_after = 0
        self.items_after = 0

    # Slide 1: Data Loading
    # Load the raw ratings and movies files into pandas DataFrames
    def load_and_preprocess(self,
                            ratings_url='https://files.grouplens.org/datasets/movielens/ml-100k/u.data',
                            movies_url='https://files.grouplens.org/datasets/movielens/ml-100k/u.item'):
        # STEP 1 — Load Raw Data
        ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
        movies_cols = [
            'item_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
            'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
            'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]

        self.ratings_raw = pd.read_csv(ratings_url, sep='\t', names=ratings_cols, encoding='latin-1')
        self.movies_raw = pd.read_csv(movies_url, sep='|', names=movies_cols, encoding='latin-1')

        # STEP 2 — Inspect & Report (before cleaning)
        print('--- BEFORE CLEANING SNAPSHOT ---')
        print('\nRatings shape:', self.ratings_raw.shape)
        print(self.ratings_raw.dtypes)
        print(self.ratings_raw.isnull().sum())
        print('\nRatings describe:\n', self.ratings_raw['rating'].describe())

        print('\nMovies shape:', self.movies_raw.shape)
        print(self.movies_raw.dtypes)
        print(self.movies_raw.isnull().sum())

        # Copy to working frames
        ratings = self.ratings_raw.copy()
        movies = self.movies_raw.copy()

        # STEP 3 — Drop Irrelevant Columns
        if 'video_release_date' in movies.columns:
            movies.drop(columns=['video_release_date'], inplace=True)
        if 'imdb_url' in movies.columns:
            movies.drop(columns=['imdb_url'], inplace=True)

        # STEP 4 — Handle Missing Values
        movies['release_date'] = movies['release_date'].fillna('Unknown')
        ratings.dropna(subset=['user_id', 'item_id', 'rating'], inplace=True)
        movies.dropna(subset=['item_id', 'title'], inplace=True)

        # STEP 5 — Fix Data Types
        ratings['user_id'] = ratings['user_id'].astype(int)
        ratings['item_id'] = ratings['item_id'].astype(int)
        ratings['rating'] = ratings['rating'].astype(float)
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

        movies['item_id'] = movies['item_id'].astype(int)

        # STEP 6 — Remove Duplicates
        before_dup = ratings.shape[0]
        ratings.drop_duplicates(subset=['user_id', 'item_id'], keep='last', inplace=True)
        after_dup = ratings.shape[0]
        self.duplicates_removed = before_dup - after_dup
        print(f'Duplicates removed: {self.duplicates_removed}')

        # STEP 7 — Filter Sparse Users and Items
        self.users_before = ratings['user_id'].nunique()
        self.items_before = ratings['item_id'].nunique()

        # Remove users with fewer than 20 ratings
        user_counts = ratings['user_id'].value_counts()
        users_to_keep = user_counts[user_counts >= 20].index
        # Remove items with fewer than 10 ratings
        item_counts = ratings['item_id'].value_counts()
        items_to_keep = item_counts[item_counts >= 10].index

        ratings = ratings[ratings['user_id'].isin(users_to_keep) & ratings['item_id'].isin(items_to_keep)].copy()

        self.users_after = ratings['user_id'].nunique()
        self.items_after = ratings['item_id'].nunique()

        print(f'Ratings shape before filter: {after_dup}, after filter: {ratings.shape[0]}')

        # STEP 8 — Validate Rating Range
        if not ratings['rating'].between(1.0, 5.0).all():
            print('Warning: some ratings out of [1,5], clipping')
            ratings['rating'] = ratings['rating'].clip(1.0, 5.0)

        # STEP 9 — Build Genre Metadata Column for Content Engine
        genre_cols = [
            'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
            'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]

        # Combine binary genres into single string
        def make_genres(row):
            tags = [g for g in genre_cols if g in row.index and row[g] == 1]
            if len(tags) == 0:
                return 'Unknown'
            return ' '.join(tags)

        movies['genres'] = movies.apply(make_genres, axis=1)
        movies.drop(columns=[c for c in genre_cols if c in movies.columns], inplace=True)

        # STEP 10 — Align Movies to Filtered Ratings
        movies = movies[movies['item_id'].isin(ratings['item_id'].unique())].copy()

        # Save cleaned frames
        self.ratings = ratings.reset_index(drop=True)
        self.movies = movies.reset_index(drop=True)

        # STEP 11 — Print Final Cleaning Report
        n_ratings_before = self.ratings_raw.shape[0]
        n_ratings_after = self.ratings.shape[0]
        n_users_before = self.users_before
        n_users_after = self.users_after
        n_items_before = self.items_before
        n_items_after = self.items_after
        n_sparse_users_removed = n_users_before - n_users_after
        n_sparse_items_removed = n_items_before - n_items_after

        sparsity = 1.0 - (n_ratings_after / (n_users_after * n_items_after))

        print('\n=== DATA CLEANING COMPLETE ===')
        print(f'Ratings before: {n_ratings_before}  after: {n_ratings_after}')
        print(f'Users before: {n_users_before}  after: {n_users_after}')
        print(f'Items before: {n_items_before}  after: {n_items_after}')
        print(f'Duplicates removed: {self.duplicates_removed}')
        print(f'Sparse users removed: {n_sparse_users_removed}')
        print(f'Sparse items removed: {n_sparse_items_removed}')
        print(f'Final sparsity: {sparsity:.4f}')
        print('\nSample cleaned ratings:')
        print(self.ratings.head(5))
        print('\nSample movies with genres:')
        print(self.movies[['item_id', 'title', 'genres']].head(5))

        # Save for downstream
        self.train_df, self.test_df = self._temporal_train_test_split(self.ratings)

        return self.ratings, self.movies

    # Slide 2: Temporal train/test split
    # Sort by timestamp and take first 80% interactions as train, last 20% as test
    def _temporal_train_test_split(self, ratings_df, train_frac=0.8):
        ratings_sorted = ratings_df.sort_values('timestamp').reset_index(drop=True)
        cutoff = int(len(ratings_sorted) * train_frac)
        train = ratings_sorted.iloc[:cutoff].copy()
        test = ratings_sorted.iloc[cutoff:].copy()
        print(f'Performed temporal split: train={train.shape[0]}, test={test.shape[0]}')
        return train, test

    # Slide 3: Offline model training
    # Train SVD on the temporal train split and TF-IDF on the movie genres
    def fit_offline_models(self, n_factors=50, lr_all=0.005, reg_all=0.02):
        if self.train_df is None or self.movies is None:
            raise RuntimeError('Data not loaded. Run load_and_preprocess() first.')

        print('\nTraining TF-IDF vectorizer on genres...')
        self.tv = TfidfVectorizer()
        self.tfidf_matrix = self.tv.fit_transform(self.movies['genres'].fillna('Unknown'))
        # item_index maps item_id to row index in movies
        self.item_index = {row['item_id']: idx for idx, row in self.movies.reset_index().iterrows()}

        print('Training SVD matrix factorization (surprise)...')
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.train_df[['user_id', 'item_id', 'rating']], reader)
        trainset = data.build_full_trainset()

        algo = SVD(n_factors=n_factors, lr_all=lr_all, reg_all=reg_all)
        algo.fit(trainset)
        self.svd_algo = algo

        print('Offline models trained.')

    # Slide 4: Candidate retrieval with SVD
    # For a user, score all unseen items with SVD and return top K candidates
    def _retrieve_candidates(self, user_id, k=50):
        # items to consider: all items in self.movies
        all_items = set(self.movies['item_id'].unique())
        user_train_items = set(self.train_df[self.train_df['user_id'] == user_id]['item_id'].unique())
        unseen = list(all_items - user_train_items)

        scores = []
        for iid in unseen:
            try:
                pred = self.svd_algo.predict(user_id, iid).est
            except Exception:
                pred = 3.0
            scores.append((iid, pred))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    # Slide 5: Hybrid scoring
    # Combine CF and CB with alpha weighting; handle cold-start rules
    def _hybrid_score(self, user_id, candidates, alpha=0.5):
        # CF_score normalized to [0,1]
        user_ratings = self.train_df[self.train_df['user_id'] == user_id]
        user_high_rated = set(user_ratings[user_ratings['rating'] >= 4.0]['item_id'].unique())

        # Determine alpha adjustment for cold-start user
        if len(user_ratings) < 5:
            alpha_user = 0.1
        else:
            alpha_user = alpha

        hybrid_scores = []
        for iid, cf_pred in candidates:
            # CF component
            cf_score = (cf_pred - 1.0) / 4.0
            # CB component: average cosine similarity to user's high-rated items
            if len(user_high_rated) == 0:
                cb_score = 0.0
            else:
                sims = []
                if iid not in self.item_index:
                    cb_score = 0.0
                else:
                    iidx = self.item_index[iid]
                    item_vec = self.tfidf_matrix[iidx]
                    for hid in user_high_rated:
                        if hid in self.item_index:
                            hid_idx = self.item_index[hid]
                            sims.append(float(cosine_similarity(item_vec, self.tfidf_matrix[hid_idx])[0, 0]))
                cb_score = np.mean(sims) if len(sims) > 0 else 0.0

            # If item has zero ratings in training, CB carries full weight
            item_pop = self.train_df[self.train_df['item_id'] == iid].shape[0]
            if item_pop == 0:
                alpha_final = 0.0
            else:
                alpha_final = alpha_user

            score = alpha_final * cf_score + (1.0 - alpha_final) * cb_score
            hybrid_scores.append((iid, score))

        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores

    # Slide 6: MMR reranking
    # Greedy MMR selection: balance relevance and diversity using cosine similarity
    def _mmr_rerank(self, candidate_list, top_n=10, lam=0.6):
        # candidate_list: list of (item_id, score) ordered by relevance
        candidates = [iid for iid, _ in candidate_list]
        scores = {iid: score for iid, score in candidate_list}

        selected = []
        while len(selected) < top_n and candidates:
            if not selected:
                # pick highest scored
                pick = candidates.pop(0)
                selected.append(pick)
                continue

            mmr_scores = []
            for iid in candidates:
                rel = scores.get(iid, 0.0)
                # similarity to selected set
                sim_to_sel = 0.0
                if iid in self.item_index:
                    v_i = self.tfidf_matrix[self.item_index[iid]]
                    sims = []
                    for j in selected:
                        if j in self.item_index:
                            v_j = self.tfidf_matrix[self.item_index[j]]
                            sims.append(float(cosine_similarity(v_i, v_j)[0, 0]))
                    sim_to_sel = max(sims) if len(sims) > 0 else 0.0

                mmr_score = lam * rel - (1.0 - lam) * sim_to_sel
                mmr_scores.append((iid, mmr_score))

            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            pick = mmr_scores[0][0]
            selected.append(pick)
            candidates.remove(pick)

        # Return selected list in order with their hybrid scores
        return [(iid, scores.get(iid, 0.0)) for iid in selected]

    # Slide 7: Execute full pipeline for a user
    # Runs retrieval, hybrid scoring, then MMR re-ranking to produce top-N
    def execute_pipeline(self, user_id, top_n=10, alpha=0.5, lam=0.6):
        candidates = self._retrieve_candidates(user_id, k=50)
        hybrid = self._hybrid_score(user_id, candidates, alpha=alpha)
        reranked = self._mmr_rerank(hybrid, top_n=top_n, lam=lam)
        # Build DataFrame with titles
        rows = []
        for iid, score in reranked:
            title = self.movies[self.movies['item_id'] == iid]['title'].values
            title = title[0] if len(title) > 0 else 'Unknown'
            rows.append({'item_id': iid, 'title': title, 'score': score})
        return pd.DataFrame(rows)

    # Slide 8: Intra-list diversity
    # ILD = 1 - average pairwise cosine similarity among recommended items
    def evaluate_intra_list_diversity(self, recommended_df):
        ids = recommended_df['item_id'].tolist()
        vecs = []
        for iid in ids:
            if iid in self.item_index:
                vecs.append(self.tfidf_matrix[self.item_index[iid]].toarray()[0])
        if len(vecs) <= 1:
            return 0.0
        sims = cosine_similarity(np.vstack(vecs))
        # take upper triangle without diagonal
        n = sims.shape[0]
        pairs = n * (n - 1) / 2
        sum_sim = np.sum(np.triu(sims, k=1))
        avg_sim = sum_sim / pairs if pairs > 0 else 0.0
        ild = 1.0 - avg_sim
        return ild

    # Slide 9: Full evaluation across test users
    # Compute Precision@10, Recall@10, NDCG@10, ILD, Catalog Coverage
    def run_full_evaluation(self, n_test_users=50):
        # collect test users with at least one test interaction
        test_users = self.test_df['user_id'].unique()
        test_users = [u for u in test_users if u in self.train_df['user_id'].unique()]
        if len(test_users) == 0:
            raise RuntimeError('No overlapping users between train and test.')
        sampled = test_users[:n_test_users]

        precs, recalls, ndcgs, ilds = [], [], [], []
        recommended_items = set()

        for u in sampled:
            recs = self.execute_pipeline(u, top_n=10)
            recommended_items.update(recs['item_id'].tolist())
            relevant = set(self.test_df[self.test_df['user_id'] == u]['item_id'].tolist())

            # Precision@10
            tp = len([i for i in recs['item_id'] if i in relevant])
            prec = tp / 10.0
            precs.append(prec)

            # Recall@10
            rec_count = min(10, len(relevant))
            rec_count = rec_count if rec_count > 0 else len(relevant)
            recall = tp / len(relevant) if len(relevant) > 0 else 0.0
            recalls.append(recall)

            # NDCG@10
            dcg = 0.0
            for i, iid in enumerate(recs['item_id'].tolist()):
                rel = 1.0 if iid in relevant else 0.0
                dcg += (2 ** rel - 1) / math.log2(i + 2)
            # ideal dcg
            ideal_rels = [1.0] * min(len(relevant), 10)
            idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels)) if len(ideal_rels) > 0 else 0.0
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)

            # ILD
            ild = self.evaluate_intra_list_diversity(recs)
            ilds.append(ild)

        total_catalog = self.movies['item_id'].nunique()
        coverage = len(recommended_items) / total_catalog

        def stats(arr):
            return np.mean(arr), np.std(arr)

        print('\n=== FULL EVALUATION ===')
        p_mean, p_std = stats(precs)
        r_mean, r_std = stats(recalls)
        n_mean, n_std = stats(ndcgs)
        i_mean, i_std = stats(ilds)

        print(f'Precision@10: {p_mean:.4f} ± {p_std:.4f}')
        print(f'Recall@10: {r_mean:.4f} ± {r_std:.4f}')
        print(f'NDCG@10: {n_mean:.4f} ± {n_std:.4f}')
        print(f'ILD@10: {i_mean:.4f} ± {i_std:.4f}')
        print(f'Catalog coverage: {coverage:.4f}')

        # Alpha sensitivity
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        print('\nAlpha sensitivity (Precision@10 and ILD):')
        for a in alphas:
            precs_a, ilds_a = [], []
            for u in sampled:
                recs = self.execute_pipeline(u, top_n=10, alpha=a)
                relevant = set(self.test_df[self.test_df['user_id'] == u]['item_id'].tolist())
                tp = len([i for i in recs['item_id'] if i in relevant])
                precs_a.append(tp / 10.0)
                ilds_a.append(self.evaluate_intra_list_diversity(recs))
            print(f'alpha={a:.2f}: Precision@10={np.mean(precs_a):.4f}, ILD={np.mean(ilds_a):.4f}')

        return {
            'precision': (p_mean, p_std),
            'recall': (r_mean, r_std),
            'ndcg': (n_mean, n_std),
            'ild': (i_mean, i_std),
            'coverage': coverage
        }

    # Slide 10: Explain latent factors
    # Print the 50-dimensional item vector and top-5 similar items by dot product
    def explain_latent_factors(self, item_id, top_k=5):
        if self.svd_algo is None:
            raise RuntimeError('Model not trained. Run fit_offline_models() first.')

        trainset = self.svd_algo.trainset
        try:
            inner_iid = trainset.to_inner_iid(str(item_id))
        except Exception:
            print('Item not in trainset; cannot explain.')
            return

        item_vec = self.svd_algo.qi[inner_iid]
        print(f'Latent factors for item {item_id}:')
        print(item_vec)

        # find top-k similar in latent space by cosine similarity
        all_q = self.svd_algo.qi
        sims = cosine_similarity([item_vec], all_q)[0]
        idxs = np.argsort(sims)[::-1][1:top_k+1]
        # map inner ids back to raw ids
        similar = []
        for ii in idxs:
            raw = trainset.to_raw_iid(ii)
            similar.append((raw, sims[ii]))

        print('Top similar items (raw id, similarity):')
        for s in similar:
            print(s)

    # Slide 11: Recommendation report for a user
    # Print recommendations and ILD for a given user
    def print_recommendation_report(self, user_id):
        recs = self.execute_pipeline(user_id, top_n=10)
        print('\nRecommendations for user', user_id)
        for i, row in recs.iterrows():
            print(f"{i+1}. {row['title']} (id={row['item_id']}) score={row['score']:.4f}")
        ild = self.evaluate_intra_list_diversity(recs)
        print(f'Intra-List Diversity (ILD): {ild:.4f}')


if __name__ == '__main__':
    # Example run sequence. The evaluator should run this script to produce outputs.
    engine = ProductionCascadeHybridEngine()
    engine.load_and_preprocess()
    engine.fit_offline_models()
    print('\nTop recommendations for user 42:')
    engine.print_recommendation_report(42)
    engine.run_full_evaluation(n_test_users=50)
