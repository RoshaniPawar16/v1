class MusicRecommender:
    def __init__(self, df):
        self.df = df
        self.user_song_matrix = None
        self.similarity_matrix = None
        self.user_profiles = None
        self.song_popularity = None
        self.fitted = False
        
    def fit(self):
        # Check if the required columns exist
        required_columns = ['user', 'song', 'title', 'artist_name', 'play_count']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Create user-song matrix with weighted play counts
        self.user_song_matrix = pd.pivot_table(
            self.df,
            values='play_count',
            index='user',
            columns=['song', 'title', 'artist_name'],
            fill_value=0,
            aggfunc='sum'
        )

        # Handle edge case where max play count is 0 (to avoid division by zero)
        play_count_max = self.df['play_count'].max()
        if play_count_max == 0:
            raise ValueError("All play counts are zero, normalization cannot be performed.")

        # Calculate normalized play counts
        normalized_matrix = self.user_song_matrix.values / play_count_max

        # Calculate user similarity using cosine similarity
        self.similarity_matrix = cosine_similarity(normalized_matrix)

        # Calculate song popularity scores
        song_plays = self.df.groupby(['song', 'title', 'artist_name'])['play_count'].agg(['sum', 'count']).reset_index()
        song_plays['popularity_score'] = (
            0.7 * song_plays['sum'] / song_plays['sum'].max() + 
            0.3 * song_plays['count'] / song_plays['count'].max()
        )
        self.song_popularity = song_plays.set_index(['song', 'title', 'artist_name'])['popularity_score']

        # Mark the recommender as fitted
        self.fitted = True
        print("Recommender system fitted successfully!")

    
    def _get_candidate_songs(self, similar_users, user_songs):
        similar_user_songs = self.df[self.df['user'].isin(similar_users)]
        candidate_songs = similar_user_songs[~similar_user_songs['song'].isin(user_songs)]
        
        candidate_pool = candidate_songs.groupby(['song', 'title', 'artist_name']).agg({
            'play_count': ['sum', 'count', 'mean']
        }).reset_index()
        
        candidate_pool.columns = ['song', 'title', 'artist_name', 'total_plays', 'n_listeners', 'avg_plays']
        return candidate_pool
    
    def get_recommendations(self, user_id, n_recommendations=5, diversity_weight=0.3):
        if not self.fitted:
            raise Exception("Call fit() before making recommendations")
            
        if user_id not in self.user_song_matrix.index:
            return self._get_popular_recommendations(n_recommendations)
        
        user_history = self.df[self.df['user'] == user_id]
        user_artists = set(user_history['artist_name'])
        user_songs = set(user_history['song'])
        
        user_idx = list(self.user_song_matrix.index).index(user_id)
        similar_users = self._get_similar_users(user_idx)
        
        candidate_pool = self._get_candidate_songs(similar_users, user_songs)
        
        recommendations = []
        for _, song_data in candidate_pool.iterrows():
            similarity_score = song_data['avg_plays'] / song_data['total_plays']
            popularity_score = self.song_popularity.get((song_data['song'], 
                                                       song_data['title'], 
                                                       song_data['artist_name']), 0)
            diversity_score = self._calculate_diversity_score(song_data, user_artists)
            
            final_score = (1 - diversity_weight) * (similarity_score + popularity_score) + \
                         diversity_weight * diversity_score
            
            recommendations.append({
                'song_id': song_data['song'],
                'title': song_data['title'],
                'artist': song_data['artist_name'],
                'score': final_score,
                'genre': self._infer_genre(song_data),
                'popularity': popularity_score,
                'novelty': diversity_score
            })
        
        if diversity_weight > 0.5:
            return self._get_diverse_selection(recommendations, n_recommendations)
        else:
            return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:n_recommendations]
    
    def _get_similar_users(self, user_idx, n=10):
        similarities = self.similarity_matrix[user_idx]
        similar_indices = np.argsort(similarities)[::-1][1:n+1]
        return [self.user_song_matrix.index[idx] for idx in similar_indices]
    
    def _calculate_diversity_score(self, song_data, user_artists):
        artist = song_data['artist_name']
        artist_novelty = 1.0 if artist not in user_artists else 0.2
        genre = self._infer_genre(song_data)
        genre_novelty = 0.5
        return (artist_novelty + genre_novelty) / 2
    
    def _get_diverse_selection(self, recommendations, n):
        selected = []
        remaining = recommendations.copy()
        
        while len(selected) < n and remaining:
            best_item = max(remaining, key=lambda x: x['score'])
            selected.append(best_item)
            remaining.remove(best_item)
            
            for item in remaining:
                if item['artist'] == best_item['artist']:
                    item['score'] *= 0.5
                if item['genre'] == best_item['genre']:
                    item['score'] *= 0.8
        
        return selected
    
    def _infer_genre(self, song_data):
        title = song_data['title'].lower()
        artist = song_data['artist_name'].lower()
        
        if any(word in title + artist for word in ['rock', 'metal', 'punk']):
            return 'Rock'
        elif any(word in title + artist for word in ['jazz', 'blues']):
            return 'Jazz/Blues'
        elif any(word in title + artist for word in ['classical', 'symphony']):
            return 'Classical'
        return 'Other'
    
    def _get_popular_recommendations(self, n_recommendations=5):
        top_songs = self.song_popularity.nlargest(n_recommendations)
        
        recommendations = []
        for (song, title, artist), score in top_songs.items():
            recommendations.append({
                'song_id': song,
                'title': title,
                'artist': artist,
                'score': score * 100,
                'genre': self._infer_genre({'title': title, 'artist_name': artist}),
                'popularity': score,
                'novelty': 0.5
            })
        
        return recommendations