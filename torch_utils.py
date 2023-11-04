import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import Dataset
from torchtext.vocab import vocab

from collections import Counter

# P.S: Change this to CUDA if you are on Linux/Windows and have a GPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

"""
    Below we have the DataSet definition for the MovieLens dataset.
"""

class MovieRatingDataset(Dataset):
    def __init__(self, tag_vocab, tokenizer, train='train', pad_token='<pad>'):
        self.train = train
        # 1. Load Data
        if train == 'train':
            self.data_df = pd.read_csv('data/train_ratings.csv')
        elif train == 'test':
            self.data_df = pd.read_csv('data/test_set_no_ratings.csv')
        elif train == 'valid':
            self.data_df = pd.read_csv('data/valid_data.csv')
        else:
            raise ValueError('train must be one of "train", "test", or "valid"')
        
        self.movies = pd.read_csv('data/movies.csv')
        # read movies from movies_df.xlsx
        movies_new_df = pd.read_excel('data/movies_df.xlsx')
        # Merge movies_new_df with movies_df on movieId
        self.movies = self.movies.merge(movies_new_df, on='movieId', how='left')
        # Fill NaN for the following Cols with 0 [budget, popularity, runtime, vote_average, vote_count, revenue]
        self.movies['budget'] = self.movies['budget'].fillna(0)
        self.movies['popularity'] = self.movies['popularity'].fillna(0)
        self.movies['runtime'] = self.movies['runtime'].fillna(0)
        self.movies['vote_average'] = self.movies['vote_average'].fillna(0)
        self.movies['vote_count'] = self.movies['vote_count'].fillna(0)
        self.movies['revenue'] = self.movies['revenue'].fillna(0)
        # 2. Pre-processing: Convert genres into a binary encoded vector
        self.movies['genres'] = self.movies['genres'].str.split('|')
        mlb = MultiLabelBinarizer()
        genres_encoded = mlb.fit_transform(self.movies['genres'])
        self.genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_, index=self.movies.movieId)
        
        # 3. Encode users and movies as integer indices
        user_enc = LabelEncoder()
        self.data_df['user_label'] = user_enc.fit_transform(self.data_df['userId'])
        movie_enc = LabelEncoder()
        all_movies = self.movies['movieId'].unique().tolist()
        movie_enc.fit(all_movies)
        self.data_df['movie_label'] = movie_enc.transform(self.data_df['movieId'])

        # 4. Encode original_language as integer indices
        lang_enc = LabelEncoder()
        self.movies['lang_label'] = lang_enc.fit_transform(self.movies['original_language'])
        # Create a dataframe called movie_tags with columns movieId and tag from movies_df
        movie_tags = self.movies[['movieId', 'tags']]
        self.data_df = self.data_df.merge(movie_tags, on='movieId', how='left')
        # fill NaNs with empty set string
        self.data_df['tags'] = self.data_df['tags'].fillna('[]')
        self.data_df['tags'] = self.data_df['tags'].apply(lambda x: eval(x))
        self.data_df['tags'] = self.data_df['tags'].apply(lambda x: ' '.join(x))
        self.data_df['tags'] = self.data_df['tags'].apply(lambda x: tokenizer(x))
        # Find max length of tags
        self.data_df['tag_len'] = self.data_df['tags'].apply(lambda x: len(x))
        self.max_tag_len = self.data_df['tag_len'].max()
        # pad tags with <pad> token
        self.data_df['tags'] = self.data_df['tags'].apply(lambda x: x + [pad_token] * (self.max_tag_len - len(x)))
        self.vocab = tag_vocab
        # 6. Encode tags as integer indices
        self.data_df['tag_label'] = self.data_df['tags'].apply(lambda x: [self.vocab[token] for token in x])
        # Make movieId the index in movies
        self.movies = self.movies.set_index('movieId')

    def get_data_df(self):
        return self.data_df

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        user = row['user_label']
        movie = row['movie_label']
        genres = self.genres_df.loc[row['movieId']].values.tolist()
        tags = row['tag_label']
        lang = self.movies.loc[row['movieId']]['lang_label']
        budget = self.movies.loc[row['movieId']]['budget']
        popularity = self.movies.loc[row['movieId']]['popularity']
        runtime = self.movies.loc[row['movieId']]['runtime']
        vote_average = self.movies.loc[row['movieId']]['vote_average']
        vote_count = self.movies.loc[row['movieId']]['vote_count']
        revenue = self.movies.loc[row['movieId']]['revenue']
        if self.train == 'train' or self.train == 'valid':
            rating = row['rating']
            return (user, movie, genres, tags, lang, budget, popularity, runtime, vote_average, vote_count, revenue, rating)
        row_id = row['Id']
        return (row_id, user, movie, genres, tags, lang, budget, popularity, runtime, vote_average, vote_count, revenue)

    @classmethod
    def build_tag_vocab(cls, tokenizer, min_freq=1, pad_token='<pad>', unk_token='<unk>', tags_df=None):
        if tags_df is None:
            tags_df = pd.read_excel('data/movies_df.xlsx')
        tags_df['tags'] = tags_df['tags'].fillna('[]')
        tags_df['tags'] = tags_df['tags'].apply(lambda x: eval(x))
        tags_df['tags'] = tags_df['tags'].apply(lambda x: ' '.join(x))
        tag_counter = Counter()
        for tag in tags_df['tags']:
            tag_counter.update(tokenizer(tag))
        tag_vocab = vocab(tag_counter, min_freq=min_freq)
        tag_vocab.insert_token(pad_token, 0)
        tag_vocab.append_token(unk_token)
        tag_vocab.set_default_index(tag_vocab[unk_token])
        return tag_vocab

"""
    Below we have the Neural Network for a Recommender System.
"""

class Recommender(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, num_tags, user_movie_embed=100, tag_embed=50, freeze=False, tag_weights=None, device='cpu', padding_idx=0):
        super(Recommender, self).__init__()
        self.device = device
        self.padding_idx = padding_idx
        self.user_embedding = nn.Embedding(num_users, user_movie_embed)
        self.movie_embedding = nn.Embedding(num_movies, user_movie_embed)
        if tag_weights is not None:
            self.tag_embedding = nn.Embedding.from_pretrained(tag_weights, freeze=freeze, padding_idx=padding_idx)
        else:
            self.tag_embedding = nn.Embedding(num_tags, tag_embed, padding_idx=padding_idx)
        self.fc = nn.Sequential(
            # we add 7 because: budget, popularity, runtime, vote_average, vote_count, revenue, lang
            nn.Linear(
                user_movie_embed * 2 + num_genres + tag_embed + 7,
                128
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        self.to(device)

    def collate_fn(self, batch):
        # get_item returns: (user, movie, genres, tags, lang, budget, popularity, runtime, vote_average, vote_count, revenue, rating)
        users, movies, genres, tags, lang, budget, popularity, runtime, vote_average, vote_count, revenue, ratings = zip(*batch)
        users = torch.LongTensor(users).to(self.device)
        movies = torch.LongTensor(movies).to(self.device)
        genres = torch.FloatTensor(genres).to(self.device)
        tags = torch.LongTensor(tags).to(self.device)
        ratings = torch.FloatTensor(ratings).to(self.device)
        lang = torch.LongTensor(lang).to(self.device)
        budget = torch.FloatTensor(budget).to(self.device)
        popularity = torch.FloatTensor(popularity).to(self.device)
        runtime = torch.FloatTensor(runtime).to(self.device)
        vote_average = torch.FloatTensor(vote_average).to(self.device)
        vote_count = torch.FloatTensor(vote_count).to(self.device)
        revenue = torch.FloatTensor(revenue).to(self.device)

        return ratings, (users, movies, genres, tags, lang, budget, popularity, runtime, vote_average, vote_count, revenue)

    def collate_fn_test(self, batch):
        row_id, users, movies, genres, tags, lang, budget, popularity, runtime, vote_average, vote_count, revenue = zip(*batch)
        row_id = torch.LongTensor(row_id).to(self.device)
        users = torch.LongTensor(users).to(self.device)
        movies = torch.LongTensor(movies).to(self.device)
        genres = torch.FloatTensor(genres).to(self.device)
        tags = torch.LongTensor(tags).to(self.device)
        lang = torch.LongTensor(lang).to(self.device)
        budget = torch.FloatTensor(budget).to(self.device)
        popularity = torch.FloatTensor(popularity).to(self.device)
        runtime = torch.FloatTensor(runtime).to(self.device)
        vote_average = torch.FloatTensor(vote_average).to(self.device)
        vote_count = torch.FloatTensor(vote_count).to(self.device)
        revenue = torch.FloatTensor(revenue).to(self.device)

        return row_id, (users, movies, genres, tags, lang, budget, popularity, runtime, vote_average, vote_count, revenue)

    def forward(self, x):
        users, movies, genres, tags, lang, budget, popularity, runtime, vote_average, vote_count, revenue = x
        user_emb = self.user_embedding(users)
        movie_emb = self.movie_embedding(movies)
        tag_emb = self.tag_embedding(tags)
        # Average tag embeddings
        tag_emb = tag_emb.mean(dim=1)
        # Concatenate all features
        # Everything after and including lang is 1D (N), the others are 2D (N, embed_dim)
        # To concatenate them, we need to unsqueeze lang to make it 2D (N, 1)
        lang = lang.unsqueeze(1)
        budget = budget.unsqueeze(1)
        popularity = popularity.unsqueeze(1)
        runtime = runtime.unsqueeze(1)
        vote_average = vote_average.unsqueeze(1)
        vote_count = vote_count.unsqueeze(1)
        revenue = revenue.unsqueeze(1)
        x = torch.cat([user_emb, movie_emb, genres, tag_emb, lang, budget, popularity, runtime, vote_average, vote_count, revenue], dim=1)
        x = self.fc(x)
        x = x.squeeze()
        # We want to predict ratings between 0.5 and 5.0
        x = torch.sigmoid(x) * 4.5 + 0.5
        return x