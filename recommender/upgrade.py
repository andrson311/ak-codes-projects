from typing import Dict, Text
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})

movies = movies.map(lambda x: x["movie_title"])

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

# Stage 1: Retrieve recommended movies
embedding_dimension = 32

def build_embedding(vocab, embedding_dimension):
    model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=vocab, mask_token=None
        ),
        tf.keras.layers.Embedding(len(vocab) + 1, embedding_dimension)
    ])
    return model


user_model = build_embedding(unique_user_ids, embedding_dimension)
movie_model = build_embedding(unique_movie_titles, embedding_dimension)

retrieval_metrics = tfrs.metrics.FactorizedTopK(
    candidates=movies.batch(128).map(movie_model)
)

retrieval_task = tfrs.tasks.Retrieval(
    metrics=retrieval_metrics
)

class RetrievalModel(tfrs.models.Model):
    def __init__(self, user_model, movie_model):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = retrieval_task
    
    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        user_embeddings = self.user_model(features['user_id'])
        positive_movie_embeddings = self.movie_model(features['movie_title'])
        return self.task(user_embeddings, positive_movie_embeddings)

retrieval_model = RetrievalModel(user_model, movie_model)
retrieval_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
retrieval_model.fit(cached_train, epochs=10)
retrieval_model.evaluate(cached_test)

# Stage 2: Rank retrieved movies
ranking_task = tfrs.tasks.Ranking(
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

class RankingModel(tfrs.models.Model):
    def __init__(self, user_model, movie_model):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = ranking_task

        self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    
    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        user_embedding = self.user_model(features['user_id'])
        movie_embedding = self.movie_model(features['movie_title'])
        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        labels = features.pop('user_rating')
        rating_predictions = self(features)

        return self.task(labels=labels, predictions=rating_predictions)
    
ranking_model = RankingModel(user_model, movie_model)
ranking_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
ranking_model.fit(cached_train, epochs=100)
ranking_model.evaluate(cached_test)

# Results
user = 42

index = tfrs.layers.factorized_top_k.BruteForce(retrieval_model.user_model)
index.index_from_dataset(
    tf.data.Dataset.zip(movies.batch(100), movies.batch(100).map(retrieval_model.movie_model))
)

_, titles = index(tf.constant([f'{user}']))
retrieved = np.asarray(titles[0, :10])
print(f'Recommendations for user with id: {user}: {retrieved}')

ratings = {}
for movie_title in retrieved:
    ratings[movie_title] = ranking_model({
        'user_id': np.array([f'{user}']),
        'movie_title': np.array([movie_title])
    })

print('Ratings:')
for title, score in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
    print(f'{title}: {score}')

