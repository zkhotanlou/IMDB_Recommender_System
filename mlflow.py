import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
import warnings; warnings.simplefilter('ignore')

"""# Read datasets"""

md=pd.read_csv('IMDB/movies_metadata.csv')
md.head(2)

credits=pd.read_csv('IMDB/credits.csv')
credits.head(2)

keywords=pd.read_csv('IMDB/keywords.csv')
keywords.head(2)

links_small=pd.read_csv('IMDB/links_small.csv')
links_small.head(2)

df_rating = pd.read_csv('IMDB/ratings_small.csv')
df_rating.head(5)

"""# Data preprocessing"""

links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
C

m = vote_counts.quantile(0.95)
m

md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified.shape

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = md.drop('genres', axis=1).join(s)
md = md.drop([19730, 29503, 35587])
md.head(5)

md['id'] = md['id'].astype('int')
smd = md[md['id'].isin(links_small)]
smd.shape

smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])
tfidf_matrix.shape

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

"""#First content based recommendation"""

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

get_recommendations('The Godfather').head(10)

"""# Some process on data to have better recommendations"""

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')
md.shape

md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')
smd = md[md['id'].isin(links_small)]
smd.shape

smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

smd['director'] = smd['crew'].apply(get_director)
smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)

smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x,x, x])

s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s[:5]

s = s[s > 1]
stemmer = SnowballStemmer('english')
stemmer.stem('dogs')

def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words

smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))

smd.head(5)

count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

df_rating = pd.read_csv('IMDB/ratings_small.csv')
df_rating.head(5)

"""# Data preprocessing and analyzing"""
rating_copy = df_rating.copy()
rating_copy['rating'] = rating_copy['rating'].apply(np.floor)
gp_by_rating = rating_copy.groupby('rating')['rating'].agg(['count'])



movie_count = df_rating['movieId'].nunique()
cust_count = df_rating['userId'].nunique()



ax = gp_by_rating.plot(kind = 'barh', legend = False, figsize = (8,8))
plt.title('{:,} Movies, {:,} customers'.format(movie_count, cust_count), fontsize=14)
plt.axis('off')

for i in range(0,6):
    ax.text(gp_by_rating.iloc[i][0]/4, i, 'Rating {}: {:.0f}%'.
            format(i, gp_by_rating.iloc[i][0]*100 / gp_by_rating.sum()[0]), color = 'black')

agg_function = ['count','mean']

gp_by_movie = df_rating.groupby('movieId')['rating'].agg(agg_function)

df_rating = pd.merge(df_rating, smd, how='right', left_on='movieId', right_on='id')
df_rating = df_rating[['movieId', 'userId', 'rating']]
pivot_rating = pd.pivot_table(df_rating, values='rating', index='userId', columns='movieId')
pivot_rating


"""# Improved recommendation"""
with mlflow.start_run():
    get_recommendations('The Dark Knight').head(10)

    def improved_recommendations(title):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]
        movie_indices = [i[0] for i in sim_scores]
        eval_cosine = sum(movie_indices) / len(movie_indices)
        movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
        vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(0.60)
        qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')
        qualified['wr'] = qualified.apply(weighted_rating, axis=1)
        qualified = qualified.sort_values('wr', ascending=False).head(10)
        return qualified, eval_cosine

    q, eval_cosine = improved_recommendations('The Dark Knight')
    mlflow.log_metric('cosine_sim',eval_cosine)
    """# Collaborative Filtering"""


    """# PearsonR recommendation"""

    df_movie_title = smd[['id', 'title']]
    df_movie_title.shape

    def corr_recommend(movie_title, min_count):
        i = int(df_movie_title[df_movie_title['title'] == movie_title]['id'])
        target = pivot_rating[i]
        similar_to_target = pivot_rating.corrwith(target)
        corr_target = pd.DataFrame(similar_to_target, columns = ['PearsonR'])
        corr_target.dropna(inplace = True)
        corr_target = corr_target.sort_values('PearsonR', ascending = False)
        corr_target.index = corr_target.index.map(int)
        corr_target = corr_target.join(df_movie_title).join(gp_by_movie)[['PearsonR', 'title', 'count', 'mean']]
        return corr_target[corr_target['count']>min_count][:10]
        
    corr_recommend('The Dark Knight', 0)

    def hybrid_recommendation(movie_name):

        soup_based = improved_recommendations(movie_name)
        corr = corr_recommend(soup_based.iloc[0]['title'],0)
        return get_recommendations(corr.iloc[0]['title'])
        

    print(hybrid_recommendation('Toy Story').head(10))