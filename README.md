# Movie Recommender System

This project implements a movie recommender system that leverages a combination of collaborative filtering, content-based algorithms, and Pearson's R correlation to provide users with personalized movie recommendations.

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Implementation](#implementation)
- [Usage](#usage)

## Introduction

The Movie Recommender System is designed to help users discover movies that match their preferences. It utilizes various algorithms and data sources to ensure a diverse and accurate set of recommendations.

## Data

The project uses several data sources, including:
- IMDB movie metadata
- Movie credits
- Movie keywords
- Movie ratings

These datasets are processed and combined to build the recommendation system.

## Implementation
### Data Preprocessing

- Data preprocessing includes handling missing values, extracting relevant features, and creating a dataset of qualified movies.
- It also calculates weighted ratings to improve the quality of recommendations.

### Content-Based Recommendation

- This system provides content-based recommendations using movie descriptions and keywords.
- It calculates the cosine similarity between movies to identify similar content.

### Improved Recommendation

- The project further improves recommendations by considering additional factors like user ratings and popularity.
- It filters movies based on user ratings and applies weighted ratings to prioritize high-quality movies.

### Collaborative Filtering

- Collaborative filtering is used to make recommendations based on user behavior, specifically their movie ratings.

## Usage

You can use the Movie Recommender System by calling the `hybrid_recommendation(movie_name)` function with a movie name as an argument. This function combines various recommendation techniques to provide a list of movies tailored to the user's preferences.

```python
print(hybrid_recommendation('Toy Story').head(10))

This project combines the power of multiple recommendation methods to offer a versatile and accurate movie recommendation experience for users. Enjoy discovering your next favorite film with the Movie Recommender System!
