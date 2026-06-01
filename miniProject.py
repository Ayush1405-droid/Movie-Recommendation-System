import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data = {
    'title': [
        'Avatar','Titanic','The Avengers','Iron Man','Interstellar',
        'Inception','The Dark Knight','Finding Nemo','Frozen','Toy Story'
    ],
    'genre': [
        'Science Fiction, Adventure, Action','Romance, Drama',
        'Action, Superhero, Science Fiction','Action, Superhero, Technology',
        'Science Fiction, Space, Drama','Science Fiction, Thriller',
        'Action, Crime, Drama','Animation, Adventure, Family',
        'Animation, Fantasy, Musical','Animation, Comedy, Family'
    ],
    'keywords': [
        'alien, planet, war, future','ship, romance, ocean, disaster',
        'heroes, battle, earth, villain','armor, inventor, billionaire, hero',
        'space, time, survival, future','dream, subconscious, heist, mind',
        'batman, joker, gotham, crime','fish, ocean, journey, family',
        'snow, magic, sister, kingdom','toys, friendship, child, adventure'
    ],
    'overview': [
        'A soldier joins a mission on a distant planet...',
        'A romantic story develops between two passengers...',
        'A team of superheroes unites...',
        'A brilliant inventor builds advanced armor...',
        'Astronauts travel through space...',
        'A thief who can enter dreams...',
        'Batman must stop the Joker...',
        'A father fish searches...',
        'A girl goes on a journey...',
        'A group of toys experience adventures...'
    ]
}

movies = pd.DataFrame(data)

movies['tags'] = movies['genre'] + " " + movies['keywords'] + " " + movies['overview']

cv = CountVectorizer(stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors) #cosine_similarity measures how similar two things are by checking angle bw their vectors

#Function
def recommend(movie_name):
    movie_name = movie_name.lower()
    matched = movies[movies['title'].str.lower() == movie_name] #searches inside dataframe
    
    if matched.empty:
        return []     #fn will return empty list and gui will show error
    
    index = matched.index[0]  # every ,ovie has position
    distances = similarity[index] # it retrieves the similarity scores of selected movie with all the other movies
    
    movie_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]
    
    return [movies.iloc[i[0]]['title'] for i in movie_list] #extract movie title using index

#GUI
def show_recommendations():
    movie_name = entry.get()
    
    results.delete(0, tk.END)  # clear previous results
    
    recs = recommend(movie_name)
    
    if not recs:
        messagebox.showerror("Error", "Movie not found!")
    else:
        for movie in recs:
            results.insert(tk.END, movie)

root = tk.Tk()
root.title("Movie Recommendation System")
root.geometry("400x400")

title_label = tk.Label(root, text="🎬 Movie Recommender", font=("Arial", 16))
title_label.pack(pady=10)

entry = tk.Entry(root, width=30)
entry.pack(pady=10)

btn = tk.Button(root, text="Recommend", command=show_recommendations)
btn.pack(pady=10)

results = tk.Listbox(root, width=50)
results.pack(pady=10)

root.mainloop()