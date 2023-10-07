# SUMMER DANCEFLOOR: THE DATA-DRIVEN DANCE PARTY 🎷

# Background 🌌

It's that time of the year when summer has come, and it brings a feeling of happiness and liveliness, especially if you're in the Northern Hemisphere. Our company has decided to throw a dance party to celebrate this cheerful season.

# Objective 🎯

We've been given a special task because we have a unique mix of creativity and the ability to use our data skills. Our job is to make a playlist for our dance party that makes everyone want to dance and have a great time.

![dance party](screenshots/dance_party.jpg)

# Executive Summary 🧾

In the pursuit of creating the perfect dance-themed playlist for a summer celebration, our journey took us through a data-driven odyssey that combined art and science. Here's a brief summary of the extensive work we undertook:

**Data Preprocessing:** We have provided a dataset of over 113,027 music tracks from various genres. This dataset included audio features such as danceability, energy, tempo, and more. Extensive data preprocessing involved handling missing values, deduplicating records, and ensuring data quality.

**Exploratory Data Analysis (EDA):** EDA was pivotal in understanding the relationships between audio features and their impact on danceability. We uncovered intriguing correlations, both Spearman and Phik, that guided our playlist curation.

**Genre Clustering:** Employing NLP techniques and Doc2Vec embeddings, we clustered 114 unique music genres into 22 distinct groups. This enhanced our understanding of genre relationships and informed our playlist creation.

**Model Building:** Leveraging machine learning, we built a predictive model for danceability using Random Forest Regressor. Hyperparameter tuning and feature importance analysis helped us optimize the model's performance.

**Track Selection:** We meticulously selected tracks based on their genre, danceability, and other feature attributes. Proportional distribution ensured that each genre was proportionally represented in the playlist.

**Playlist Curation:** The culmination of our efforts led to the creation of a sensational playlist of 50 tracks. This playlist, a harmonious blend of rhythm, energy, and musical diversity, promises an electrifying dance experience.


```python
import pandas as pd
final_playlist = pd.read_csv('DancefloorAnthems.csv')
final_playlist[['track_name', 'artists', 'album_name']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track_name</th>
      <th>artists</th>
      <th>album_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hope</td>
      <td>The Chainsmokers;Winona Oak</td>
      <td>Sick Boy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Class-Sikh Maut, Vol. II</td>
      <td>Prabh Deep;Seedhe Maut;Sez on the Beat</td>
      <td>Class-Sikh Maut, Vol. II</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Parabéns Meu Amor</td>
      <td>Catuaba Com Amendoim</td>
      <td>O Tesão do Forró</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Machu Picchu</td>
      <td>The Strokes</td>
      <td>Angles</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Padre Nuestro - Remasterizado 2008</td>
      <td>Los Fabulosos Cadillacs</td>
      <td>Rey Azúcar</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ways &amp; Means</td>
      <td>The Green</td>
      <td>Ways &amp; Means</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Perseverança</td>
      <td>Xande De Pilares</td>
      <td>Perseverança</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Nüchtern</td>
      <td>Anstandslos &amp; Durchgeknallt;Emi Flemming</td>
      <td>Nüchtern</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Lover Chanting - Edit</td>
      <td>Little Dragon</td>
      <td>Lover Chanting (Jayda G Remix)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>I Was Made For Lovin' You - Single Mix</td>
      <td>KISS</td>
      <td>KISS 40</td>
    </tr>
    <tr>
      <th>10</th>
      <td>I Hope That It Hurts</td>
      <td>Nicky Romero;Norma Jean Martine</td>
      <td>I Hope That It Hurts</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Eve Of Destruction</td>
      <td>The Chemical Brothers</td>
      <td>No Geography</td>
    </tr>
    <tr>
      <th>12</th>
      <td>I Can't Explain - Stereo Version</td>
      <td>The Who</td>
      <td>My Generation (Stereo Version)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>A Little Less Conversation</td>
      <td>Elvis Presley</td>
      <td>The Essential Elvis Presley</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Blackwater - 128 full strings vocal mix</td>
      <td>Octave One</td>
      <td>One Black Water (feat. Ann Saunderson) [Full S...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>beside you</td>
      <td>keshi</td>
      <td>beside you</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Sunny - Summer Vibe Mix</td>
      <td>Blank &amp; Jones;Boney M.</td>
      <td>Relax Edition 9</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Free tibet</td>
      <td>Hilight Tribe</td>
      <td>Love medicine &amp; natural trance</td>
    </tr>
    <tr>
      <th>18</th>
      <td>So Far so Good</td>
      <td>Gabrielle Aplin</td>
      <td>Dear Happy</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Soy Yo</td>
      <td>Bomba Estéreo</td>
      <td>Amanecer</td>
    </tr>
    <tr>
      <th>20</th>
      <td>The Sibbi Song</td>
      <td>Abid Brohi;SomeWhatSuper</td>
      <td>The Sibbi Song</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Dame Tus Ojos (feat. Laurie Colón)</td>
      <td>Laurie Colon;Reynaldo Santiago " Chino "</td>
      <td>Chino " De Viaje " ..... Camino Al Cielo</td>
    </tr>
    <tr>
      <th>22</th>
      <td>NOBODY</td>
      <td>COUCOU CHLOE</td>
      <td>NOBODY</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Delírio</td>
      <td>Roberta Sá</td>
      <td>Delírio</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Meu Bem Querer - JAH-VAN</td>
      <td>BiD;Black Alien;Fernando Nunes;Seu Jorge</td>
      <td>JAH-VAN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>My Humps</td>
      <td>Joshwa;Lee Foss</td>
      <td>My Humps</td>
    </tr>
    <tr>
      <th>26</th>
      <td>I Wanna - Tchami Remix</td>
      <td>Shiba San;Tchami</td>
      <td>I Wanna (Tchami Remix)</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Spies Are Watching Me</td>
      <td>Sir Jean;Voilaaa</td>
      <td>On te l'avait dit</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Samba Sambei</td>
      <td>Criolo</td>
      <td>Nó na Orelha</td>
    </tr>
    <tr>
      <th>29</th>
      <td>CONVERSAR COM O MAR</td>
      <td>Flora Matos</td>
      <td>DO LADO DE FLORA</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Lembrança</td>
      <td>Consciência Humana</td>
      <td>Entre a Adolescência e o Crime</td>
    </tr>
    <tr>
      <th>31</th>
      <td>La Veleta</td>
      <td>Oliver Koletzki</td>
      <td>Made of Wood</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Bocat - Michael Bibi Remix</td>
      <td>Albertina;Guy Gerber;Michael Bibi</td>
      <td>Bocat (Michael Bibi Remix)</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Could You Be Loved</td>
      <td>Bob Marley &amp; The Wailers</td>
      <td>Uprising</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Mambo Para Los Presos</td>
      <td>Bayron Fire;Yiordano Ignacio</td>
      <td>Mambo Para Los Presos</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Locco</td>
      <td>Biscits</td>
      <td>Locco</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Memories</td>
      <td>Eden Prince;Nonô</td>
      <td>Memories</td>
    </tr>
    <tr>
      <th>37</th>
      <td>How Long</td>
      <td>Charlie Puth</td>
      <td>Voicenotes</td>
    </tr>
    <tr>
      <th>38</th>
      <td>CUFF IT</td>
      <td>Beyoncé</td>
      <td>RENAISSANCE</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Ride or Die</td>
      <td>T.K. Soul</td>
      <td>Untouchable</td>
    </tr>
    <tr>
      <th>40</th>
      <td>MEGA FUNK - TOMA CATUCADA</td>
      <td>DJ Bratti SC</td>
      <td>MEGA FUNK - TOMA CATUCADA</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Toma! Mega Rave</td>
      <td>DJ Ghost Floripa</td>
      <td>Toma! Mega Rave</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Genda Phool (feat. Payal Dev)</td>
      <td>Badshah;Payal Dev</td>
      <td>Genda Phool (feat. Payal Dev)</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Untitled</td>
      <td>KR$NA</td>
      <td>Untitled</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Mirchi</td>
      <td>DIVINE;MC Altaf;Phenom;Stylo G</td>
      <td>Mirchi</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Drogba (Joanna)</td>
      <td>Afro B</td>
      <td>Afrowave 2</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Whap Whap (feat. Fivio Foreign &amp; French Montana)</td>
      <td>Fivio Foreign;French Montana;Skillibeng</td>
      <td>Whap Whap (feat. Fivio Foreign &amp; French Montana)</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Joanna (Drogba) - Remix</td>
      <td>Afro B;French Montana</td>
      <td>Joanna (Drogba) [Remix]</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Get Down On It - Single Version</td>
      <td>Kool &amp; The Gang</td>
      <td>Collected</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Upside Down</td>
      <td>Diana Ross</td>
      <td>Diana</td>
    </tr>
  </tbody>
</table>
</div>



# 1. Introduction 🌟

This report is like a journey where we try to find the answer to a question: How can we choose songs that make people want to dance and have fun at our party? We'll use our data skills and sense of what sounds good to do this. We'll show you the steps we took, the things we learned from the data, and how we used both creativity and science to make a playlist that will make our party unforgettable. So, let's start this journey into the world of music, numbers, and good times!

# 2. Data Description 💿
You have assembled information on more than `125` genres of Spotify music tracks in a file called `spotify.csv`, with each genre containing approximately `1000` tracks. All tracks, from all time, have been taken into account without any time period limitations. However, the data collection was concluded in `October 2022`.
Each row represents a track that has some audio features associated with it.

![spotify-logo](screenshots/spotify-logo.jpg)

| Column     | Description              |
|------------|--------------------------|
| `track_id` | The Spotify ID number of the track. |
| `artists` | Names of the artists who performed the track, separated by a `;` if there's more than one.|
| `album_name` | The name of the album that includes the track.|
| `track_name` | The name of the track.|
| `popularity` | Numerical value ranges from `0` to `100`, with `100` being the highest popularity. This is calculated based on the number of times the track has been played recently, with more recent plays contributing more to the score. Duplicate tracks are scored independently.|
| `duration_ms` | The length of the track, measured in milliseconds.|
| `explicit` | Indicates whether the track contains explicit lyrics. `true` means it does, `false` means it does not or it's unknown.|
| `danceability` | A score ranges between `0.0` and `1.0` that represents the track's suitability for dancing. This is calculated by algorithm and is determined by factors like tempo, rhythm stability, beat strength, and regularity.|
| `energy` | A score ranges between `0.0` and `1.0` indicating the track's intensity and activity level. Energetic tracks tend to be fast, loud, and noisy.|
| `key` | The key the track is in. Integers map to pitches using standard Pitch class notation. E.g.`0 = C`, `1 = C♯/D♭`, `2 = D`, and so on. If no key was detected, the value is `-1`.| 
| `loudness` | The overall loudness, measured in decibels (dB).|
| `mode` |  The modality of a track, represented as `1` for major and `0` for minor.| 
| `speechiness` | Measures the amount of spoken words in a track. A value close to `1.0` denotes speech-based content, while `0.33` to `0.66` indicates a mix of speech and music like rap. Values below `0.33` are usually music and non-speech tracks.| 
| `acousticness` | A confidence measure ranges from `0.0` to `1.0`, with `1.0` representing the highest confidence that the track is acoustic.|
| `instrumentalness` | Instrumentalness estimates the likelihood of a track being instrumental. Non-lyrical sounds such as "ooh" and "aah" are considered instrumental, whereas rap or spoken word tracks are classified as "vocal". A value closer to `1.0` indicates a higher probability that the track lacks vocal content.|
| `liveness` | A measure of the probability that the track was performed live. Scores above `0.8` indicate a high likelihood of the track being live.|
| `valence` | A score from `0.0` to `1.0` representing the track's positiveness. High scores suggest a more positive or happier track.|
| `tempo` | The track's estimated tempo, measured in beats per minute (BPM).|
| `time_signature` | An estimate of the track's time signature (meter), which is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from `3` to `7` indicating time signatures of `3/4`, to `7/4`.|
| `track_genre` |  The genre of the track.|

[Source](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (data has been modified)


```python
# Installing Some External Libraries and Importing Necessary Libraries

# Comment out if not installed
# %pip install phik --quiet
# %pip install spacy --quiet
print("Installed")

import os
import json
import math
import phik
import spacy
import numpy as np
import pandas as pd
import seaborn as sns
from functools import reduce
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
```

    Installed
    


```python
#The following flags indicate whether certain processes need to be run again. If False, it means that the process has already been handled by either saving it as a file or taking an output screenshot. Therefore, there is no need to go through these processes again and again.

PROCESSED_DATA_SAVE = False
ML_HYPERPARAMETER_TUNING = False
ML_TRAINING = False
FEATURE_IMPORANCE = False
SAVE_FINAL_PLAYLIST = False
if ML_TRAINING:
    FEATURE_IMPORANCE = True
```


```python
spotify = pd.read_csv('data/spotify.csv')
spotify.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track_id</th>
      <th>artists</th>
      <th>album_name</th>
      <th>track_name</th>
      <th>popularity</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>track_genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5SuOikwiRyPMVoIQDJUgSV</td>
      <td>Gen Hoshino</td>
      <td>Comedy</td>
      <td>Comedy</td>
      <td>73</td>
      <td>230666.0</td>
      <td>False</td>
      <td>0.676</td>
      <td>0.4610</td>
      <td>1</td>
      <td>-6.746</td>
      <td>0</td>
      <td>0.1430</td>
      <td>0.0322</td>
      <td>0.000001</td>
      <td>0.3580</td>
      <td>0.715</td>
      <td>87.917</td>
      <td>4</td>
      <td>acoustic</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4qPNDBW1i3p13qLCt0Ki3A</td>
      <td>Ben Woodward</td>
      <td>Ghost (Acoustic)</td>
      <td>Ghost - Acoustic</td>
      <td>55</td>
      <td>149610.0</td>
      <td>False</td>
      <td>0.420</td>
      <td>0.1660</td>
      <td>1</td>
      <td>-17.235</td>
      <td>1</td>
      <td>0.0763</td>
      <td>0.9240</td>
      <td>0.000006</td>
      <td>0.1010</td>
      <td>0.267</td>
      <td>77.489</td>
      <td>4</td>
      <td>acoustic</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1iJBSr7s7jYXzM8EGcbK5b</td>
      <td>Ingrid Michaelson;ZAYN</td>
      <td>To Begin Again</td>
      <td>To Begin Again</td>
      <td>57</td>
      <td>210826.0</td>
      <td>False</td>
      <td>0.438</td>
      <td>0.3590</td>
      <td>0</td>
      <td>-9.734</td>
      <td>1</td>
      <td>0.0557</td>
      <td>0.2100</td>
      <td>0.000000</td>
      <td>0.1170</td>
      <td>0.120</td>
      <td>76.332</td>
      <td>4</td>
      <td>acoustic</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6lfxq3CG4xtTiEg7opyCyx</td>
      <td>Kina Grannis</td>
      <td>Crazy Rich Asians (Original Motion Picture Sou...</td>
      <td>Can't Help Falling In Love</td>
      <td>71</td>
      <td>201933.0</td>
      <td>False</td>
      <td>0.266</td>
      <td>0.0596</td>
      <td>0</td>
      <td>-18.515</td>
      <td>1</td>
      <td>0.0363</td>
      <td>0.9050</td>
      <td>0.000071</td>
      <td>0.1320</td>
      <td>0.143</td>
      <td>181.740</td>
      <td>3</td>
      <td>acoustic</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5vjLSffimiIP26QG5WcN2K</td>
      <td>Chord Overstreet</td>
      <td>Hold On</td>
      <td>Hold On</td>
      <td>82</td>
      <td>198853.0</td>
      <td>False</td>
      <td>0.618</td>
      <td>0.4430</td>
      <td>2</td>
      <td>-9.681</td>
      <td>1</td>
      <td>0.0526</td>
      <td>0.4690</td>
      <td>0.000000</td>
      <td>0.0829</td>
      <td>0.167</td>
      <td>119.949</td>
      <td>4</td>
      <td>acoustic</td>
    </tr>
  </tbody>
</table>
</div>



# 3. Data Preprocessing 🧹

Our dataset comprises a total of `113,027` entries, each representing a unique music track. These entries are organized into 20 columns, each containing specific information about the tracks.


```python
spotify.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 113027 entries, 0 to 113026
    Data columns (total 20 columns):
     #   Column            Non-Null Count   Dtype  
    ---  ------            --------------   -----  
     0   track_id          113027 non-null  object 
     1   artists           113026 non-null  object 
     2   album_name        113026 non-null  object 
     3   track_name        113026 non-null  object 
     4   popularity        113027 non-null  int64  
     5   duration_ms       113027 non-null  float64
     6   explicit          113027 non-null  bool   
     7   danceability      113027 non-null  float64
     8   energy            113027 non-null  float64
     9   key               113027 non-null  int64  
     10  loudness          113027 non-null  float64
     11  mode              113027 non-null  int64  
     12  speechiness       113027 non-null  float64
     13  acousticness      113027 non-null  float64
     14  instrumentalness  113027 non-null  float64
     15  liveness          113027 non-null  float64
     16  valence           113027 non-null  float64
     17  tempo             113027 non-null  float64
     18  time_signature    113027 non-null  int64  
     19  track_genre       113027 non-null  object 
    dtypes: bool(1), float64(10), int64(4), object(5)
    memory usage: 16.5+ MB
    

This dataset provides a rich array of information about various musical attributes, and we will leverage this data to curate a dance-themed playlist that resonates with the energy and spirit of our summer party. Before we proceed with playlist curation, we will perform necessary data preprocessing steps to ensure the dataset is ready for analysis and selection.

## 3.1 Handling Missing Values

Upon examining the dataset, we identified that there are a few missing values in specific columns: `artists`, `album_name`, and `track_name`.


```python
spotify.isnull().sum()[spotify.isnull().sum()>0]
```




    artists       1
    album_name    1
    track_name    1
    dtype: int64



Given the small number of missing values (only three in total) and to maintain data consistency, we decided that the best course of action is to remove the records containing these missing values. This approach ensures that the dataset remains complete and aligned with the overall structure, making it suitable for further analysis and playlist curation.


```python
# Rows where 'artists', 'album_name', or 'track_name' is null
index_to_drop = (spotify.loc[spotify['artists'].isnull() | 
                             spotify['album_name'].isnull() | 
                             spotify['track_name'].isnull()]).index

# Remove the identified rows with null values and reset the index
spotify = spotify.drop(index_to_drop).reset_index(drop=True)

# This is a double-check to ensure all rows with null values are removed
spotify.loc[spotify['artists'].isnull() | spotify['album_name'].isnull() | spotify['track_name'].isnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track_id</th>
      <th>artists</th>
      <th>album_name</th>
      <th>track_name</th>
      <th>popularity</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>track_genre</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



## 3.2 Handling Duplicate Records

During our examination of the dataset, it came to our attention that there were instances of duplicate records. Specifically, we identified a total of `444 records` that were exact duplicates of one another. These duplicates arose due to factors such as multiple appearances of the same track in different playlists or catalog entries.


```python
spotify[spotify.duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track_id</th>
      <th>artists</th>
      <th>album_name</th>
      <th>track_name</th>
      <th>popularity</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>track_genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1911</th>
      <td>0CDucx9lKxuCZplLXUz0iX</td>
      <td>Buena Onda Reggae Club</td>
      <td>Disco 2</td>
      <td>Song for Rollins</td>
      <td>16</td>
      <td>219346.0</td>
      <td>False</td>
      <td>0.841</td>
      <td>0.577</td>
      <td>0</td>
      <td>-7.544</td>
      <td>1</td>
      <td>0.0438</td>
      <td>0.238000</td>
      <td>0.860000</td>
      <td>0.0571</td>
      <td>0.843</td>
      <td>90.522</td>
      <td>4</td>
      <td>afrobeat</td>
    </tr>
    <tr>
      <th>2141</th>
      <td>2aibwv5hGXSgw7Yru8IYTO</td>
      <td>Red Hot Chili Peppers</td>
      <td>Stadium Arcadium</td>
      <td>Snow (Hey Oh)</td>
      <td>80</td>
      <td>334666.0</td>
      <td>False</td>
      <td>0.427</td>
      <td>0.900</td>
      <td>11</td>
      <td>-3.674</td>
      <td>1</td>
      <td>0.0499</td>
      <td>0.116000</td>
      <td>0.000017</td>
      <td>0.1190</td>
      <td>0.599</td>
      <td>104.655</td>
      <td>4</td>
      <td>alt-rock</td>
    </tr>
    <tr>
      <th>3723</th>
      <td>7mULVp0DJrI2Nd6GesLvxn</td>
      <td>Joy Division</td>
      <td>Timeless Rock Hits</td>
      <td>Love Will Tear Us Apart</td>
      <td>0</td>
      <td>204621.0</td>
      <td>False</td>
      <td>0.524</td>
      <td>0.902</td>
      <td>2</td>
      <td>-8.662</td>
      <td>1</td>
      <td>0.0368</td>
      <td>0.000989</td>
      <td>0.695000</td>
      <td>0.1370</td>
      <td>0.907</td>
      <td>146.833</td>
      <td>4</td>
      <td>alternative</td>
    </tr>
    <tr>
      <th>4599</th>
      <td>6d3RIvHfVkoOtW1WHXmbX3</td>
      <td>Little Symphony</td>
      <td>Serenity</td>
      <td>Margot</td>
      <td>27</td>
      <td>45714.0</td>
      <td>False</td>
      <td>0.269</td>
      <td>0.142</td>
      <td>0</td>
      <td>-23.695</td>
      <td>1</td>
      <td>0.0509</td>
      <td>0.866000</td>
      <td>0.904000</td>
      <td>0.1140</td>
      <td>0.321</td>
      <td>67.872</td>
      <td>3</td>
      <td>ambient</td>
    </tr>
    <tr>
      <th>5708</th>
      <td>481beimUiUnMUzSbOAFcUT</td>
      <td>SUPER BEAVER</td>
      <td>突破口 / 自慢になりたい</td>
      <td>突破口</td>
      <td>54</td>
      <td>255080.0</td>
      <td>False</td>
      <td>0.472</td>
      <td>0.994</td>
      <td>8</td>
      <td>-1.786</td>
      <td>1</td>
      <td>0.1140</td>
      <td>0.025900</td>
      <td>0.000000</td>
      <td>0.0535</td>
      <td>0.262</td>
      <td>103.512</td>
      <td>4</td>
      <td>anime</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>110287</th>
      <td>0sSjIvTvd6fUSZZ5rnTPDW</td>
      <td>Everything But The Girl</td>
      <td>Eden (Deluxe Edition)</td>
      <td>Another Bridge - 2012 Remaster</td>
      <td>26</td>
      <td>132826.0</td>
      <td>False</td>
      <td>0.480</td>
      <td>0.853</td>
      <td>0</td>
      <td>-6.276</td>
      <td>1</td>
      <td>0.0734</td>
      <td>0.030600</td>
      <td>0.000001</td>
      <td>0.3200</td>
      <td>0.775</td>
      <td>85.181</td>
      <td>4</td>
      <td>trip-hop</td>
    </tr>
    <tr>
      <th>110402</th>
      <td>2zg3iJW4fK7KZgHOvJU67z</td>
      <td>Faithless</td>
      <td>Faithless 2.0</td>
      <td>Tarantula</td>
      <td>21</td>
      <td>398152.0</td>
      <td>False</td>
      <td>0.622</td>
      <td>0.816</td>
      <td>6</td>
      <td>-11.095</td>
      <td>0</td>
      <td>0.0483</td>
      <td>0.009590</td>
      <td>0.578000</td>
      <td>0.0991</td>
      <td>0.427</td>
      <td>136.007</td>
      <td>4</td>
      <td>trip-hop</td>
    </tr>
    <tr>
      <th>111017</th>
      <td>46FPub2Fewe7XrgM0smTYI</td>
      <td>Morcheeba</td>
      <td>Parts of the Process</td>
      <td>Undress Me Now</td>
      <td>17</td>
      <td>203773.0</td>
      <td>False</td>
      <td>0.576</td>
      <td>0.352</td>
      <td>7</td>
      <td>-10.773</td>
      <td>0</td>
      <td>0.0268</td>
      <td>0.700000</td>
      <td>0.270000</td>
      <td>0.1600</td>
      <td>0.360</td>
      <td>95.484</td>
      <td>4</td>
      <td>trip-hop</td>
    </tr>
    <tr>
      <th>112001</th>
      <td>6qVA1MqDrDKfk9144bhoKp</td>
      <td>Acil Servis</td>
      <td>Küçük Adam</td>
      <td>Bebek</td>
      <td>38</td>
      <td>319933.0</td>
      <td>False</td>
      <td>0.486</td>
      <td>0.485</td>
      <td>5</td>
      <td>-12.391</td>
      <td>0</td>
      <td>0.0331</td>
      <td>0.004460</td>
      <td>0.000017</td>
      <td>0.3690</td>
      <td>0.353</td>
      <td>120.095</td>
      <td>4</td>
      <td>turkish</td>
    </tr>
    <tr>
      <th>112378</th>
      <td>5WaioelSGekDk3UNQy8zaw</td>
      <td>Matt Redman</td>
      <td>Sing Like Never Before: The Essential Collection</td>
      <td>Our God - New Recording</td>
      <td>34</td>
      <td>265373.0</td>
      <td>False</td>
      <td>0.487</td>
      <td>0.895</td>
      <td>11</td>
      <td>-5.061</td>
      <td>1</td>
      <td>0.0413</td>
      <td>0.000183</td>
      <td>0.000000</td>
      <td>0.3590</td>
      <td>0.384</td>
      <td>105.021</td>
      <td>4</td>
      <td>world-music</td>
    </tr>
  </tbody>
</table>
<p>444 rows × 20 columns</p>
</div>



To ensure the integrity and accuracy of our data analysis, we have chosen to address this issue by removing the duplicate records. This action eliminates redundancy and streamlines our dataset to contain only unique entries, thus preventing any potential skewing of our analysis or playlist curation process.


```python
#dropping duplicated records
spotify.drop_duplicates(inplace=True)

#ensuring all duplicates are removed
('Duplicated Records', spotify[spotify.duplicated()].shape[0])
```




    ('Duplicated Records', 0)



Before moving ahead, there are fields like `artist names` can pose challenges in finding duplicated records, as their names may appear in different orders across records. To ensure accurate analysis, it's essential to **standardize the artist names** within each record.

Consider the following scenario:

| Record | Artist            | Song        |
| ------ | ----------------- | ----------- |
| 1      | Aaron; Paul       | HelloWorld  |
| 2      | Paul; Aaron       | HelloWorld  |

Although both records represent the same song, they are treated as distinct entries due to the inconsistent order of artist names. To effectively identify and manage duplicate records, we must standardize the artist names by ordering them consistently.

By standardizing the artist names, the program can recognize these as duplicate records:

| Record | Artist            | Song        |
| ------ | ----------------- | ----------- |
| 1      | Aaron; Paul       | HelloWorld  |
| 2      | Aaron; Paul       | HelloWorld  |

This enables us to confidently remove duplicates from the dataset, ensuring data integrity and accurate analysis.


```python
def orderDelimiterText(text,delimiter):
    # space remove
    text = " ".join(text.split())

    # Convert Text to List
    array = text.split(delimiter)
    
    # strip removing start-end whitespaces from a string
    array = list ( map(str.strip, array) )
    
    # Sort elements
    array.sort()
    
    # Convert Back to String
    return ";".join(array)


print("Number of Duplicated Records by artists & track_name Before Name Standardization are:",spotify[['artists','track_name']].duplicated().sum())
ordered_artists = spotify['artists'].apply(orderDelimiterText, args=(";",))
spotify['artists'] = ordered_artists
print("Number of Duplicated Records by artists & track_name After Name Standardization are:",spotify[['artists','track_name']].duplicated().sum())

print("Number of Duplicated Records are:",spotify[spotify.duplicated()].size)
spotify = spotify.drop_duplicates()
spotify = spotify.reset_index(drop=True)
print("Duplicated Records after deleting:", spotify.duplicated().sum())
```

    Number of Duplicated Records by artists & track_name Before Name Standardization are: 32009
    Number of Duplicated Records by artists & track_name After Name Standardization are: 32018
    Number of Duplicated Records are: 0
    Duplicated Records after deleting: 0
    

We also observed that for the same `track_id`. There can be multiple entries of **same song with different genres** assigned to them. 

**Example** 

| track id |   Artist   | Song |     Genre     |
|:--------:|:----------:|:----:|:-------------:|
|     1    | Jack; Ryan |  ABC |      rock     | 
|     1    | Jack; Ryan |  ABC |      punk     |

It is important to note that some songs may belong to multiple genres, and removing these duplicate entries could result in the loss of valuable genre information. 

As such, we have **decided not to delete duplicates in cases where the track ID remains the same**, preserving the diversity of genre associations for each song.


```python
print('Baby Blue Tracks With Different Genres:', spotify[spotify['track_id']=='6S3JlDAGk3uu3NtZbPnuhS'].shape[0] )
spotify[spotify['track_id']=='6S3JlDAGk3uu3NtZbPnuhS'][['track_id', 'artists', 'track_name', 'track_genre']]
```

    Baby Blue Tracks With Different Genres: 9
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track_id</th>
      <th>artists</th>
      <th>track_name</th>
      <th>track_genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8214</th>
      <td>6S3JlDAGk3uu3NtZbPnuhS</td>
      <td>Badfinger</td>
      <td>Baby Blue - Remastered 2010</td>
      <td>blues</td>
    </tr>
    <tr>
      <th>19434</th>
      <td>6S3JlDAGk3uu3NtZbPnuhS</td>
      <td>Badfinger</td>
      <td>Baby Blue - Remastered 2010</td>
      <td>country</td>
    </tr>
    <tr>
      <th>34272</th>
      <td>6S3JlDAGk3uu3NtZbPnuhS</td>
      <td>Badfinger</td>
      <td>Baby Blue - Remastered 2010</td>
      <td>folk</td>
    </tr>
    <tr>
      <th>61420</th>
      <td>6S3JlDAGk3uu3NtZbPnuhS</td>
      <td>Badfinger</td>
      <td>Baby Blue - Remastered 2010</td>
      <td>j-pop</td>
    </tr>
    <tr>
      <th>62273</th>
      <td>6S3JlDAGk3uu3NtZbPnuhS</td>
      <td>Badfinger</td>
      <td>Baby Blue - Remastered 2010</td>
      <td>j-rock</td>
    </tr>
    <tr>
      <th>80990</th>
      <td>6S3JlDAGk3uu3NtZbPnuhS</td>
      <td>Badfinger</td>
      <td>Baby Blue - Remastered 2010</td>
      <td>power-pop</td>
    </tr>
    <tr>
      <th>83043</th>
      <td>6S3JlDAGk3uu3NtZbPnuhS</td>
      <td>Badfinger</td>
      <td>Baby Blue - Remastered 2010</td>
      <td>psych-rock</td>
    </tr>
    <tr>
      <th>98443</th>
      <td>6S3JlDAGk3uu3NtZbPnuhS</td>
      <td>Badfinger</td>
      <td>Baby Blue - Remastered 2010</td>
      <td>singer-songwriter</td>
    </tr>
    <tr>
      <th>101384</th>
      <td>6S3JlDAGk3uu3NtZbPnuhS</td>
      <td>Badfinger</td>
      <td>Baby Blue - Remastered 2010</td>
      <td>songwriter</td>
    </tr>
  </tbody>
</table>
</div>



In our data exploration, we also found that utilizing a composite key comprised of the attributes `artists`, `track_name`, and `track_genre` proved effective in identifying and addressing duplicate records. While these tracks might share similar names and artists, they often appeared across multiple albums, introducing redundancy to the dataset.


```python
filter = spotify[['track_name','artists', 'track_genre']]
same_track = filter.value_counts()
mask = filter.value_counts() > 1
print('Duplicated Clusters of Records by track_name, artists, track_genre:', same_track[mask].size)
print('Duplicated Records by track_name, artists, track_genre:', filter.duplicated().size)
```

    Duplicated Clusters of Records by track_name, artists, track_genre: 6766
    Duplicated Records by track_name, artists, track_genre: 112582
    

To tackle this issue, we chose a **deduplication approach**, systematically removing duplicate entries while retaining a single, best representative record within each duplicate group. Our objectives with this approach are to maintain data consistency, reduce redundancy, and preserve valuable genre information associated with tracks appearing in multiple albums.

**Euclidean distance-based record deduplication**
>
> **Overview**
> 
> In our deduplication process, our main goal is to find the **best single record** within a group of duplicate entries. We do this by comparing the attributes of each record to the **highest values** found in that same group. This helps us pick the record that comes **closest to the ideal attribute values**, making it the most representative choice for that specific group of duplicates.
>
> **Methodology:**
> 
> Features: `ID`, `Artist`, `Song`, `Genre`, `popularity`, `danceability`, `energy`, `tempo`
> 
> On the basis of features **Artist, Song and, Genre** duplicated rows are identified and retained. However, when considering features popularity, danceability, energy and tempo, we select a single representative record to avoid duplicates. 
> 
> 
> | ID | Artist | Song | Genre | popularity | danceability | energy | tempo |
> |:--:|:------:|:----:|:-----:|:----------:|:------------:|:------:|:-----:|
> | 1  |   A    | B    | rock  |     10     |      0.5     |   0.3  | 4     |
> | 2  |   A    | B    | rock  |     30     |      0.2     |   0.2  | 3     |
> | 3  |   A    | B    | rock  |     70     |      0.8     |   0.1  | 3     |
> 
> 
> 
> **Step 1: Define the Maximum Point (Max)**
> 
> - Which consists of the maximum values for each relevant attribute.
> - $ Max = max_{popularity}, max_{danceability}, max_{energy}, max_{tempo} = (70,0.8,0.3,4) $
> 
> **Step 2: Calculate Euclidean Distance for Each Record**
> 
> - For each duplicate record, calculate the Euclidean distance between the record's attribute values and the maximum point Max. 
> 
> $$ Distance_{Record}= \sqrt( \sum_{i=1}^{N=columns} (Max_i - Record_{i} )^2 ) $$
> where columns = relevant attributes
>
> - $ Distance_{1} =\sqrt((70-10)^2 + (0.8-0.5)^2 + (0.3-0.3)^2) + (4-4)^2) \approx 60 $
> - $ Distance_{2} =\sqrt((70-30)^2 + (0.8-0.2)^2 + (0.3-0.2)^2) + (4-3)^2) \approx 40 $
> - $ Distance_{3} =\sqrt((70-10)^2 + (0.8-0.8)^2 + (0.3-0.1)^2) + (4-3)^2) \approx 1 $
> 
> **Step 3: Select the Representative Record**
> 
> - Choose the record with the smallest Euclidean distance as the representative record for that group of duplicate tracks. The record that is closest to the maximum point Max is selected.
> $$ Selected = Distance_{Record} < OthersDistance $$
> - $ Selected_{3} = Distance_{3} < \{Distance_{1},Distance_{2}\} $
> 
> **Step 4: Repeat for All Groups of Duplicates**
> 
> - Iterated through all groups of duplicate tracks based on the specified attributes and apply the same process to select representative records for each group.


```python
# Return Duplicated Tracks From a Group
def selectOneTrackFromDuplicated(data, attributes):
    
    # Create Max Point For Each Attribute
    max_point = data[attributes].max()
    
    # Calculate Euclidean Distance from Record To Max Point
    euclidean_distances = np.sqrt(((data[attributes] - max_point) ** 2).sum(axis=1))
    
    # Find the index of the record with the minimum distance
    min_distance_idx = euclidean_distances.idxmin()

    # Every Record Having Not The Minimum Distance
    dropping_idx = data.index[data.index != min_distance_idx].tolist()
    
    # Return Index of Those Which Needed To Be Drop
    return dropping_idx


# Return Duplicated Tracks From All Groups
def getDroppingIdx(data, attributes, base_attr):

    dropping_idx = []
    
    # Identify clusters with the same attribute values 
    # (e.g., 'track_name', 'artists', 'track_genre')
    clusters = data.groupby(attributes).filter(lambda group: len(group) > 1)
    
    # Iterate through each cluster of duplicate records
    for _, cluster in clusters.groupby(attributes):
        # Apply selectOneTrackFromDuplicated to each cluster to select the record to be dropped
        dropping_idx.extend(selectOneTrackFromDuplicated(data=cluster, attributes=base_attr))

    return dropping_idx
```


```python
# Calculating duplicated tracks indexes
dropping_idx = getDroppingIdx(data=spotify, 
                                attributes=['track_name','artists', 'track_genre'], 
                                base_attr=["popularity", "danceability", "energy", "tempo"])
    
# Dropping duplicated tracks
spotify = spotify.drop(dropping_idx).reset_index(drop=True)

# Ensuring all the duplicated are dropped
same_track = spotify[['track_name','artists', 'track_genre']].value_counts()
mask = spotify[['track_name','artists', 'track_genre']].value_counts() > 1
print('Duplicated Records by track_name, artists, track_genre:', same_track[mask].size)
```

    Duplicated Records by track_name, artists, track_genre: 0
    


```python
total_data = filter.groupby(['track_genre']).agg(count=('track_genre', 'count'))
duplicated_data = filter[filter.duplicated()].groupby(['track_genre']).agg(count=('track_genre', 'count'))
duplicated_data['count']  = total_data['count'] - duplicated_data['count'] 
order = duplicated_data.sort_values(by='count', ascending=False).index

f, ax = plt.subplots(figsize=(14, 20))

# Create the first barplot for 'total_data' with a 'Total' label
sns.barplot(x="count", y="track_genre", data=total_data, label="Duplicated Tracks Drop", color="lightblue", orient='h', order=order)
# Create the second barplot for 'duplicated_data' with a 'Duplicated' label,
sns.barplot(x="count", y="track_genre", data=duplicated_data, label="Total Track", color="dimgray", orient='h', order=order)

ax.legend(loc="upper center")
ax.set(xlabel="Count", ylabel="Track Genre", title="Total vs. Duplicated Tracks by Track Name, Genre, and Album")
plt.show()
```


    
![png](readme_files/output_30_0.png)
    


This strategy ensures that our curated dance-themed playlist is comprehensive, free from replication-related inconsistencies, and rich in diverse genre representation.


```python
"Records",spotify.shape[0]
```




    ('Records', 99312)



After handling of missing values and the removal of duplicate records, we have successfully refined our dataset. As a result, we now possess a dataset that comprises a total of **99,312 unique entries**. These entries represent distinct music tracks, each contributing to the rich tapestry of our dataset. This streamlined dataset is now well-prepared and primed for the subsequent stages of our analysis and the essential task of crafting the perfect dance-themed playlist for our upcoming summer party.

# 4. Exploratory Data Analysis 📚

## 4.1 Exploring Variable Relationships: Spearman Correlation Heatmap

In our exploratory journey into the dataset, we employed a valuable analytical tool, the Spearman correlation heatmap. This heatmap unveiled insights into the relationships between various musical attributes, shedding light on how they interact within the context of our dataset.


```python
attr = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

sns.set_theme()

# Spearman’s ρ 
x = spotify[attr].corr()
fig = plt.figure(figsize =(16, 6)) 
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
sns.heatmap(x, cmap='YlGnBu', annot=True, ax=ax)
plt.xticks(rotation = 45)
plt.title('Spearman Correlation Between Each Variables', fontsize = 40)
plt.show()
```


    
![png](readme_files/output_35_0.png)
    


Notable findings include a strong positive correlation **_0.76_** between `loudness` and `energy`, indicating louder tracks are often more energetic. A positive correlation **_0.47_** between `valence` and `danceability` suggests cheerful tracks are dance-friendly. `Speechiness` and `explicit` exhibit a positive correlation **_0.32_**, indicating explicit tracks tend to have more spoken content. Additionally, `valence` and `loudness` show a positive correlation **_0.27_**, implying happier tracks may be louder. These insights inform our playlist curation, ensuring an enjoyable and harmonious selection for our summer party celebration.

## 4.2 Exploring Variable Relationships: Phik Correlation Heatmap

In our data exploration, we utilized the Phik correlation analysis to delve into the relationships between musical attributes. Notably, the Phik analysis unveiled distinctive insights into these connections. For instance, it revealed a particularly robust positive correlation of **_0.80_** between `loudness` and `energy`, emphasizing that louder tracks often correspond to higher energy levels, aligning with our objective of curating an energetic playlist. Similarly, it showcased a significant correlation of **_0.74_** between `acousticness` and `energy`, indicating that tracks with lower acoustic qualities tend to possess greater energy, a valuable finding for our upbeat playlist. Additionally, it highlighted correlations such as **_0.62_** between `acousticness` and `loudness`, which underscores the connection between acoustic qualities and volume in music.


```python
# Phik (𝜙k)
x = spotify[attr].phik_matrix(interval_cols=attr)
fig = plt.figure(figsize =(16, 6)) 
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
sns.heatmap(x, cmap='YlGnBu', annot=True, ax=ax)
plt.xticks(rotation = 45)
plt.title('Phik Correlation Between Each Variables', fontsize = 40)
plt.show()
```


    
![png](readme_files/output_38_0.png)
    


What sets the Phik correlation analysis apart is its focus on specific attributes' correlation with `danceability`. It revealed a correlation of **_0.55_** between `valence` and `danceability`, suggesting that tracks with a more positive mood are often well-suited for dancing. Moreover, it emphasized the correlation of **_0.56_** between `tempo` and `danceability`, indicating that faster tempos are linked to greater danceability, and the correlation of **_0.49_** between `time_signature` and `danceability`, highlighting that certain time signatures may contribute to a track's dance-friendly nature.

## 4.3 Analyzing Danceability Across Musical Attributes

As we embark on our journey to curate the perfect dance-themed playlist, we recognize the importance of understanding how various musical attributes contribute to the danceability of a track. To gain deeper insights, we conducted a thorough analysis by plotting point graphs of `danceability` in relation to key attributes, specifically `energy`, `loudness`, `speechiness`, `acousticness`, `tempo`, `valence`, `popularity`, `liveness`, and `duration_ms`. These plots are segmented into quartiles, allowing us to explore how danceability varies across different ranges of these attributes.


```python
# Create a copy of the Spotify dataset
spotify_copy = spotify.copy()

# Convert selected attributes to quartiles
attributes_to_convert = ['energy', 'loudness', 'speechiness', 'acousticness', 'tempo', 'valence', 'popularity', 'liveness']

# Create quartile columns for each attribute
for attribute in attributes_to_convert:
    spotify_copy[f'{attribute}_quartile'] = pd.qcut(spotify_copy[attribute], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Create a bar plot for each attribute quartile
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))

for i, attribute in enumerate(attributes_to_convert):
    sns.pointplot(data=spotify_copy, x=f'{attribute}_quartile', y='danceability', ax=axes[i // 2, i % 2])
    axes[i // 2, i % 2].set_title(f'{attribute[0].upper()+attribute[1:]} Quartile Distribution')

plt.subplots_adjust(hspace=0.5)    
plt.show()
```


    
![png](readme_files/output_41_0.png)
    


These point plots offer a valuable visual representation of how different musical attributes impact danceability. This comprehensive understanding will be instrumental in developing a predictive model that accurately anticipates the danceability of tracks.

# 5. Enhancing Genre Analysis with Clustering ♣️♠️♥️♦️

In our pursuit of understanding the intricate relationship between music genres and danceability, we recognized the need for a more refined approach. Initially, we attempted a basic analysis of danceability across **_114 unique music genres_**, but creating a barplot was quite overwhelming.


```python
spotify.groupby('track_genre')['danceability'].agg(['mean', 'median', ('mode',lambda x: x.mode().iloc[0])])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mode</th>
    </tr>
    <tr>
      <th>track_genre</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acoustic</th>
      <td>0.550380</td>
      <td>0.5580</td>
      <td>0.582</td>
    </tr>
    <tr>
      <th>afrobeat</th>
      <td>0.669148</td>
      <td>0.6890</td>
      <td>0.677</td>
    </tr>
    <tr>
      <th>alt-rock</th>
      <td>0.538030</td>
      <td>0.5480</td>
      <td>0.557</td>
    </tr>
    <tr>
      <th>alternative</th>
      <td>0.562526</td>
      <td>0.5580</td>
      <td>0.487</td>
    </tr>
    <tr>
      <th>ambient</th>
      <td>0.371284</td>
      <td>0.3700</td>
      <td>0.324</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>techno</th>
      <td>0.685112</td>
      <td>0.7030</td>
      <td>0.800</td>
    </tr>
    <tr>
      <th>trance</th>
      <td>0.574771</td>
      <td>0.5820</td>
      <td>0.518</td>
    </tr>
    <tr>
      <th>trip-hop</th>
      <td>0.639358</td>
      <td>0.6495</td>
      <td>0.641</td>
    </tr>
    <tr>
      <th>turkish</th>
      <td>0.615769</td>
      <td>0.6330</td>
      <td>0.650</td>
    </tr>
    <tr>
      <th>world-music</th>
      <td>0.414235</td>
      <td>0.4235</td>
      <td>0.389</td>
    </tr>
  </tbody>
</table>
<p>114 rows × 3 columns</p>
</div>





## 5.1 Incorporating External Genre Definitions

To overcome this challenge, we decided to employ a more sophisticated method. We harnessed the power of _Natural Language Processing (NLP)_ to extract and analyze the definitions associated with each of the 114 music genres. This meticulous process enabled us to create a foundation for a more profound exploration of genre relationships.

To harness the full potential of these genres and derive meaningful insights, we embarked on the task of defining each genre. Our approach involved the utilization of a Large Language Model (LLM), a tool renowned for its ability to generate succinct and contextually relevant definitions for diverse genres.

**_Why Use an LLM?_**

We chose to leverage an LLM for several reasons:
- **Consistency**: Using an LLM ensured that the genre definitions shared a consistent structure and length, allowing for easier comparison and clustering.
- **Reduced Vocabulary Diversity**: By constraining the vocabulary used in the definitions, we aimed to achieve a level of uniformity in the language used to describe different genres.


```python
#Once we had generated these genre definitions, we saved them in a structured format. We created a CSV file named `genre_dictionary.csv` and stored it in the 'files' folder.
genre_dictionary = pd.read_csv('processed_data/genre_dictionary.csv')
genre_dictionary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre</th>
      <th>definition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Acoustic</td>
      <td>Music produced primarily using acoustic instru...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afrobeat</td>
      <td>A genre blending African rhythms and Western j...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alt-Rock</td>
      <td>Alternative rock, characterized by a non-mains...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alternative</td>
      <td>A broad category of non-mainstream music with ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ambient</td>
      <td>Music often characterized by soothing and atmo...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>109</th>
      <td>Techno</td>
      <td>Electronic dance music style with repetitive b...</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Trance</td>
      <td>Electronic music genre known for its hypnotic ...</td>
    </tr>
    <tr>
      <th>111</th>
      <td>Trip-Hop</td>
      <td>Electronic music genre with downtempo beats an...</td>
    </tr>
    <tr>
      <th>112</th>
      <td>Turkish</td>
      <td>Music originating from Turkey, reflecting cult...</td>
    </tr>
    <tr>
      <th>113</th>
      <td>World-Music</td>
      <td>Diverse music styles from around the world, of...</td>
    </tr>
  </tbody>
</table>
<p>114 rows × 2 columns</p>
</div>



## 5.2 Transforming Genre Definitions into Vector

By infusing the genre definitions with the richness of _Doc2Vec embeddings_ and the linguistic context provided by spaCy's open-source word embedding model, we were equipped with a powerful tool for clustering the genres based on their definitions. This approach not only enabled us to unveil cohesive genre groups but also emphasized our dedication to incorporating accessible and state-of-the-art resources in our analytical process.


```python
spacy.cli.download("en_core_web_md")
nlp_model = spacy.load("en_core_web_md")
nlp_model

def lemmatizer(text):
    # Convert to lowercase
    lower_case = text.lower()

    # Lemmatization and Remove Stop Words
    tokens = [token.lemma_ for token in nlp_model(lower_case) if not token.is_stop]

    # Sentence Formation
    sentence = " ".join(tokens)
    return sentence

def doc2vec(text):
    # Lemmatization
    text = lemmatizer(text)

    # Initialize a vector with zeros
    vector = np.zeros(len(nlp_model('').vector))
    
    # Tokenize the lemmatized text
    tokens = nlp_model(text)
    len_of_text = len(tokens)

    if len_of_text > 0:
        # Calculate the weighted vector
        for token in tokens:            
            vector = vector + token.vector
        vector = vector / len_of_text
    return vector


def map_genre_to_doc2vec(df):
    # Extract the genre definitions from the DataFrame
    definition = list(df['definition'])
    
    # Get the dimensions for the resulting Doc2Vec vectors
    rows, cols = len(definition), len(nlp_model('').vector)
    
    # Initialize an array to store the Doc2Vec vectors
    doc2vec_genre = np.zeros(rows*cols).reshape(rows, cols)

    # Calculate the Doc2Vec vector for each genre definition
    for i, text in enumerate(definition):
        vector = doc2vec(text)
        doc2vec_genre[i] = vector
    return doc2vec_genre
```


```python
genre_vector = map_genre_to_doc2vec(genre_dictionary)
print("doc2vec representation of genre#1: ", genre_vector[0][0:5])
print("doc2vec representation of genre#2: ", genre_vector[1][0:5])
print("doc2vec representation of genre#3: ", genre_vector[2][0:5])
print("doc2vec representation of genre#4: ", genre_vector[3][0:5])
print(genre_vector.shape)
```

    doc2vec representation of genre#1:  [-2.71545241 -1.35414001 -1.28834057 -0.35858012  2.50714705]
    doc2vec representation of genre#2:  [-0.94569867 -0.89432041  0.11685249 -1.32520501  2.72854838]
    doc2vec representation of genre#3:  [-2.93170039 -0.61801003  0.00729301  0.66736898  4.06898808]
    doc2vec representation of genre#4:  [-2.83153338 -0.257912   -0.58647201  0.26961697  4.73906107]
    (114, 300)
    

## 5.3 Optimal Genre Clustering with K-Means

In our endeavor to cluster the 114 music genres based on their definitions using NLP techniques, an essential step was to determine the optimal number of clusters. To achieve this, we leveraged the **Elbow Method**, a widely recognized technique in clustering analysis. The essence of the Elbow Method lies in identifying a point in the graph where there is a sudden change in the loss or distortion function. This point is often referred to as the 'elbow' of the graph.


```python
# Initialize empty lists to store sum of squares and cluster values
sum_of_squares = []
clusters = []

# Define a range of cluster values to test (from 20 to 39)
iteration = range(20,40)
clusters.extend(iteration)

# Iterate through different cluster values
for k in iteration:
    # Create a KMeans model with specified parameters
    model = KMeans(
        n_clusters=k,             # Number of clusters
        init='k-means++',         # Initialization method
        n_init=10,                # Number of re-run with different initializations
        max_iter=100,             # Maximum number of iterations for each run
        tol=1e-3,                 # Tolerance to declare convergence
        random_state=k*2,         # Random seed for reproducibility
    )

    # Fit the KMeans model to the genre vectors
    model.fit(genre_vector)
    # Append the sum of squares error to the list
    sum_of_squares.append(model.inertia_)

# Plotting elbow curve
sns.set(style='darkgrid')
fig = plt.figure(figsize=(17, 10))
plt.plot(clusters, sum_of_squares)
plt.grid(True)
plt.xticks(np.arange(min(clusters), max(clusters) + 1, 1))
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squares Error')
plt.title('Elbow curve')
```




    Text(0.5, 1.0, 'Elbow curve')




    
![png](readme_files/output_51_1.png)
    


After careful analysis, we found that this sudden change occurred precisely at **22 clusters**. This number stood out as the optimal choice for grouping the 114 music genres based on their definitions. The decision to select 22 clusters was deliberate and strategic. It allowed us to strike the right balance between granularity and coherence, ensuring that each cluster captured a meaningful set of genres while avoiding excessive fragmentation.


```python
model = KMeans(
    n_clusters=22, 
    init='k-means++',
    n_init=10, 
    max_iter=100,
    tol=1e-3,
    random_state=78, 
    )
model.fit(genre_vector)

genre_dictionary['cluster'] = model.labels_
genre_dictionary = genre_dictionary[['genre','cluster','definition']]
genre_dictionary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre</th>
      <th>cluster</th>
      <th>definition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Acoustic</td>
      <td>9</td>
      <td>Music produced primarily using acoustic instru...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afrobeat</td>
      <td>6</td>
      <td>A genre blending African rhythms and Western j...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alt-Rock</td>
      <td>16</td>
      <td>Alternative rock, characterized by a non-mains...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alternative</td>
      <td>16</td>
      <td>A broad category of non-mainstream music with ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ambient</td>
      <td>9</td>
      <td>Music often characterized by soothing and atmo...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>109</th>
      <td>Techno</td>
      <td>17</td>
      <td>Electronic dance music style with repetitive b...</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Trance</td>
      <td>2</td>
      <td>Electronic music genre known for its hypnotic ...</td>
    </tr>
    <tr>
      <th>111</th>
      <td>Trip-Hop</td>
      <td>2</td>
      <td>Electronic music genre with downtempo beats an...</td>
    </tr>
    <tr>
      <th>112</th>
      <td>Turkish</td>
      <td>1</td>
      <td>Music originating from Turkey, reflecting cult...</td>
    </tr>
    <tr>
      <th>113</th>
      <td>World-Music</td>
      <td>1</td>
      <td>Diverse music styles from around the world, of...</td>
    </tr>
  </tbody>
</table>
<p>114 rows × 3 columns</p>
</div>



These clusters provide a structured framework that enhances our comprehension of the intricate relationships between genres. The benefits of this genre clustering are profound. It not only simplifies the analysis but also opens doors to more insightful and effective subsequent analyses. With these cohesive genre clusters in hand, we are better equipped to curate a dance-themed playlist that is not only harmonious but also diverse, catering to a wide spectrum of musical tastes and preferences at our upcoming summer party celebration.


```python
# Generate the dictionary
result_dict = genre_dictionary.groupby('cluster')['genre'].apply(list).to_dict()

# Convert the dictionary to a DataFrame
genre_cluster = pd.DataFrame(result_dict.items(), columns=['Cluster', 'Genres'])

# Display the DataFrame
genre_cluster
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cluster</th>
      <th>Genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>[Honky-Tonk, Reggaeton]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>[Brazil, British, Classical, Folk, French, Ger...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>[Bluegrass, Blues, Breakbeat, Drum-and-Bass, D...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>[Anime, Indie, Opera, Piano]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>[Club, Dancehall, Disco, Forro, Groove, Guitar...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>[Black-Metal, Death-Metal, Grindcore, Metal]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>[Afrobeat, Comedy, Detroit-Techno, Disney, Hap...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>[Cantopop, Mandopop]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>[Children, Kids, Sleep, Study]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>[Acoustic, Ambient, Dub, EDM, Electro, Electro...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>[Pop, Romance, Sad, Singer-Songwriter, Songwri...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>[Punk-Rock, Rockabilly]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>[Country, Gospel, Goth, New-Age, Reggae, Salsa...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>[Garage, Power-Pop, R-n-B, Rock-n-Roll, Rock]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>[Show-Tunes]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>[Indie-Pop, Punk]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>[Alt-Rock, Alternative]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>[Chicago-House, Chill, Dance, Deep-House, Hard...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>[J-Pop, K-Pop, Synth-Pop]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>[Emo, Grunge, Hardcore, Metalcore]</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>[House, J-Dance]</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>[Pop-Film]</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Convert genre names to lowercase for consistent mapping
genre_dictionary['genre'] = genre_dictionary['genre'].apply(str.lower)

# Create a mapping dictionary to map lowercase genre names to cluster labels as integers
# This dictionary associates each genre with its corresponding cluster label
mapping_dict = genre_dictionary.groupby('genre')['cluster'].apply(int).to_dict()

# Map the cluster labels to the 'track_genre' column in the Spotify dataset
# This adds a new column 'track_genre_cluster' with cluster labels for each track genre
spotify['track_genre_cluster'] = spotify['track_genre'].map(lambda x: mapping_dict[x])

spotify.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track_id</th>
      <th>artists</th>
      <th>album_name</th>
      <th>track_name</th>
      <th>popularity</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>track_genre</th>
      <th>track_genre_cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5SuOikwiRyPMVoIQDJUgSV</td>
      <td>Gen Hoshino</td>
      <td>Comedy</td>
      <td>Comedy</td>
      <td>73</td>
      <td>230666.0</td>
      <td>False</td>
      <td>0.676</td>
      <td>0.4610</td>
      <td>1</td>
      <td>-6.746</td>
      <td>0</td>
      <td>0.1430</td>
      <td>0.0322</td>
      <td>0.000001</td>
      <td>0.3580</td>
      <td>0.715</td>
      <td>87.917</td>
      <td>4</td>
      <td>acoustic</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4qPNDBW1i3p13qLCt0Ki3A</td>
      <td>Ben Woodward</td>
      <td>Ghost (Acoustic)</td>
      <td>Ghost - Acoustic</td>
      <td>55</td>
      <td>149610.0</td>
      <td>False</td>
      <td>0.420</td>
      <td>0.1660</td>
      <td>1</td>
      <td>-17.235</td>
      <td>1</td>
      <td>0.0763</td>
      <td>0.9240</td>
      <td>0.000006</td>
      <td>0.1010</td>
      <td>0.267</td>
      <td>77.489</td>
      <td>4</td>
      <td>acoustic</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1iJBSr7s7jYXzM8EGcbK5b</td>
      <td>Ingrid Michaelson;ZAYN</td>
      <td>To Begin Again</td>
      <td>To Begin Again</td>
      <td>57</td>
      <td>210826.0</td>
      <td>False</td>
      <td>0.438</td>
      <td>0.3590</td>
      <td>0</td>
      <td>-9.734</td>
      <td>1</td>
      <td>0.0557</td>
      <td>0.2100</td>
      <td>0.000000</td>
      <td>0.1170</td>
      <td>0.120</td>
      <td>76.332</td>
      <td>4</td>
      <td>acoustic</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6lfxq3CG4xtTiEg7opyCyx</td>
      <td>Kina Grannis</td>
      <td>Crazy Rich Asians (Original Motion Picture Sou...</td>
      <td>Can't Help Falling In Love</td>
      <td>71</td>
      <td>201933.0</td>
      <td>False</td>
      <td>0.266</td>
      <td>0.0596</td>
      <td>0</td>
      <td>-18.515</td>
      <td>1</td>
      <td>0.0363</td>
      <td>0.9050</td>
      <td>0.000071</td>
      <td>0.1320</td>
      <td>0.143</td>
      <td>181.740</td>
      <td>3</td>
      <td>acoustic</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5vjLSffimiIP26QG5WcN2K</td>
      <td>Chord Overstreet</td>
      <td>Hold On</td>
      <td>Hold On</td>
      <td>82</td>
      <td>198853.0</td>
      <td>False</td>
      <td>0.618</td>
      <td>0.4430</td>
      <td>2</td>
      <td>-9.681</td>
      <td>1</td>
      <td>0.0526</td>
      <td>0.4690</td>
      <td>0.000000</td>
      <td>0.0829</td>
      <td>0.167</td>
      <td>119.949</td>
      <td>4</td>
      <td>acoustic</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



## 5.4 Analyzing Danceability in Genre Clusters


```python
plt.figure(figsize=(14,6))
order = spotify.groupby('track_genre_cluster')['danceability'].mean().sort_values().index
plot = sns.barplot(data=spotify, y='track_genre_cluster', x='danceability', orient='h', palette='Blues', order=order)
plot.set_title('Average Danceability by Genre Cluster')
plot.bar_label(plot.containers[0])
plt.show()
```


    
![png](readme_files/output_58_0.png)
    


The resulting barplot provides a visual glimpse into the danceability levels associated with each cluster. Notably, several clusters stood out for their impressive danceability scores. `Cluster 18`, boasting an average danceability score of **0.606**, emerged as a vibrant choice, offering genres that encourage dance enthusiasts to hit the floor. Close behind, `Cluster 4`, with an average score of **0.611**, and `Cluster 0`, reaching **0.649**, both promise engaging and rhythm-driven musical experiences. `Cluster 17`, with an average danceability score of **0.664**, and `Cluster 20`, topping the chart at **0.675**, further solidify the notion that genres within these clusters are ideal candidates for our dance-themed playlist. These insights guide us in selecting the most dance-friendly genres, ensuring our playlist sets the perfect mood for an energetic and unforgettable summer party celebration.


```python
# Saving Processed Data
if PROCESSED_DATA_SAVE:
    spotify.to_csv("processed_data/processed_spotify.csv",index=False)
```

# 6. Predictive Modeling for Danceability 🎶📈

With a profound understanding of our music genres, their attributes, and danceability patterns, the next exciting phase of our journey involves predictive modeling. Our goal is to harness the insights gleaned from extensive data exploration and genre clustering to create a robust predictive model for danceability.

## 6.1 Handling Non-related & Bias Attributes

In our commitment to constructing an accurate and impartial AI model for predicting danceability, we have taken a deliberate and mindful step. Recognizing the potential for bias to influence predictive models, particularly when artist-related data is included, we have made a conscious choice to exclude specific data attributes from our model's training dataset. 

These exclusions encompass `track names`, `album information`, and `artist names`. The central objective behind this exclusion is to shield our AI model from potential biases that could arise during training. Bias in AI frequently emerges from the training data itself, and including artist information could inadvertently lead to recommendations that disproportionately favor well-known artists. 

By concentrating solely on the intrinsic characteristics of songs, such as their sound, tempo, rhythm, and other audio features, we aim to create recommendations that are firmly rooted in the qualities of the music itself, ensuring that our model remains fair, impartial, and open to both established and emerging artists who produce similar styles of music. This approach underscores our commitment to curating a dance-themed playlist that honors the essence of the music itself, transcending external influences.

## 6.2 Standardization for Modeling

A crucial aspect of our data preparation process involved standardization. Specifically, we focused on standardizing key numeric attributes essential for our predictive modeling endeavor. These attributes encompassed critical factors such as `loudness`, `duration_ms`, `energy`, `speechiness`, `acousticness`, and `instrumentalness`. To achieve this, we employed the _StandardScaler_, a standardization technique that transforms these attributes to a common scale, ensuring they contribute equally to our modeling process.


```python
def applyStandardScaler(data, attributes):
    for attr in attributes:
        data[attr+'_scaled'] = StandardScaler().fit_transform(data[[attr]])
        
attributes = ['loudness', 'duration_ms', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']  
applyStandardScaler(data=spotify, attributes=attributes)
```

## 6.3 Categorical Data Transformation with One-Hot Encoding

As part of our comprehensive data preprocessing, we employed one-hot encoding to effectively handle categorical data within our dataset. Specifically, we applied this technique to two columns, namely `key` and `track_genre_cluster`.

During our analysis, we conducted a thorough assessment of the `key` feature to gauge its impact on predicting danceability. However, our findings revealed that this particular attribute did not exert a significant influence on accurately predicting danceability scores. Consequently, we made the informed decision to remove the 'key' feature from our dataset. This streamlined our data, allowing us to focus exclusively on attributes that demonstrated more substantial relevance to the prediction of danceability.

In stark contrast, our analysis highlighted the meaningful impact of the `track_genre_cluster` feature on the prediction of danceability. Given its relevance and ability to enhance our predictive model's accuracy, we retained this attribute in our dataset. 


```python
one_hot_genre = pd.get_dummies(spotify['track_genre_cluster'], prefix='g_cluster')
one_hot_genre.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>g_cluster_0</th>
      <th>g_cluster_1</th>
      <th>g_cluster_2</th>
      <th>g_cluster_3</th>
      <th>g_cluster_4</th>
      <th>g_cluster_5</th>
      <th>g_cluster_6</th>
      <th>g_cluster_7</th>
      <th>g_cluster_8</th>
      <th>g_cluster_9</th>
      <th>g_cluster_10</th>
      <th>g_cluster_11</th>
      <th>g_cluster_12</th>
      <th>g_cluster_13</th>
      <th>g_cluster_14</th>
      <th>g_cluster_15</th>
      <th>g_cluster_16</th>
      <th>g_cluster_17</th>
      <th>g_cluster_18</th>
      <th>g_cluster_19</th>
      <th>g_cluster_20</th>
      <th>g_cluster_21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 6.4 Feature Selection for Model Training

In the preparation of our predictive model for danceability, a critical step involved the selection of features that would serve as inputs for the model, while danceability itself was designated as the target label for prediction. The features identified for model training encompass a comprehensive array of attributes:


```python
attr = ['popularity', 'explicit', 'mode'] + ['duration_ms_scaled', 'energy_scaled', 'loudness_scaled',
        'speechiness_scaled','acousticness_scaled', 'instrumentalness_scaled', 'liveness_scaled',
       'valence_scaled', 'tempo_scaled', 'time_signature_scaled']

y = spotify['danceability']

#features
x1 = spotify[attr] 
x1[one_hot_genre.columns] = one_hot_genre

x1.columns
```




    Index(['popularity', 'explicit', 'mode', 'duration_ms_scaled', 'energy_scaled',
           'loudness_scaled', 'speechiness_scaled', 'acousticness_scaled',
           'instrumentalness_scaled', 'liveness_scaled', 'valence_scaled',
           'tempo_scaled', 'time_signature_scaled', 'g_cluster_0', 'g_cluster_1',
           'g_cluster_2', 'g_cluster_3', 'g_cluster_4', 'g_cluster_5',
           'g_cluster_6', 'g_cluster_7', 'g_cluster_8', 'g_cluster_9',
           'g_cluster_10', 'g_cluster_11', 'g_cluster_12', 'g_cluster_13',
           'g_cluster_14', 'g_cluster_15', 'g_cluster_16', 'g_cluster_17',
           'g_cluster_18', 'g_cluster_19', 'g_cluster_20', 'g_cluster_21'],
          dtype='object')



By utilizing this extensive set of features, we are poised to create a model that not only understands the inherent characteristics of the music but also interprets how they collectively influence danceability. This comprehensive approach aligns with our objective of developing a robust and accurate AI model capable of generating precise predictions, ultimately enhancing the quality of the dance-themed playlist for our forthcoming summer party celebration.

## 6.5 Hyperparameter Tuning for Model Optimization

In our quest to fine-tune and optimize our predictive model for danceability, we undertook a meticulous process of hyperparameter tuning. The objective was to identify the ideal combination of hyperparameters that would maximize the model's performance and predictive accuracy.


```python
# Define the desired percentage for cross-validation
percent = 30
# Calculate the number of cross-validation (CV) folds based on the percentage
cv_30percent = int(x1.shape[0]//(x1.shape[0]*percent/100))

if ML_HYPERPARAMETER_TUNING:
    # Define the hyperparameters to tune
    param_grid = {
        'n_estimators': [400],
        'max_depth': [None],
        'min_samples_split': [2,4],
        'min_samples_leaf': [4],
        'n_jobs': [-1]
    }

    # Create the random forest regressor
    rf = RandomForestRegressor()

    # Perform grid search cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv_30percent, verbose=1)
    grid_search.fit(x1, y)

    # Get the best hyperparameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Print the best hyperparameters and the best score
    best_params, best_score
```

({'max_depth': None,
  'min_samples_leaf': 4,
  'min_samples_split': 2,
  'n_estimators': 400,
  'n_jobs': -1},
 0.6237475896626198)

After rigorous experimentation and evaluation, our hyperparameter tuning efforts culminated in a set of optimized hyperparameters:

| Parameters             | Values |
|------------------------|--------|
| **max_depth**          | None   |
| **min_samples_leaf**   | 4      |
| **min_samples_split**  | 2      |
| **n_estimators**       | 400    |
| **n_jobs**             | -1     |

This configuration represents the culmination of our iterative tuning process, meticulously chosen to strike a balance between model complexity and predictive precision. The resulting model, with these optimized hyperparameters, exhibited an impressive performance metric _**Mean Absolute Error (MAE)**_ of **0.6237**.

These results signify a notable achievement in the pursuit of an AI model that can accurately predict danceability, providing invaluable insights for our playlist curation.

## 6.6 Model Training and Evaluation

After meticulous experimentation and evaluation, a clear and compelling pattern emerged that the Random Forest Regressor consistently outperformed the Decision Tree Regressor in terms of predictive accuracy and robustness The Random Forest Regressor was selected as the exclusive model for our final analysis and predictive tasks.

Following the process of hyperparameter tuning, we proceeded to train our predictive model for danceability on the optimal parameters. This rigorous training process resulted in a model that demonstrates remarkable predictive capabilities, as evidenced by the following evaluation metrics:


```python
if ML_TRAINING:

    best_params = {
        'max_depth': None,             # Maximum depth of each tree in the forest
        'min_samples_leaf': 4,         # Minimum no of samples required to be at a leaf node
        'min_samples_split': 2,        # Minimum no of samples required to split an internal node
        'n_estimators': 400,           # Number of trees in the random forest
        'random_state': 128,           # Seed for random number generation 
        'n_jobs': -1                   # Number of CPU cores to use for parallel computation
    }

    rf1 = RandomForestRegressor(    
        max_depth=best_params['max_depth'],
        n_estimators=best_params['n_estimators'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=best_params['random_state'],
        n_jobs=-1
    )

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.2, random_state=42)

    # Fit the decision tree regressor on the training data
    rf1.fit(x_train, y_train)

    # Predict on the testing data
    y_pred = rf1.predict(x_test)

    # Calculate the mean squared error and root mean squared error
    mse = skm.mean_squared_error(y_test, y_pred)
    rmse = skm.mean_squared_error(y_test, y_pred, squared=False)

    # Calculate the mean absolute error
    mae = skm.mean_absolute_error(y_test, y_pred)

    # Calculate the max error
    mxe = skm.max_error(y_test, y_pred)

    r2 = skm.r2_score(y_test, y_pred)

    rmse, mse, mae, mxe, r2
```

| Error & Score     | Value                  |
|-------------------|------------------------|
| Root Mean Square  | 0.09457460304385866    |
| Mean Square       | 0.00894435554090344    |
| Mean Absolute     | 0.07140843682128345    |
| Max  Error        | 0.4800830022823025     |
| R2   Score        | 0.7035212083361053     |

These metrics collectively underscore the model's ability to make accurate predictions of danceability, reflecting its impressive performance. The RMSE, MSE, and MAE values demonstrate the model's precision in estimating danceability scores, with the RMSE highlighting the model's ability to maintain a low level of prediction error. The Max Error provides an upper bound on prediction errors, revealing that the model rarely deviates significantly from the actual danceability scores. Finally, the R2 Score showcases the model's capacity to explain a substantial portion of the variance in danceability, indicating a strong correlation between predicted and actual values.

## 6.7 Feature Importance Analysis

In our relentless pursuit of model transparency and interpretability, we conducted a comprehensive analysis of feature importance following the training of our Random Forest Regressor model. This analysis aimed to discern which attributes exerted the most significant influence on the prediction of danceability, shedding light on the factors that play a pivotal role in shaping the musical essence of our curated playlist.


```python
if FEATURE_IMPORANCE:
    # Get the feature importances from the decision tree model
    feature_importances = rf1.feature_importances_

    # Get the names of the features
    feature_names = x1.columns

    # Sort the feature importances and feature names in descending order
    sorted_indices = feature_importances.argsort()[::-1]
    sorted_feature_importances = feature_importances[sorted_indices]
    sorted_feature_names = feature_names[sorted_indices]

    # Create a horizontal bar graph of the feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_feature_importances)), sorted_feature_importances, align='center')
    plt.yticks(range(len(sorted_feature_importances)), sorted_feature_names)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.show()
```

![Feature Importances](screenshots/feature_importances.png)

These importance scores serve as a valuable guide, revealing the extent to which each attribute contributes to our model's predictions. Notably, `valence_scaled` emerges as the most influential attribute, underscoring its role in shaping the danceability of a track. `Tempo_scaled` also plays a substantial role, emphasizing its impact on the overall danceability prediction.

As we delved into the feature importance analysis, it became evident that the genre clusters had a significant influence on danceability predictions. While individual genres bring their unique attributes to the table, the clusters offered a more holistic perspective by encapsulating shared characteristics and patterns among multiple genres within each cluster.

For instance, `genre_cluster_5`, `genre_cluster_17`, and `genre_cluster_8`, played a pivotal role in shaping our understanding of feature importance. This holistic approach to feature importance, taking into account the influence of genre clusters, empowered us to make more informed decisions in curating our dance-themed playlist.


```python
#Import Processed Data
spotify = pd.read_csv("processed_data/processed_spotify.csv")
spotify.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track_id</th>
      <th>artists</th>
      <th>album_name</th>
      <th>track_name</th>
      <th>popularity</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>...</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>track_genre</th>
      <th>track_genre_cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5SuOikwiRyPMVoIQDJUgSV</td>
      <td>Gen Hoshino</td>
      <td>Comedy</td>
      <td>Comedy</td>
      <td>73</td>
      <td>230666.0</td>
      <td>False</td>
      <td>0.676</td>
      <td>0.4610</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0.1430</td>
      <td>0.0322</td>
      <td>0.000001</td>
      <td>0.3580</td>
      <td>0.715</td>
      <td>87.917</td>
      <td>4</td>
      <td>acoustic</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4qPNDBW1i3p13qLCt0Ki3A</td>
      <td>Ben Woodward</td>
      <td>Ghost (Acoustic)</td>
      <td>Ghost - Acoustic</td>
      <td>55</td>
      <td>149610.0</td>
      <td>False</td>
      <td>0.420</td>
      <td>0.1660</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0.0763</td>
      <td>0.9240</td>
      <td>0.000006</td>
      <td>0.1010</td>
      <td>0.267</td>
      <td>77.489</td>
      <td>4</td>
      <td>acoustic</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1iJBSr7s7jYXzM8EGcbK5b</td>
      <td>Ingrid Michaelson;ZAYN</td>
      <td>To Begin Again</td>
      <td>To Begin Again</td>
      <td>57</td>
      <td>210826.0</td>
      <td>False</td>
      <td>0.438</td>
      <td>0.3590</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0.0557</td>
      <td>0.2100</td>
      <td>0.000000</td>
      <td>0.1170</td>
      <td>0.120</td>
      <td>76.332</td>
      <td>4</td>
      <td>acoustic</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6lfxq3CG4xtTiEg7opyCyx</td>
      <td>Kina Grannis</td>
      <td>Crazy Rich Asians (Original Motion Picture Sou...</td>
      <td>Can't Help Falling In Love</td>
      <td>71</td>
      <td>201933.0</td>
      <td>False</td>
      <td>0.266</td>
      <td>0.0596</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0.0363</td>
      <td>0.9050</td>
      <td>0.000071</td>
      <td>0.1320</td>
      <td>0.143</td>
      <td>181.740</td>
      <td>3</td>
      <td>acoustic</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5vjLSffimiIP26QG5WcN2K</td>
      <td>Chord Overstreet</td>
      <td>Hold On</td>
      <td>Hold On</td>
      <td>82</td>
      <td>198853.0</td>
      <td>False</td>
      <td>0.618</td>
      <td>0.4430</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>0.0526</td>
      <td>0.4690</td>
      <td>0.000000</td>
      <td>0.0829</td>
      <td>0.167</td>
      <td>119.949</td>
      <td>4</td>
      <td>acoustic</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



# 7. Curating the Playlist 📀

## 7.1 Clusters of Significance and Rejection

Our journey to curate the perfect dance-themed playlist was not solely based on feature importance; it also took into account the essence and suitability of musical genres. To achieve this balance, we meticulously examined the clusters that emerged from our analysis, carefully selecting those that aligned with our goal of creating an electrifying atmosphere for our summer party celebration.

**_Selected Clusters_**
- Clusters: **[2, 4, 6, 9, 12, 13, 17]**
- These clusters have demonstrated significant feature importance in predicting danceability, as determined by the Random Forest Regressor model.

**_Rejected Clusters_**

While certain clusters exhibited notable feature importance, they were excluded from our final selection due to genre considerations:

- _Rejected Cluster 8_
    - Genres: _Children_, _Kids_, _Sleep_, _Study_
    - Despite their high feature importance, these clusters were deemed unsuitable for a party atmosphere and were therefore removed.


- _Rejected Cluster 3_
    - Genres: _Anime_, _Indie_, _Opera_, _Piano_
    - Although they had a standard impact on danceability, these clusters were also removed due to their unconventional genres.


- _Rejected Cluster 1_
    - Genres: _Brazil_, _British_, _Classical_, _Folk_, _French_, _German_, _Indian, Iranian, Latin, Latino, Malay, Spanish, Swedish, Turkish, World-Music_
    - This cluster exhibited a low effect on danceability and lacked genres suitable for a party, resulting in its removal.


- _Rejected Cluster 5_
    - Genres: _Black-metal, Death-metal, Grindcore, Metal_
    - This cluster exhibited a significant influence on predicting danceability. However, it did not demonstrate a high average danceability, as previously identified in the "Average Danceability by Genre Cluster" analysis.


```python
# Low Feature Importance
low_importance_cluster = [15, 21, 16, 0, 18, 20, 7, 11, 14, 19, 10]
# Unsuitable genre cluster
not_suitable_cluster = [1, 3, 8, 5]
drop_cluster = set(low_importance_cluster + not_suitable_cluster)

clusters_selection = [ i for i in range(22) if i not in drop_cluster]
clusters_selection
```




    [2, 4, 6, 9, 12, 13, 17]




```python
spotify = spotify.drop_duplicates(subset=['track_id'])
spotify.reset_index()

# Filtering data based on selected clusters
filtered_by_genre = spotify[spotify['track_genre_cluster'].isin(clusters_selection)]
filtered_by_genre
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track_id</th>
      <th>artists</th>
      <th>album_name</th>
      <th>track_name</th>
      <th>popularity</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>...</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>track_genre</th>
      <th>track_genre_cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5SuOikwiRyPMVoIQDJUgSV</td>
      <td>Gen Hoshino</td>
      <td>Comedy</td>
      <td>Comedy</td>
      <td>73</td>
      <td>230666.0</td>
      <td>False</td>
      <td>0.676</td>
      <td>0.4610</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0.1430</td>
      <td>0.0322</td>
      <td>0.000001</td>
      <td>0.3580</td>
      <td>0.7150</td>
      <td>87.917</td>
      <td>4</td>
      <td>acoustic</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4qPNDBW1i3p13qLCt0Ki3A</td>
      <td>Ben Woodward</td>
      <td>Ghost (Acoustic)</td>
      <td>Ghost - Acoustic</td>
      <td>55</td>
      <td>149610.0</td>
      <td>False</td>
      <td>0.420</td>
      <td>0.1660</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0.0763</td>
      <td>0.9240</td>
      <td>0.000006</td>
      <td>0.1010</td>
      <td>0.2670</td>
      <td>77.489</td>
      <td>4</td>
      <td>acoustic</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1iJBSr7s7jYXzM8EGcbK5b</td>
      <td>Ingrid Michaelson;ZAYN</td>
      <td>To Begin Again</td>
      <td>To Begin Again</td>
      <td>57</td>
      <td>210826.0</td>
      <td>False</td>
      <td>0.438</td>
      <td>0.3590</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0.0557</td>
      <td>0.2100</td>
      <td>0.000000</td>
      <td>0.1170</td>
      <td>0.1200</td>
      <td>76.332</td>
      <td>4</td>
      <td>acoustic</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6lfxq3CG4xtTiEg7opyCyx</td>
      <td>Kina Grannis</td>
      <td>Crazy Rich Asians (Original Motion Picture Sou...</td>
      <td>Can't Help Falling In Love</td>
      <td>71</td>
      <td>201933.0</td>
      <td>False</td>
      <td>0.266</td>
      <td>0.0596</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0.0363</td>
      <td>0.9050</td>
      <td>0.000071</td>
      <td>0.1320</td>
      <td>0.1430</td>
      <td>181.740</td>
      <td>3</td>
      <td>acoustic</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5vjLSffimiIP26QG5WcN2K</td>
      <td>Chord Overstreet</td>
      <td>Hold On</td>
      <td>Hold On</td>
      <td>82</td>
      <td>198853.0</td>
      <td>False</td>
      <td>0.618</td>
      <td>0.4430</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>0.0526</td>
      <td>0.4690</td>
      <td>0.000000</td>
      <td>0.0829</td>
      <td>0.1670</td>
      <td>119.949</td>
      <td>4</td>
      <td>acoustic</td>
      <td>9</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>97388</th>
      <td>0rGllPwXKYxS07xZ0osn22</td>
      <td>Leftfield;Quiet Village</td>
      <td>Leftism 22</td>
      <td>Melt - Quiet Village Mix</td>
      <td>17</td>
      <td>469906.0</td>
      <td>False</td>
      <td>0.333</td>
      <td>0.3780</td>
      <td>6</td>
      <td>...</td>
      <td>1</td>
      <td>0.0541</td>
      <td>0.5140</td>
      <td>0.910000</td>
      <td>0.0828</td>
      <td>0.0689</td>
      <td>169.993</td>
      <td>4</td>
      <td>trip-hop</td>
      <td>2</td>
    </tr>
    <tr>
      <th>97389</th>
      <td>3TpcGANz2N705Bq4zc982H</td>
      <td>Roots Manuva</td>
      <td>Run Come Save Me</td>
      <td>Trim Body</td>
      <td>17</td>
      <td>215720.0</td>
      <td>True</td>
      <td>0.798</td>
      <td>0.6840</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>0.1680</td>
      <td>0.0307</td>
      <td>0.000159</td>
      <td>0.4790</td>
      <td>0.5570</td>
      <td>94.936</td>
      <td>4</td>
      <td>trip-hop</td>
      <td>2</td>
    </tr>
    <tr>
      <th>97390</th>
      <td>57qWtXga1hMwSfkhLDJCKQ</td>
      <td>Everything But The Girl</td>
      <td>Worldwide (Deluxe Edition)</td>
      <td>My Head Is My Only House Unless It Rains - 201...</td>
      <td>18</td>
      <td>178586.0</td>
      <td>False</td>
      <td>0.590</td>
      <td>0.1200</td>
      <td>5</td>
      <td>...</td>
      <td>1</td>
      <td>0.0374</td>
      <td>0.8990</td>
      <td>0.000023</td>
      <td>0.1090</td>
      <td>0.3690</td>
      <td>107.099</td>
      <td>4</td>
      <td>trip-hop</td>
      <td>2</td>
    </tr>
    <tr>
      <th>97391</th>
      <td>0xcDUsknTawAv5VZKQ62aZ</td>
      <td>Wax Tailor</td>
      <td>Dusty Rainbow from the Dark</td>
      <td>From The Dark</td>
      <td>17</td>
      <td>202960.0</td>
      <td>False</td>
      <td>0.460</td>
      <td>0.2920</td>
      <td>11</td>
      <td>...</td>
      <td>1</td>
      <td>0.0281</td>
      <td>0.3730</td>
      <td>0.229000</td>
      <td>0.1700</td>
      <td>0.3570</td>
      <td>158.906</td>
      <td>3</td>
      <td>trip-hop</td>
      <td>2</td>
    </tr>
    <tr>
      <th>97392</th>
      <td>2Mt8qdFRaaIgzsf2bxwH4Y</td>
      <td>Lovage;Nathaniel Merriweather</td>
      <td>Music to Make Love to Your Old Lady By (Instru...</td>
      <td>Strangers on a Train - Instrumental</td>
      <td>16</td>
      <td>278000.0</td>
      <td>False</td>
      <td>0.843</td>
      <td>0.4860</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0.0445</td>
      <td>0.0753</td>
      <td>0.773000</td>
      <td>0.0961</td>
      <td>0.9240</td>
      <td>86.316</td>
      <td>4</td>
      <td>trip-hop</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>44910 rows × 21 columns</p>
</div>



## 7.2 Track Selection Based on Quartiles

With the clusters of significant importance in hand and less fitting clusters duly rejected, our quest to curate the ultimate dance-themed playlist entered a more refined phase. To ensure the highest quality and consistency in track selection, we turned our attention to the quartiles within the clusters with the best danceability averages. This strategic move allowed us to pinpoint the tracks that not only contributed to our desired danceable vibe but also maintained a high standard of musical excellence.

We identified a set of attributes - `valence`, `tempo`, `acousticness`, `loudness`, `speechiness`, `energy`, `duration_ms`, `liveness`, and `popularity` - and dissected their quartiles. Within these quartiles, we pinpointed the specific records that boasted not only danceability but also a remarkable blend of musical qualities. By doing so, we are ensuring that each track in our playlist possesses the perfect balance of these critical attributes. 


```python
def selectDataOnFeatureQuartiles(data, feature, against_feature, select_q=2, q=4):
    
    # Quartiles Label list
    labels = ['Q'+str(i+1) for i in range(q)]
   
    filtered_data = data.copy(deep=True)
    feature_quartile = feature+'_quartile'
    
    # Calculate quartiles for 'valence'
    filtered_data[feature_quartile] = pd.qcut(data[feature], q=4, labels=labels)

    # Calculate the average danceability for each quartile
    average_danceability_by_quartile = filtered_data.groupby(feature_quartile)[against_feature].mean()

    # Find the two quartiles with the highest average danceability
    selected_quartiles = average_danceability_by_quartile.nlargest(select_q).index

    # Filter the data based on the selected quartiles
    filtered_data = filtered_data[filtered_data[feature_quartile].isin(selected_quartiles)]

    # Now 'filtered_data' contains the rows with the selected number of quartiles that have the highest average danceability
    return filtered_data.index
```


```python
valence = selectDataOnFeatureQuartiles(data=filtered_by_genre, feature='valence', against_feature='danceability')

tempo = selectDataOnFeatureQuartiles(data=filtered_by_genre, feature='tempo', against_feature='danceability')

acousticness = selectDataOnFeatureQuartiles(data=filtered_by_genre, feature='acousticness', against_feature='danceability')

energy = selectDataOnFeatureQuartiles(data=filtered_by_genre, feature='energy', against_feature='danceability')

speechiness = selectDataOnFeatureQuartiles(data=filtered_by_genre, feature='speechiness', against_feature='danceability')

liveness = selectDataOnFeatureQuartiles(data=filtered_by_genre, feature='liveness', against_feature='danceability', select_q=1)

loudness = selectDataOnFeatureQuartiles(data=filtered_by_genre, feature='loudness', against_feature='danceability')

# Filter the playlist to include only tracks with feature values greater than the median
median = filtered_by_genre['popularity'].quantile(0.5)
popularity = filtered_by_genre[filtered_by_genre['popularity'] > median].index
```


```python
# Create a list of all the index sets
index_sets = [valence, tempo, acousticness, energy, speechiness, liveness, loudness, popularity]

# Use reduce to find the intersection of all sets
intersection_indexes = reduce(lambda x, y: x.intersection(y), index_sets)

# Now 'intersection_indexes' contains the common indexes from all sets
intersection_indexes
```




    Index([  459,   952,   958,   962,  1261,  1312,  1835,  7342,  7520,  7558,
           ...
           84108, 84147, 87006, 87022, 87311, 94814, 95832, 96829, 97016, 97056],
          dtype='int64', length=163)




```python
playlist = spotify[spotify.index.isin(intersection_indexes)]
playlist
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track_id</th>
      <th>artists</th>
      <th>album_name</th>
      <th>track_name</th>
      <th>popularity</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>...</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>track_genre</th>
      <th>track_genre_cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>459</th>
      <td>3BztDPhRhfwdwUndTlg45h</td>
      <td>Gabrielle Aplin</td>
      <td>Dear Happy</td>
      <td>So Far so Good</td>
      <td>37</td>
      <td>198040.0</td>
      <td>False</td>
      <td>0.751</td>
      <td>0.565</td>
      <td>5</td>
      <td>...</td>
      <td>1</td>
      <td>0.0513</td>
      <td>0.0777</td>
      <td>0.012100</td>
      <td>0.0855</td>
      <td>0.679</td>
      <td>119.996</td>
      <td>4</td>
      <td>acoustic</td>
      <td>9</td>
    </tr>
    <tr>
      <th>952</th>
      <td>1YuVO6AipAn5U1VoG1RBGN</td>
      <td>Criolo</td>
      <td>Ainda Há Tempo</td>
      <td>Demorô</td>
      <td>37</td>
      <td>138480.0</td>
      <td>False</td>
      <td>0.755</td>
      <td>0.709</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0.2890</td>
      <td>0.0528</td>
      <td>0.000000</td>
      <td>0.0825</td>
      <td>0.657</td>
      <td>140.989</td>
      <td>4</td>
      <td>afrobeat</td>
      <td>6</td>
    </tr>
    <tr>
      <th>958</th>
      <td>3AJ0MFDt4OSknQ5D9Jlo4l</td>
      <td>Criolo</td>
      <td>Nó na Orelha</td>
      <td>Samba Sambei</td>
      <td>37</td>
      <td>222560.0</td>
      <td>False</td>
      <td>0.870</td>
      <td>0.737</td>
      <td>11</td>
      <td>...</td>
      <td>1</td>
      <td>0.1260</td>
      <td>0.2680</td>
      <td>0.003690</td>
      <td>0.0504</td>
      <td>0.952</td>
      <td>136.070</td>
      <td>4</td>
      <td>afrobeat</td>
      <td>6</td>
    </tr>
    <tr>
      <th>962</th>
      <td>31RfYWiC3WcExsZtqDMvdS</td>
      <td>Criolo</td>
      <td>Sobre Viver</td>
      <td>Aprendendo a Sobreviver</td>
      <td>41</td>
      <td>212826.0</td>
      <td>False</td>
      <td>0.817</td>
      <td>0.619</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0.0542</td>
      <td>0.2630</td>
      <td>0.003170</td>
      <td>0.0889</td>
      <td>0.614</td>
      <td>103.041</td>
      <td>4</td>
      <td>afrobeat</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1261</th>
      <td>3VDMcm2WqOQdNpzLqlbp9B</td>
      <td>Nero X</td>
      <td>Yawa Dey</td>
      <td>Yawa Dey</td>
      <td>38</td>
      <td>260220.0</td>
      <td>False</td>
      <td>0.858</td>
      <td>0.634</td>
      <td>9</td>
      <td>...</td>
      <td>1</td>
      <td>0.0498</td>
      <td>0.4060</td>
      <td>0.000000</td>
      <td>0.0697</td>
      <td>0.824</td>
      <td>105.018</td>
      <td>4</td>
      <td>afrobeat</td>
      <td>6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>94814</th>
      <td>12Dov40hSJdrgS5uHGKNP3</td>
      <td>Blank &amp; Jones;Boney M.</td>
      <td>Relax Edition 9</td>
      <td>Sunny - Summer Vibe Mix</td>
      <td>53</td>
      <td>259934.0</td>
      <td>False</td>
      <td>0.824</td>
      <td>0.693</td>
      <td>8</td>
      <td>...</td>
      <td>1</td>
      <td>0.0631</td>
      <td>0.0382</td>
      <td>0.291000</td>
      <td>0.0950</td>
      <td>0.521</td>
      <td>113.995</td>
      <td>4</td>
      <td>techno</td>
      <td>17</td>
    </tr>
    <tr>
      <th>95832</th>
      <td>1form6XxaJCcp9CoU3Qxos</td>
      <td>Hilight Tribe</td>
      <td>Love medicine &amp; natural trance</td>
      <td>Free tibet</td>
      <td>42</td>
      <td>359533.0</td>
      <td>False</td>
      <td>0.487</td>
      <td>0.861</td>
      <td>11</td>
      <td>...</td>
      <td>1</td>
      <td>0.0516</td>
      <td>0.4750</td>
      <td>0.753000</td>
      <td>0.0720</td>
      <td>0.776</td>
      <td>137.927</td>
      <td>4</td>
      <td>trance</td>
      <td>2</td>
    </tr>
    <tr>
      <th>96829</th>
      <td>3ceihRUljV1eSM1plIjsB6</td>
      <td>Little Dragon</td>
      <td>Lover Chanting (Jayda G Remix)</td>
      <td>Lover Chanting - Edit</td>
      <td>54</td>
      <td>205464.0</td>
      <td>True</td>
      <td>0.865</td>
      <td>0.762</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0.0762</td>
      <td>0.0166</td>
      <td>0.795000</td>
      <td>0.0233</td>
      <td>0.896</td>
      <td>118.005</td>
      <td>4</td>
      <td>trip-hop</td>
      <td>2</td>
    </tr>
    <tr>
      <th>97016</th>
      <td>2QtYCTIYO6WrHcFmcPtVYX</td>
      <td>Coldcut;Roots Manuva</td>
      <td>Sound Mirrors</td>
      <td>True Skool</td>
      <td>39</td>
      <td>214066.0</td>
      <td>False</td>
      <td>0.673</td>
      <td>0.838</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>0.1710</td>
      <td>0.0133</td>
      <td>0.000218</td>
      <td>0.0534</td>
      <td>0.695</td>
      <td>104.953</td>
      <td>4</td>
      <td>trip-hop</td>
      <td>2</td>
    </tr>
    <tr>
      <th>97056</th>
      <td>6rU0dLXyWbXqti69nrQP4H</td>
      <td>Gui Boratto;Massive Attack</td>
      <td>Heligoland</td>
      <td>Paradise Circus</td>
      <td>46</td>
      <td>487574.0</td>
      <td>False</td>
      <td>0.836</td>
      <td>0.811</td>
      <td>8</td>
      <td>...</td>
      <td>1</td>
      <td>0.0515</td>
      <td>0.1620</td>
      <td>0.814000</td>
      <td>0.0818</td>
      <td>0.534</td>
      <td>126.017</td>
      <td>4</td>
      <td>trip-hop</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>163 rows × 21 columns</p>
</div>



As we near the final stage of playlist curation, we've implemented a sophisticated algorithm known as the _Proportional Distribution Algorithm_. This ingenious approach is tailored to allocate a specific number of tracks to various music genres within our playlist, ensuring that each genre is represented in proportion to its prevalence in our dataset.

### Proportional Distribution Algorithm
> 
> This algorithm is designed to allocate a specific number of tracks to various music genres in a playlist, ensuring that each genre is represented in proportion to its prevalence in the dataset. The algorithm is especially useful when the desired total number of tracks in the playlist exceeds the number of unique tracks genres available.
> 
> **_Proportional Distribution Example_**
> 
> Imagine you have a dataset of items with associated quantities, and you want to allocate a specific number of items proportionally to each item category. Here's a step-by-step example:
> 
> **Data:**
> ```js
> names:      ['a', 'b', 'c', 'd', 'e'],
> quantities: [30, 20, 70, 10, 10]
> allocation: [ ]
> ```
> 
> **Objective:** Allocate 10 items proportionally to categories 'a' through 'e' based on their quantities.
> 
> **Step 1 - Calculate Ratio:**
> - Calculate the total quantity in the dataset: 30 + 20 + 70 + 10 + 10 = 140.
> - Calculate the allocation ratio: 10 / 140 = 1/14.
> 
> **Step 2 - Initial Allocation:**
> - Multiply each category's quantity by the allocation ratio and round to the nearest integer:
>    ```js
>    - 'a': 30 * (1/14) ≈ 2.14 → Rounded up to 3.
>    - 'b': 20 * (1/14) ≈ 1.43 → Rounded up to 2.
>    - 'c': 70 * (1/14) ≈ 5.00 → Rounded up to 5.
>    - 'd': 10 * (1/14) ≈ 0.71 → Rounded up to 1.
>    - 'e': 10 * (1/14) ≈ 0.71 → Rounded up to 1.
>    ```
>
> 
> **Step 3 - Updated Data:**
> - Include the minimum required quantity category, which is 'e' with 1 item, and remove it from further allocation.
> ```js
> names:      ['a', 'b', 'c', 'd'],
> quantities: [30, 20, 70, 10]
> ```
> ```js
> allocation: [ ('e',1) ]
> ```
> 
> **Step 1 - Calculate Ratio:**
> 1. Recalculate the allocation ratio using the updated total quantity: 30 + 20 + 70 + 10 = 130.
> 2. New allocation ratio: 9 / 130 ≈ 0.07.
> 
> **Step 2 - Allocation:**
> 1. Multiply quantities by the new allocation ratio and round to the nearest integer:
>    ```js
>    - 'a': 30 * 0.07 ≈ 2.1 → Rounded up to 3.
>    - 'b': 20 * 0.07 ≈ 1.4 → Rounded up to 3.
>    - 'c': 70 * 0.07 ≈ 4.9 → Rounded up to 5.
>    - 'd': 10 * 0.07 ≈ 0.7 → Rounded up to 1.
>    ```
> 2. Include the minimum required quantity category, which is 'd' with 1 item, and remove it from further allocation.
> 
> **Step 3 - Updated Data:**
> ```js
> names:      ['a', 'b', 'c']
> quantities: [30, 20, 70]
> ```
> ```js
> allocation: [ ('d',1), ('e',1)  ]
> ```
> 
> Repeat Steps 1 to 3 until all categories have been allocated the desired quantity. This process ensures that items are distributed proportionally while maintaining the minimum required quantity for each category.
> 
> Final allocation would look like this
> ```js
> allocation: [ ('a',2), ('b',2), ('c',4), ('d',1), ('e',1)  ]
> ```


```python
# This algorithm only works when playlist size is greater than unique number of tracks
def proportionalDistribution(playlist_size, playlist, attribute):
    # Set the desired total number of tracks in the playlist
    total = playlist_size
    no_unique_tracks = len(playlist[attribute].unique())

    if total > no_unique_tracks:
        # Create a Series with genre counts
        quantity = pd.Series(data=playlist[attribute].value_counts())
        data = []

        # Continue allocating tracks until the desired total is reached
        while total > 0:
            # Calculate the allocation ratio based on the remaining tracks to allocate
            ratio = total / quantity.sum()

            # Calculate the quantity of tracks to allocate to each genre
            quantity_include = (quantity * ratio).apply(math.ceil)

            # Find the genre with the minimum allocation requirement
            genre_name, genre_counts = quantity_include.idxmin(), quantity_include.min()

            # Append the genre name and allocated quantity to the data list
            data.append([genre_name, genre_counts])
            total -= genre_counts

            # Remove the selected genre from the quantity Series
            quantity = quantity.drop(genre_name)

        allocation = pd.DataFrame(data=data, columns=[attribute,'counts'])
        return allocation
    
    return False
```


```python
playlist['track_genre'].value_counts().to_frame()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>track_genre</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>dancehall</th>
      <td>16</td>
    </tr>
    <tr>
      <th>hip-hop</th>
      <td>15</td>
    </tr>
    <tr>
      <th>disco</th>
      <td>11</td>
    </tr>
    <tr>
      <th>funk</th>
      <td>10</td>
    </tr>
    <tr>
      <th>groove</th>
      <td>9</td>
    </tr>
    <tr>
      <th>electronic</th>
      <td>7</td>
    </tr>
    <tr>
      <th>deep-house</th>
      <td>7</td>
    </tr>
    <tr>
      <th>afrobeat</th>
      <td>6</td>
    </tr>
    <tr>
      <th>r-n-b</th>
      <td>6</td>
    </tr>
    <tr>
      <th>minimal-techno</th>
      <td>5</td>
    </tr>
    <tr>
      <th>reggae</th>
      <td>5</td>
    </tr>
    <tr>
      <th>dance</th>
      <td>5</td>
    </tr>
    <tr>
      <th>blues</th>
      <td>5</td>
    </tr>
    <tr>
      <th>salsa</th>
      <td>4</td>
    </tr>
    <tr>
      <th>club</th>
      <td>4</td>
    </tr>
    <tr>
      <th>samba</th>
      <td>4</td>
    </tr>
    <tr>
      <th>mpb</th>
      <td>4</td>
    </tr>
    <tr>
      <th>edm</th>
      <td>3</td>
    </tr>
    <tr>
      <th>electro</th>
      <td>3</td>
    </tr>
    <tr>
      <th>forro</th>
      <td>3</td>
    </tr>
    <tr>
      <th>garage</th>
      <td>3</td>
    </tr>
    <tr>
      <th>ska</th>
      <td>3</td>
    </tr>
    <tr>
      <th>j-rock</th>
      <td>3</td>
    </tr>
    <tr>
      <th>pagode</th>
      <td>3</td>
    </tr>
    <tr>
      <th>party</th>
      <td>3</td>
    </tr>
    <tr>
      <th>trip-hop</th>
      <td>3</td>
    </tr>
    <tr>
      <th>hard-rock</th>
      <td>2</td>
    </tr>
    <tr>
      <th>progressive-house</th>
      <td>2</td>
    </tr>
    <tr>
      <th>breakbeat</th>
      <td>2</td>
    </tr>
    <tr>
      <th>psych-rock</th>
      <td>1</td>
    </tr>
    <tr>
      <th>rock-n-roll</th>
      <td>1</td>
    </tr>
    <tr>
      <th>detroit-techno</th>
      <td>1</td>
    </tr>
    <tr>
      <th>chill</th>
      <td>1</td>
    </tr>
    <tr>
      <th>techno</th>
      <td>1</td>
    </tr>
    <tr>
      <th>trance</th>
      <td>1</td>
    </tr>
    <tr>
      <th>acoustic</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# After applying proportional Distribution

genre_selection = proportionalDistribution(50, playlist, 'track_genre')
print("Song Quanitity Selected on basis of each genre type selected from filtered playlist",genre_selection.counts.sum())
genre_selection
```

    Song Quanitity Selected on basis of each genre type selected from filtered playlist 50
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track_genre</th>
      <th>counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>edm</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>electro</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>forro</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>garage</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ska</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>j-rock</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pagode</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>party</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>trip-hop</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>hard-rock</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>progressive-house</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>breakbeat</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>psych-rock</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>rock-n-roll</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>detroit-techno</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>chill</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>techno</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>trance</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>acoustic</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>electronic</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>salsa</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>club</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>samba</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>mpb</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>deep-house</td>
      <td>2</td>
    </tr>
    <tr>
      <th>25</th>
      <td>afrobeat</td>
      <td>2</td>
    </tr>
    <tr>
      <th>26</th>
      <td>r-n-b</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>minimal-techno</td>
      <td>2</td>
    </tr>
    <tr>
      <th>28</th>
      <td>reggae</td>
      <td>2</td>
    </tr>
    <tr>
      <th>29</th>
      <td>groove</td>
      <td>2</td>
    </tr>
    <tr>
      <th>30</th>
      <td>dance</td>
      <td>2</td>
    </tr>
    <tr>
      <th>31</th>
      <td>blues</td>
      <td>1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>funk</td>
      <td>2</td>
    </tr>
    <tr>
      <th>33</th>
      <td>hip-hop</td>
      <td>3</td>
    </tr>
    <tr>
      <th>34</th>
      <td>dancehall</td>
      <td>3</td>
    </tr>
    <tr>
      <th>35</th>
      <td>disco</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



The outcome of this meticulous process is a playlist that not only mirrors the genre distribution in our dataset but also ensures that each genre receives a proportional representation. This algorithm ensured that each genre type was proportionally represented within the 187 records. Through careful allocation, we achieved the desired genre counts that summed up to 50.

# 8. The Dancefloor Anthems: Your Ultimate Summer Party Playlist 💃🏻

After an exhaustive journey through data analysis, model training, and meticulous track selection, we've arrived at the grand finale – the creation of the ultimate dance-themed playlist. Our goal was to craft a collection of tracks that would set the dancefloor ablaze and ensure an unforgettable night of celebration.


```python
# Sort the original playlist DataFrame by 'danceability' in descending order
sorted_playlist = playlist.sort_values(by='danceability', ascending=False)

# Create an empty DataFrame to store the selected tracks
playlist_50 = pd.DataFrame(data=[], columns=spotify.columns)

# Iterate through each row in the genre_selection DataFrame
for _, row in genre_selection.iterrows():
    
    genre = row['track_genre'] # Extract the genre name
    count = row['counts']      # Extract the desired count for this genre
    
    # Select the top 'count' tracks of the specified genre and concatenate them to playlist_50
    genre_tracks = sorted_playlist[sorted_playlist['track_genre'] == genre].head(count)
    playlist_50 = pd.concat([playlist_50, genre_tracks])

playlist_50
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track_id</th>
      <th>artists</th>
      <th>album_name</th>
      <th>track_name</th>
      <th>popularity</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>...</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>track_genre</th>
      <th>track_genre_cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26728</th>
      <td>4BiiOzZCrXEzHRLYcYFiD5</td>
      <td>The Chainsmokers;Winona Oak</td>
      <td>Sick Boy</td>
      <td>Hope</td>
      <td>65</td>
      <td>180120.0</td>
      <td>False</td>
      <td>0.773</td>
      <td>0.699</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0.0958</td>
      <td>0.4880</td>
      <td>0.000004</td>
      <td>0.0814</td>
      <td>0.513</td>
      <td>104.941</td>
      <td>4</td>
      <td>edm</td>
      <td>9</td>
    </tr>
    <tr>
      <th>27755</th>
      <td>4lupTQUPLUwzlQoyqCzURl</td>
      <td>Prabh Deep;Seedhe Maut;Sez on the Beat</td>
      <td>Class-Sikh Maut, Vol. II</td>
      <td>Class-Sikh Maut, Vol. II</td>
      <td>42</td>
      <td>142163.0</td>
      <td>True</td>
      <td>0.855</td>
      <td>0.680</td>
      <td>8</td>
      <td>...</td>
      <td>1</td>
      <td>0.4280</td>
      <td>0.0601</td>
      <td>0.000000</td>
      <td>0.0417</td>
      <td>0.818</td>
      <td>129.843</td>
      <td>4</td>
      <td>electro</td>
      <td>9</td>
    </tr>
    <tr>
      <th>31408</th>
      <td>0o6lcPso94JOf4F56xsRgv</td>
      <td>Catuaba Com Amendoim</td>
      <td>O Tesão do Forró</td>
      <td>Parabéns Meu Amor</td>
      <td>37</td>
      <td>188400.0</td>
      <td>False</td>
      <td>0.742</td>
      <td>0.806</td>
      <td>7</td>
      <td>...</td>
      <td>1</td>
      <td>0.0510</td>
      <td>0.1340</td>
      <td>0.000000</td>
      <td>0.0945</td>
      <td>0.883</td>
      <td>119.697</td>
      <td>4</td>
      <td>forro</td>
      <td>4</td>
    </tr>
    <tr>
      <th>33411</th>
      <td>6mVD1SfTvlFAPVi7txFL5H</td>
      <td>The Strokes</td>
      <td>Angles</td>
      <td>Machu Picchu</td>
      <td>64</td>
      <td>209626.0</td>
      <td>False</td>
      <td>0.695</td>
      <td>0.817</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0.0659</td>
      <td>0.0132</td>
      <td>0.097000</td>
      <td>0.0601</td>
      <td>0.874</td>
      <td>105.014</td>
      <td>4</td>
      <td>garage</td>
      <td>13</td>
    </tr>
    <tr>
      <th>87311</th>
      <td>5ZG83RxgtxxTKfXQ3TmhT0</td>
      <td>Los Fabulosos Cadillacs</td>
      <td>Rey Azúcar</td>
      <td>Padre Nuestro - Remasterizado 2008</td>
      <td>37</td>
      <td>213866.0</td>
      <td>False</td>
      <td>0.794</td>
      <td>0.759</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>0.2740</td>
      <td>0.3910</td>
      <td>0.000000</td>
      <td>0.0809</td>
      <td>0.940</td>
      <td>106.093</td>
      <td>4</td>
      <td>ska</td>
      <td>2</td>
    </tr>
    <tr>
      <th>56014</th>
      <td>239pFI8OWahs9u71849Lts</td>
      <td>The Green</td>
      <td>Ways &amp; Means</td>
      <td>Ways &amp; Means</td>
      <td>35</td>
      <td>271853.0</td>
      <td>False</td>
      <td>0.892</td>
      <td>0.610</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0.0549</td>
      <td>0.0712</td>
      <td>0.000000</td>
      <td>0.0523</td>
      <td>0.958</td>
      <td>124.945</td>
      <td>4</td>
      <td>j-rock</td>
      <td>9</td>
    </tr>
    <tr>
      <th>68076</th>
      <td>2RoTNWss4JqUwI1R19rhGW</td>
      <td>Xande De Pilares</td>
      <td>Perseverança</td>
      <td>Perseverança</td>
      <td>43</td>
      <td>199186.0</td>
      <td>False</td>
      <td>0.644</td>
      <td>0.600</td>
      <td>9</td>
      <td>...</td>
      <td>1</td>
      <td>0.1770</td>
      <td>0.4240</td>
      <td>0.000000</td>
      <td>0.0927</td>
      <td>0.810</td>
      <td>103.469</td>
      <td>4</td>
      <td>pagode</td>
      <td>6</td>
    </tr>
    <tr>
      <th>68449</th>
      <td>0L50h51ICmM3WrLagIFPJC</td>
      <td>Anstandslos &amp; Durchgeknallt;Emi Flemming</td>
      <td>Nüchtern</td>
      <td>Nüchtern</td>
      <td>35</td>
      <td>159375.0</td>
      <td>False</td>
      <td>0.864</td>
      <td>0.842</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0.0502</td>
      <td>0.1900</td>
      <td>0.000230</td>
      <td>0.0795</td>
      <td>0.844</td>
      <td>127.970</td>
      <td>4</td>
      <td>party</td>
      <td>6</td>
    </tr>
    <tr>
      <th>96829</th>
      <td>3ceihRUljV1eSM1plIjsB6</td>
      <td>Little Dragon</td>
      <td>Lover Chanting (Jayda G Remix)</td>
      <td>Lover Chanting - Edit</td>
      <td>54</td>
      <td>205464.0</td>
      <td>True</td>
      <td>0.865</td>
      <td>0.762</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0.0762</td>
      <td>0.0166</td>
      <td>0.795000</td>
      <td>0.0233</td>
      <td>0.896</td>
      <td>118.005</td>
      <td>4</td>
      <td>trip-hop</td>
      <td>2</td>
    </tr>
    <tr>
      <th>41988</th>
      <td>7dEKwMoUZCpvUFLLuJQjju</td>
      <td>KISS</td>
      <td>KISS 40</td>
      <td>I Was Made For Lovin' You - Single Mix</td>
      <td>63</td>
      <td>241106.0</td>
      <td>False</td>
      <td>0.708</td>
      <td>0.856</td>
      <td>9</td>
      <td>...</td>
      <td>1</td>
      <td>0.0537</td>
      <td>0.0287</td>
      <td>0.000085</td>
      <td>0.0727</td>
      <td>0.902</td>
      <td>128.518</td>
      <td>4</td>
      <td>hard-rock</td>
      <td>2</td>
    </tr>
    <tr>
      <th>73005</th>
      <td>7xi3mpdF9tC1SLrbkossDu</td>
      <td>Nicky Romero;Norma Jean Martine</td>
      <td>I Hope That It Hurts</td>
      <td>I Hope That It Hurts</td>
      <td>54</td>
      <td>160214.0</td>
      <td>False</td>
      <td>0.723</td>
      <td>0.771</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>0.1520</td>
      <td>0.0450</td>
      <td>0.000000</td>
      <td>0.0809</td>
      <td>0.533</td>
      <td>110.048</td>
      <td>4</td>
      <td>progressive-house</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9626</th>
      <td>6WKlbp24TraMPpWhQtU1Kp</td>
      <td>The Chemical Brothers</td>
      <td>No Geography</td>
      <td>Eve Of Destruction</td>
      <td>40</td>
      <td>280400.0</td>
      <td>False</td>
      <td>0.801</td>
      <td>0.857</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0.0604</td>
      <td>0.0760</td>
      <td>0.004580</td>
      <td>0.0847</td>
      <td>0.633</td>
      <td>125.966</td>
      <td>4</td>
      <td>breakbeat</td>
      <td>2</td>
    </tr>
    <tr>
      <th>74228</th>
      <td>0xHxeH4QTqlfNrQvczkoTA</td>
      <td>The Who</td>
      <td>My Generation (Stereo Version)</td>
      <td>I Can't Explain - Stereo Version</td>
      <td>53</td>
      <td>125967.0</td>
      <td>False</td>
      <td>0.607</td>
      <td>0.783</td>
      <td>9</td>
      <td>...</td>
      <td>1</td>
      <td>0.0505</td>
      <td>0.0409</td>
      <td>0.000000</td>
      <td>0.0461</td>
      <td>0.843</td>
      <td>138.825</td>
      <td>4</td>
      <td>psych-rock</td>
      <td>17</td>
    </tr>
    <tr>
      <th>78876</th>
      <td>47y8hRlWLEJ4VafE6LMjEZ</td>
      <td>Elvis Presley</td>
      <td>The Essential Elvis Presley</td>
      <td>A Little Less Conversation</td>
      <td>59</td>
      <td>132013.0</td>
      <td>False</td>
      <td>0.584</td>
      <td>0.802</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>0.0784</td>
      <td>0.1490</td>
      <td>0.000000</td>
      <td>0.0683</td>
      <td>0.835</td>
      <td>114.172</td>
      <td>4</td>
      <td>rock-n-roll</td>
      <td>13</td>
    </tr>
    <tr>
      <th>20939</th>
      <td>3829oHimRwuUHePcD2Jj7S</td>
      <td>Octave One</td>
      <td>One Black Water (feat. Ann Saunderson) [Full S...</td>
      <td>Blackwater - 128 full strings vocal mix</td>
      <td>52</td>
      <td>518945.0</td>
      <td>False</td>
      <td>0.763</td>
      <td>0.810</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0.0545</td>
      <td>0.2640</td>
      <td>0.000563</td>
      <td>0.0823</td>
      <td>0.496</td>
      <td>128.240</td>
      <td>4</td>
      <td>detroit-techno</td>
      <td>6</td>
    </tr>
    <tr>
      <th>13547</th>
      <td>1Fhb9iJPufNMZSwupsXiRe</td>
      <td>keshi</td>
      <td>beside you</td>
      <td>beside you</td>
      <td>75</td>
      <td>166023.0</td>
      <td>False</td>
      <td>0.711</td>
      <td>0.747</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0.0523</td>
      <td>0.4590</td>
      <td>0.000000</td>
      <td>0.0957</td>
      <td>0.852</td>
      <td>136.000</td>
      <td>4</td>
      <td>chill</td>
      <td>17</td>
    </tr>
    <tr>
      <th>94814</th>
      <td>12Dov40hSJdrgS5uHGKNP3</td>
      <td>Blank &amp; Jones;Boney M.</td>
      <td>Relax Edition 9</td>
      <td>Sunny - Summer Vibe Mix</td>
      <td>53</td>
      <td>259934.0</td>
      <td>False</td>
      <td>0.824</td>
      <td>0.693</td>
      <td>8</td>
      <td>...</td>
      <td>1</td>
      <td>0.0631</td>
      <td>0.0382</td>
      <td>0.291000</td>
      <td>0.0950</td>
      <td>0.521</td>
      <td>113.995</td>
      <td>4</td>
      <td>techno</td>
      <td>17</td>
    </tr>
    <tr>
      <th>95832</th>
      <td>1form6XxaJCcp9CoU3Qxos</td>
      <td>Hilight Tribe</td>
      <td>Love medicine &amp; natural trance</td>
      <td>Free tibet</td>
      <td>42</td>
      <td>359533.0</td>
      <td>False</td>
      <td>0.487</td>
      <td>0.861</td>
      <td>11</td>
      <td>...</td>
      <td>1</td>
      <td>0.0516</td>
      <td>0.4750</td>
      <td>0.753000</td>
      <td>0.0720</td>
      <td>0.776</td>
      <td>137.927</td>
      <td>4</td>
      <td>trance</td>
      <td>2</td>
    </tr>
    <tr>
      <th>459</th>
      <td>3BztDPhRhfwdwUndTlg45h</td>
      <td>Gabrielle Aplin</td>
      <td>Dear Happy</td>
      <td>So Far so Good</td>
      <td>37</td>
      <td>198040.0</td>
      <td>False</td>
      <td>0.751</td>
      <td>0.565</td>
      <td>5</td>
      <td>...</td>
      <td>1</td>
      <td>0.0513</td>
      <td>0.0777</td>
      <td>0.012100</td>
      <td>0.0855</td>
      <td>0.679</td>
      <td>119.996</td>
      <td>4</td>
      <td>acoustic</td>
      <td>9</td>
    </tr>
    <tr>
      <th>28778</th>
      <td>4Egb5xP6cniUx0kgZd5zLB</td>
      <td>Bomba Estéreo</td>
      <td>Amanecer</td>
      <td>Soy Yo</td>
      <td>61</td>
      <td>159800.0</td>
      <td>False</td>
      <td>0.887</td>
      <td>0.581</td>
      <td>11</td>
      <td>...</td>
      <td>0</td>
      <td>0.1720</td>
      <td>0.0396</td>
      <td>0.000198</td>
      <td>0.0788</td>
      <td>0.786</td>
      <td>117.042</td>
      <td>4</td>
      <td>electronic</td>
      <td>9</td>
    </tr>
    <tr>
      <th>28130</th>
      <td>0tTOD94y0Qvp9aF8eiATH0</td>
      <td>Abid Brohi;SomeWhatSuper</td>
      <td>The Sibbi Song</td>
      <td>The Sibbi Song</td>
      <td>40</td>
      <td>153218.0</td>
      <td>False</td>
      <td>0.887</td>
      <td>0.722</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0.0790</td>
      <td>0.0200</td>
      <td>0.000000</td>
      <td>0.0660</td>
      <td>0.697</td>
      <td>119.970</td>
      <td>4</td>
      <td>electronic</td>
      <td>9</td>
    </tr>
    <tr>
      <th>82318</th>
      <td>3jPaLIbWPyfS5GZluhxh4D</td>
      <td>Laurie Colon;Reynaldo Santiago " Chino "</td>
      <td>Chino " De Viaje " ..... Camino Al Cielo</td>
      <td>Dame Tus Ojos (feat. Laurie Colón)</td>
      <td>40</td>
      <td>250493.0</td>
      <td>False</td>
      <td>0.813</td>
      <td>0.747</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0.0691</td>
      <td>0.0271</td>
      <td>0.000009</td>
      <td>0.0490</td>
      <td>0.787</td>
      <td>130.051</td>
      <td>4</td>
      <td>salsa</td>
      <td>12</td>
    </tr>
    <tr>
      <th>15377</th>
      <td>3YoXksESEwIkPsH1gZBq4r</td>
      <td>COUCOU CHLOE</td>
      <td>NOBODY</td>
      <td>NOBODY</td>
      <td>52</td>
      <td>116756.0</td>
      <td>True</td>
      <td>0.844</td>
      <td>0.788</td>
      <td>11</td>
      <td>...</td>
      <td>1</td>
      <td>0.1360</td>
      <td>0.0905</td>
      <td>0.000656</td>
      <td>0.0839</td>
      <td>0.536</td>
      <td>110.988</td>
      <td>4</td>
      <td>club</td>
      <td>4</td>
    </tr>
    <tr>
      <th>84108</th>
      <td>6xljIkWM1sM84RPw4coZWZ</td>
      <td>Roberta Sá</td>
      <td>Delírio</td>
      <td>Delírio</td>
      <td>37</td>
      <td>174186.0</td>
      <td>False</td>
      <td>0.834</td>
      <td>0.649</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0.0588</td>
      <td>0.4510</td>
      <td>0.000731</td>
      <td>0.0636</td>
      <td>0.962</td>
      <td>112.038</td>
      <td>4</td>
      <td>samba</td>
      <td>4</td>
    </tr>
    <tr>
      <th>64999</th>
      <td>2FlIG6qvcKd73lqqedlgEX</td>
      <td>BiD;Black Alien;Fernando Nunes;Seu Jorge</td>
      <td>JAH-VAN</td>
      <td>Meu Bem Querer - JAH-VAN</td>
      <td>42</td>
      <td>284466.0</td>
      <td>False</td>
      <td>0.796</td>
      <td>0.545</td>
      <td>9</td>
      <td>...</td>
      <td>1</td>
      <td>0.0603</td>
      <td>0.1160</td>
      <td>0.015700</td>
      <td>0.0812</td>
      <td>0.620</td>
      <td>140.013</td>
      <td>4</td>
      <td>mpb</td>
      <td>6</td>
    </tr>
    <tr>
      <th>20076</th>
      <td>3BbD2sqk7P7Rc9V0KF9o4s</td>
      <td>Joshwa;Lee Foss</td>
      <td>My Humps</td>
      <td>My Humps</td>
      <td>67</td>
      <td>186124.0</td>
      <td>False</td>
      <td>0.862</td>
      <td>0.790</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>0.0672</td>
      <td>0.0418</td>
      <td>0.006150</td>
      <td>0.0746</td>
      <td>0.587</td>
      <td>129.005</td>
      <td>4</td>
      <td>deep-house</td>
      <td>17</td>
    </tr>
    <tr>
      <th>20460</th>
      <td>4Z0OyS2aO8BXWY5sXSTwQc</td>
      <td>Shiba San;Tchami</td>
      <td>I Wanna (Tchami Remix)</td>
      <td>I Wanna - Tchami Remix</td>
      <td>52</td>
      <td>215433.0</td>
      <td>False</td>
      <td>0.833</td>
      <td>0.783</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0.1710</td>
      <td>0.0151</td>
      <td>0.379000</td>
      <td>0.0965</td>
      <td>0.624</td>
      <td>127.002</td>
      <td>4</td>
      <td>deep-house</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1312</th>
      <td>1UIqpCB0b56K7U0JJPfskN</td>
      <td>Sir Jean;Voilaaa</td>
      <td>On te l'avait dit</td>
      <td>Spies Are Watching Me</td>
      <td>48</td>
      <td>432963.0</td>
      <td>False</td>
      <td>0.892</td>
      <td>0.708</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>0.0782</td>
      <td>0.0395</td>
      <td>0.001470</td>
      <td>0.0335</td>
      <td>0.949</td>
      <td>127.002</td>
      <td>4</td>
      <td>afrobeat</td>
      <td>6</td>
    </tr>
    <tr>
      <th>958</th>
      <td>3AJ0MFDt4OSknQ5D9Jlo4l</td>
      <td>Criolo</td>
      <td>Nó na Orelha</td>
      <td>Samba Sambei</td>
      <td>37</td>
      <td>222560.0</td>
      <td>False</td>
      <td>0.870</td>
      <td>0.737</td>
      <td>11</td>
      <td>...</td>
      <td>1</td>
      <td>0.1260</td>
      <td>0.2680</td>
      <td>0.003690</td>
      <td>0.0504</td>
      <td>0.952</td>
      <td>136.070</td>
      <td>4</td>
      <td>afrobeat</td>
      <td>6</td>
    </tr>
    <tr>
      <th>76893</th>
      <td>0NPQCs25OPNgYOFhCXL8kk</td>
      <td>Flora Matos</td>
      <td>DO LADO DE FLORA</td>
      <td>CONVERSAR COM O MAR</td>
      <td>41</td>
      <td>199232.0</td>
      <td>False</td>
      <td>0.867</td>
      <td>0.696</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>0.0847</td>
      <td>0.1800</td>
      <td>0.000010</td>
      <td>0.0725</td>
      <td>0.818</td>
      <td>133.008</td>
      <td>4</td>
      <td>r-n-b</td>
      <td>13</td>
    </tr>
    <tr>
      <th>76870</th>
      <td>5z2JnOo0ZxrZVkaFpeV5AJ</td>
      <td>Consciência Humana</td>
      <td>Entre a Adolescência e o Crime</td>
      <td>Lembrança</td>
      <td>42</td>
      <td>574173.0</td>
      <td>True</td>
      <td>0.856</td>
      <td>0.601</td>
      <td>8</td>
      <td>...</td>
      <td>1</td>
      <td>0.0856</td>
      <td>0.0468</td>
      <td>0.000000</td>
      <td>0.0707</td>
      <td>0.632</td>
      <td>136.898</td>
      <td>4</td>
      <td>r-n-b</td>
      <td>13</td>
    </tr>
    <tr>
      <th>64060</th>
      <td>3ov4vgaxgFhSJwejMCpZUY</td>
      <td>Oliver Koletzki</td>
      <td>Made of Wood</td>
      <td>La Veleta</td>
      <td>41</td>
      <td>390371.0</td>
      <td>False</td>
      <td>0.831</td>
      <td>0.823</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>0.0571</td>
      <td>0.0295</td>
      <td>0.773000</td>
      <td>0.0838</td>
      <td>0.495</td>
      <td>109.977</td>
      <td>4</td>
      <td>minimal-techno</td>
      <td>17</td>
    </tr>
    <tr>
      <th>64062</th>
      <td>1pea4MX31SSMy0PeQsme33</td>
      <td>Albertina;Guy Gerber;Michael Bibi</td>
      <td>Bocat (Michael Bibi Remix)</td>
      <td>Bocat - Michael Bibi Remix</td>
      <td>51</td>
      <td>367086.0</td>
      <td>False</td>
      <td>0.807</td>
      <td>0.766</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0.0783</td>
      <td>0.0145</td>
      <td>0.749000</td>
      <td>0.0812</td>
      <td>0.599</td>
      <td>126.997</td>
      <td>4</td>
      <td>minimal-techno</td>
      <td>17</td>
    </tr>
    <tr>
      <th>77336</th>
      <td>5O4erNlJ74PIF6kGol1ZrC</td>
      <td>Bob Marley &amp; The Wailers</td>
      <td>Uprising</td>
      <td>Could You Be Loved</td>
      <td>78</td>
      <td>237000.0</td>
      <td>False</td>
      <td>0.916</td>
      <td>0.720</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0.1000</td>
      <td>0.3600</td>
      <td>0.000160</td>
      <td>0.0958</td>
      <td>0.760</td>
      <td>103.312</td>
      <td>4</td>
      <td>reggae</td>
      <td>12</td>
    </tr>
    <tr>
      <th>77630</th>
      <td>14yYVlxDi7orbwheBXqIPt</td>
      <td>Bayron Fire;Yiordano Ignacio</td>
      <td>Mambo Para Los Presos</td>
      <td>Mambo Para Los Presos</td>
      <td>44</td>
      <td>187201.0</td>
      <td>False</td>
      <td>0.884</td>
      <td>0.693</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>0.0981</td>
      <td>0.1440</td>
      <td>0.000007</td>
      <td>0.0858</td>
      <td>0.587</td>
      <td>129.989</td>
      <td>4</td>
      <td>reggae</td>
      <td>12</td>
    </tr>
    <tr>
      <th>38648</th>
      <td>1MtswMNmlmCn0fl0xf8qB1</td>
      <td>Biscits</td>
      <td>Locco</td>
      <td>Locco</td>
      <td>54</td>
      <td>183875.0</td>
      <td>False</td>
      <td>0.956</td>
      <td>0.861</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0.2060</td>
      <td>0.1040</td>
      <td>0.897000</td>
      <td>0.0717</td>
      <td>0.872</td>
      <td>125.930</td>
      <td>4</td>
      <td>groove</td>
      <td>4</td>
    </tr>
    <tr>
      <th>38645</th>
      <td>6j4j6iRR0Ema531o5Yxr2T</td>
      <td>Eden Prince;Nonô</td>
      <td>Memories</td>
      <td>Memories</td>
      <td>55</td>
      <td>149395.0</td>
      <td>False</td>
      <td>0.909</td>
      <td>0.651</td>
      <td>11</td>
      <td>...</td>
      <td>0</td>
      <td>0.1050</td>
      <td>0.1610</td>
      <td>0.043700</td>
      <td>0.0923</td>
      <td>0.895</td>
      <td>124.026</td>
      <td>4</td>
      <td>groove</td>
      <td>4</td>
    </tr>
    <tr>
      <th>18032</th>
      <td>6wmAHw1szh5RCKSRjiXhPe</td>
      <td>Charlie Puth</td>
      <td>Voicenotes</td>
      <td>How Long</td>
      <td>73</td>
      <td>200853.0</td>
      <td>False</td>
      <td>0.845</td>
      <td>0.561</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0.0778</td>
      <td>0.2110</td>
      <td>0.000003</td>
      <td>0.0383</td>
      <td>0.811</td>
      <td>109.974</td>
      <td>4</td>
      <td>dance</td>
      <td>17</td>
    </tr>
    <tr>
      <th>18112</th>
      <td>1xzi1Jcr7mEi9K2RfzLOqS</td>
      <td>Beyoncé</td>
      <td>RENAISSANCE</td>
      <td>CUFF IT</td>
      <td>93</td>
      <td>225388.0</td>
      <td>True</td>
      <td>0.780</td>
      <td>0.689</td>
      <td>7</td>
      <td>...</td>
      <td>1</td>
      <td>0.1410</td>
      <td>0.0368</td>
      <td>0.000010</td>
      <td>0.0698</td>
      <td>0.642</td>
      <td>115.042</td>
      <td>4</td>
      <td>dance</td>
      <td>17</td>
    </tr>
    <tr>
      <th>7520</th>
      <td>2jbZNxygeXpRxRuKJeEQHn</td>
      <td>T.K. Soul</td>
      <td>Untouchable</td>
      <td>Ride or Die</td>
      <td>36</td>
      <td>250802.0</td>
      <td>False</td>
      <td>0.878</td>
      <td>0.602</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0.0871</td>
      <td>0.2500</td>
      <td>0.000005</td>
      <td>0.0389</td>
      <td>0.964</td>
      <td>141.973</td>
      <td>4</td>
      <td>blues</td>
      <td>2</td>
    </tr>
    <tr>
      <th>33110</th>
      <td>1ekeVWwvCSbyLbt9yLawK6</td>
      <td>DJ Bratti SC</td>
      <td>MEGA FUNK - TOMA CATUCADA</td>
      <td>MEGA FUNK - TOMA CATUCADA</td>
      <td>44</td>
      <td>386634.0</td>
      <td>True</td>
      <td>0.927</td>
      <td>0.618</td>
      <td>11</td>
      <td>...</td>
      <td>1</td>
      <td>0.1940</td>
      <td>0.0259</td>
      <td>0.000007</td>
      <td>0.0628</td>
      <td>0.571</td>
      <td>130.031</td>
      <td>4</td>
      <td>funk</td>
      <td>2</td>
    </tr>
    <tr>
      <th>32844</th>
      <td>4UympsZgnf2To9Guid2Gwn</td>
      <td>DJ Ghost Floripa</td>
      <td>Toma! Mega Rave</td>
      <td>Toma! Mega Rave</td>
      <td>46</td>
      <td>155602.0</td>
      <td>True</td>
      <td>0.892</td>
      <td>0.560</td>
      <td>6</td>
      <td>...</td>
      <td>1</td>
      <td>0.1280</td>
      <td>0.1900</td>
      <td>0.006600</td>
      <td>0.0963</td>
      <td>0.692</td>
      <td>129.981</td>
      <td>4</td>
      <td>funk</td>
      <td>2</td>
    </tr>
    <tr>
      <th>45613</th>
      <td>0gzu5mm36VJH2Zqu8sQPTf</td>
      <td>Badshah;Payal Dev</td>
      <td>Genda Phool (feat. Payal Dev)</td>
      <td>Genda Phool (feat. Payal Dev)</td>
      <td>62</td>
      <td>170769.0</td>
      <td>False</td>
      <td>0.955</td>
      <td>0.538</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0.0734</td>
      <td>0.1320</td>
      <td>0.001460</td>
      <td>0.0704</td>
      <td>0.837</td>
      <td>116.977</td>
      <td>4</td>
      <td>hip-hop</td>
      <td>17</td>
    </tr>
    <tr>
      <th>45679</th>
      <td>4dCLUBJwtZwc4HYXvfcpyP</td>
      <td>KR$NA</td>
      <td>Untitled</td>
      <td>Untitled</td>
      <td>59</td>
      <td>151384.0</td>
      <td>False</td>
      <td>0.906</td>
      <td>0.811</td>
      <td>7</td>
      <td>...</td>
      <td>1</td>
      <td>0.2780</td>
      <td>0.3930</td>
      <td>0.000013</td>
      <td>0.0703</td>
      <td>0.667</td>
      <td>130.024</td>
      <td>4</td>
      <td>hip-hop</td>
      <td>17</td>
    </tr>
    <tr>
      <th>45911</th>
      <td>5KkCsYIgMK5H7Ih31Dnd2Q</td>
      <td>DIVINE;MC Altaf;Phenom;Stylo G</td>
      <td>Mirchi</td>
      <td>Mirchi</td>
      <td>57</td>
      <td>192923.0</td>
      <td>False</td>
      <td>0.871</td>
      <td>0.608</td>
      <td>6</td>
      <td>...</td>
      <td>1</td>
      <td>0.4000</td>
      <td>0.1420</td>
      <td>0.000000</td>
      <td>0.0876</td>
      <td>0.818</td>
      <td>129.784</td>
      <td>4</td>
      <td>hip-hop</td>
      <td>17</td>
    </tr>
    <tr>
      <th>18308</th>
      <td>4E0teOQQQwagLVvQ7VfYm1</td>
      <td>Afro B</td>
      <td>Afrowave 2</td>
      <td>Drogba (Joanna)</td>
      <td>64</td>
      <td>199000.0</td>
      <td>False</td>
      <td>0.966</td>
      <td>0.633</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>0.1010</td>
      <td>0.0206</td>
      <td>0.000004</td>
      <td>0.0715</td>
      <td>0.767</td>
      <td>108.011</td>
      <td>4</td>
      <td>dancehall</td>
      <td>4</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>0Mg5cbRrpU5VF3tV90hmvk</td>
      <td>Fivio Foreign;French Montana;Skillibeng</td>
      <td>Whap Whap (feat. Fivio Foreign &amp; French Montana)</td>
      <td>Whap Whap (feat. Fivio Foreign &amp; French Montana)</td>
      <td>58</td>
      <td>161495.0</td>
      <td>True</td>
      <td>0.950</td>
      <td>0.825</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0.2680</td>
      <td>0.3820</td>
      <td>0.000012</td>
      <td>0.0867</td>
      <td>0.733</td>
      <td>106.978</td>
      <td>4</td>
      <td>dancehall</td>
      <td>4</td>
    </tr>
    <tr>
      <th>18580</th>
      <td>10dVVfHlZYjV4Qv9uNVgrA</td>
      <td>Afro B;French Montana</td>
      <td>Joanna (Drogba) [Remix]</td>
      <td>Joanna (Drogba) - Remix</td>
      <td>55</td>
      <td>207763.0</td>
      <td>True</td>
      <td>0.934</td>
      <td>0.642</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>0.1170</td>
      <td>0.0370</td>
      <td>0.000000</td>
      <td>0.0754</td>
      <td>0.748</td>
      <td>108.027</td>
      <td>4</td>
      <td>dancehall</td>
      <td>4</td>
    </tr>
    <tr>
      <th>22179</th>
      <td>2je56j0xuoTi1gYkLLMlJU</td>
      <td>Kool &amp; The Gang</td>
      <td>Collected</td>
      <td>Get Down On It - Single Version</td>
      <td>68</td>
      <td>211680.0</td>
      <td>False</td>
      <td>0.877</td>
      <td>0.560</td>
      <td>7</td>
      <td>...</td>
      <td>1</td>
      <td>0.0621</td>
      <td>0.1560</td>
      <td>0.000139</td>
      <td>0.0535</td>
      <td>0.969</td>
      <td>110.860</td>
      <td>4</td>
      <td>disco</td>
      <td>4</td>
    </tr>
    <tr>
      <th>22234</th>
      <td>3pbtBomO4Zt5gGiqsYeiBH</td>
      <td>Diana Ross</td>
      <td>Diana</td>
      <td>Upside Down</td>
      <td>69</td>
      <td>245600.0</td>
      <td>False</td>
      <td>0.873</td>
      <td>0.855</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0.0615</td>
      <td>0.1790</td>
      <td>0.028500</td>
      <td>0.0377</td>
      <td>0.884</td>
      <td>107.868</td>
      <td>4</td>
      <td>disco</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>50 rows × 21 columns</p>
</div>




```python
if SAVE_FINAL_PLAYLIST:
    playlist_50.to_csv('DancefloorAnthems.csv', index=False)
```

With unwavering dedication, we've carefully curated a playlist of 50 tracks that embody the very essence of rhythm, energy, and pure musical bliss. Each track has been selected not only for its danceability but also for its unique blend of attributes that promises an electrifying experience.

From heart-pounding beats to soul-stirring melodies, from high-energy anthems to smooth grooves, this playlist has it all. It's a sonic journey that transcends genres, bringing together the best of what music has to offer. Whether you're a fan of pop, hip-hop, electronic, rock, or any other genre, you'll find something here that moves your soul and compels your feet to the dancefloor.

As you hit play on this carefully crafted playlist, get ready to immerse yourself in a world of music that knows no boundaries. It's a playlist designed to make you dance, celebrate, and savor every moment of the summer night. So gather your friends, put on your dancing shoes, and let the music take control. The dancefloor anthems await, and the party is just getting started.


```python
# Create a copy of the Spotify dataset
playlist_50_copy = playlist_50.copy()

# Convert selected attributes to quartiles
attributes_to_convert = ['energy', 'loudness', 'speechiness', 'acousticness', 'tempo', 'valence', 'liveness']
attributes_last = attributes_to_convert[-1]

# Create quartile columns for each attribute
for attribute in attributes_to_convert:
    playlist_50_copy[f'{attribute}_quartile'] = pd.qcut(playlist_50_copy[attribute], q=3, labels=['Q1','Q2','Q3'])

# Create a bar plot for each attribute quartile
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 25))

for i, attribute in enumerate(attributes_to_convert[:-1]):
    sns.boxplot(data=playlist_50_copy, x=f'{attribute}_quartile', y='danceability', ax=axes[i // 2, i % 2])
    axes[i // 2, i % 2].set_title(f'{attribute[0].upper()+attribute[1:]} Quartile Distribution')

plt.subplots_adjust(hspace=0.2)
plt.show()

sns.boxplot(data=playlist_50_copy, x=f'{attributes_last}_quartile', y='danceability').set_title(f'{attributes_last[0].upper()+attribute[1:]} Quartile Distribution')
plt.show()
```


    
![png](readme_files/output_99_0.png)
    



    
![png](readme_files/output_99_1.png)
    


# Acknowledgments

As we wrap up this report, we extend our heartfelt thanks to you, our esteemed readers, for joining us on this musical journey. This project would not have been possible without the collaborative efforts of both **Muhammad Umar Anzar and Muhammad Faizan**. 

You can connect with Muhammad Umar Anzar via email at [omer.anzar2@gmail.com](omer.anzar2@gmail.com) and LinkedIn [https://www.linkedin.com/in/umar-anzar/](https://www.linkedin.com/in/umar-anzar/). You can also reach out to Muhammad Faizan at [faizanwaseem476@gmail.com](faizanwaseem476@gmail.com) and connect on LinkedIn [https://www.linkedin.com/in/muhammad-faizan-51a892202](https://www.linkedin.com/in/muhammad-faizan-51a892202).

Should you have any questions, feedback, or simply wish to share your experiences, please feel free to contact either of us.
