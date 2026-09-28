# Music Clustering from Spotify Audio Features

Unsupervised analysis of music tracks: two separate sources are merged into one enriched knowledge base, clustered, and then compared across four different feature spaces to see which one actually preserves the cluster structure.

Team project, Big Data course, MSc Artificial Intelligence and Robotics, CY Cergy Paris Université and ENSEA.

## The question

Streaming platforms describe a track in two very different ways. One source gives **metadata**: artist, album, genre, popularity. Another gives **acoustic features**: danceability, energy, loudness, valence, tempo, acousticness and so on. Neither alone says much about how music organises itself, so this project joins them and asks whether natural groups appear in the result.

## Building the knowledge base

The two datasets have no shared key, only track and artist names, which are written inconsistently between sources. The merge therefore had to be done by **text normalisation**: lowercasing, and stripping version descriptors such as remaster and remix annotations, so that the same song written two different ways resolves to one entity.

From the first dataset only descriptive attributes were kept, from the second only quantitative acoustic features, with text, links, duplicates and unusable metadata removed. The result is **228 enriched tracks described by 15 numerical acoustic features**.

## Exploration

- **Popularity** is not normally distributed, it is right skewed and multi modal, which is useful here because it means there is genuine variation to separate.
- **Genre** is multi genre and moderately imbalanced rather than dominated by one style.
- **Correlations** show the expected strong positive relationship between energy and loudness.
- **Outliers and scale**: `duration_ms` sits in the millions while most audio features lie between 0 and 1. Left alone this would dominate any distance based method, so scaling was applied before clustering.

## Clustering

**K-Means** and **DBSCAN** were both run on the scaled features.

| Method | Silhouette |
|---|---|
| K-Means | 0.223 |
| DBSCAN | 0.202 |

Silhouette scores in this range mean the groups are real but soft, with overlapping boundaries rather than cleanly separated clusters. That is an honest outcome for music: genres blend into each other.

## Which feature space preserves structure best

The same clustering was then evaluated on the original scaled features and on three reduced spaces, scored with three internal validation indices:

| Feature space | Silhouette (higher better) | Davies-Bouldin (lower better) | Calinski-Harabasz (higher better) |
|---|---|---|---|
| **PCA** | **0.236** | **1.217** | **119.76** |
| Original | 0.223 | 1.536 | 41.19 |
| t-SNE | 0.127 | 2.404 | 38.35 |
| ISOMAP | -0.008 | 1.897 | 63.62 |

**PCA wins on all three metrics**, and is the only reduction that improves on the original feature space rather than degrading it.

ISOMAP produces a negative silhouette, meaning the average point sits closer to a neighbouring cluster than to its own: its manifold assumption does not hold for this data. t-SNE is worth reading carefully, because it is built to preserve local neighbourhoods for visualisation and is not designed to preserve the global geometry these indices measure. A poor score here is not evidence that t-SNE is a poor method, only that it is the wrong tool for this particular question.

## Files

```
music_clustering.ipynb        full analysis, 59 code cells and 50 markdown cells
data/main_dataset.csv         source dataset
data/Enriched_Dataset.csv     merged knowledge base, 228 tracks
report/report.pdf             written report
report/presentation.pdf       slides
```

## Running it

```
pip install numpy pandas scikit-learn matplotlib seaborn
jupyter notebook music_clustering.ipynb
```

The notebook was originally written in Google Colab, so the first cells mount Drive. Replace those paths with the local files in `data/` to run it elsewhere.
