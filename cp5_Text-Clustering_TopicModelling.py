"""
In addition to the text classification, we can also use clustering to group similar documents together.

We can use the embeddings of the documents to cluster them into different groups.

For example cat,dog => animal

soccer basketball => sports

pasta pizza => food
"""
import os


os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_LOCAL_API_KEY")


# Load data from Hugging Face
from datasets import load_dataset
dataset = load_dataset("maartengr/arxiv_nlp")["train"]

# Extract metadata
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]


from sentence_transformers import SentenceTransformer

# Create an embedding for each abstract
embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
print(f"shape of our embeddings are {embeddings.shape}")


"""
In order to work with those embeddings we need to reduce the dimensionality of the embeddings.

We can use PCA or UMAP to reduce the dimensionality of the embeddings.

"""

from umap import UMAP

# We reduce the input embeddings from 384 dimensions to 5 dimensions
umap_model = UMAP(
    n_components=2, min_dist=0.0, metric='cosine')
reduced_embeddings = umap_model.fit_transform(embeddings)

print(reduced_embeddings.shape)
from hdbscan import HDBSCAN

# We fit the model and extract the clusters
hdbscan_model = HDBSCAN(
    min_cluster_size=50, metric="euclidean", cluster_selection_method="eom"
).fit(reduced_embeddings)
clusters = hdbscan_model.labels_

# How many clusters did we generate?
len(set(clusters))



import numpy as np

# Print first three documents in cluster 0
cluster = 0
for index in np.where(clusters==cluster)[0][:3]:
    print(abstracts[index][:300] + "... \n")


import pandas as pd

df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
df["title"] = titles
df["cluster"] = [str(c) for c in clusters]

# Select outliers and non-outliers (clusters)
to_plot = df.loc[df.cluster != "-1", :]
outliers = df.loc[df.cluster == "-1", :]



import matplotlib.pyplot as plt

# Plot outliers and non-outliers separately
plt.scatter(outliers.x, outliers.y, alpha=0.05, s=2, c="grey")
plt.scatter(
    to_plot.x, to_plot.y, c=to_plot.cluster.astype(int),
    alpha=0.6, s=2, cmap="tab20b"
)
plt.axis("off")

plt.show()

"""
After clustering we can assign a topic to each cluster.

Classic topic modelling is done using Latent Dirichlet Allocation (LDA).

LDA assumes that each topic is characterized by a distribution of words.This usually uses bag-of-words 
which does not take into account the context of the words.

Instead we can use BERTopic which uses a transformer model to generate topics.

After getting the clusters we calculate bag of words counts for each cluster but while 

doing so we multiply the counts by the inverse document frequency (IDF) of the words.

IDF is calculated as:

IDF(t) = log((A/cf_x(t)) + 1)

where A is the total number of documents and cf_x(t) is the number of documents that contain the word t.

This allows us to penalize words that are common across all documents.

For example if a word is present in all documents then its IDF will be 0 and thus it will not contribute to the topic.
So we will have c-tf x idf which is called c-tf-idf.

"""

from bertopic import BERTopic

# Train our model with our previously defined models
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True
).fit(abstracts, embeddings)


print(topic_model.get_topic_info())


# Visualize topics and documents
fig = topic_model.visualize_documents(
    titles, 
    reduced_embeddings=reduced_embeddings, 
    width=1200, 
    hide_annotations=True
)

# Update fonts of legend for easier visualization
fig.update_layout(font=dict(size=16))


# Visualize barchart with ranked keywords
topic_model.visualize_barchart()

# Visualize relationships between topics
topic_model.visualize_heatmap(n_clusters=30)

# Visualize the potential hierarchical structure of topics
topic_model.visualize_hierarchy()


"""
We are still using the bag of words approach to generate topics.

additionally we can use a fine-tune approach on top of the previously found topics.

"""
# Save original representations
from copy import deepcopy
original_topics = deepcopy(topic_model.topic_representations_)


def topic_differences(model, original_topics, nr_topics=5):
    """Show the differences in topic representations between two models """
    df = pd.DataFrame(columns=["Topic", "Original", "Updated"])
    for topic in range(nr_topics):

        # Extract top 5 words per topic per model
        og_words = " | ".join(list(zip(*original_topics[topic]))[0][:5])
        new_words = " | ".join(list(zip(*model.get_topic(topic)))[0][:5])
        df.loc[len(df)] = [topic, og_words, new_words]
    
    return df


"""
KeyBertInspired:

Inspired from KeyBERT, which extracts keywords from documents by comparing the embeddings of the document and the words.

By using the embedding of the documents in to the cluster we can check how similar are the documents in the cluster with the candidate words.
"""

from bertopic.representation import KeyBERTInspired

# Update our topic representations using KeyBERTInspired
representation_model = KeyBERTInspired()
topic_model.update_topics(abstracts, representation_model=representation_model)

# Show topic differences
topic_differences(topic_model, original_topics)


"""
With this approach we still can't get rid of the words that are very similar like summaries and summary

Even though both of them are good at representing the documents having additional word does not contribute to the meaning

That is why we will use maximal marginal relevance

Instead of 30 words to represent the topic we will use  diversified 10 words.
"""


from bertopic.representation import MaximalMarginalRelevance

# Update our topic representations to MaximalMarginalRelevance
representation_model = MaximalMarginalRelevance(diversity=0.2)
topic_model.update_topics(abstracts, representation_model=representation_model)

# Show topic differences
topic_differences(topic_model, original_topics)













