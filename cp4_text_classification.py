from datasets import load_dataset


data = load_dataset("rotten_tomatoes")
print(f"our data is {data}")

print(f"Last element of the training set is {data['train'][0]}")



from transformers import pipeline

# Path to our HF model
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Load model into pipeline
pipe = pipeline(
    model=model_path,
    tokenizer=model_path,
    return_all_scores=True,
    device="cuda:0"
)


import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

# Run inference
y_pred = []
for output in tqdm(pipe(KeyDataset(data["test"], "text")), total=len(data["test"])):
    negative_score = output[0]["score"]
    positive_score = output[2]["score"]
    assignment = np.argmax([negative_score, positive_score])
    y_pred.append(assignment)


from sklearn.metrics import classification_report

def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)

evaluate_performance(data["test"]["label"], y_pred)


print("Getting embeddings for the training and test sets")


from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Convert text to embeddings
train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

print(f"for each of our {train_embeddings.shape[0]} training reviews, we have an embedding of size {train_embeddings.shape[1]}")
print(f"for each of our {test_embeddings.shape[0]} test reviews, we have an embedding of size {test_embeddings.shape[1]}")

""" sentence => embedding => vector => logistic regression => sentiment indicator """

from sklearn.linear_model import LogisticRegression


clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, data["train"]["label"])

y_pred = clf.predict(test_embeddings)
evaluate_performance(data["test"]["label"], y_pred)

"""
What if we dont have the labels?

In this case we can use zero shot classification

f: input sentence + candidate labels => predict which label is true

In order to this our model needs to output 3 things:
- embedding of the input sentence
- embedding for the positive label ( a positive movie review)
- embedding for the negative label ( a negative movie review)

Then we can use the cosine similarity to find the label that is most similar to the input sentence's embedding


"""
label_embeddings = model.encode(["A negative review",  "A positive review"])

print(f"shape of our label embeddings are {label_embeddings.shape}")


from sklearn.metrics.pairwise import cosine_similarity

# Find the best matching label for each document
sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
y_pred = np.argmax(sim_matrix, axis=1)

evaluate_performance(data["test"]["label"], y_pred)






