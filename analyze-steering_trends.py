import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize the model for embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Caches for embeddings and cosine similarities
embedding_cache = {}
cosine_sim_cache = {}

# Function to compute and cache embeddings
def get_embedding(text):
    if text not in embedding_cache:
        embedding_cache[text] = model.encode(text)
    return embedding_cache[text]

# Function to compute cosine similarity with caching
def get_cosine_sim(text1, text2):
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    pair = (text1, text2)
    if pair not in cosine_sim_cache:
        cosine_sim_cache[pair] = cosine_similarity([embedding1], [embedding2])[0][0]
    else:
        print('pair in cache')
    return cosine_sim_cache[pair]

# Load files from subdirectories
def load_files_from_subdirectories(base_directory):
    data = []
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                # Rename the first two columns to 'factor' and 'text'
                if df.shape[1] >= 2:
                    df.columns = ['factor', 'text'] + list(df.columns[2:])
                else:
                    continue  # Skip files that don't have at least two columns

                # Extract emotion and data type from subdirectory structure
                parts = os.path.relpath(root, base_directory).split(os.sep)
                if len(parts) >= 2:
                    data_type = parts[0]  # First subdirectory
                    emotion = parts[1]    # Deepest subdirectory
                else:
                    data_type = 'Unknown'
                    emotion = 'Unknown'

                df['filename'] = file
                df['data_type'] = data_type
                df['emotion'] = emotion
                data.append(df)
    return pd.concat(data, ignore_index=True)

# Directory containing the files
base_directory = 'results/steering_vectors'
df_all = load_files_from_subdirectories(base_directory)

# Define text at the 152nd row (1-indexed)
text_at_factor_0 = df_all.iloc[151]['text']

# Compute and save cosine similarity
def compute_and_save_cosine_sim(df, text_at_factor_0, save_path):
    results = []
    for _, row in df.iterrows():
        factor = row['factor']
        text = row['text']
        cosine_sim_value = get_cosine_sim(text, text_at_factor_0)
        results.append({'factor': factor, 'data_type': row['data_type'], 'emotion': row['emotion'], 'cosine_sim': cosine_sim_value})
    cosine_sim_df = pd.DataFrame(results)
    cosine_sim_df.to_csv(save_path, index=False)

# Path to save the cosine similarity DataFrame
cosine_sim_file = 'cosine_sim_results.csv'

# Compute and save cosine similarity if not already saved
if not os.path.exists(cosine_sim_file):
    compute_and_save_cosine_sim(df_all, text_at_factor_0, cosine_sim_file)

# Load the precomputed cosine similarity DataFrame
cosine_sim_df = pd.read_csv(cosine_sim_file)

# Compute the difference between love and anger
def compute_difference(df):
    love_df = df[df['emotion'] == 'love'].copy()
    anger_df = df[df['emotion'] == 'anger'].copy()

    # Merge on factor and data_type to compute the difference
    merged_df = pd.merge(love_df, anger_df, on=['factor', 'data_type'], suffixes=('_love', '_anger'))
    merged_df['cosine_sim_diff'] = merged_df['cosine_sim_love'] - merged_df['cosine_sim_anger']
    return merged_df

# Compute differences
difference_df = compute_difference(cosine_sim_df)

# Plotting the differences
data_types = difference_df['data_type'].unique()

plt.figure(figsize=(14, 10))

for i, data_type in enumerate(data_types):
    plt.subplot(2, 2, i + 1)
    subset = difference_df[difference_df['data_type'] == data_type]
    if not subset.empty:
        plt.plot(
            subset['factor'],
            subset['cosine_sim_diff'],
            # marker='o',
            linestyle='-',
            label=f"Difference for {data_type}"
        )
    plt.xlabel('Factor')
    plt.ylabel('Cosine Similarity Difference')
    plt.title(f'Cosine Similarity Difference ({data_type})')
    plt.legend()

plt.tight_layout()
plt.savefig(' /cosine_similarity_difference_by_data_type_wild.png')
plt.show()
