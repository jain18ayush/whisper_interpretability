import torch
import torch.nn.functional as F
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_util(vector1, vector2):
    v1 = vector1.squeeze(0)
    v2 = vector2.squeeze(0)

    similarity = F.cosine_similarity(v1, v2, dim=-1)
    return similarity

def create_df_from_filenames(directory):
    data = []

    for filename in os.listdir(directory):
        if filename.endswith('.pt'):
            # Split the filename
            parts = filename[:-3].split('_')  # Remove '.pt' and split

            if len(parts) == 5:
                audio_path, data_type, analysis_type, encoder, steering_factor = parts
                data.append({
                    'Audio': audio_path,
                    'DataType': data_type,
                    'AnalysisType': analysis_type,
                    'Encoder': int(encoder),
                    'Factor': float(steering_factor),
                    'Path': os.path.join(directory, filename)
                })

    # Create DataFrame
    df = pd.DataFrame(data)
    return df

#activation vectors
anger_neg = torch.load('gpt2_example_activations/anger_-15.pt')
anger_base = torch.load('gpt2_example_activations/anger_0.pt')
anger_pos = torch.load('gpt2_example_activations/anger_30.pt')
love_neg = torch.load('gpt2_example_activations/love_-10.pt')
love_base = torch.load('gpt2_example_activations/love_0.pt')
love_pos = torch.load('gpt2_example_activations/love_10.pt')

#gpt2 base steering vector
anger_gpt2 = torch.load('crosssteering_vectors/basic_gpt2/anger-vector.pt')
love_gpt2 = torch.load('crosssteering_vectors/basic_gpt2/love-vector.pt')

#translated steering vectors
anger_encoded = torch.load('crosssteering_vectors/anger/anger-encoded.pt')
anger_gaussian = torch.load('crosssteering_vectors/anger/anger-gaussian.pt')
anger_pca = torch.load('crosssteering_vectors/anger/anger-pca.pt')
anger_sparse = torch.load('crosssteering_vectors/anger/anger-sparse.pt')

love_encoded = torch.load('crosssteering_vectors/love/love-encoded.pt')
love_gaussian = torch.load('crosssteering_vectors/love/love-gaussian.pt')
love_pca = torch.load('crosssteering_vectors/love/love-pca.pt')
love_sparse = torch.load('crosssteering_vectors/love/love-sparse.pt')

cross_steering_directory = 'results/cross_steering'
activations = pd.read_csv('tables/cross_steering_data.csv')


#encoder steering vectors
en_0 = torch.load('out/tiny_encoder.blocks.0_en')
en_1 = torch.load('out/tiny_encoder.blocks.1_en')
en_2 = torch.load('out/tiny_encoder.blocks.2_en')
en_3 = torch.load('out/tiny_encoder.blocks.3_en')
fr_0 = torch.load('out/tiny_encoder.blocks.0_fr')
fr_1 = torch.load('out/tiny_encoder.blocks.1_fr')
fr_2 = torch.load('out/tiny_encoder.blocks.2_fr')
fr_3 = torch.load('out/tiny_encoder.blocks.3_fr')


#* how far are the different vector translation factos from each other? (VERY CLOSE)
#* how far are the different ranslation factors from the english and french steering factors? (EXTREMELY CLOSE)



def plot_vector_similarities(vectors, vector_names=None):
    if vector_names is None:
        vector_names = [f'Vector {i+1}' for i in range(len(vectors))]


    # Ensure vectors are 2D arrays
    vectors = [v.reshape(-1).numpy() for v in vectors]  # Shape becomes (1, 1500 * 384) = (1, 576000)
    # Compute cosine similarity matrix
    for vector in vectors:
        print(f'Shape: {vector.shape}')

    similarity_matrix = cosine_similarity(vectors)

    # Use default names if not provided

    # Create a heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Cosine Similarity')

    # Add labels and title
    plt.xticks(range(len(vectors)), vector_names, rotation=45, ha='right')
    plt.yticks(range(len(vectors)), vector_names)
    plt.title('Cosine Similarity between Vectors of Anger')

    # Add text annotations
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                     ha='center', va='center', color='white')

    plt.tight_layout()
    plt.savefig('images/cross_steering/anger_similarity_constructed_heatmap.png')
    plt.show()

# how different are thye from french and english steering vectors?
# anger_vectors = [anger_encoded, anger_gaussian, anger_pca, anger_sparse, en_0, en_1, en_2, en_3, fr_0, fr_1, fr_2, fr_3]
# vector_names = ['Encoded', 'Gaussian', 'PCA', 'Sparse', 'en_0', 'en_1', 'en_2', 'en_3', 'fr_0', 'fr_1', 'fr_2', 'fr_3']

anger_vectors = [anger_encoded, anger_gaussian, anger_pca, anger_sparse]
vector_names = ['Encoded', 'Gaussian', 'PCA', 'Sparse']


plot_vector_similarities(vectors=anger_vectors, vector_names=vector_names)

#* How far the the activation vectors from the base by type?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_vector(path):
    # Placeholder function to load vectors, replace with actual loading logic
    # Assume vectors are stored in a numpy-compatible format, e.g., .npy
    return torch.load(path)

def calculate_differences(df):
    similarities = []

    for analysis_type in df['AnalysisType'].unique():
        for encoder in df['Encoder'].unique():
            # Filter vectors for steering factor 0
            base_vectors = df[(df['Factor'] == 0) &
                              (df['AnalysisType'] == analysis_type) &
                              (df['Encoder'] == encoder)]

            other_vectors = df[(df['Factor'] != 0) &
                               (df['AnalysisType'] == analysis_type) &
                               (df['Encoder'] == encoder)]

            # Load vectors
            base_vectors_loaded = [load_vector(path) for path in base_vectors['Path']]
            other_vectors_loaded = [load_vector(path) for path in other_vectors['Path']]

            # Compute cosine similarities
            for base_vec in base_vectors_loaded:
                for other_vec in other_vectors_loaded:
                    similarity = cosine_similarity_util(base_vec, other_vec)

                    if isinstance(similarity, torch.Tensor):
                        similarity = similarity.numpy()  # Convert to numpy array

                    similarities.append({
                        'AnalysisType': analysis_type,
                        'Encoder': encoder,
                        'CosineSimilarity': similarity
                    })

    out = pd.DataFrame(similarities)
    # Convert to DataFrame
    return out

def plot_boxplot_by_encoder(similarity_df, analysis_type):
    filtered_df = similarity_df[similarity_df['AnalysisType'] == analysis_type]
    # Convert arrays in 'CosineSimilarity' to their mean values
    filtered_df['CosineSimilarityMean'] = filtered_df['CosineSimilarity'].apply(np.mean)

    plt.figure(figsize=(10, 6))
    filtered_df.boxplot(column='CosineSimilarityMean', by='Encoder', grid=False)
    plt.xlabel('Encoder')
    plt.ylabel('Mean Cosine Similarity')
    plt.title(f'{analysis_type.capitalize()} Cosine Similarity to Base Vector')
    plt.suptitle('')  # Suppress the automatic title
    # plt.savefig(f'images/cosine_similarity_{analysis_type}.png')
    plt.close()

# Assuming df is your input DataFrame
# similarity_df = calculate_differences(activations)

# for type_name in ['pca', 'gaussian', 'sparse', 'encoded']:
#     plot_boxplot_by_encoder(similarity_df=similarity_df, analysis_type=type_name)
