import os
import pandas as pd

def read_and_tag_vectors(directory):
    vectors = []
    for filename in os.listdir(directory):
        if 'encoder.blocks' in filename:
            # Disperse the path into its components
            path_components = filename.split('_')
            language = path_components[2]  # Assuming language is the third component (e.g., "en" in "common_voice_en_...")
            encoder = path_components[4]
            steering_vector = filename.split('_sf')[-1].split('.')[0]  # Assuming steering vector is the last component before the extension (e.g., "sf0" in "sf0.pt")
            vectors.append({
                'filename': filename,
                'block_number': encoder,
                'language': language,
                'steering_factor': steering_vector
            })
    return vectors

# Example usage
directory = '/Users/ayushjain/Development/Interp/audio-steering/whisper_interpretability/results/activations'
tagged_vectors = read_and_tag_vectors(directory)
tagged_vectors_df = pd.DataFrame(tagged_vectors)


def read_and_tag_steering_vectors(directory):
    steering_vectors = []
    for filename in os.listdir(directory):
        if 'tiny_encoder.blocks' in filename:
            # Disperse the path into its components
            path_components = filename.split('_')
            block_number = path_components[-2]  # Assuming block number is the second last component (e.g., "1" in "tiny_encoder.blocks.1_fr")
            language = path_components[-1]  # Assuming language is the last component (e.g., "fr" in "tiny_encoder.blocks.1_fr")
            steering_vectors.append({
                'filename': filename,
                'block_number': block_number,
                'language': language
            })
    return steering_vectors

tagged_steering_vectors = read_and_tag_steering_vectors('out/')
tagged_steering_vectors_df = pd.DataFrame(tagged_steering_vectors)

import torch
import torch.nn.functional as F

def cosine_similarity(vector1, vector2):
    v1 = vector1.squeeze(0)
    v2 = vector2.squeeze(0)

    similarity = F.cosine_similarity(v1, v2, dim=-1)
    return similarity

ACTIVATION_ROOT = 'results/activations/'
STEER_ROOT = 'out/'

# print(tagged_vectors_df)
# print(tagged_steering_vectors_df)
import matplotlib.pyplot as plt

def compare_to_steering_vectors(vectors, steering):
    transformed = vectors[vectors['steering_factor'] == '1']

    similarity_scores = []

    for index, row in transformed.iterrows():
        filename = row['filename']
        encoder = filename.split('_')[4]  # Assuming encoder is the third component of the filename (e.g., "encoder" in "common_voice_en_encoder.blocks.0_sf1.pt")
        language = row['language']
        opposite_language = 'fr' if language == 'en' else 'en'
        steering_path = steering[(steering['block_number'] == row['block_number']) & (steering['language'] == opposite_language)]['filename'].values[0]


        activation = torch.load(ACTIVATION_ROOT + filename)
        steering_vector = torch.load(STEER_ROOT + steering_path)
        similarity = cosine_similarity(activation, steering_vector)
        avg_similarity = similarity.mean().item()

        similarity_scores.append({
            'encoder': encoder,
            'language': language,
            'filename': filename,
            'similarity': avg_similarity
        })

    # Convert to DataFrame for easy manipulation
    similarity_df = pd.DataFrame(similarity_scores)

    return similarity_df

def get_audio(filepath):
    path_components = filepath.split('_encoder')
    return path_components[0]

def get_encoder(filePath):
    path_components = filePath.split('_')
    return path_components[4]

def compare_to_self(vectors, steering):
    # Load all activation vectors with steering_factor == '0'
    zero_vectors = vectors[vectors['steering_factor'] == '0']
    zero_activations = {}
    for index, row in zero_vectors.iterrows():
        filename = row['filename']
        zero_activations[filename] = torch.load(ACTIVATION_ROOT + filename)

    # Prepare to store similarity scores
    similarity_scores = []

    # Iterate over activation vectors with steering_factor == '1'
    transformed = vectors[vectors['steering_factor'] == '1']
    for index, row in transformed.iterrows():
        filename = row['filename']
        encoder = get_encoder(filename)
        language = row['language']

        # Compare to all activation vectors with steering_factor == '0'
        activation = torch.load(ACTIVATION_ROOT + filename)

        audioFile = get_audio(filename)

        for zero_filename, zero_activation in zero_activations.items():
            # Compute cosine similarity
            zeroAudio = get_audio(zero_filename)
            zero_encoder = get_encoder(zero_filename)

            if audioFile != zeroAudio or zero_encoder != encoder:
                continue

            similarity = cosine_similarity(activation, zero_activation).mean().item()

            similarity_scores.append({
                'encoder': encoder,
                'language': language,
                'filename': filename,
                'zero_filename': zero_filename,
                'similarity': similarity
            })

    # Convert to DataFrame for easy manipulation
    similarity_df = pd.DataFrame(similarity_scores)

    return similarity_df

#* This is for comparing to the steering vector that was used
# steering_data = compare_to_steering_vectors(tagged_vectors_df, tagged_steering_vectors_df)
#* This is for comparing to the other factor (ie steered french compared to base french )
steering_data = compare_to_self(tagged_vectors_df, tagged_steering_vectors_df)
print(steering_data)

def plot_boxplot_by_encoder(similarity_df):
    plt.figure(figsize=(10, 6))
    similarity_df.boxplot(column='similarity', by='encoder', grid=False)
    plt.xlabel('Encoder')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity by Encoder')
    plt.suptitle('')  # Suppress the automatic title
    plt.savefig('images/cosine_similarity_by_encoder.png')
    plt.close()

# Box plot of similarity by language
def plot_boxplot_by_language(similarity_df):
    plt.figure(figsize=(10, 6))
    similarity_df.boxplot(column='similarity', by='language', grid=False)
    plt.xlabel('Language')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity by Language')
    plt.suptitle('')  # Suppress the automatic title
    plt.savefig('images/cosine_similarity_by_language.png')
    plt.close()

# Plotting average similarity value for each filepath
def plot_average_similarity_by_filepath(similarity_df):
    # Extract the part before 'encoder.blocks' from the filename
    similarity_df['filename'] = similarity_df['filename'].apply(lambda x: x.split('encoder.blocks')[0])
    avg_similarity_df = similarity_df.groupby('filename')['similarity'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_similarity_df['filename'], avg_similarity_df['similarity'], marker='o')
    plt.xticks(rotation=90)
    plt.xlabel('Filename')
    plt.ylabel('Average Cosine Similarity')
    plt.title('Average Cosine Similarity by Filepath')
    plt.tight_layout()  # Adjust layout for better fit
    plt.savefig('images/average_cosine_similarity_by_filepath.png')
    plt.close()

plot_boxplot_by_encoder(steering_data)
plot_boxplot_by_language(steering_data)
plot_average_similarity_by_filepath(steering_data)
