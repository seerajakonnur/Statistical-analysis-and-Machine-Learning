import os
# Set environment variables before imports to prevent threading issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
# Disable triton for macOS compatibility
os.environ["DISABLE_TRITON"] = "1"
os.environ["TRITON_DISABLE"] = "1"
# Force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from Bio import SeqIO
import glob
import warnings
warnings.filterwarnings("ignore")

# Configure matplotlib to avoid Type-3 fonts for conference submission
matplotlib.rcParams['pdf.fonttype'] = 42  # Use Type-42 fonts (TrueType)
matplotlib.rcParams['ps.fonttype'] = 42   # Use Type-42 fonts for PostScript
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # Use a standard font
matplotlib.rcParams['axes.unicode_minus'] = False  # Use ASCII minus sign

# Set multiprocessing start method to avoid fork issues on macOS
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

print("STEP 1: Loading DNABERT2 model and tokenizer...")
print("="*60)

# Try to load DNABERT2 model with fallback options
model_name = "zhihan1996/DNABERT-2-117M"

# Try to load DNABERT2 model with fallback options
model_name = "zhihan1996/DNABERT-2-117M"

try:
    print("Attempting to load DNABERT2...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    print("✅ DNABERT2 model loaded successfully")
except ImportError as e:
    if "triton" in str(e):
        print("⚠ Triton dependency issue. Trying alternative model...")
        try:
            # Fallback to original DNABERT (smaller, no triton dependency)
            model_name = "armheb/DNA_bert_6"
            print("Loading DNA_bert_6 as alternative...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            print("✅ DNA_bert_6 loaded successfully (triton-free alternative)")
        except Exception as e3:
            print(f"⚠ Alternative model also failed: {e3}")
            print("Please try: pip install transformers==4.30.0 torch==2.0.0")
            exit(1)
    else:
        raise e

def read_fasta_files_recursive(base_folder):
    """Read all FASTA files from a base folder and all its subfolders, keeping segments separate"""
    all_segment_sequences = []  # List of dictionaries, each containing 8 segments
    filenames = []
    
    print(f"\nSearching for FASTA files in {base_folder} and all subfolders...")
    
    # Get all FASTA files recursively from all subfolders
    fasta_patterns = ["*.fasta", "*.fa", "*.fas"]
    fasta_files = []
    
    for pattern in fasta_patterns:
        fasta_files.extend(glob.glob(os.path.join(base_folder, "**", pattern), recursive=True))
    
    print(f"Found {len(fasta_files)} FASTA files")
    
    for fasta_file in sorted(fasta_files):
        # Create a filename from the relative path
        rel_path = os.path.relpath(fasta_file, base_folder)
        filename = os.path.splitext(rel_path)[0].replace(os.sep, '_')
        
        # Read all segments from the FASTA file and keep them separate
        segments = []
        segment_count = 0
        
        try:
            for record in SeqIO.parse(fasta_file, "fasta"):
                segments.append(str(record.seq).upper())
                segment_count += 1
        except Exception as e:
            print(f"Error reading {fasta_file}: {e}")
            continue
        
        if segment_count == 0:
            print(f"Warning: No sequences found in {fasta_file}")
            continue
        
        # Store segments as separate sequences (expecting 8 segments)
        if segment_count != 8:
            print(f"Warning: {filename} has {segment_count} segments (expected 8)")
        
        all_segment_sequences.append(segments)
        filenames.append(filename)
        
        total_length = sum(len(seg) for seg in segments)
        print(f"  {filename}: {segment_count} segments, total length: {total_length} bp")
    
    return all_segment_sequences, filenames

print("\nSTEP 2: Reading FASTA sequences from folder structure...")
print("="*60)

# Read sequences from the two main folders
print("Reading reassortant sequences from 'reassortant_all' folder...")
reassortant_sequences, reassortant_names = read_fasta_files_recursive("reassortant_all")

print("\nReading non-reassortant sequences from 'nonreassortant_all' folder...")
nonreassortant_sequences, nonreassortant_names = read_fasta_files_recursive("nonreassortant_all")

# Combine all data
all_sequences = reassortant_sequences + nonreassortant_sequences
all_names = reassortant_names + nonreassortant_names
labels = ['Reassortant'] * len(reassortant_sequences) + ['Non-Reassortant'] * len(nonreassortant_sequences)

print(f"\n✅ Data loading complete:")
print(f"  Reassortant sequences: {len(reassortant_sequences)}")
print(f"  Non-reassortant sequences: {len(nonreassortant_sequences)}")
print(f"  Total sequences: {len(all_sequences)}")

def chunk_sequence(sequence, chunk_size=400, overlap=100):
    """Split long sequence into overlapping chunks for DNABERT2 processing"""
    chunks = []
    start = 0
    while start < len(sequence):
        end = min(start + chunk_size, len(sequence))
        chunks.append(sequence[start:end])
        if end == len(sequence):
            break
        start += (chunk_size - overlap)
    return chunks

def get_segment_embedding(segment_sequence, model, tokenizer, chunk_size=400, overlap=100):
    """Get DNABERT2 embedding for a single segment"""
    model.eval()
    
    with torch.no_grad():
        # If segment is short enough, process directly
        if len(segment_sequence) <= chunk_size:
            inputs = tokenizer(segment_sequence, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            
            if isinstance(outputs, tuple):
                last_hidden_state = outputs[0]
            elif hasattr(outputs, 'last_hidden_state'):
                last_hidden_state = outputs.last_hidden_state
            else:
                last_hidden_state = outputs[0] if hasattr(outputs, '__getitem__') else outputs
            
            return last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # For longer segments, use chunking
        chunks = chunk_sequence(segment_sequence, chunk_size, overlap)
        chunk_embeddings = []
        
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            
            if isinstance(outputs, tuple):
                last_hidden_state = outputs[0]
            elif hasattr(outputs, 'last_hidden_state'):
                last_hidden_state = outputs.last_hidden_state
            else:
                last_hidden_state = outputs[0] if hasattr(outputs, '__getitem__') else outputs
            
            chunk_embedding = last_hidden_state.mean(dim=1).squeeze().numpy()
            chunk_embeddings.append(chunk_embedding)
        
        # Average all chunk embeddings for this segment
        return np.mean(chunk_embeddings, axis=0)

def get_dnabert2_embeddings_segment_specific(all_segment_sequences, model, tokenizer):
    """Extract segment-specific embeddings from DNABERT2"""
    
    print(f"\nProcessing {len(all_segment_sequences)} genomes with segment-specific embeddings...")
    
    # Define expected segment order (you may need to adjust this based on your data)
    segment_names = ['PB2', 'PB1', 'PA', 'HA', 'NP', 'NA', 'MP', 'NS']
    
    all_genome_embeddings = []
    
    for i, segments in enumerate(all_segment_sequences):
        print(f"Processing genome {i+1}/{len(all_segment_sequences)}")
        
        if len(segments) != 8:
            print(f"  Warning: Genome has {len(segments)} segments (expected 8)")
        
        # Get embedding for each segment
        segment_embeddings = []
        for j, segment_seq in enumerate(segments):
            segment_name = segment_names[j] if j < len(segment_names) else f"Segment_{j+1}"
            print(f"  Processing {segment_name} (length: {len(segment_seq)} bp)")
            
            segment_emb = get_segment_embedding(segment_seq, model, tokenizer)
            segment_embeddings.append(segment_emb)
            print(f"    ✅ {segment_name} embedding shape: {segment_emb.shape}")
        
        # Concatenate all segment embeddings for this genome
        genome_embedding = np.concatenate(segment_embeddings)
        all_genome_embeddings.append(genome_embedding)
        print(f"  ✅ Complete genome embedding shape: {genome_embedding.shape}")
    
    return np.array(all_genome_embeddings)

print("\nSTEP 3: Creating segment-specific DNABERT2 embeddings...")
print("="*60)

# Test the model output format first
print("Testing model with a small sequence...")
test_sequence = "ATCGATCGATCG"
test_inputs = tokenizer(test_sequence, return_tensors="pt", padding=True, truncation=True, max_length=512)
with torch.no_grad():
    test_outputs = model(**test_inputs)
    print(f"✅ Model output type: {type(test_outputs)}")
    if isinstance(test_outputs, tuple):
        print(f"  Tuple length: {len(test_outputs)}")
        print(f"  First element shape: {test_outputs[0].shape}")
    elif hasattr(test_outputs, 'last_hidden_state'):
        print(f"  Last hidden state shape: {test_outputs.last_hidden_state.shape}")

# Extract embeddings using segment-specific approach
print("\nNow processing all sequences with segment-specific embeddings...")
embeddings = get_dnabert2_embeddings_segment_specific(all_sequences, model, tokenizer)
print(f"✅ All segment-specific embeddings created! Final shape: {embeddings.shape}")

def plot_clusters_simple(embeddings, labels, method='PCA'):
    """Plot 2D clusters without sequence names (simplified for 240 sequences)"""
    
    print(f"\nCreating {method} plot with {len(embeddings)} samples...")
    
    if method == 'PCA':
        reducer = PCA(n_components=2, random_state=42)
    elif method == 'tSNE':
        # Use safer parameters for t-SNE
        n_samples = len(embeddings)
        perplexity = min(30, max(5, n_samples // 4))
        
        print(f"Using t-SNE with perplexity={perplexity}")
        
        reducer = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=perplexity,
            n_iter=1000,
            learning_rate='auto',
            init='pca',
            n_jobs=1
        )
    
    try:
        # Reduce dimensions to 2D
        print(f"Fitting {method} reducer...")
        reduced_embeddings = reducer.fit_transform(embeddings)
        print(f"✅ {method} reduction completed successfully")
    except Exception as e:
        print(f"Error during {method} reduction: {e}")
        print("Falling back to PCA...")
        reducer = PCA(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        method = 'PCA (fallback)'
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot each class with different colors and markers
    colors = ['red', 'blue']
    markers = ['o', 's']
    
    for i, label in enumerate(['Reassortant', 'Non-Reassortant']):
        mask = np.array(labels) == label
        plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], 
                   label=f'{label} (n={np.sum(mask)})', alpha=0.7, s=60, 
                   c=colors[i], marker=markers[i])
    
    plt.xlabel(f'{method} Component 1', fontsize=12)
    plt.ylabel(f'{method} Component 2', fontsize=12)
    plt.title(f'Influenza Genome Clustering - {method}\nDNABERT2 Segment-Specific Embeddings', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'influenza_clustering_segment_specific_{method.lower().replace(" ", "_").replace("(", "").replace(")", "")}'
    plt.savefig(f'{plot_filename}.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(f'{plot_filename}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Plot saved as: {plot_filename}.pdf and {plot_filename}.png")
    plt.show()

print("\nSTEP 4: Creating clustering visualizations...")
print("="*60)

# PCA visualization
print("Creating PCA visualization...")
plot_clusters_simple(embeddings, labels, method='PCA')

# t-SNE visualization
print("\nCreating t-SNE visualization...")
try:
    plot_clusters_simple(embeddings, labels, method='tSNE')
except Exception as e:
    print(f"t-SNE failed with error: {e}")
    print("Skipping t-SNE visualization...")

print("\nSTEP 5: Dataset statistics...")
print("="*60)

# Calculate and display sequence length statistics
segment_lengths = []
for segments in all_sequences:
    for segment in segments:
        segment_lengths.append(len(segment))

print(f"Segment length statistics:")
print(f"  Mean: {np.mean(segment_lengths):.0f} bp")
print(f"  Median: {np.median(segment_lengths):.0f} bp")
print(f"  Std: {np.std(segment_lengths):.0f} bp")
print(f"  Range: {min(segment_lengths)} - {max(segment_lengths)} bp")

print(f"\nDataset composition:")
print(f"  Total sequences: {len(all_sequences)}")
print(f"  Reassortant sequences: {len(reassortant_sequences)}")
print(f"  Non-reassortant sequences: {len(nonreassortant_sequences)}")
print(f"  Embedding dimension: {embeddings.shape[1]} (8 segments × {embeddings.shape[1]//8} features each)")

print("\nSTEP 6: Saving embeddings for machine learning...")
print("="*60)

# Save the embeddings and metadata for the ML script
np.save('influenza_embeddings_segment_specific.npy', embeddings)
np.save('sequence_names.npy', np.array(all_names))
np.save('sequence_labels.npy', np.array(labels))

print("✅ Segment-specific embeddings saved successfully!")
print(f"  💾 influenza_embeddings_segment_specific.npy - Shape: {embeddings.shape}")
print(f"  💾 sequence_names.npy - {len(all_names)} sequence names")
print(f"  💾 sequence_labels.npy - {len(labels)} labels")

# Create a summary file for ML use
print("\nCreating summary file for ML analysis...")
with open('dataset_summary_segment_specific.txt', 'w') as f:
    f.write("INFLUENZA REASSORTMENT DATASET SUMMARY - SEGMENT-SPECIFIC EMBEDDINGS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Total samples: {len(all_sequences)}\n")
    f.write(f"Reassortant samples: {len(reassortant_sequences)}\n")
    f.write(f"Non-reassortant samples: {len(nonreassortant_sequences)}\n")
    f.write(f"Embedding dimension: {embeddings.shape[1]} (8 segments × {embeddings.shape[1]//8} features per segment)\n")
    f.write(f"Average segment length: {np.mean(segment_lengths):.0f} bp\n\n")
    
    f.write("SEGMENT-SPECIFIC APPROACH:\n")
    f.write("  - Each of 8 influenza segments processed separately\n")
    f.write("  - Individual segment embeddings concatenated\n")
    f.write("  - Preserves segment-level information for reassortment detection\n\n")
    
    f.write("CLASS DISTRIBUTION:\n")
    f.write(f"  Positive class (Reassortant): {len(reassortant_sequences)} samples\n")
    f.write(f"  Negative class (Non-Reassortant): {len(nonreassortant_sequences)} samples\n\n")
    
    f.write("FILES INCLUDED:\n")
    f.write("Reassortant files:\n")
    for i, name in enumerate(reassortant_names, 1):
        f.write(f"  {i:3d}. {name}\n")
    
    f.write("\nNon-reassortant files:\n")
    for i, name in enumerate(nonreassortant_names, 1):
        f.write(f"  {i:3d}. {name}\n")

print("✅ Summary saved to dataset_summary_segment_specific.txt")

print(f"\n{'='*60}")
print("SEGMENT-SPECIFIC EMBEDDING CREATION COMPLETED!")
print(f"{'='*60}")
print("Next steps:")
print("1. ✅ DNABERT2 segment-specific embeddings created")
print("2. ✅ PCA and t-SNE visualizations generated") 
print("3. ✅ Data saved for machine learning")
print("4. ➜ Now you can run the Random Forest classification script!")
print("5. ➜ Compare results with whole genome approach")
print(f"{'='*60}")

# Optional: Print first few sample names to verify correct loading
print("\nSample verification (first 5 from each class):")
print("Reassortant samples:")
for i, name in enumerate(reassortant_names[:5]):
    print(f"  {i+1}. {name}")
if len(reassortant_names) > 5:
    print(f"  ... and {len(reassortant_names)-5} more")

print("\nNon-reassortant samples:")
for i, name in enumerate(nonreassortant_names[:5]):
    print(f"  {i+1}. {name}")
if len(nonreassortant_names) > 5:
    print(f"  ... and {len(nonreassortant_names)-5} more")