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
from Bio import SeqIO
import glob
import warnings
warnings.filterwarnings("ignore")

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

print("\nSTEP 2: Reading TEST sequences from folder structure...")
print("="*60)

# Read test sequences from the two test folders
print("Reading test non-reassortant sequences from 'test_nonreassortant' folder...")
test_nonreassortant_sequences, test_nonreassortant_names = read_fasta_files_recursive("test_nonreassortant")

print("\nReading test reassortant sequences from 'test_reassortant' folder...")
test_reassortant_sequences, test_reassortant_names = read_fasta_files_recursive("test_reassortant")

# Combine all test data
all_test_sequences = test_nonreassortant_sequences + test_reassortant_sequences
all_test_names = test_nonreassortant_names + test_reassortant_names
test_labels = ['Non-Reassortant'] * len(test_nonreassortant_sequences) + ['Reassortant'] * len(test_reassortant_sequences)

print(f"\n✅ Test data loading complete:")
print(f"  Test Non-reassortant sequences: {len(test_nonreassortant_sequences)}")
print(f"  Test Reassortant sequences: {len(test_reassortant_sequences)}")
print(f"  Total test sequences: {len(all_test_sequences)}")

print("\nSTEP 3: Creating segment-specific DNABERT2 embeddings for TEST data...")
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

# Extract embeddings using segment-specific approach for test data
print("\nNow processing all TEST sequences with segment-specific embeddings...")
test_embeddings = get_dnabert2_embeddings_segment_specific(all_test_sequences, model, tokenizer)

print(f"\n✅ Test embeddings extraction complete!")
print(f"  Shape: {test_embeddings.shape}")
print(f"  Sequences processed: {len(all_test_sequences)}")

print("\nSTEP 4: Test dataset statistics...")
print("="*60)

# Calculate and display test sequence length statistics
test_segment_lengths = []
for segments in all_test_sequences:
    for segment in segments:
        test_segment_lengths.append(len(segment))

print(f"Test segment length statistics:")
print(f"  Mean: {np.mean(test_segment_lengths):.0f} bp")
print(f"  Median: {np.median(test_segment_lengths):.0f} bp")
print(f"  Std: {np.std(test_segment_lengths):.0f} bp")
print(f"  Range: {min(test_segment_lengths)} - {max(test_segment_lengths)} bp")

print(f"\nTest dataset composition:")
print(f"  Total test sequences: {len(all_test_sequences)}")
print(f"  Test Non-reassortant sequences: {len(test_nonreassortant_sequences)}")
print(f"  Test Reassortant sequences: {len(test_reassortant_sequences)}")
print(f"  Embedding dimension: {test_embeddings.shape[1]} (8 segments × {test_embeddings.shape[1]//8} features each)")

print("\nSTEP 5: Saving TEST embeddings for machine learning...")
print("="*60)

# Save the test embeddings and metadata for the ML script
np.save('influenza_test_embeddings_segment_specific.npy', test_embeddings)
np.save('test_sequence_names.npy', np.array(all_test_names))
np.save('test_sequence_labels.npy', np.array(test_labels))

print("✅ Test segment-specific embeddings saved successfully!")
print(f"  💾 influenza_test_embeddings_segment_specific.npy - Shape: {test_embeddings.shape}")
print(f"  💾 test_sequence_names.npy - {len(all_test_names)} sequence names")
print(f"  💾 test_sequence_labels.npy - {len(test_labels)} labels")

# Create a summary file for ML use
summary_data = {
    'test_embeddings_file': 'influenza_test_embeddings_segment_specific.npy',
    'test_names_file': 'test_sequence_names.npy', 
    'test_labels_file': 'test_sequence_labels.npy',
    'num_test_samples': len(all_test_sequences),
    'embedding_dimension': test_embeddings.shape[1],
    'num_test_nonreassortant': len(test_nonreassortant_sequences),
    'num_test_reassortant': len(test_reassortant_sequences),
    'model_used': model_name,
    'segments_per_genome': 8,
    'segment_names': ['PB2', 'PB1', 'PA', 'HA', 'NP', 'NA', 'MP', 'NS']
}

import json
with open('test_embeddings_summary.json', 'w') as f:
    json.dump(summary_data, f, indent=2)

print("\n" + "="*60)
print("🎉 TEST EMBEDDING PIPELINE COMPLETED SUCCESSFULLY! 🎉")
print("="*60)
print("1. ✅ DNABERT2 segment-specific embeddings created for TEST data")
print("2. ✅ Test data saved for machine learning")
print("3. ➜ Now you can use these embeddings with your trained ML model for prediction!")
print("4. ➜ Files ready: influenza_test_embeddings_segment_specific.npy")
print(f"{'='*60}")

# Optional: Print first few test sample names to verify correct loading
print("\nTest sample verification (first 5 from each class):")
print("Test Non-reassortant samples:")
for i, name in enumerate(test_nonreassortant_names[:5]):
    print(f"  {i+1}. {name}")
if len(test_nonreassortant_names) > 5:
    print(f"  ... and {len(test_nonreassortant_names)-5} more")

print("\nTest Reassortant samples:")
for i, name in enumerate(test_reassortant_names[:5]):
    print(f"  {i+1}. {name}")
if len(test_reassortant_names) > 5:
    print(f"  ... and {len(test_reassortant_names)-5} more")
