# Predicting Reassortment potential in Influenza A virus using foundation models (DNABERT2) and genetic algorithms
🚧 **Project Status:** Under active development.  
Expect frequent updates and changes.
![Alt text](assets/to_use_reassortment_image.png)

## Background and Objective
Influenza A virus (IAV) poses a persistent global threat due to its ability to evolve
rapidly through reassortment. This project presents a computational framework that inte-
grates DNABERT-2, a transformer-based foundation model for genomic sequences,
with machine learning and genetic algorithms to predict reassortment events.

## Concept and Framework
This work-in-progress presents a four-stage computational pipeline designed to identify and assess
influenza virus reassortment potential from environmental surveillance data. 

### * Feature extraction using foundation model
Genomic sequences (influenza RNA sequences across all 8 segments) are processed segment-wise through DNABERT2 to generate embeddings. Embeddings are created for each segment because reassortment occurs at the segment level, where whole genome segments are exchanged between two influenza viruses.
