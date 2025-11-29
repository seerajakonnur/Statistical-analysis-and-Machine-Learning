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
This work-in-progress presents a 3-stage computational pipeline designed to identify and assess
influenza virus reassortment potential from environmental surveillance data. 

### * Feature extraction using foundation model
Genomic sequences (influenza RNA sequences across all 8 segments) are processed segment-wise through DNABERT2 to generate embeddings. Embeddings are created for each segment because reassortment occurs at the segment level, where whole genome segments are exchanged between two influenza viruses.

### * Machine learning classifier
The sequences are classified into reassortant and non-reassortant categories using a Random Forest classifier trained on the DNABERT2-derived embeddings.

### * Genetic algorithm based candidate search
Influenza virus reassortment is not entirely random; it is shaped by factors such as host species, viral subtypes, and compatible segment combinations. This component of the project is still under development and requires further refinement before a full-scale genetic algorithm can be implemented.
The goal is to simulate reassortment by generating multiple potential reassortant genomes and evaluating them using the trained classifier to determine whether they qualify as reassortants or non-reassortants. For newly collected environmental sequencing samples, the workflow would first classify whether the samples are reassortant or non-reassortant. Next, for those identified as non-reassortants, potential reassortant combinations will be computationally generated and passed through the classifier again to identify viable reassortant candidates.

## Data
