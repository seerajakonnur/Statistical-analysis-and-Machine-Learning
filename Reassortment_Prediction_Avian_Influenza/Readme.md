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
For generating the initial results, H5N1 clade 2.3.4.4b sequences from the United States (2021–2022)—a period marked by major reassortment events were used. This dataset includes both non-reassortant and reassortant genotypes circulating during this time. Specifically, it contains non-reassortant genotypes such as A1, A2, and A3, along with reassortant genotypes including B1.1, B1.2, B2, B3.1, B3.2, B4, B5, and several minor reassortants that represent subsets of these major groups.
For model development, 120 non-reassortant sequences from genotype A1 were selected as the negative training set, and 119 reassortant sequences spanning genotypes B1.1, B1.2, B2, B3.1, B3.2, B4, and B5 were used as the positive training set, preserving proportional representation across classes. To evaluate model generalization on unseen data, non-reassortant genotypes A2 and A3 (25 sequences combined) were reserved exclusively for testing, while 30 minor reassortants served as the reassortant test set. 
