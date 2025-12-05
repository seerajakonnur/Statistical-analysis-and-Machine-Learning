## DNABERT2 (Foundation Model)
t-SNE was applied to the DNABERT2-derived embeddings (from training sequences) to project the high-dimensional sequence representations into two dimensions. The resulting visualization showed clear separation between reassortant and non-reassortant sequences, demonstrating that DNABERT2 successfully captures reassortment-related genomic features even without task-specific fine-tuning.

![t-SNE visualization](../assets/influenza_clustering_segment_specific_tsne.png)

## ML Classifier
The classifier correctly identified all 30 reassortant sequences and 25 non-reassortant sequences in the test set, demonstrating strong generalization to previously unseen mixed genotypes. Why 100% accuracy is expected here and not Overfitting: The segment-specific approach is particularly well-suited for reassortment detection because influenza reassortment occurs through exchange of entire intact segments rather than point mutations, creating categorical genomic signatures. During model development, cross-validation analysis on training data showed zero overfitting gap (mean training score: 100.0%, mean validation score: 100.0%, gap: 0.0%), indicating excellent generalization capacity. This perfect generalization was confirmed on 55 completely unseen test samples from different genotypes, achieving 100.0% accuracy with high prediction confidence (mean = 0.96, all probabilities > 0.7). Statistical hypothesis
testing indicates this performance is highly unlikely by chance (p = 0.006, binomial exact test).
Since reassortment involves complete segment exchanges between North American and Eurasian
genotypes, each segment retains its distinct evolutionary signature, enabling clear discrimination
between reassorted and non-reassorted genomes. The combination of zero overfitting gap during
training, uniformly high test prediction confidence, cross-genotype validation, and biological mecha-
nism alignment demonstrates that segment-specific DNABERT-2 embeddings capture fundamental
reassortment signatures.
![t-SNE visualization](../assets/prediction_probabilities.png)

## Genetic Algorithm

