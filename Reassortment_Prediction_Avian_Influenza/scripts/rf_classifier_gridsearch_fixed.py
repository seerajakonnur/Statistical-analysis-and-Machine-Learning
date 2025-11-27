
# Random Forest Classifier with Grid Search for DNABERT2 Embeddings
# Optimized for 240 training samples to prevent overfitting

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading DNABERT2 embeddings...")

# Load training data
train_embeddings = np.load('influenza_embeddings_segment_specific.npy')
train_labels = np.load('sequence_labels.npy')
train_names = np.load('sequence_names.npy')

# Load test data
test_embeddings = np.load('influenza_test_embeddings_segment_specific.npy')
test_labels = np.load('test_sequence_labels.npy')
test_names = np.load('test_sequence_names.npy')

print(f"Training data: {train_embeddings.shape}")
print(f"Test data: {test_embeddings.shape}")

# Grid search parameters - optimized for 240 samples
param_grid = {
    'n_estimators': [200, 300, 400, 500],           # More trees help with small datasets
    'max_depth': [5, 8, 10],              # Prevent overfitting
    'min_samples_split': [5, 10, 15],        # Higher values prevent overfitting
    'min_samples_leaf': [5, 8, 10],           # Ensure meaningful leaf nodes
    'max_features': ['sqrt', 'log2']     # Feature subsampling
}

print("\nPerforming Grid Search with Cross-Validation...")
print("This may take a few minutes...")

# Stratified K-Fold for small dataset
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search with cross-validation
rf_base = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',  # Handle class imbalance
    n_jobs=-1,               # Use all cores
    bootstrap=True,          # Bootstrap sampling
    oob_score=True          # Out-of-bag score for overfitting check
)

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    return_train_score=True  # To check for overfitting
)

# Fit grid search
grid_search.fit(train_embeddings, train_labels)

# Get best model
best_rf = grid_search.best_estimator_

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
print(f"OOB Score: {best_rf.oob_score_:.3f}")

# Check for overfitting
results_df = pd.DataFrame(grid_search.cv_results_)
best_idx = grid_search.best_index_
train_score = results_df.loc[best_idx, 'mean_train_score']
val_score = results_df.loc[best_idx, 'mean_test_score']
overfitting_gap = train_score - val_score

print(f"\nOverfitting Analysis:")
print(f"Mean training score: {train_score:.3f}")
print(f"Mean validation score: {val_score:.3f}")
print(f"Overfitting gap: {overfitting_gap:.3f}")

if overfitting_gap > 0.05:
    print("⚠️  WARNING: Potential overfitting detected!")
else:
    print("✅ Good generalization - low overfitting risk")

# Final cross-validation on best model
print("\nFinal Cross-Validation on Best Model:")
final_cv_scores = cross_val_score(best_rf, train_embeddings, train_labels, 
                                 cv=cv_strategy, scoring='accuracy')
print(f"CV Accuracy: {final_cv_scores.mean():.3f} (+/- {final_cv_scores.std() * 2:.3f})")

# Train final model and predict
print("\nTraining final model and evaluating on test set...")
best_rf.fit(train_embeddings, train_labels)

# Test set predictions
test_predictions = best_rf.predict(test_embeddings)
test_probabilities = best_rf.predict_proba(test_embeddings)

# ===== ENHANCED: Display prediction probabilities for each test sample =====
print("\n" + "="*60)
print("PREDICTION PROBABILITIES FOR EACH TEST SAMPLE")
print("="*60)

# Create detailed predictions DataFrame
detailed_predictions = pd.DataFrame({
    'Sample_ID': range(1, len(test_names) + 1),
    'Sequence_Name': test_names,
    'True_Label': test_labels,
    'Predicted_Label': test_predictions,
    'Prob_Non_Reassortant': test_probabilities[:, 0],
    'Prob_Reassortant': test_probabilities[:, 1],
    'Confidence': np.max(test_probabilities, axis=1),
    'Correct_Prediction': test_labels == test_predictions
})

# Display all predictions
print(detailed_predictions.to_string(index=False))

# Summary statistics for probabilities
print("\n" + "="*50)
print("PREDICTION PROBABILITY STATISTICS")
print("="*50)
print(f"Average confidence: {detailed_predictions['Confidence'].mean():.3f}")
print(f"Minimum confidence: {detailed_predictions['Confidence'].min():.3f}")
print(f"Maximum confidence: {detailed_predictions['Confidence'].max():.3f}")
print(f"Standard deviation of confidence: {detailed_predictions['Confidence'].std():.3f}")

# Identify uncertain predictions (low confidence)
uncertain_threshold = 0.6
uncertain_predictions = detailed_predictions[detailed_predictions['Confidence'] < uncertain_threshold]
if len(uncertain_predictions) > 0:
    print(f"\n⚠️  {len(uncertain_predictions)} predictions with confidence < {uncertain_threshold}:")
    print(uncertain_predictions[['Sample_ID', 'Sequence_Name', 'Predicted_Label', 'Confidence']].to_string(index=False))
else:
    print(f"\n✅ All predictions have confidence >= {uncertain_threshold}")

# Calculate metrics
test_accuracy = accuracy_score(test_labels, test_predictions)

# FIX: Use pos_label parameter for string labels in ROC calculation
#test_auc = roc_auc_score(test_labels, test_probabilities[:, 1], pos_label='Reassortant')

print(f"\n=== FINAL RESULTS ===")
print(f"Test Accuracy: {test_accuracy:.3f}")
#print(f"Test AUC: {test_auc:.3f}")
print(f"\nDetailed Classification Report:")
print(classification_report(test_labels, test_predictions, 
                          target_names=['Non-Reassortant', 'Reassortant']))

# Feature importance analysis
feature_importance = best_rf.feature_importances_
top_features_idx = np.argsort(feature_importance)[-20:]  # Top 20 features

plt.figure(figsize=(15, 12))
plt.subplot(2, 3, 1)
sns.barplot(x=feature_importance[top_features_idx], y=range(len(top_features_idx)))
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance Score')

# Confusion Matrix
plt.subplot(2, 3, 2)
cm = confusion_matrix(test_labels, test_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Non-Reassortant', 'Reassortant'],
           yticklabels=['Non-Reassortant', 'Reassortant'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')


# Precision-Recall Curve
#plt.subplot(2, 3, 4)
#precision, recall, _ = precision_recall_curve(test_labels, test_probabilities[:, 1], pos_label='Reassortant')
#plt.plot(recall, precision, 'g-', label='PR Curve')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Precision-Recall Curve')
#plt.legend()

# Prediction Confidence Distribution
plt.subplot(2, 3, 5)
plt.hist(detailed_predictions['Confidence'], bins=20, alpha=0.7, edgecolor='black')
plt.axvline(detailed_predictions['Confidence'].mean(), color='red', linestyle='--', 
           label=f'Mean: {detailed_predictions["Confidence"].mean():.3f}')
plt.xlabel('Prediction Confidence')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Confidence')
plt.legend()

# Probability Distribution by True Label
plt.subplot(2, 3, 6)
reassortant_probs = detailed_predictions[detailed_predictions['True_Label'] == 'Reassortant']['Prob_Reassortant']
non_reassortant_probs = detailed_predictions[detailed_predictions['True_Label'] == 'Non-Reassortant']['Prob_Reassortant']

plt.hist(reassortant_probs, bins=15, alpha=0.7, label='True Reassortant', color='red')
plt.hist(non_reassortant_probs, bins=15, alpha=0.7, label='True Non-Reassortant', color='blue')
plt.xlabel('Probability of Reassortant')
plt.ylabel('Frequency')
plt.title('Probability Distribution by True Label')
plt.legend()

plt.tight_layout()
plt.savefig('rf_results_with_gridsearch_enhanced.png', dpi=300, bbox_inches='tight')
plt.show()

# Save results
print("\nSaving results...")

# Save best model
joblib.dump(best_rf, 'rf_model_optimized.pkl')

# Save detailed predictions (enhanced)
detailed_predictions.to_csv('test_predictions_detailed.csv', index=False)

# Save parameter search results
grid_results = pd.DataFrame(grid_search.cv_results_)
grid_results.to_csv('grid_search_results.csv', index=False)

# Save probability analysis
#prob_summary = pd.DataFrame({
#    'Metric': ['Mean Confidence', 'Min Confidence', 'Max Confidence', 'Std Confidence', 
#               'Uncertain Predictions (< 0.6)', 'Test Accuracy'],
#    'Value': [detailed_predictions['Confidence'].mean(),
#              detailed_predictions['Confidence'].min(),
#              detailed_predictions['Confidence'].max(),
#              detailed_predictions['Confidence'].std(),
#              len(uncertain_predictions),
#              test_accuracy,
#              #test_auc]
#})
#prob_summary.to_csv('probability_analysis_summary.csv', index=False)

print("\n✅ Analysis complete!")
print("Files saved:")
print("- rf_model_optimized.pkl (trained model)")
print("- test_predictions_detailed.csv (detailed predictions with probabilities)")
print("- grid_search_results.csv (grid search details)")
print("- probability_analysis_summary.csv (probability statistics)")
print("- rf_results_with_gridsearch_enhanced.png (enhanced visualization)")

print(f"\n📊 Summary:")
print(f"Best model achieves {test_accuracy:.1%} accuracy on test set")
print(f"Average prediction confidence: {detailed_predictions['Confidence'].mean():.1%}")
print(f"Overfitting gap: {overfitting_gap:.3f} ({'LOW' if overfitting_gap < 0.05 else 'HIGH'})")

