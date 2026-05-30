import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
import os

def identify_incorrect_classifications(df_actual, y_test, y_pred, algorithm_name, y_pred_proba=None):
    results_df = pd.DataFrame({
        'Tweet': df_actual['Tweet'],
        'Actual Label': y_test,
        'Predicted Label': y_pred
    })
    results_df['Incorrectly Classified'] = results_df['Actual Label'] != results_df['Predicted Label']
    incorrect_predictions = results_df[results_df['Incorrectly Classified']]
    
    # Save with algorithm name in filename
    filename = f'incorrectly_classified_rows_{algorithm_name}.csv'
    incorrect_predictions.to_csv(filename, index=False)
    print(f"Incorrectly classified rows saved to '{filename}'")
    
    return incorrect_predictions, results_df

def class_wise_misclassification(incorrect_predictions, algorithm_name):
    class_misclassifications = incorrect_predictions.groupby(['Actual Label', 'Predicted Label']).size().reset_index(name='Count')
    
    # Save with algorithm name in filename
    filename = f'class_wise_misclassifications_{algorithm_name}.csv'
    class_misclassifications.to_csv(filename, index=False)
    print(f"Class-wise misclassification counts saved to '{filename}'")
    
    return class_misclassifications

def plot_confusion_matrix(y_test, y_pred, labels, algorithm_name):
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(f"Confusion Matrix with Misclassifications - {algorithm_name}")
    
    # Save with algorithm name in filename
    filename = f"confusion_matrix_{algorithm_name}.png"
    plt.savefig(filename)
    plt.show()
    print(f"Confusion matrix saved as '{filename}'")

def most_common_misclassified_texts(incorrect_predictions, algorithm_name):
    most_common_misclassifications = incorrect_predictions['Tweet'].value_counts().reset_index()
    most_common_misclassifications.columns = ['Tweet', 'Misclassification Count']
    
    # Save with algorithm name in filename
    filename = f'most_common_misclassified_texts_{algorithm_name}.csv'
    most_common_misclassifications.to_csv(filename, index=False)
    print(f"Most commonly misclassified texts saved to '{filename}'")
    
    return most_common_misclassifications

def misclassification_by_text_length(results_df, algorithm_name):
    results_df['Text Length'] = results_df['Tweet'].apply(len)
    length_analysis = results_df.groupby(['Incorrectly Classified'])['Text Length'].describe()
    
    # Save with algorithm name in filename
    filename = f'misclassification_by_text_length_{algorithm_name}.csv'
    length_analysis.to_csv(filename)
    print(f"Misclassification by text length saved to '{filename}'")
    
    return length_analysis

def common_misclassified_words(incorrect_predictions, algorithm_name):
    all_misclassified_texts = " ".join(incorrect_predictions['Tweet'])
    misclassified_words = Counter(all_misclassified_texts.split())
    most_common_misclassified_words = pd.DataFrame(misclassified_words.most_common(20), columns=['Word', 'Frequency'])
    
    # Save with algorithm name in filename
    filename = f'common_misclassified_words_{algorithm_name}.csv'
    most_common_misclassified_words.to_csv(filename, index=False)
    print(f"Common misclassified words saved to '{filename}'")
    
    return most_common_misclassified_words

# def confidence_analysis_misclassifications(incorrect_predictions, y_pred_proba, algorithm_name):
#     if y_pred_proba is not None:
#         #incorrect_predictions['Confidence'] = incorrect_predictions.apply(
#         #    lambda row: y_pred_proba[row.name, row['Predicted Label']], axis=1)
#         #incorrect_predictions['Confidence'] = incorrect_predictions.apply(
#         #    lambda row: y_pred_proba[row.Index, row['Predicted Label']], axis=1)
#         incorrect_predictions['Confidence'] = incorrect_predictions.apply(
#             lambda row: y_pred_proba[row.name, row['Predicted Label']], axis=1)
#         confidence_analysis = incorrect_predictions[['Tweet', 'Actual Label', 'Predicted Label', 'Confidence']]
        
#         # Save with algorithm name in filename
#         filename = f'confidence_analysis_misclassifications_{algorithm_name}.csv'
#         confidence_analysis.to_csv(filename, index=False)
#         print(f"Confidence analysis for misclassifications saved to '{filename}'")
        
#         return confidence_analysis
#     else:
#         print("Confidence analysis skipped because y_pred_proba is not provided.")
#         return None

def confidence_analysis_misclassifications(incorrect_predictions, y_pred_proba, algorithm_name, label_encoder):
    if y_pred_proba is not None:
        # Encode 'Predicted Label' using the label encoder to align with y_pred_proba indices
        # incorrect_predictions['Predicted Label Encoded'] = label_encoder.transform(incorrect_predictions['Predicted Label'])

        # # Assign confidence scores based on the encoded labels
        # incorrect_predictions['Confidence'] = incorrect_predictions.apply(
        #     lambda row: y_pred_proba[row.name, row['Predicted Label Encoded']], axis=1
        # )

        #incorrect_predictions.loc[:, 'Predicted Label Encoded'] = label_encoder.transform(incorrect_predictions['Predicted Label'])
        #incorrect_predictions.loc[:, 'Confidence'] = incorrect_predictions.apply(
        #    lambda row: y_pred_proba[row.name, row['Predicted Label Encoded']], axis=1
        
        # Ensure you're using `.loc` to set values
        # incorrect_predictions.loc[:, 'Predicted Label Encoded'] = label_encoder.transform(incorrect_predictions['Predicted Label'])
        # incorrect_predictions.loc[:, 'Confidence'] = incorrect_predictions.apply(
        #     lambda row: y_pred_proba[row.name, row['Predicted Label Encoded']], axis=1)
        

        # Ensure incorrect_predictions is a copy of the DataFrame slice
        incorrect_predictions = incorrect_predictions.copy()

        # Use .loc to set values explicitly
        incorrect_predictions.loc[:, 'Predicted Label Encoded'] = label_encoder.transform(incorrect_predictions['Predicted Label'])
        incorrect_predictions.loc[:, 'Confidence'] = incorrect_predictions.apply(
            lambda row: y_pred_proba[row.name, row['Predicted Label Encoded']], axis=1
        )


        confidence_analysis = incorrect_predictions[['Tweet', 'Actual Label', 'Predicted Label', 'Confidence']]
        
        # Save with algorithm name in filename
        filename = f'confidence_analysis_misclassifications_{algorithm_name}.csv'
        confidence_analysis.to_csv(filename, index=False)
        print(f"Confidence analysis for misclassifications saved to '{filename}'")
        
        return confidence_analysis
    else:
        print("Confidence analysis skipped because y_pred_proba is not provided.")
        return None

def perform_misclassification_analysis(df_actual, y_test, y_pred, algorithm_name, y_pred_proba=None, labels=None, label_encoder=None):
    """
    Performs a comprehensive misclassification analysis by calling individual analysis functions.
    """
    print(f"Starting misclassification analysis for {algorithm_name}...")

    # Identify incorrect classifications
    incorrect_predictions, results_df = identify_incorrect_classifications(df_actual, y_test, y_pred, algorithm_name, y_pred_proba)
    
    # Class-wise misclassification analysis
    class_wise_misclassification(incorrect_predictions, algorithm_name)
    
    # Plot confusion matrix if labels are provided
    if labels is not None and len(labels) > 0:
        plot_confusion_matrix(y_test, y_pred, labels, algorithm_name)
    else:
        print("Skipping confusion matrix plot as 'labels' parameter is missing.")
    
    # Most common misclassified texts
    most_common_misclassified_texts(incorrect_predictions, algorithm_name)
    
    # Misclassification analysis by text length
    misclassification_by_text_length(results_df, algorithm_name)
    
    # Common misclassified words
    common_misclassified_words(incorrect_predictions, algorithm_name)
    
    # Confidence analysis if probability predictions are provided
    if label_encoder is not None:
        confidence_analysis_misclassifications(incorrect_predictions, y_pred_proba, algorithm_name, label_encoder)
    else:
        print("Skipping confidence analysis as 'label_encoder' is missing.")

    print(f"Misclassification analysis completed for {algorithm_name}.")


# def perform_misclassification_analysis(df_actual, y_test, y_pred, algorithm_name, y_pred_proba=None, labels=None):
#     """
#     Performs a comprehensive misclassification analysis by calling individual analysis functions.
#     """
#     print(f"Starting misclassification analysis for {algorithm_name}...")

#     # Identify incorrect classifications
#     incorrect_predictions, results_df = identify_incorrect_classifications(df_actual, y_test, y_pred, algorithm_name, y_pred_proba)
    
#     # Class-wise misclassification analysis
#     class_wise_misclassification(incorrect_predictions, algorithm_name)
    
#     # Plot confusion matrix if labels are provided
#     #if labels:
#     if labels is not None and len(labels) > 0:
#         plot_confusion_matrix(y_test, y_pred, labels, algorithm_name)
#     else:
#         print("Skipping confusion matrix plot as 'labels' parameter is missing.")
    
#     # Most common misclassified texts
#     most_common_misclassified_texts(incorrect_predictions, algorithm_name)
    
#     # Misclassification analysis by text length
#     misclassification_by_text_length(results_df, algorithm_name)
    
#     # Common misclassified words
#     common_misclassified_words(incorrect_predictions, algorithm_name)
    
#     # Confidence analysis if probability predictions are provided
#     confidence_analysis_misclassifications(incorrect_predictions, y_pred_proba, algorithm_name)

#     print(f"Misclassification analysis completed for {algorithm_name}.")
