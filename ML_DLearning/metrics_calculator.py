import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score, matthews_corrcoef, confusion_matrix,
    precision_recall_fscore_support, average_precision_score,
    log_loss, fbeta_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

# def calculate_metrics(y_true, y_pred, y_pred_proba=None, classes=None):
#     metrics = {}
    
#     # Encode y_true and y_pred if they are not numeric
#     label_encoder = LabelEncoder()
#     y_true_encoded = label_encoder.fit_transform(y_true)
#     y_pred_encoded = label_encoder.transform(y_pred)
#     encoded_classes = label_encoder.classes_ if classes is None else classes
    
#     # Basic Metrics
#     metrics['accuracy'] = accuracy_score(y_true_encoded, y_pred_encoded)
#     metrics['f1_weighted'] = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
#     metrics['precision_weighted'] = precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
#     metrics['recall_weighted'] = recall_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)

#     # Additional Metrics
#     metrics['cohen_kappa'] = cohen_kappa_score(y_true_encoded, y_pred_encoded)
#     metrics['mcc'] = matthews_corrcoef(y_true_encoded, y_pred_encoded)

#     # PR AUC and ROC AUC for multi-class if probabilities are provided
#     if y_pred_proba is not None and len(encoded_classes) > 2:
#         # Calculate PR AUC for each class and average
#         pr_auc_scores = []
#         for i in range(len(encoded_classes)):
#             pr_auc = average_precision_score((y_true_encoded == i).astype(int), y_pred_proba[:, i])
#             pr_auc_scores.append(pr_auc)
#         metrics['pr_auc'] = np.mean(pr_auc_scores)  # Weighted average PR AUC
        
#         metrics['roc_auc'] = roc_auc_score(y_true_encoded, y_pred_proba, multi_class="ovr", average="weighted")
#         metrics['log_loss'] = log_loss(y_true_encoded, y_pred_proba)
#     elif y_pred_proba is not None:
#         # Binary case
#         metrics['pr_auc'] = average_precision_score(y_true_encoded, y_pred_proba[:, 1])
#         metrics['roc_auc'] = roc_auc_score(y_true_encoded, y_pred_proba[:, 1])
#         metrics['log_loss'] = log_loss(y_true_encoded, y_pred_proba)
#     else:
#         # Fallback when probabilities are not provided
#         metrics['pr_auc'] = average_precision_score(y_true_encoded, y_pred_encoded, average="weighted", zero_division=0)

#     # F2 Score if recall is prioritized
#     metrics['f2_score'] = fbeta_score(y_true_encoded, y_pred_encoded, beta=2, average='weighted', zero_division=0)

#     # Class-wise Metrics
#     precision, recall, fscore, _ = precision_recall_fscore_support(y_true_encoded, y_pred_encoded, average=None, labels=range(len(encoded_classes)), zero_division=0)
#     metrics['class_precision'] = precision
#     metrics['class_recall'] = recall
#     metrics['class_fscore'] = fscore

#     # Display Metrics
#     print("\n--- Model Metrics ---")
#     for metric, value in metrics.items():
#         print(f"{metric}: {value}")
#     print("\nClass-wise Precision, Recall, F1:")
#     for i, label in enumerate(encoded_classes):
#         print(f"Class: {label}")
#         print(f"Precision: {precision[i]}, Recall: {recall[i]}, F1-Score: {fscore[i]}")
#         print()

#     # Confusion Matrix
#     cm = confusion_matrix(y_true_encoded, y_pred_encoded, normalize='true')
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=encoded_classes, yticklabels=encoded_classes)
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title('Normalized Confusion Matrix')
#     plt.show()

#     return metrics

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score, matthews_corrcoef, confusion_matrix,
    precision_recall_fscore_support, average_precision_score,
    log_loss, fbeta_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

# def calculate_metrics(y_true, y_pred, y_pred_proba=None, classes=None):
#     metrics = {}
    
#     # Encode y_true and y_pred if they are not numeric
#     label_encoder = LabelEncoder()
#     y_true_encoded = label_encoder.fit_transform(y_true)
#     y_pred_encoded = label_encoder.transform(y_pred)
#     encoded_classes = label_encoder.classes_ if classes is None else classes
    
#     # Basic Metrics
#     metrics['accuracy'] = accuracy_score(y_true_encoded, y_pred_encoded)
#     metrics['f1_weighted'] = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
#     metrics['precision_weighted'] = precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
#     metrics['recall_weighted'] = recall_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)

#     # Additional Metrics
#     metrics['cohen_kappa'] = cohen_kappa_score(y_true_encoded, y_pred_encoded)
#     metrics['mcc'] = matthews_corrcoef(y_true_encoded, y_pred_encoded)

#     # PR AUC and ROC AUC for multi-class if probabilities are provided
#     if y_pred_proba is not None and len(encoded_classes) > 2:
#         pr_auc_scores = [average_precision_score((y_true_encoded == i).astype(int), y_pred_proba[:, i]) for i in range(len(encoded_classes))]
#         metrics['pr_auc'] = np.mean(pr_auc_scores)  # Weighted average PR AUC
        
#         metrics['roc_auc'] = roc_auc_score(y_true_encoded, y_pred_proba, multi_class="ovr", average="weighted")
#         metrics['log_loss'] = log_loss(y_true_encoded, y_pred_proba)
#     elif y_pred_proba is not None:
#         metrics['pr_auc'] = average_precision_score(y_true_encoded, y_pred_proba[:, 1])
#         metrics['roc_auc'] = roc_auc_score(y_true_encoded, y_pred_proba[:, 1])
#         metrics['log_loss'] = log_loss(y_true_encoded, y_pred_proba)
#     else:
#         metrics['pr_auc'] = average_precision_score(y_true_encoded, y_pred_encoded, average="weighted", zero_division=0)

#     # F2 Score if recall is prioritized
#     metrics['f2_score'] = fbeta_score(y_true_encoded, y_pred_encoded, beta=2, average='weighted', zero_division=0)

#     # Class-wise Metrics
#     precision, recall, fscore, _ = precision_recall_fscore_support(
#         y_true_encoded, y_pred_encoded, average=None, labels=range(len(encoded_classes)), zero_division=0
#     )
#     metrics['class_precision'] = precision
#     metrics['class_recall'] = recall
#     metrics['class_fscore'] = fscore

#     # Display Metrics
#     print("\n--- Model Metrics ---")
#     for metric, value in metrics.items():
#         print(f"{metric}: {value}")

#     # Class-wise Precision, Recall, F1 table
#     classwise_df = pd.DataFrame({
#         'Class': encoded_classes,
#         'Precision': precision,
#         'Recall': recall,
#         'F1-Score': fscore
#     })
    
#     print("\nClass-wise Precision, Recall, and F1-Score:")
#     print(classwise_df.to_string(index=False))

#     # Confusion Matrix
#     cm = confusion_matrix(y_true_encoded, y_pred_encoded, normalize='true')
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=encoded_classes, yticklabels=encoded_classes)
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title('Normalized Confusion Matrix')
#     plt.show()

#     return metrics


# def calculate_metrics(y_true, y_pred, y_pred_proba=None, classes=None):
#     metrics = {}
    
#     # Encode y_true and y_pred if they are not numeric
#     label_encoder = LabelEncoder()
#     y_true_encoded = label_encoder.fit_transform(y_true)
#     y_pred_encoded = label_encoder.transform(y_pred)
#     encoded_classes = label_encoder.classes_ if classes is None else classes
    
#     # Basic Metrics
#     metrics['accuracy'] = accuracy_score(y_true_encoded, y_pred_encoded)
#     metrics['f1_weighted'] = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=1)
#     metrics['precision_weighted'] = precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=1)
#     metrics['recall_weighted'] = recall_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=1)

#     # Additional Metrics
#     metrics['cohen_kappa'] = cohen_kappa_score(y_true_encoded, y_pred_encoded)
#     metrics['mcc'] = matthews_corrcoef(y_true_encoded, y_pred_encoded)

#     # PR AUC and ROC AUC for multi-class if probabilities are provided
#     if y_pred_proba is not None and len(encoded_classes) > 2:
#         pr_auc_scores = [average_precision_score((y_true_encoded == i).astype(int), y_pred_proba[:, i]) for i in range(len(encoded_classes))]
#         metrics['pr_auc'] = np.mean(pr_auc_scores)  # Weighted average PR AUC
        
#         metrics['roc_auc'] = roc_auc_score(y_true_encoded, y_pred_proba, multi_class="ovr", average="weighted")
#         metrics['log_loss'] = log_loss(y_true_encoded, y_pred_proba)
#     elif y_pred_proba is not None:
#         metrics['pr_auc'] = average_precision_score(y_true_encoded, y_pred_proba[:, 1])
#         metrics['roc_auc'] = roc_auc_score(y_true_encoded, y_pred_proba[:, 1])
#         metrics['log_loss'] = log_loss(y_true_encoded, y_pred_proba)
#     else:
#         metrics['pr_auc'] = average_precision_score(y_true_encoded, y_pred_encoded, average="weighted", zero_division=1)

#     # F2 Score if recall is prioritized
#     metrics['f2_score'] = fbeta_score(y_true_encoded, y_pred_encoded, beta=2, average='weighted', zero_division=1)

#     # Class-wise Metrics
#     precision, recall, fscore, _ = precision_recall_fscore_support(
#         y_true_encoded, y_pred_encoded, average=None, labels=range(len(encoded_classes)), zero_division=1
#     )
#     metrics['class_precision'] = precision
#     metrics['class_recall'] = recall
#     metrics['class_fscore'] = fscore

#     # Display Metrics
#     print("\n--- Model Metrics ---")
#     for metric, value in metrics.items():
#         print(f"{metric}: {value}")

#     # Class-wise Precision, Recall, F1 table
#     classwise_df = pd.DataFrame({
#         'Class': encoded_classes,
#         'Precision': precision,
#         'Recall': recall,
#         'F1-Score': fscore
#     })
    
#     print("\nClass-wise Precision, Recall, and F1-Score:")
#     print(classwise_df.to_string(index=False))

#     # Confusion Matrix
#     cm = confusion_matrix(y_true_encoded, y_pred_encoded, normalize='true')
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=encoded_classes, yticklabels=encoded_classes)
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title('Normalized Confusion Matrix')
#     plt.show()

#     return metrics

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd
# from sklearn.metrics import (
#     accuracy_score, f1_score, precision_score, recall_score,
#     cohen_kappa_score, matthews_corrcoef, confusion_matrix,
#     precision_recall_fscore_support, average_precision_score,
#     log_loss, fbeta_score, roc_auc_score
# )
# from sklearn.preprocessing import LabelEncoder

# def calculate_metrics(y_true, y_pred, y_pred_proba=None, classes=None):
#     metrics = {}
    
#     # Encode y_true and y_pred if they are not numeric
#     label_encoder = LabelEncoder()
#     y_true_encoded = label_encoder.fit_transform(y_true)
#     y_pred_encoded = label_encoder.transform(y_pred)
#     encoded_classes = label_encoder.classes_ if classes is None else classes
    
#     # Basic Metrics with zero_division set to handle undefined metrics
#     metrics['accuracy'] = accuracy_score(y_true_encoded, y_pred_encoded)
#     metrics['f1_weighted'] = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=1)
#     metrics['precision_weighted'] = precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=1)
#     metrics['recall_weighted'] = recall_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=1)

#     # Additional Metrics
#     metrics['cohen_kappa'] = cohen_kappa_score(y_true_encoded, y_pred_encoded)
#     metrics['mcc'] = matthews_corrcoef(y_true_encoded, y_pred_encoded)

#     # PR AUC and ROC AUC for multi-class if probabilities are provided
#     if y_pred_proba is not None and len(encoded_classes) > 2:
#         pr_auc_scores = [
#             average_precision_score((y_true_encoded == i).astype(int), y_pred_proba[:, i], zero_division=1)
#             for i in range(len(encoded_classes))
#         ]
#         metrics['pr_auc'] = np.mean(pr_auc_scores)  # Weighted average PR AUC
#         metrics['roc_auc'] = roc_auc_score(y_true_encoded, y_pred_proba, multi_class="ovr", average="weighted")
#         metrics['log_loss'] = log_loss(y_true_encoded, y_pred_proba)
#     elif y_pred_proba is not None:
#         metrics['pr_auc'] = average_precision_score(y_true_encoded, y_pred_proba[:, 1], zero_division=1)
#         metrics['roc_auc'] = roc_auc_score(y_true_encoded, y_pred_proba[:, 1])
#         metrics['log_loss'] = log_loss(y_true_encoded, y_pred_proba)
#     else:
#         metrics['pr_auc'] = average_precision_score(y_true_encoded, y_pred_encoded, average="weighted", zero_division=1)

#     # F2 Score if recall is prioritized
#     metrics['f2_score'] = fbeta_score(y_true_encoded, y_pred_encoded, beta=2, average='weighted', zero_division=1)

#     # Class-wise Metrics
#     precision, recall, fscore, _ = precision_recall_fscore_support(
#         y_true_encoded, y_pred_encoded, average=None, labels=range(len(encoded_classes)), zero_division=1
#     )
#     metrics['class_precision'] = precision
#     metrics['class_recall'] = recall
#     metrics['class_fscore'] = fscore

#     # Display Metrics
#     print("\n--- Model Metrics ---")
#     for metric, value in metrics.items():
#         print(f"{metric}: {value}")

#     # Class-wise Precision, Recall, F1 table
#     classwise_df = pd.DataFrame({
#         'Class': encoded_classes,
#         'Precision': precision,
#         'Recall': recall,
#         'F1-Score': fscore
#     })
    
#     print("\nClass-wise Precision, Recall, and F1-Score:")
#     print(classwise_df.to_string(index=False))

#     # Confusion Matrix
#     cm = confusion_matrix(y_true_encoded, y_pred_encoded, normalize='true')
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=encoded_classes, yticklabels=encoded_classes)
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title('Normalized Confusion Matrix')
#     plt.show()

#     return metrics


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score, matthews_corrcoef, confusion_matrix,
    precision_recall_fscore_support, average_precision_score,
    log_loss, fbeta_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

def calculate_metrics(y_true, y_pred, y_pred_proba=None, classes=None):
    metrics = {}
    
    # Encode y_true and y_pred if they are not numeric
    label_encoder = LabelEncoder()
    y_true_encoded = label_encoder.fit_transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)
    encoded_classes = label_encoder.classes_ if classes is None else classes
    
    # Basic Metrics with zero_division set to handle undefined metrics
    metrics['accuracy'] = accuracy_score(y_true_encoded, y_pred_encoded)
    metrics['f1_weighted'] = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=1)
    metrics['precision_weighted'] = precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=1)
    metrics['recall_weighted'] = recall_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=1)

    # Additional Metrics
    metrics['cohen_kappa'] = cohen_kappa_score(y_true_encoded, y_pred_encoded)
    metrics['mcc'] = matthews_corrcoef(y_true_encoded, y_pred_encoded)

    # PR AUC and ROC AUC for multi-class if probabilities are provided
    if y_pred_proba is not None and len(encoded_classes) > 2:
        pr_auc_scores = []
        for i in range(len(encoded_classes)):
            # Check if there are any positive instances for the current class
            if np.any(y_true_encoded == i):
                pr_auc = average_precision_score((y_true_encoded == i).astype(int), y_pred_proba[:, i])
                pr_auc_scores.append(pr_auc)
            else:
                pr_auc_scores.append(0.0)  # Assign 0 if no instances of the class
        metrics['pr_auc'] = np.mean(pr_auc_scores)  # Weighted average PR AUC

        metrics['roc_auc'] = roc_auc_score(y_true_encoded, y_pred_proba, multi_class="ovr", average="weighted")
        metrics['log_loss'] = log_loss(y_true_encoded, y_pred_proba)
    elif y_pred_proba is not None:
        metrics['pr_auc'] = average_precision_score(y_true_encoded, y_pred_proba[:, 1])
        metrics['roc_auc'] = roc_auc_score(y_true_encoded, y_pred_proba[:, 1])
        metrics['log_loss'] = log_loss(y_true_encoded, y_pred_proba)
    else:
        metrics['pr_auc'] = average_precision_score(y_true_encoded, y_pred_encoded, average="weighted", zero_division=1)

    # F2 Score if recall is prioritized
    metrics['f2_score'] = fbeta_score(y_true_encoded, y_pred_encoded, beta=2, average='weighted', zero_division=1)

    # Class-wise Metrics
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true_encoded, y_pred_encoded, average=None, labels=range(len(encoded_classes)), zero_division=1
    )
    metrics['class_precision'] = precision
    metrics['class_recall'] = recall
    metrics['class_fscore'] = fscore

    # Display Metrics
    print("\n--- Model Metrics ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # Class-wise Precision, Recall, F1 table
    classwise_df = pd.DataFrame({
        'Class': encoded_classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': fscore
    })
    
    print("\nClass-wise Precision, Recall, and F1-Score:")
    print(classwise_df.to_string(index=False))

    # Confusion Matrix
    cm = confusion_matrix(y_true_encoded, y_pred_encoded, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=encoded_classes, yticklabels=encoded_classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Normalized Confusion Matrix')
    plt.show()

    return metrics
