import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve, roc_curve, auc

def accuracy(y_predicted: np.ndarray, y_real: np.ndarray) -> float:
    TP_TN = sum([y_i == y_j for (y_i, y_j) in zip(y_predicted, y_real)]) 
    P_N = len(y_real)
    return TP_TN / P_N

def get_df_metricas(predictions, scores, true_labels):
    k = len(predictions)

    rows = []
    y_true_all = []
    y_pred_all = []
    scores_all = []

    for i in range(k):
        y_train, y_test = true_labels[i]
        y_pred_train, y_pred_test = predictions[i]
        y_score_train, y_score_test = scores[i]
        
        y_true_all.append(y_test)
        y_pred_all.append(y_pred_test)
        scores_all.append(y_score_test)

        # AUPRC training
        prec_tr, rec_tr, _ = precision_recall_curve(y_train, y_score_train)
        auprc_train = auc(rec_tr, prec_tr)

        # AUPRC validación
        prec_te, rec_te, _ = precision_recall_curve(y_test, y_score_test)
        auprc_val = auc(rec_te, prec_te)

        # AUC ROC training
        fpr_tr, tpr_tr, _ = roc_curve(y_train, y_score_train)
        aucroc_train = auc(fpr_tr, tpr_tr)

        # AUC ROC validación
        fpr_te, tpr_te, _ = roc_curve(y_test, y_score_test)
        aucroc_val = auc(fpr_te, tpr_te)

        row = {
            "Permutación": i + 1,
            "Accuracy (training)": accuracy(y_pred_train, y_train),
            "Accuracy (validación)": accuracy(y_pred_test, y_test),
            "AUPRC (training)": auprc_train,
            "AUPRC (validación)": auprc_val,
            "AUC ROC (training)": aucroc_train,
            "AUC ROC (validación)": aucroc_val,
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    promedio = df.mean(numeric_only=True)
    promedio["Permutación"] = "Promedios"
    df = pd.concat([df, pd.DataFrame([promedio])], ignore_index=True)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    scores_all = np.concatenate(scores_all)

    prec_g, rec_g, _ = precision_recall_curve(y_true_all, scores_all)
    auprc_global = auc(rec_g, prec_g)

    fpr_g, tpr_g, _ = roc_curve(y_true_all, scores_all)
    aucroc_global = auc(fpr_g, tpr_g)

    fila_global = {
        "Permutación": "Global",
        "Accuracy (training)": "(NO)",
        "Accuracy (validación)": accuracy(y_pred_all, y_true_all),
        "AUPRC (training)": "(NO)",
        "AUPRC (validación)": auprc_global,
        "AUC ROC (training)": "(NO)",
        "AUC ROC (validación)": aucroc_global,
    }

    df = pd.concat([df, pd.DataFrame([fila_global])], ignore_index=True)

    return df