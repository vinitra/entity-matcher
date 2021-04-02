#!/usr/bin/env python3

import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score


def get_scores(actual, pred):
    """
    It computes the classification performance score between the true and the predicted pairs.
    This method uses both actual pairs and reversed ones and joins them with the actual data
    (i.e. actual data - left join - concatenation of predicted and reversed predicted data).

    :param actual: pd.DataFrame, all possible pairs with their labels (1 if pair else 0)
    :param pred: pd.DataFrame, predicted pairs
    :return: dict, the classification scores
    """
    # create a reversed dataset
    pred_reversed = pd.DataFrame()
    pred_reversed['left_instance_id'] = pred['right_instance_id']
    pred_reversed['right_instance_id'] = pred['left_instance_id']

    # add label to all the predicted ones
    pred_reversed['pred_label'] = int(1)
    pred['pred_label'] = int(1)

    pred_final = pd.concat([pred, pred_reversed], ignore_index=True)

    # perform left join between the target dataset and the predicted one
    merged = pd.merge(actual, pred_final,
                      on=["left_instance_id", 'right_instance_id'],
                      how='left')

    # fill null values from the left join with 0 label
    merged = merged.fillna(int(0))
    merged['pred_label'] = merged['pred_label'].apply(lambda x: int(x))

    # extract label pd.Series
    y_true = merged['label']
    y_pred = merged['pred_label']

    # compute scores
    return {
        'confusion_matrix': confusion_matrix(y_true=y_true, y_pred=y_pred),
        'precision_score': precision_score(y_true=y_true, y_pred=y_pred),
        'recall_score': recall_score(y_true=y_true, y_pred=y_pred),
        'f1_score': f1_score(y_true=y_true, y_pred=y_pred),
    }
