import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, recall_score, average_precision_score
import warnings
from sklearn.preprocessing import label_binarize
from collections import defaultdict
warnings.simplefilter("ignore")


COLOR_MAPPING = ['#3288bd', '#99d594', '#fee08b', '#F13C26', '#f768a1', '#000000']


def calc_roc_pr_per_class(y_true, probs, label_names):
    n_classes = len(label_names)
    y_true = label_binarize(y_true, classes=list(range(n_classes)))
    fprs, tprs, precisions, recalls, rocs, auprs, thresholds = [], [], [], [], [], [], []

    for i in range(n_classes+1):
        # Last iteration is the overall stats
        curr_y = y_true.ravel(order='F') if i == n_classes else y_true[:, i]
        curr_prob = probs.ravel(order='F') if i == n_classes else probs[:, i]
        fpr, tpr, _ = roc_curve(curr_y, curr_prob)
        precision, recall, threshold = precision_recall_curve(curr_y, curr_prob)
        rocauc_score = roc_auc_score(curr_y, curr_prob)
        prauc_score = auc(recall, precision)
        fprs.append(fpr)
        tprs.append(tpr)
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(threshold)
        rocs.append(rocauc_score)
        auprs.append(prauc_score)

    return fprs, tprs, precisions, recalls, rocs, auprs, thresholds


def get_threshold_dict(precision, threshold, min_threshold, threshold_dict=None):
    flatten_dict = threshold_dict is None
    threshold_dict = threshold_dict if threshold_dict is not None else defaultdict(list)
    rel_pres = sorted(list(np.arange(min_threshold, 1, 0.05)) + [0.5, 0.75, 0.975, 0.99])
    for t in rel_pres:
        relevant_ind = np.where(precision >= t)[0][0]
        if relevant_ind == len(threshold):
            # no threshold fits the wanted precision, so last index of precision array was returned (always 1)
            continue
        threshold_dict[t].append(threshold[relevant_ind])
    if flatten_dict:
        threshold_dict = {key: value[0] for key, value in threshold_dict.items()}
    return threshold_dict


def finish_aupr_roc_plot(f, axes, location='best'):
    axes[0].set_xlabel('FPR')
    axes[1].set_xlabel('Recall')
    axes[0].set_ylabel('TPR')
    axes[1].set_ylabel('Precision')
    roc_location = location if location == 'best' else location + ' right'
    pr_location = location if location == 'best' else location + ' left'
    axes[0].legend(loc=roc_location, fontsize='small')
    axes[1].legend(loc=pr_location, fontsize='small')
    axes[0].plot([0, 1], [0, 1], '--', color='grey')
    axes[0].set_title(f"ROC curve")
    axes[1].set_title(f"Precision Recall curve")
    f.tight_layout()


def create_cv_ROC_AUPR_plot_multiclass(cvs, label_names, min_threshold=0.8, save_figs=False, prefix=''):
    fprs, tprs, precisions, recalls, rocs, auprs, accuracies, average_recalls, thresholds = [], [], [], [], [], [], [], [], []
    threshold_dict = {i: defaultdict(list) for i in range(len(label_names))}
    f, axes = plt.subplots(1, 2, figsize=(10, 5))

    for amr_model, X_test, y_test, fold_label in cvs:
        model = amr_model.model
        probs = model.predict_proba(X_test)
        pred = model.predict(X_test)
        average_recalls.append(recall_score(y_test, pred, average='micro')) # Chose micro since the overall curves are micro average
        accuracies.append(model.score(X_test, y_test))
        stats = calc_roc_pr_per_class(y_test, probs, label_names)
        curr_thresholds = stats[-1]
        for i, lst in enumerate([fprs, tprs, precisions, recalls, rocs, auprs, thresholds]):
            lst.append(stats[i])

        for i in range(len(label_names)):
            get_threshold_dict(precisions[-1][i], curr_thresholds[i], min_threshold, threshold_dict[i])

    mean_fpr = np.linspace(0, 1, 100)
    mean_rec = np.linspace(0, 1, 100)
    for i in range(len(label_names)+1):
        label_prefix = f'{label_names[i].title().replace(" To ", " to ")}' if i < len(label_names) else 'Overall'
        curr_tprs = [np.interp(mean_fpr, fpr[i], tpr[i]) for tpr, fpr in zip(tprs, fprs)]
        mean_tpr = np.mean(np.array(curr_tprs), axis=0)
        mean_tpr[-1] = 1
        mean_auc = np.mean([roc[i] for roc in rocs])

        axes[0].plot(mean_fpr, mean_tpr, label=f'{label_prefix} ({round(mean_auc, 3)})', lw=1.5, color=COLOR_MAPPING[i], alpha=.8)
        axes[0].plot([0, 1], [0, 1], '--', color='grey')

        std_tpr = np.std(curr_tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        axes[0].fill_between(mean_fpr, tprs_lower, tprs_upper, color=COLOR_MAPPING[i], alpha=.2)

        curr_pre = [np.interp(mean_rec, np.fliplr([rec[i]])[0], np.fliplr([prec[i]])[0]) for prec, rec in zip(precisions, recalls)]
        mean_pre = np.mean(np.array(curr_pre), axis=0)
        mean_pre[0] = 1
        mean_aupr = np.mean([aupr[i] for aupr in auprs])
        axes[1].plot(mean_rec, mean_pre, label=f'{label_prefix} ({round(mean_aupr , 3)})', lw=1.5, color=COLOR_MAPPING[i], alpha=.8)

        std_pre = np.std(curr_pre, axis=0)
        std_pre[0] = 0.0
        pre_upper = np.minimum(mean_pre + std_pre, 1)
        pre_lower = np.maximum(mean_pre - std_pre, 0)
        axes[1].fill_between(mean_rec, pre_lower, pre_upper, color=COLOR_MAPPING[i], alpha=.2)

    finish_aupr_roc_plot(f, axes, 'lower')
    plt.savefig(f"{prefix}_multiclass_cv_AUPR_AUROC.pdf", bbox_inches='tight') if save_figs else plt.show()
    plt.close()
    return np.mean(rocs), np.mean(auprs), np.mean(accuracies), np.mean(average_recalls), threshold_dict


def create_cv_ROC_AUPR_plot(cvs, min_threshold=0.8, save_figs=False, prefix=''):
    all_probs, all_y, rocs, auprs, accuracies, recalls = [], [], [], [], [], []
    threshold_dict = defaultdict(list)
    f, axes = plt.subplots(1, 2, figsize=(10, 5))
    fold = 0

    for amr_model, X_test, y_test, fold_label in cvs:
        pos_label = int(y_test.max())
        fold += 1
        fold_label = f"{fold_label} ({round(y_test.sum()/len(y_test), 2)})" if fold_label else f"Fold {fold}"
        model = amr_model.model
        accuracies.append(model.score(X_test, y_test))
        probs = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        all_probs.append(probs)
        all_y.append(y_test)
        precision, recall, thresholds = precision_recall_curve(y_test, probs, pos_label=pos_label)
        get_threshold_dict(precision, thresholds, min_threshold, threshold_dict)

        recalls.append(recall_score(y_test, pred, pos_label=pos_label))
        fpr, tpr, _ = roc_curve(y_test, probs, pos_label=pos_label)
        prauc_score = auc(recall, precision)
        auprs.append(prauc_score)
        rocauc_score = roc_auc_score(y_test, probs)
        rocs.append(rocauc_score)
        lab0 = f'{fold_label} ROC AUC={round(rocauc_score, 4)}'
        lab1 = f'{fold_label} PR AUC={round(prauc_score, 4)}'
        axes[0].plot(fpr, tpr, label=lab0, lw=2)
        axes[1].plot(recall, precision, label=lab1, lw=2)

    # creating a curve that contains all the results
    y_proba = np.concatenate(all_probs)
    y_real = np.concatenate(all_y)
    fpr, tpr, _ = roc_curve(y_real, y_proba, pos_label=int(y_real.max()))
    precision, recall, _ = precision_recall_curve(y_real, y_proba, pos_label=int(y_real.max()))
    lab0 = 'Overall ROC AUC=%.3f' % (roc_auc_score(y_real, y_proba))
    lab1 = 'Overall PR AUC=%.3f' % (auc(recall, precision))
    axes[0].plot(fpr, tpr, label=lab0, lw=2, color='black')
    axes[1].plot(recall, precision, label=lab1, lw=2, color='black')

    finish_aupr_roc_plot(f, axes)
    plt.savefig(f"{prefix}cv_AUPR_AUROC.pdf") if save_figs else plt.show()
    plt.close()
    return np.mean(rocs), np.mean(auprs), np.mean(accuracies), np.mean(recalls), threshold_dict


def get_model_quality_stats(model, cvs, X_test, y_test, min_threshold=0.8, save_figs=False, display=True, prefix='', label_names=()):
    if cvs:
        crosstab = pd.DataFrame([])
        if len(label_names) > 0:  # multiclass case
            roc, aupr, accuracy, rec, thresholds_dict = create_cv_ROC_AUPR_plot_multiclass(cvs, label_names, min_threshold, save_figs, prefix)
        else:
            roc, aupr, accuracy, rec, thresholds_dict = create_cv_ROC_AUPR_plot(cvs, min_threshold, save_figs=save_figs, prefix=prefix)
        if display:
            print(f"Average ROC AUC Score: {roc}")
            print(f"Average PR AUC Score: {aupr}")
            print(f"Average Accuracy: {accuracy}")
            print(f"Average Recall: {rec}")

    else:
        pred = model.predict(X_test)
        crosstab = pd.crosstab(np.array(list(y_test.values.ravel())), np.array(pred), rownames=['Actual'], colnames=['Predicted'], margins=True)

        if len(label_names) > 0: # multiclass case
            thresholds_dict = {}
            test_probabilities = model.predict_proba(X_test)
            if display:
                aupr, roc = plot_roc_pr_curve_per_class(y_test, test_probabilities, label_names, save_figs=save_figs, img_path=f"{prefix}_all_data_AUPR_AUROC_per_class.pdf")
                print(f"ROC AUC Score: {roc}")
                print(f"PR AUC Score: {aupr}")
        else:
            pos_label = int(y_test.max())
            test_probabilities = model.predict_proba(X_test)[:, 1]
            if display:
                plot_all_roc_pr_curve(y_test, test_probabilities, save_figs=save_figs, out_file=f"{prefix}all_data_AUPR_AUROC.pdf")
            roc = roc_auc_score(y_test, test_probabilities)
            precision, recall, thresholds = precision_recall_curve(y_test, test_probabilities, pos_label=pos_label)
            aupr = auc(recall, precision)
            thresholds_dict = get_threshold_dict(precision, thresholds, min_threshold)

            if display:
                print(f"ROC AUC Score: {roc}")
                print(f"PR AUC Score: {aupr}")
                print(f"Accuracy: {model.score(X_test, y_test)}")
                print(f"Recall is: {recall_score(y_test, pred, pos_label=pos_label)}")
    return crosstab, roc, aupr, thresholds_dict


def plot_all_roc_pr_curve(labels, probs, save_figs=False, out_file="all_data_AUPR_AUROC.pdf"):
    f, axes = plt.subplots(1, 2, figsize=(10, 5))

    pos_label = int(labels.max())
    fpr, tpr, roc_thresholds = roc_curve(labels, probs, pos_label=pos_label)
    precision, recall, aupr_thresholds = precision_recall_curve(labels, probs, pos_label=pos_label)

    axes[0].plot(fpr, tpr, label=f'ROC Curve (AUC={round(roc_auc_score(labels, probs), 2)})')
    axes[1].plot(recall, precision, label=f'PR Curve (AUC={round(auc(recall, precision), 2)})')
    finish_aupr_roc_plot(f, axes)

    plt.savefig(out_file) if save_figs else plt.show()
    plt.close()


def plot_roc_pr_curve_per_class(labels, probs, label_names, save_figs=False, img_path="AUPR_AUROC_per_class.pdf"):
    f, axes = plt.subplots(1, 2, figsize=(10, 5))

    fprs, tprs, precisions, recalls, rocs, auprs, _ = calc_roc_pr_per_class(labels, probs, label_names)
    for i in range(len(label_names)+1):
        curr_label =f'{label_names[i].title().replace(" To ", " to ")}' if i < len(label_names) else 'Overall'
        axes[0].plot(fprs[i], tprs[i], lw=2, label=f'{curr_label} ({round(rocs[i], 2)})', color=COLOR_MAPPING[i])
        axes[1].plot(recalls[i], precisions[i], lw=2, label=f'{curr_label} ({round(auprs[i], 2)})', color=COLOR_MAPPING[i])

    finish_aupr_roc_plot(f, axes, 'lower')
    plt.savefig(img_path, bbox_inches='tight') if save_figs else plt.show()
    plt.close()
    return np.mean(auprs), np.mean(rocs)