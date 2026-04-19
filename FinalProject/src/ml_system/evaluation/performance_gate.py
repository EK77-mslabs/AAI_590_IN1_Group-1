import logging

from sklearn.metrics import f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(y_true, y_pred, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    return {"auc": auc, "f1": f1}


def check_performance_gate(metrics: dict, baseline_auc: float) -> bool:
    auc = metrics.get("auc", 0)
    logger.info(f"Model AUC: {auc: .4f} | Baseline AUC: {baseline_auc: .4f}")
    if auc >= baseline_auc:
        logger.info("Performance gate PASSED.")
        return True
    else:
        logger.error("Performance gate FAILED. Model is worse than baseline.")
        return False
