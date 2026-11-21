"""
Business Analytics & Managerial Decision Framework:
Calculates cost-matrix threshold optimization, Three-Tier Triage operating bands,
and Unit Economics / Monthly ROI for Fintech KYC.
"""
import numpy as np
import pandas as pd

def calculate_cost_curve(y_true, y_probs, cost_fn=2500, cost_fp=15, thresholds=None):
    """
    Computes total enterprise cost across decision thresholds.
    y_true: 0 (Real), 1 (Manipulated)
    y_probs: P(Manipulated)
    cost_fn: Financial fraud loss per missed fake ($)
    cost_fp: Manual review cost / friction per false alarm ($)
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    results = []
    for tau in thresholds:
        y_pred = (y_probs >= tau).astype(int)
        
        # Confusion matrix elements
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        total_cost = (fn * cost_fn) + (fp * cost_fp)
        
        acc = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            'threshold': tau,
            'total_cost': total_cost,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'tn': tn,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    df = pd.DataFrame(results)
    best_idx = df['total_cost'].idxmin()
    best_row = df.loc[best_idx]

    return df, best_row

def evaluate_three_tier_triage(y_true, y_probs, tau_low=0.20, tau_high=0.75):
    """
    Simulates the Three-Tier Human-in-the-Loop triage model.
    Green Zone: p < tau_low  (Auto-Approve)
    Yellow Zone: tau_low <= p <= tau_high (Manual Review)
    Red Zone: p > tau_high   (Auto-Reject)
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    n_total = len(y_true)

    green_mask = (y_probs < tau_low)
    yellow_mask = (y_probs >= tau_low) & (y_probs <= tau_high)
    red_mask = (y_probs > tau_high)

    green_vol = np.sum(green_mask)
    yellow_vol = np.sum(yellow_mask)
    red_vol = np.sum(red_mask)

    # In Green zone (auto-approve): any fake that slipped through is a costly FN
    green_leaked_fakes = np.sum((y_true[green_mask] == 1))
    
    # In Red zone (auto-reject): any real person falsely rejected is an FP
    red_blocked_reals = np.sum((y_true[red_mask] == 0))

    return {
        'total_volume': n_total,
        'green_volume': green_vol,
        'green_pct': (green_vol / n_total) * 100,
        'green_leaked_fakes': green_leaked_fakes,
        'yellow_volume': yellow_vol,
        'yellow_pct': (yellow_vol / n_total) * 100,
        'red_volume': red_vol,
        'red_pct': (red_vol / n_total) * 100,
        'red_blocked_reals': red_blocked_reals
    }

def calculate_monthly_roi(
    monthly_applicants=100000,
    manual_review_cost=6.00,
    fraud_loss_per_fake=2500.0,
    fraud_incidence_rate=0.04,  # 4% of incoming applicants are synthetic/deepfake
    auto_approve_pct=0.82,
    yellow_review_pct=0.14,
    red_reject_pct=0.04,
    leak_rate=0.005 # 0.5% of fakes leak through green
):
    """
    Computes P&L comparison between Status Quo (100% manual check) vs. AI + HITL.
    """
    # 1. Status Quo:
    status_quo_review_cost = monthly_applicants * manual_review_cost
    # Assume humans catch 90% of fakes, 10% slip through due to fatigue
    total_fakes = monthly_applicants * fraud_incidence_rate
    status_quo_fraud_loss = (total_fakes * 0.10) * fraud_loss_per_fake
    status_quo_total_cost = status_quo_review_cost + status_quo_fraud_loss

    # 2. Proposed AI + Three-Tier Model:
    # Only yellow zone requires human review:
    ai_review_volume = monthly_applicants * (yellow_review_pct)
    ai_review_cost = ai_review_volume * manual_review_cost
    
    # Fraud loss under AI:
    ai_leaked_fakes = total_fakes * leak_rate
    ai_fraud_loss = ai_leaked_fakes * fraud_loss_per_fake
    ai_total_cost = ai_review_cost + ai_fraud_loss

    monthly_savings = status_quo_total_cost - ai_total_cost
    annual_savings = monthly_savings * 12

    return {
        'status_quo_review_cost': status_quo_review_cost,
        'status_quo_fraud_loss': status_quo_fraud_loss,
        'status_quo_total_cost': status_quo_total_cost,
        'ai_review_cost': ai_review_cost,
        'ai_fraud_loss': ai_fraud_loss,
        'ai_total_cost': ai_total_cost,
        'monthly_savings': monthly_savings,
        'annual_savings': annual_savings,
        'instant_onboarding_users': int(monthly_applicants * auto_approve_pct)
    }
