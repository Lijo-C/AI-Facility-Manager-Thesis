from sklearn.metrics import classification_report

def get_model_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    print("--- MODEL PERFORMANCE REPORT ---")
    print(report)
    return report