from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss

# ===============================
# FEATURE MATRIX (SIMPLIFIED)
# ===============================

X = team_game[["dxg", "dgsax", "dfinish", "home"]]
y = team_game["win"]

# ===============================
# LOGISTIC WITH CV TUNING
# ===============================

logreg = LogisticRegression(max_iter=2000)

param_grid = {
    "C": [0.01, 0.1, 0.5, 1, 2, 5, 10]
}

grid = GridSearchCV(
    logreg,
    param_grid,
    scoring="neg_log_loss",
    cv=5
)

grid.fit(X, y)

best_model = grid.best_estimator_

probs = best_model.predict_proba(X)[:,1]
preds = best_model.predict(X)

print("\n=== Simplified Logistic (No Suppression) ===")
print("Best C:", grid.best_params_)
print("Accuracy:", round(accuracy_score(y, preds),4))
print("Log Loss:", round(log_loss(y, probs),4))

print("\nCoefficients:")
for name, coef in zip(X.columns, best_model.coef_[0]):
    print(f"{name}: {round(coef,4)}")