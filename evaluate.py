from sklearn.metrics import ndcg_score

def evaluate_ranker(ranker, X, y):
    preds = ranker.predict(X)

    ndcg = ndcg_score(
        [y],
        [preds],
        k=10
    )

    print("\n" + "=" * 50)
    print(f"NDCG@10: {ndcg:.4f}")
    print("=" * 50 + "\n")

    return ndcg