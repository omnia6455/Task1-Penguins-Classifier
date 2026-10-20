import os
import pandas as pd
from src.preparingData import get_train_test_split
from src.models import Perceptron, Adaline, evaluate_predictions
from src.visualization import plot_decision_boundary

os.makedirs('saved_plots', exist_ok=True)
data = pd.read_csv('penguins.csv')
for col in ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']:
    data[col] = data.groupby('Species')[col].transform(lambda x: x.fillna(x.mean()))
data['OriginLocation'] = data['OriginLocation'].map({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})
combinations = [
    ('Adelie', 'Chinstrap', 'CulmenDepth', 'BodyMass', 'Bad_Overlap'),
    ('Adelie', 'Chinstrap', 'CulmenDepth', 'FlipperLength', 'Mediocre_Overlap'),
    ('Adelie', 'Gentoo', 'CulmenLength', 'CulmenDepth', 'Good_Separable'),
    ('Adelie', 'Chinstrap', 'CulmenLength', 'CulmenDepth', 'Good_Separable'),
    ('Chinstrap', 'Gentoo', 'FlipperLength', 'BodyMass', 'Good_Separable')
]

def run_tests():
    print(f"{'='*60}\nGenerating 5 Test Combinations for Report...\n{'='*60}")

    for i, (c1, c2, f1, f2, quality) in enumerate(combinations, start=1):
        print(f"\nTest {i}: {c1} vs {c2} | {f1} & {f2} ({quality})")
        X_tr, y_tr, X_te, y_te = get_train_test_split(data, c1, c2, f1, f2)

        p_model = Perceptron(learning_rate=0.01, epochs=1000, use_bias=True)
        p_model.fit(X_tr, y_tr)
        p_preds = p_model.predict(X_te)
        _, p_acc = evaluate_predictions(y_te, p_preds)

        fig_p, _ = plot_decision_boundary(X_tr, y_tr, X_te, y_te, p_model, f1, f2, c1, c2, title=f"Perceptron: {c1} vs {c2}")
        fig_p.savefig(f"saved_plots/Perceptron_Comb{i}_{quality}.png", bbox_inches='tight')
        print(f"  -> Perceptron Accuracy: {p_acc*100:.1f}% (Plot saved)")

        a_model = Adaline(learning_rate=0.001, epochs=1000, mse_threshold=0.05, use_bias=True)
        a_model.fit(X_tr, y_tr)
        a_preds = a_model.predict_class(X_te)
        _, a_acc = evaluate_predictions(y_te, a_preds)

        fig_a, _ = plot_decision_boundary(X_tr, y_tr, X_te, y_te, a_model, f1, f2, c1, c2, title=f"Adaline: {c1} vs {c2}")
        fig_a.savefig(f"saved_plots/Adaline_Comb{i}_{quality}.png", bbox_inches='tight')
        print(f"  -> Adaline Accuracy:    {a_acc*100:.1f}% (Plot saved)")

    print(f"\n{'='*60}\nAll tests complete! Check the 'saved_plots/' folder.\n{'='*60}")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    run_tests()