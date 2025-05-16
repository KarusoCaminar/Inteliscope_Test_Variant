# data_manager.py
"""
Modul zur Bereitstellung von Trainings- und Testdaten
für Machine-Learning-Beispiele.
"""
import numpy as np

def get_linear_regression_data(n_samples=100, true_w0=2.0, true_w1=1.5, noise_std=1.5, x_range=(0, 10), seed=42):
    """ Erzeugt Daten für eine Lineare Regression. """
    np.random.seed(seed)
    X = np.linspace(x_range[0], x_range[1], n_samples)
    Y = true_w0 + true_w1 * X + np.random.normal(0, noise_std, size=X.shape)
    info = {'true_coeffs': np.array([true_w0, true_w1])}
    return X, Y, info

def get_poly_regression_data(n_samples=100, coeffs=(1.0, -2.0, 0.5), noise_std=5.0, x_range=(-5, 5), seed=42):
    """ 
     Erzeugt Daten für eine Polynomielle Regression.
     coeffs: [w0, w1, w2, ...] 
    """
    np.random.seed(seed)
    X = np.linspace(x_range[0], x_range[1], n_samples)
    Y = np.polynomial.polynomial.polyval(X, coeffs) + np.random.normal(0, noise_std, size=X.shape)
    info = {'true_coeffs': np.array(coeffs)}
    return X, Y, info

if __name__ == '__main__':
     print("DataManager: Teste Datengenerierung...")
     X_lin, Y_lin, info_lin = get_linear_regression_data()
     print(f"Lineare Regression: {X_lin.shape[0]} Datenpunkte generiert.")
     
     X_poly, Y_poly, info_poly = get_poly_regression_data()
     print(f"Polynom. Regression: {X_poly.shape[0]} Datenpunkte generiert.")
     print("DataManager bereit.")