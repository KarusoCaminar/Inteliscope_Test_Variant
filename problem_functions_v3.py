# problem_functions_v3.py
"""
Modul, das Optimierungsprobleme bzw. Testfunktionen bereitstellt.
Jede Funktion hat eine einheitliche Schnittstelle:
- Input: x als numpy-Array mit n Elementen
- Output: Dict mit 'value' und 'gradient'
"""
import numpy as np
import sympy as sp
from typing import Dict, Any, Callable, List, Tuple, Optional

# Symbolische Variablen für die automatische Gradientenberechnung
X, Y = sp.symbols('X Y')

def _sympy_to_numpy_func(sympy_expr, variables=[X, Y]):
    """Konvertiert einen sympy-Ausdruck in eine numpy-kompatible Funktion."""
    numpy_func = sp.lambdify(variables, sympy_expr, 'numpy')
    return numpy_func

def _sympy_gradient_to_numpy(sympy_expr, variables=[X, Y]):
    """Berechnet den Gradienten eines sympy-Ausdrucks und konvertiert ihn in eine numpy-Funktion."""
    grads = [sp.diff(sympy_expr, var) for var in variables]
    numpy_grads = [sp.lambdify(variables, grad, 'numpy') for grad in grads]
    
    def gradient_func(x, y):
        return np.array([g(x, y) for g in numpy_grads])
    
    return gradient_func

def _create_function_from_sympy(sympy_expr, name="", tooltip="", x_range=(-5, 5), y_range=(-5, 5), minima=None):
    """Erstellt eine Funktion aus einem sympy-Ausdruck mit einheitlicher Schnittstelle."""
    func = _sympy_to_numpy_func(sympy_expr)
    grad_func = _sympy_gradient_to_numpy(sympy_expr)
    
    def wrapper(x: np.ndarray) -> Dict[str, Any]:
        # Stelle sicher, dass x ein numpy array ist
        x = np.asarray(x, dtype=float)
        
        if x.size == 1:
            # Spezialfall: 1D-Funktion
            value = float(func(x[0], 0))
            gradient = np.array([float(grad_func(x[0], 0)[0])])
        else:
            # Standardfall: 2D-Funktion
            value = float(func(x[0], x[1]))
            gradient = grad_func(x[0], x[1])
            
        result = {
            'value': value,
            'gradient': gradient,
            'name': name,
            'tooltip': tooltip,
            'x_range': x_range,
            'y_range': y_range
        }
        
        if minima is not None:
            result['minima'] = minima
            
        return result
    
    return wrapper

# Rosenbrock-Funktion (a=1, b=100)
rosenbrock_expr = (1 - X)**2 + 100 * (Y - X**2)**2
rosenbrock_func = _create_function_from_sympy(
    rosenbrock_expr,
    name="Rosenbrock",
    tooltip="Die Rosenbrock-Funktion ist ein klassisches Testbeispiel für Optimierungsalgorithmen. "
            "Sie hat ein globales Minimum bei (1, 1) mit f(1, 1) = 0. Die Funktion hat ein langes, "
            "schmales, parabelförmiges Tal, was die Optimierung besonders schwierig macht.",
    x_range=(-2, 2),
    y_range=(-1, 3),
    minima=[(1.0, 1.0)]
)

# Himmelblau-Funktion
himmelblau_expr = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2
himmelblau_func = _create_function_from_sympy(
    himmelblau_expr,
    name="Himmelblau",
    tooltip="Die Himmelblau-Funktion ist eine Testfunktion mit vier identischen lokalen Minima "
            "bei (3, 2), (-2.81, 3.13), (-3.78, -3.28) und (3.58, -1.85), jeweils mit f = 0.",
    x_range=(-5, 5),
    y_range=(-5, 5),
    minima=[(3.0, 2.0), (-2.81, 3.13), (-3.78, -3.28), (3.58, -1.85)]
)

# Rastrigin-Funktion (erweiterte Visualisierungsreichweite)
def rastrigin_func(x: np.ndarray) -> Dict[str, Any]:
    """
    Rastrigin-Funktion:
    f(x,y) = 20 + x^2 + y^2 - 10*cos(2π*x) - 10*cos(2π*y)
    """
    A = 10
    n = len(x)
    value = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    gradient = 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)
    
    return {
        'value': float(value),
        'gradient': gradient,
        'name': "Rastrigin",
        'tooltip': "Die Rastrigin-Funktion hat viele lokale Minima, aber nur ein globales Minimum "
                  "bei (0, 0) mit f(0, 0) = 0. Sie ist hochgradig multimodal und stellt eine große "
                  "Herausforderung für Optimierungsalgorithmen dar.",
        'x_range': (-5.12, 5.12),
        'y_range': (-5.12, 5.12),
        'minima': [(0.0, 0.0)]
    }

# Ackley-Funktion (erweiterte Visualisierungsreichweite)
def ackley_func(x: np.ndarray) -> Dict[str, Any]:
    """
    Ackley-Funktion:
    f(x,y) = -20 * exp(-0.2 * sqrt(0.5 * (x^2 + y^2))) - exp(0.5 * (cos(2π*x) + cos(2π*y))) + e + 20
    """
    a, b, c = 20, 0.2, 2 * np.pi
    n = len(x)
    
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    
    value = -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.exp(1)
    
    # Berechnung des Gradienten
    term1 = a * b * np.exp(-b * np.sqrt(sum1 / n)) / (2 * n * np.sqrt(sum1 / n)) * (2 * x)
    term2 = c * np.sin(c * x) * np.exp(sum2 / n) / n
    gradient = term1 + term2
    
    return {
        'value': float(value),
        'gradient': gradient,
        'name': "Ackley",
        'tooltip': "Die Ackley-Funktion ist eine multimodale Testfunktion mit vielen lokalen Minima "
                  "und einem globalen Minimum bei (0, 0) mit f(0, 0) = 0. Die Funktion hat eine "
                  "nahezu flache äußere Region und einen tiefen Trichter in der Mitte.",
        'x_range': (-5, 5),
        'y_range': (-5, 5),
        'minima': [(0.0, 0.0)]
    }

# Schwefel-Funktion (erweiterte Visualisierungsreichweite)
def schwefel_func(x: np.ndarray) -> Dict[str, Any]:
    """
    Schwefel-Funktion:
    f(x) = 418.9829 * n - sum(x_i * sin(sqrt(|x_i|)))
    """
    n = len(x)
    value = 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    # Berechnung des Gradienten
    gradient = -np.sin(np.sqrt(np.abs(x))) - 0.5 * x * np.cos(np.sqrt(np.abs(x))) / np.sqrt(np.abs(x) + 1e-10)
    
    return {
        'value': float(value),
        'gradient': gradient,
        'name': "Schwefel",
        'tooltip': "Die Schwefel-Funktion ist eine komplexe multimodale Funktion mit einem "
                  "globalen Minimum bei (420.9687, ..., 420.9687) und vielen lokalen Minima, "
                  "die weit vom globalen Minimum entfernt liegen.",
        'x_range': (-500, 500),
        'y_range': (-500, 500),
        'minima': [(420.9687, 420.9687)]
    }

# Eggcrate-Funktion (einfaches Beispiel mit vielen lokalen Minima)
eggcrate_expr = X**2 + Y**2 + 25 * (sp.sin(X)**2 + sp.sin(Y)**2)
eggcrate_func = _create_function_from_sympy(
    eggcrate_expr,
    name="Eggcrate",
    tooltip="Die Eggcrate-Funktion hat ein globales Minimum bei (0, 0) mit f(0, 0) = 0 "
            "und viele lokale Minima in einem regelmäßigen Muster.",
    x_range=(-5, 5),
    y_range=(-5, 5),
    minima=[(0.0, 0.0)]
)

# Einheitliche Bibliothek aller Funktionen
MATH_FUNCTIONS_LIB = {
    "Rosenbrock": {
        "func": rosenbrock_func,
        "default_range": [(-2, 2), (-1, 3)],
        "contour_levels": 50
    },
    "Himmelblau": {
        "func": himmelblau_func,
        "default_range": [(-6, 6), (-6, 6)],  # Erweitert für bessere Sichtbarkeit
        "contour_levels": 40
    },
    "Rastrigin": {
        "func": rastrigin_func,
        "default_range": [(-5.12, 5.12), (-5.12, 5.12)],
        "contour_levels": 50
    },
    "Ackley": {
        "func": ackley_func,
        "default_range": [(-5, 5), (-5, 5)],
        "contour_levels": 50
    },
    "Schwefel": {
        "func": schwefel_func,
        "default_range": [(-500, 500), (-500, 500)],
        "contour_levels": 40
    },
    "Eggcrate": {
        "func": eggcrate_func,
        "default_range": [(-5, 5), (-5, 5)],
        "contour_levels": 30
    }
}

# Helper-Funktion, um benutzerdefinierte Funktionen zu erstellen
def create_custom_function(expr_str, name="Custom", x_range=(-5, 5), y_range=(-5, 5)):
    """
    Erstellt eine benutzerdefinierte Funktion aus einem String-Ausdruck.
    
    Args:
        expr_str: String-Darstellung der mathematischen Funktion (mit x und y als Variablen)
        name: Name der Funktion
        x_range: Bereich der x-Achse für die Visualisierung
        y_range: Bereich der y-Achse für die Visualisierung
        
    Returns:
        Funktion mit der üblichen Schnittstelle
    """
    try:
        # Ersetze x, y durch die Sympy-Symbole X, Y
        expr_str = expr_str.replace('x', 'X').replace('y', 'Y')
        
        # Parse den Ausdruck
        expr = sp.sympify(expr_str)
        
        # Erstelle die Funktion
        func = _create_function_from_sympy(
            expr,
            name=name,
            tooltip=f"Benutzerdefinierte Funktion: {expr_str}",
            x_range=x_range,
            y_range=y_range
        )
        
        return func
    except Exception as e:
        # Bei Fehler gib eine Fehlermeldung zurück
        def error_func(x):
            return {
                'value': float('nan'),
                'gradient': np.array([float('nan')] * len(x)),
                'name': name,
                'tooltip': f"Fehler beim Erstellen der Funktion: {e}",
                'x_range': x_range,
                'y_range': y_range
            }
        return error_func