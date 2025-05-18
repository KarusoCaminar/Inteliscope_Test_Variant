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
    
    def gradient_func(x_val, y_val): # Renamed parameters to avoid conflict with global X, Y
        return np.array([g(x_val, y_val) for g in numpy_grads])
    
    return gradient_func

def _create_function_from_sympy(sympy_expr, name="", tooltip="", x_range=(-5, 5), y_range=(-5, 5), minima=None):
    """Erstellt eine Funktion aus einem sympy-Ausdruck mit einheitlicher Schnittstelle."""
    func_eval = _sympy_to_numpy_func(sympy_expr) # Renamed to func_eval
    grad_func_eval = _sympy_gradient_to_numpy(sympy_expr) # Renamed to grad_func_eval
    
    def wrapper(x_input: np.ndarray) -> Dict[str, Any]: # Renamed x to x_input
        # Stelle sicher, dass x_input ein numpy array ist
        x_input = np.asarray(x_input, dtype=float)
        
        value = float(func_eval(x_input[0], x_input[1]))
        gradient = grad_func_eval(x_input[0], x_input[1])
            
        result = {
            'value': value,
            'gradient': gradient,
            'name': name,
            'tooltip': tooltip,
            'x_range': x_range,
            'y_range': y_range
        }
        
        if minima is not None:
            # Immer als Liste von Listen zurückgeben!
            if isinstance(minima, (list, tuple)):
                result['minima'] = [list(m) for m in minima]
            else:
                result['minima'] = [list(minima)]
            
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
    minima=[(3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)] # More precise minima
)

# Rastrigin-Funktion
def rastrigin_func(x_input: np.ndarray) -> Dict[str, Any]:
    """
    Rastrigin-Funktion:
    f(x,y) = 20 + x^2 + y^2 - 10*cos(2π*x) - 10*cos(2π*y)
    """
    x_input = np.asarray(x_input, dtype=float)
    A = 10
    n = len(x_input)
    value = A * n + np.sum(x_input**2 - A * np.cos(2 * np.pi * x_input))
    
    gradient = 2 * x_input + 2 * np.pi * A * np.sin(2 * np.pi * x_input)
    
    return {
        'value': float(value),
        'gradient': gradient,
        'name': "Rastrigin",
        'tooltip': "Die Rastrigin-Funktion hat viele lokale Minima, aber nur ein globales Minimum "
                  "bei (0, ..., 0) mit f(0, ..., 0) = 0. Sie ist hochgradig multimodal und stellt eine große "
                  "Herausforderung für Optimierungsalgorithmen dar.",
        'x_range': (-5.12, 5.12),
        'y_range': (-5.12, 5.12), # Assuming 2D for default y_range
        'minima': [[0.0,] * n] # Global minimum at origin for n dimensions
    }

# Ackley-Funktion
def ackley_func(x_input: np.ndarray) -> Dict[str, Any]:
    """
    Ackley-Funktion:
    f(x,y) = -20 * exp(-0.2 * sqrt(0.5 * (x^2 + y^2))) - exp(0.5 * (cos(2π*x) + cos(2π*y))) + e + 20
    """
    x_input = np.asarray(x_input, dtype=float)
    a, b, c = 20, 0.2, 2 * np.pi
    n = len(x_input)
    
    sum_sq_term = np.sum(x_input**2)
    sum_cos_term = np.sum(np.cos(c * x_input))
    
    term1_val = -a * np.exp(-b * np.sqrt(sum_sq_term / n))
    term2_val = -np.exp(sum_cos_term / n)
    
    value = term1_val + term2_val + a + np.exp(1)
    
    # Berechnung des Gradienten
    # Derivative of the first term part: -a * exp(-b * sqrt(sum_sq_term / n))
    # d/dx_j (sqrt(sum_sq_term / n)) = 1/(2*sqrt(sum_sq_term/n)) * (2*x_j/n) = x_j / (n*sqrt(sum_sq_term/n))
    # If sum_sq_term is zero, gradient is zero for this part.
    if sum_sq_term == 0:
        grad_term1 = np.zeros_like(x_input)
    else:
        grad_term1 = term1_val * (-b) * (x_input / (n * np.sqrt(sum_sq_term / n))) # Corrected chain rule application

    # Derivative of the second term part: -exp(sum_cos_term / n)
    # d/dx_j (sum_cos_term / n) = (-c * sin(c*x_j)) / n
    grad_term2 = term2_val * ((-c * np.sin(c * x_input)) / n) # Corrected chain rule application
    
    gradient = grad_term1 + grad_term2
    
    return {
        'value': float(value),
        'gradient': gradient,
        'name': "Ackley",
        'tooltip': "Die Ackley-Funktion ist eine multimodale Testfunktion mit vielen lokalen Minima "
                  "und einem globalen Minimum bei (0, ..., 0) mit f(0, ..., 0) = 0. Die Funktion hat eine "
                  "nahezu flache äußere Region und einen tiefen Trichter in der Mitte.",
        'x_range': (-5, 5),
        'y_range': (-5, 5), # Assuming 2D for default y_range
        'minima': [[0.0,] * n] # Global minimum at origin for n dimensions
    }

# Schwefel-Funktion
def schwefel_func(x_input: np.ndarray) -> Dict[str, Any]:
    """
    Schwefel-Funktion:
    f(x) = 418.9829 * n - sum(x_i * sin(sqrt(|x_i|)))
    Minimum bei x_i = 420.9687 für alle i.
    """
    x_input = np.asarray(x_input, dtype=float)
    n = len(x_input)
    value = 418.9829 * n - np.sum(x_input * np.sin(np.sqrt(np.abs(x_input))))
    
    # Berechnung des Gradienten
    # d/dx (x * sin(sqrt(|x|))) = sin(sqrt(|x|)) + x * cos(sqrt(|x|)) * (sgn(x) / (2*sqrt(|x|)))
    # = sin(sqrt(|x|)) + (x * sgn(x) * cos(sqrt(|x|))) / (2*sqrt(|x|))
    # = sin(sqrt(|x|)) + (|x| * cos(sqrt(|x|))) / (2*sqrt(|x|))
    # = sin(sqrt(|x|)) + (sqrt(|x|) * cos(sqrt(|x|))) / 2
    # Gradient of sum term is -(sin(sqrt(|x|)) + (sqrt(|x|) * cos(sqrt(|x|))) / 2)
    
    abs_x = np.abs(x_input)
    sqrt_abs_x = np.sqrt(abs_x)
    
    # Add a small epsilon to sqrt_abs_x in denominator for cos term if x_input can be zero,
    # but since it's a multiplier, sqrt(0)*cos(0) = 0, so it's fine.
    # However, for the original formula's structure, let's be careful.
    # The derivative of x*sin(sqrt|x|) is sin(sqrt|x|) + (sqrt|x|/2)*cos(sqrt|x|)
    # So the gradient of the function is the negative of this.
    
    grad_term_sum = np.sin(sqrt_abs_x) + 0.5 * sqrt_abs_x * np.cos(sqrt_abs_x)
    gradient = -grad_term_sum
    
    # Handle cases where x_input is exactly zero to avoid NaN if any intermediate step leads to 0/0
    # For x_i = 0, the derivative of x_i*sin(sqrt|x_i|) is 0.
    gradient[x_input == 0] = 0.0

    return {
        'value': float(value),
        'gradient': gradient,
        'name': "Schwefel",
        'tooltip': "Die Schwefel-Funktion ist eine komplexe multimodale Funktion mit einem "
                  "globalen Minimum bei (420.9687, ..., 420.9687) und vielen lokalen Minima, "
                  "die weit vom globalen Minimum entfernt liegen.",
        'x_range': (-500, 500),
        'y_range': (-500, 500), # Assuming 2D for default y_range
        'minima': [[420.9687,] * n] # Global minimum for n dimensions
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
        "default_range": [(-2, 2), (-1, 3)], # x_range, y_range
        "contour_levels": 50,
        "dimensions": 2 # Explicitly state dimensions
    },
    "Himmelblau": {
        "func": himmelblau_func,
        "default_range": [(-6, 6), (-6, 6)],
        "contour_levels": 40,
        "dimensions": 2
    },
    "Rastrigin": {
        "func": rastrigin_func,
        "default_range": [(-5.12, 5.12), (-5.12, 5.12)],
        "contour_levels": 50,
        "dimensions": 2 # Default for visualization, function can handle n-D
    },
    "Ackley": {
        "func": ackley_func,
        "default_range": [(-5, 5), (-5, 5)],
        "contour_levels": 50,
        "dimensions": 2 # Default for visualization, function can handle n-D
    },
    "Schwefel": {
        "func": schwefel_func,
        "default_range": [(-500, 500), (-500, 500)],
        "contour_levels": 40,
        "dimensions": 2 # Default for visualization, function can handle n-D
    },
    "Eggcrate": {
        "func": eggcrate_func,
        "default_range": [(-5, 5), (-5, 5)],
        "contour_levels": 30,
        "dimensions": 2
    }
}

# Helper-Funktion, um benutzerdefinierte Funktionen zu erstellen
def create_custom_function(expr_str, name="Custom", x_range=(-5, 5), y_range=(-5, 5)):
    """
    Erstellt eine benutzerdefinierte Funktion aus einem String-Ausdruck.
    Akzeptiert viele mathematische Schreibweisen und Synonyme.
    
    Args:
        expr_str: String-Darstellung der mathematischen Funktion (mit X und Y als Variablen)
        name: Name der Funktion
        x_range: Bereich der x-Achse für die Visualisierung
        y_range: Bereich der y-Achse für die Visualisierung
        
    Returns:
        Funktion mit der üblichen Schnittstelle
    """
    try:
        # Automatische Ersetzung typischer Synonyme für mathematische Funktionen
        expr_str = expr_str.replace('ln', 'log').replace('tg', 'tan')
        expr_str = expr_str.replace('abs', 'Abs')  # Für SymPy ist Abs(x) richtig
        
        # Sehr großzügiges local_dict (SymPy + Standard)
        local_dict = {
            "X": X, "Y": Y, "x": X, "y": Y,  # Variablen, egal ob groß/klein
            "e": sp.E, "E": sp.E,            # Eulersche Zahl
            "pi": sp.pi, "Pi": sp.pi, "π": sp.pi,
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
            "sqrt": sp.sqrt, "Abs": sp.Abs, "abs": sp.Abs,
            "log": sp.log, "ln": sp.log,     # ln als log
            "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
            "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
            "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
            "min": min, "max": max,
            # Sicherheitsmaßnahmen (optional): keine eval, keine os, keine sys usw.
        }
        # SymPy-Parsing
        expr = sp.sympify(expr_str, locals=local_dict)
        custom_func_obj = _create_function_from_sympy(
            expr,
            name=name,
            tooltip=f"Benutzerdefinierte Funktion: {expr_str}",
            x_range=x_range,
            y_range=y_range
        )
        return custom_func_obj
    except Exception as e:
        print(f"Error creating custom function '{name}' with expr '{expr_str}': {e}")
        def error_func_wrapper(x_input_err: np.ndarray) -> Dict[str, Any]:
            return {
                'value': float('nan'),
                'gradient': np.array([float('nan')] * len(x_input_err)),
                'name': name,
                'tooltip': f"Fehler beim Erstellen der Funktion: {e}. Ausdruck: {expr_str}",
                'x_range': x_range,
                'y_range': y_range,
                'error': str(e)
            }
        return error_func_wrapper

if __name__ == '__main__':
    # Test der Funktionen
    test_point_2d = np.array([1.0, 1.5])
    
    print("Testing Rosenbrock function:")
    rosenbrock_res = rosenbrock_func(test_point_2d)
    print(f"Value: {rosenbrock_res['value']}, Gradient: {rosenbrock_res['gradient']}")

    print("\nTesting Himmelblau function:")
    himmelblau_res = himmelblau_func(test_point_2d)
    print(f"Value: {himmelblau_res['value']}, Gradient: {himmelblau_res['gradient']}")

    print("\nTesting Rastrigin function:")
    rastrigin_res = rastrigin_func(test_point_2d)
    print(f"Value: {rastrigin_res['value']}, Gradient: {rastrigin_res['gradient']}")

    print("\nTesting Ackley function:")
    ackley_res = ackley_func(test_point_2d)
    print(f"Value: {ackley_res['value']}, Gradient: {ackley_res['gradient']}")
    # Test Ackley at origin
    ackley_origin_res = ackley_func(np.array([0.0, 0.0]))
    print(f"Ackley at origin: Value: {ackley_origin_res['value']}, Gradient: {ackley_origin_res['gradient']}")


    print("\nTesting Schwefel function:")
    schwefel_res = schwefel_func(test_point_2d)
    print(f"Value: {schwefel_res['value']}, Gradient: {schwefel_res['gradient']}")
    schwefel_test_val = np.array([420.9687, 420.9687])
    schwefel_min_res = schwefel_func(schwefel_test_val)
    print(f"Schwefel at minimum ({schwefel_test_val}): Value: {schwefel_min_res['value']}, Gradient: {schwefel_min_res['gradient']}")


    print("\nTesting Eggcrate function:")
    eggcrate_res = eggcrate_func(test_point_2d)
    print(f"Value: {eggcrate_res['value']}, Gradient: {eggcrate_res['gradient']}")

    print("\nTesting custom function creation:")
    custom_f = create_custom_function("X**3 + Y**2 - sin(X*Y)")
    custom_res = custom_f(test_point_2d)
    print(f"Custom Value: {custom_res['value']}, Gradient: {custom_res['gradient']}")
    
    custom_f_error = create_custom_function("X^^2") # Syntax error
    custom_res_error = custom_f_error(test_point_2d)
    print(f"Custom Error Value: {custom_res_error.get('value')}, Error: {custom_res_error.get('error')}")

