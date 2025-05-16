# optimization_algorithms_v3.py
"""
Modul, das Optimierungsalgorithmen bereitstellt. Version 3.
Callback-Signatur: callback(iteration, params, value, grad_norm, status_msg)
"""
import numpy as np
import sys 

def _linesearch(fun, initial_alpha=1e-3, max_ls_steps=20):
    """
    Führt eine einfache Liniensuche durch.
    
    Args:
        fun: Funktion, die für alpha-Wert einen Funktionswert zurückgibt
        initial_alpha: Startwert für alpha
        max_ls_steps: Maximale Anzahl an Schritten für die Liniensuche
    
    Returns:
        optimal_alpha: Bester gefundener alpha-Wert
        status: String, der den Status der Liniensuche beschreibt
    """
    alpha = initial_alpha
    try:
        current_f_val = fun(alpha)
        if not np.isfinite(current_f_val): # Prüfen ob NaN oder Inf
            return initial_alpha, "initial_alpha_non_finite"
    except Exception:
         return initial_alpha, "initial_alpha_exception"

    optimal_alpha = alpha
    optimal_f_val = current_f_val
    status = "max_ls_steps_reached"
   
    for _ in range(max_ls_steps): 
        next_alpha = 2 * alpha
        try:
            next_f_val = fun(next_alpha)
            if not np.isfinite(next_f_val):
                status = "next_alpha_non_finite"
                break
        except (OverflowError, ValueError, TypeError): # TypeError für unpassende Operationen
            status = "next_alpha_exception"
            break 

        if next_f_val < optimal_f_val:
            optimal_f_val = next_f_val
            optimal_alpha = next_alpha
            alpha = next_alpha 
            status = "improved" # Letzter guter Status vor potentiellem Abbruch
        else:
            status = "no_improvement"
            break 
    return optimal_alpha, status
    
def gradientDescent(obj_fun, initial_x, max_iter=500, 
                    step_norm_tol=1e-6, func_impr_tol=1e-8, 
                    initial_t_ls=1e-3, callback=None):
    """
    Gradient Descent mit einfacher Liniensuche.
    
    Args:
        obj_fun: Zielfunktion, die 'value' und 'gradient' im Rückgabedict hat
        initial_x: Startpunkt
        max_iter: Maximale Anzahl an Iterationen
        step_norm_tol: Toleranz für die Schrittnorm zum Abbruch
        func_impr_tol: Toleranz für die Funktionsverbesserung zum Abbruch
        initial_t_ls: Initialer Schritt für die Liniensuche
        callback: Funktion, die in jeder Iteration aufgerufen wird
    
    Returns:
        x_current: Endpunkt
        x_history: Liste aller besuchten Punkte
        loss_history: Liste aller berechneten Funktionswerte
        status: Statusmeldung zum Abbruchgrund
    """
    x_current = np.asarray(initial_x, dtype=float).copy()
    status = "Max Iter erreicht"
    
    try:
       eval_current = obj_fun(x_current)
       if not (np.isfinite(eval_current['value']) and np.all(np.isfinite(eval_current['gradient']))):
           raise ValueError("Nicht-finite Werte von obj_fun am Startpunkt.")
    except Exception as e:
       if callback: callback(0, initial_x, np.nan, np.nan, f"Startpunktfehler: {e}")
       return initial_x, [initial_x], [np.nan], f"Startpunktfehler: {e}"
       
    f_current = eval_current['value']
    x_history = [x_current.copy()]
    loss_history = [f_current]
    
    grad_norm = np.linalg.norm(eval_current.get('gradient', np.array([np.nan])))
    if callback:
         callback(0, x_current, f_current, grad_norm, "Initial")

    for iteration in range(1, max_iter + 1):
        gradient = eval_current.get('gradient', np.zeros_like(x_current))
        grad_norm = np.linalg.norm(gradient)

        if grad_norm < 1e-9: 
            status = "Gradientennorm minimal"
            break

        # Liniensuche: Definiere Funktion, die nur von t abhängt
        line_objective_func = lambda t: obj_fun(x_current - t * gradient)['value']
        t_optimal, ls_status = _linesearch(line_objective_func, initial_alpha=initial_t_ls)

        if abs(t_optimal) < 1e-12 or "non_finite" in ls_status or "exception" in ls_status: 
            # Liniensuche findet keinen sinnvollen Schritt mehr
            if status == "Max Iter erreicht": 
                status = f"Liniensuche Problem: {ls_status}"
            break

        step_vector = -t_optimal * gradient
        x_next = x_current + step_vector
        
        try:
            eval_next = obj_fun(x_next)
            # Gradient kann fehlen, wenn nur Wert evaluiert wird (z.B. bei Ende Liniensuche)
            if not (np.isfinite(eval_next['value']) and np.all(np.isfinite(eval_next.get('gradient', x_next)))): 
                status = "Nicht-finite Werte nach Schritt"
                break
        except (OverflowError, ValueError, TypeError) as e:
            status = f"Numerisches Problem nach Schritt: {e}"
            break
            
        f_next = eval_next['value']
        next_grad_norm = np.linalg.norm(eval_next.get('gradient', np.array([np.nan])))
        
        x_history.append(x_next.copy())
        loss_history.append(f_next)
            
        step_size_norm = np.linalg.norm(step_vector)
        improvement = f_current - f_next 
        
        x_current = x_next
        f_current = f_next
        eval_current = eval_next
        
        if callback:
            callback(iteration, x_current, f_current, next_grad_norm, "Iteriert")
        
        if step_size_norm < step_norm_tol:
             status = f"Schrittnorm < Tol."
             break
        if improvement < func_impr_tol and improvement >= 0 and iteration > 5 : 
             status = f"Verbesserung < Tol."
             break
        if f_next > f_current + func_impr_tol*100 and iteration > 3 and improvement < 0 : 
            # Wert wird signifikant schlechter
            status = "Funktionswert stark verschlechtert"
            loss_history[-1] = loss_history[-2] # Letzten (schlechten) Loss nicht behalten
            x_history[-1] = x_history[-2]       # Letzten (schlechten) Punkt nicht behalten
            x_current = x_history[-1]           # Zurück zum vorherigen Punkt
            break            
    
    return x_current, x_history, loss_history, status

def gradientDescentWithMomentum(obj_fun, initial_x, learning_rate=0.01, momentum_beta=0.9,
                                max_iter=500, grad_norm_tol=1e-6, callback=None):
    """
    Gradient Descent mit Momentum.
    
    Args:
        obj_fun: Zielfunktion, die 'value' und 'gradient' im Rückgabedict hat
        initial_x: Startpunkt
        learning_rate: Lernrate
        momentum_beta: Momentum-Parameter (0 = kein Momentum, 0.9 = starkes Momentum)
        max_iter: Maximale Anzahl an Iterationen
        grad_norm_tol: Toleranz für die Gradientennorm zum Abbruch
        callback: Funktion, die in jeder Iteration aufgerufen wird
    
    Returns:
        x_current: Endpunkt
        x_history: Liste aller besuchten Punkte
        loss_history: Liste aller berechneten Funktionswerte
        status: Statusmeldung zum Abbruchgrund
    """
    x_current = np.asarray(initial_x, dtype=float).copy()
    status = "Max Iter erreicht"
    
    try:
        eval_current = obj_fun(x_current)
        if not (np.isfinite(eval_current['value']) and np.all(np.isfinite(eval_current['gradient']))):
            raise ValueError("Nicht-finite Werte von obj_fun am Startpunkt.")
    except Exception as e:
        if callback: callback(0, initial_x, np.nan, np.nan, f"Startpunktfehler: {e}")
        return initial_x, [initial_x], [np.nan], f"Startpunktfehler: {e}"
    
    f_current = eval_current['value']
    x_history = [x_current.copy()]
    loss_history = [f_current]
    
    # Initialisiere Momentum mit Null
    velocity = np.zeros_like(x_current)
    
    grad_norm = np.linalg.norm(eval_current.get('gradient', np.array([np.nan])))
    if callback:
        callback(0, x_current, f_current, grad_norm, "Initial")
    
    for iteration in range(1, max_iter + 1):
        gradient = eval_current.get('gradient', np.zeros_like(x_current))
        grad_norm = np.linalg.norm(gradient)
        
        if grad_norm < grad_norm_tol:
            status = "Gradientennorm < Tol."
            break
        
        # Momentum-Update
        velocity = momentum_beta * velocity - learning_rate * gradient
        x_next = x_current + velocity
        
        try:
            eval_next = obj_fun(x_next)
            if not (np.isfinite(eval_next['value']) and np.all(np.isfinite(eval_next.get('gradient', np.zeros_like(x_next))))):
                status = "Nicht-finite Werte nach Schritt"
                break
        except Exception as e:
            status = f"Numerisches Problem nach Schritt: {e}"
            break
        
        f_next = eval_next['value']
        next_grad_norm = np.linalg.norm(eval_next.get('gradient', np.array([np.nan])))
        
        x_history.append(x_next.copy())
        loss_history.append(f_next)
        
        x_current = x_next
        f_current = f_next
        eval_current = eval_next
        
        if callback:
            callback(iteration, x_current, f_current, next_grad_norm, "Iteriert")
    
    return x_current, x_history, loss_history, status

def adam_optimizer(obj_fun, initial_x, learning_rate=0.001, beta1=0.9, beta2=0.999, 
                  epsilon=1e-8, max_iter=1000, grad_norm_tol=1e-6, callback=None):
    """
    Adam Optimizer - Adaptive Moment Estimation.
    
    Args:
        obj_fun: Zielfunktion, die 'value' und 'gradient' im Rückgabedict hat
        initial_x: Startpunkt
        learning_rate: Lernrate (alpha)
        beta1: Exponentieller Zerfallsrate für den ersten Moment (0.9 empfohlen)
        beta2: Exponentieller Zerfallsrate für den zweiten Moment (0.999 empfohlen)
        epsilon: Kleiner Wert zur Vermeidung von Division durch Null (1e-8 empfohlen)
        max_iter: Maximale Anzahl an Iterationen
        grad_norm_tol: Toleranz für die Gradientennorm zum Abbruch
        callback: Funktion, die in jeder Iteration aufgerufen wird
    
    Returns:
        x_current: Endpunkt
        x_history: Liste aller besuchten Punkte
        loss_history: Liste aller berechneten Funktionswerte
        status: Statusmeldung zum Abbruchgrund
    """
    x_current = np.asarray(initial_x, dtype=float).copy()
    status = "Max Iter erreicht"
    
    try:
        eval_current = obj_fun(x_current)
        if not (np.isfinite(eval_current['value']) and np.all(np.isfinite(eval_current['gradient']))):
            raise ValueError("Nicht-finite Werte von obj_fun am Startpunkt.")
    except Exception as e:
        if callback: callback(0, initial_x, np.nan, np.nan, f"Startpunktfehler: {e}")
        return initial_x, [initial_x], [np.nan], f"Startpunktfehler: {e}"
    
    f_current = eval_current['value']
    x_history = [x_current.copy()]
    loss_history = [f_current]
    
    # Initialisiere Momente
    m = np.zeros_like(x_current)
    v = np.zeros_like(x_current)
    
    grad_norm = np.linalg.norm(eval_current.get('gradient', np.array([np.nan])))
    if callback:
        callback(0, x_current, f_current, grad_norm, "Initial")
    
    for iteration in range(1, max_iter + 1):
        gradient = eval_current.get('gradient', np.zeros_like(x_current))
        grad_norm = np.linalg.norm(gradient)
        
        if grad_norm < grad_norm_tol:
            status = "Gradientennorm < Tol."
            break
        
        # Update Biased First Moment Estimate (Momentum)
        m = beta1 * m + (1 - beta1) * gradient
        # Update Biased Second Moment Estimate (RMSProp)
        v = beta2 * v + (1 - beta2) * gradient**2
        
        # Korrigierte Schätzungen
        m_corrected = m / (1 - beta1**(iteration))
        v_corrected = v / (1 - beta2**(iteration))
        
        # Update Parameter
        step = learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
        x_next = x_current - step
        
        try:
            eval_next = obj_fun(x_next)
            if not (np.isfinite(eval_next['value']) and np.all(np.isfinite(eval_next.get('gradient', np.zeros_like(x_next))))):
                status = "Nicht-finite Werte nach Schritt"
                break
        except Exception as e:
            status = f"Numerisches Problem nach Schritt: {e}"
            break
        
        f_next = eval_next['value']
        next_grad_norm = np.linalg.norm(eval_next.get('gradient', np.array([np.nan])))
        
        x_history.append(x_next.copy())
        loss_history.append(f_next)
        
        x_current = x_next
        f_current = f_next
        eval_current = eval_next
        
        if callback:
            callback(iteration, x_current, f_current, next_grad_norm, "Iteriert")
    
    return x_current, x_history, loss_history, status