# improved_optimizer.py
"""
Verbesserte Implementierung von Optimierungsalgorithmen mit Multi-Start-Strategie
zur zuverlässigeren Findung des globalen Minimums.
"""
import numpy as np
import optimization_algorithms_v3 as oa
from typing import Dict, List, Tuple, Callable, Optional, Any

def multi_start_optimization(
    obj_fun: Callable, 
    optimizer_func: Callable, 
    n_starts: int = 5, 
    x_range: Tuple[float, float] = (-2, 2), 
    y_range: Tuple[float, float] = (-2, 2),
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[Tuple]]:
    """
    Führt Multi-Start-Optimierung mit verschiedenen Startpunkten durch, 
    um das globale Minimum zuverlässiger zu finden.
    
    Args:
        obj_fun: Die zu optimierende Funktion
        optimizer_func: Die zu verwendende Optimierungsfunktion
        n_starts: Anzahl der verschiedenen Startpunkte
        x_range, y_range: Bereich für die zufälligen Startpunkte
        seed: Seed für den Zufallszahlengenerator
        
    Returns:
        best_x: Der beste gefundene Punkt
        all_results: Liste aller Optimierungsergebnisse (x_opt, x_history, loss_history, status) für jeden Start
    """
    if seed is not None:
        np.random.seed(seed)
    
    all_results = []
    best_value = float('inf')
    best_x = None
    
    for i in range(n_starts):
        # Generiere Startpunkt
        x_init = np.random.uniform(x_range[0], x_range[1])
        y_init = np.random.uniform(y_range[0], y_range[1])
        initial_x = np.array([x_init, y_init])
        
        try:
            # Führe Optimierung durch
            result = optimizer_func(obj_fun, initial_x)
            x_opt, x_history, loss_history, status = result
            
            # Evaluiere das Ergebnis
            try:
                final_value = obj_fun(x_opt)['value']
            except Exception:
                # Bei Fehler in der Funktionsevaluation verwende den letzten Wert der Historie
                final_value = loss_history[-1] if loss_history else float('inf')
            
            all_results.append((x_opt, x_history, loss_history, status))
            
            # Prüfe, ob dies das beste Ergebnis ist
            if final_value < best_value:
                best_value = final_value
                best_x = x_opt
        except Exception as e:
            print(f"Fehler bei Start {i+1}: {e}")
            continue
    
    if best_x is None and all_results:
        # Wenn keine gültige Evaluation, aber Ergebnisse vorhanden sind, verwende das erste
        best_x = all_results[0][0]
    elif best_x is None:
        # Fallback, wenn gar nichts funktioniert hat
        best_x = np.array([0.0, 0.0])
    
    # Stelle sicher, dass all_results nicht leer ist
    if not all_results and best_x is not None:
        # Füge einen Dummy-Eintrag hinzu, damit die Visualisierung nicht abstürzt
        all_results.append((best_x, [best_x], [0.0], "Fallback-Ergebnis"))
    
    return best_x, all_results

def adaptive_multistart_optimization(
    obj_fun: Callable, 
    optimizer_func: Callable, 
    initial_starts: int = 5, 
    max_starts: int = 20,
    convergence_threshold: float = 1e-6,
    x_range: Tuple[float, float] = (-5, 5), 
    y_range: Tuple[float, float] = (-5, 5),
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[Tuple]]:
    """
    Führt eine adaptive Multi-Start-Optimierung durch. Beginnt mit einer kleineren Anzahl von Starts
    und erhöht diese, wenn keine Konvergenz zu einem gemeinsamen Minimum gefunden wird.
    
    Args:
        obj_fun: Die zu optimierende Funktion
        optimizer_func: Die zu verwendende Optimierungsfunktion
        initial_starts: Anfängliche Anzahl der Starts
        max_starts: Maximale Anzahl der Starts
        convergence_threshold: Schwellenwert für die Konvergenz (Funktionswertdifferenz)
        x_range, y_range: Bereich für die zufälligen Startpunkte
        seed: Seed für den Zufallszahlengenerator
        
    Returns:
        best_x: Der beste gefundene Punkt
        all_results: Liste aller Optimierungsergebnisse
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Starte mit der initialen Anzahl
    n_starts = initial_starts
    best_x, all_results = multi_start_optimization(
        obj_fun, optimizer_func, n_starts, x_range, y_range, seed
    )
    
    # Sammle die Funktionswerte der besten Punkte
    function_values = []
    for x_opt, _, loss_history, _ in all_results:
        try:
            val = obj_fun(x_opt)['value']
            function_values.append(val)
        except Exception:
            # Bei Fehler verwende den letzten Wert aus der Historie
            if loss_history:
                function_values.append(loss_history[-1])
    
    # Überprüfe, ob wir konvergieren
    if function_values:
        min_val = min(function_values)
        max_val = max(function_values)
        
        # Solange die Differenz größer als der Schwellenwert ist und wir unter dem Maximum liegen
        while (max_val - min_val > convergence_threshold) and (n_starts < max_starts):
            # Erhöhe die Anzahl der Starts
            additional_starts = min(5, max_starts - n_starts)
            n_starts += additional_starts
            
            # Führe weitere Optimierungen durch
            _, additional_results = multi_start_optimization(
                obj_fun, optimizer_func, additional_starts, x_range, y_range
            )
            
            # Füge die neuen Ergebnisse hinzu
            all_results.extend(additional_results)
            
            # Aktualisiere die Funktionswerte
            function_values = []
            for x_opt, _, loss_history, _ in all_results:
                try:
                    val = obj_fun(x_opt)['value']
                    function_values.append(val)
                except Exception:
                    if loss_history:
                        function_values.append(loss_history[-1])
            
            if function_values:
                min_val = min(function_values)
                max_val = max(function_values)
            else:
                break
    
    # Finde den besten Punkt aus allen Durchläufen
    best_value = float('inf')
    best_x = None
    
    for x_opt, _, loss_history, _ in all_results:
        try:
            val = obj_fun(x_opt)['value']
            if val < best_value:
                best_value = val
                best_x = x_opt
        except Exception:
            if loss_history and loss_history[-1] < best_value:
                best_value = loss_history[-1]
                best_x = x_opt
    
    # Prüfe auf leere Ergebnisse
    if not all_results:
        # Fallback wenn keine Ergebnisse vorhanden sind
        dummy_point = np.array([0.0, 0.0])
        all_results.append((dummy_point, [dummy_point], [0.0], "Keine Konvergenz gefunden"))
        best_x = dummy_point
    elif best_x is None:
        # Wenn kein bester Punkt gefunden wurde, nehme den ersten
        best_x = all_results[0][0]
        
    return best_x, all_results

def random_restart_gd(
    obj_fun: Callable,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    max_restarts: int = 5,
    max_iter_per_restart: int = 100,
    step_norm_tol: float = 1e-6,
    func_impr_tol: float = 1e-8,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[np.ndarray], List[float], str]:
    """
    Implementiert einen Gradientenabstieg mit zufälligen Neustarts, 
    wenn die Konvergenz in ein lokales Minimum vermutet wird.
    
    Args:
        obj_fun: Die zu optimierende Funktion
        x_range, y_range: Bereich für die zufälligen Neustarts
        max_restarts: Maximale Anzahl von Neustarts
        max_iter_per_restart: Maximale Anzahl an Iterationen pro Neustart
        step_norm_tol: Toleranz für die Schrittnorm zum Abbruch
        func_impr_tol: Toleranz für die Funktionsverbesserung zum Abbruch
        seed: Seed für den Zufallszahlengenerator
        
    Returns:
        x_best: Der beste gefundene Punkt
        x_history_combined: Kombinierte Geschichte aller Pfade
        loss_history_combined: Kombinierte Geschichte aller Funktionswerte
        status: Statusmeldung
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_history_combined = []
    loss_history_combined = []
    best_x = None
    best_loss = float('inf')
    status = "Max restarts erreicht"
    
    for restart in range(max_restarts):
        # Generiere Startpunkt
        x_init = np.random.uniform(x_range[0], x_range[1])
        y_init = np.random.uniform(y_range[0], y_range[1])
        initial_x = np.array([x_init, y_init])
        
        # Führe Optimierung durch
        x_opt, x_history, loss_history, curr_status = oa.gradientDescent(
            obj_fun, initial_x, max_iter=max_iter_per_restart,
            step_norm_tol=step_norm_tol, func_impr_tol=func_impr_tol
        )
        
        # Füge die Geschichte hinzu (mit Markierung für Neustart)
        if x_history_combined:
            # Füge ein NaN ein, um den Neustart zu markieren
            x_history_combined.append(np.array([np.nan, np.nan]))
            loss_history_combined.append(np.nan)
        
        x_history_combined.extend(x_history)
        loss_history_combined.extend(loss_history)
        
        # Prüfe, ob dies der beste Run ist
        current_loss = loss_history[-1]
        if current_loss < best_loss:
            best_loss = current_loss
            best_x = x_opt
            
        # Wenn wir ein gutes Minimum gefunden haben, beenden wir frühzeitig
        if best_loss < func_impr_tol * 10:
            status = "Gutes Minimum gefunden"
            break
    
    # Stelle sicher, dass best_x nicht None ist
    if best_x is None:
        best_x = np.array([0.0, 0.0])
        status = "Keine Konvergenz"
    
    # Stelle sicher, dass Historien nicht leer sind
    if not x_history_combined:
        x_history_combined = [best_x]  # Zumindest einen Punkt als Geschichte
    if not loss_history_combined:
        loss_history_combined = [0.0]  # Dummy-Wert für leere Loss-Historie
    
    return best_x, x_history_combined, loss_history_combined, status

# Hier fügen wir die verbesserten Algorithmen in einer einheitlichen Bibliothek zusammen
OPTIMIZERS_EXTENDED = {
    "GD_Simple_LS": oa.gradientDescent,
    "GD_Momentum": oa.gradientDescentWithMomentum,
    "Adam": oa.adam_optimizer,
    "GD_Multi_Start": lambda obj_fun, initial_x, **kwargs: multi_start_optimization(
        obj_fun, 
        lambda f, x: oa.gradientDescent(f, x, **kwargs),
        n_starts=5
    )[0],
    "GD_Adaptive_Multi": lambda obj_fun, initial_x, **kwargs: adaptive_multistart_optimization(
        obj_fun,
        lambda f, x: oa.gradientDescent(f, x, **kwargs),
        initial_starts=3,
        max_starts=10
    )[0],
    "GD_Random_Restart": random_restart_gd
}

# Funktionen zur Erzeugung von Optimierungsfunktionen mit angepassten Parametern

def create_gd_simple_ls(max_iter=500, step_norm_tol=1e-6, func_impr_tol=1e-8, initial_t_ls=1e-3):
    """Erzeugt einen Gradientenabstieg mit Liniensuche mit benutzerdefinierten Parametern."""
    return lambda obj_fun, initial_x: oa.gradientDescent(
        obj_fun, initial_x, 
        max_iter=max_iter,
        step_norm_tol=step_norm_tol,
        func_impr_tol=func_impr_tol,
        initial_t_ls=initial_t_ls
    )

def create_gd_momentum(max_iter=500, learning_rate=0.01, momentum_beta=0.9, grad_norm_tol=1e-6):
    """Erzeugt einen Gradientenabstieg mit Momentum mit benutzerdefinierten Parametern."""
    return lambda obj_fun, initial_x: oa.gradientDescentWithMomentum(
        obj_fun, initial_x, 
        learning_rate=learning_rate,
        momentum_beta=momentum_beta,
        max_iter=max_iter,
        grad_norm_tol=grad_norm_tol
    )

def create_adam(max_iter=1000, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, grad_norm_tol=1e-6):
    """Erzeugt einen Adam-Optimierer mit benutzerdefinierten Parametern."""
    return lambda obj_fun, initial_x: oa.adam_optimizer(
        obj_fun, initial_x, 
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        max_iter=max_iter,
        grad_norm_tol=grad_norm_tol
    )

def create_multi_start_optimizer(base_optimizer, n_starts=5, x_range=(-5, 5), y_range=(-5, 5)):
    """Erzeugt einen Multi-Start-Optimierer mit dem angegebenen Basisoptimierer."""
    return lambda obj_fun, initial_x: multi_start_optimization(
        obj_fun, base_optimizer, n_starts, x_range, y_range
    )[0]

def create_adaptive_multi_start(base_optimizer, initial_starts=5, max_starts=20, 
                              convergence_threshold=1e-6, x_range=(-5, 5), y_range=(-5, 5)):
    """Erzeugt einen adaptiven Multi-Start-Optimierer mit dem angegebenen Basisoptimierer."""
    return lambda obj_fun, initial_x: adaptive_multistart_optimization(
        obj_fun, base_optimizer, initial_starts, max_starts, 
        convergence_threshold, x_range, y_range
    )[0]