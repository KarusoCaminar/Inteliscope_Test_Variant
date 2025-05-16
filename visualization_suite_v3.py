# visualization_suite_v3.py
"""
Modul zur Visualisierung von Optimierungslandschaften und -pfaden.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Any, Callable

def generate_2d_grid(x_range, y_range, n_points=100):
    """Erzeugt ein 2D-Gitter für Oberflächenplots."""
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    return X, Y

def evaluate_function_on_grid(func, X, Y):
    """Wertet eine Funktion auf einem 2D-Gitter aus."""
    Z = np.zeros_like(X)
    n_rows, n_cols = X.shape
    
    for i in range(n_rows):
        for j in range(n_cols):
            point = np.array([X[i, j], Y[i, j]])
            result = func(point)
            Z[i, j] = result['value']
            
    return Z

def compute_gradient_field(func, x_range, y_range, n_points=20):
    """Berechnet ein Gradientenfeld für einen Konturplot."""
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(n_points):
        for j in range(n_points):
            point = np.array([X[i, j], Y[i, j]])
            result = func(point)
            grad = result['gradient']
            # Negatives Gradientenfeld (Abstiegsrichtung)
            U[i, j] = -grad[0]
            V[i, j] = -grad[1]
            
    # Normalisieren für bessere Visualisierung
    magnitude = np.sqrt(U**2 + V**2)
    nonzero = magnitude > 1e-10
    U[nonzero] = U[nonzero] / magnitude[nonzero]
    V[nonzero] = V[nonzero] / magnitude[nonzero]
    
    return X, Y, U, V

def plot_loss_curves(ax, loss_histories_dict, title="Verlauf des Loss"):
    """
    Zeichnet Verlaufskurven für Loss/Funktionswerte.
    
    Args:
        ax: Matplotlib-Achse
        loss_histories_dict: Dictionary mit Label als Key und Loss-History als Value
        title: Titel des Plots
    
    Returns:
        ax: Die aktualisierte Achse
    """
    ax.clear()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Loss / Funktionswert", fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True, linestyle=':', alpha=0.7)

    has_data_to_plot = False
    all_finite_loss_values = []

    for label, loss_history_raw in loss_histories_dict.items():
        if loss_history_raw and isinstance(loss_history_raw, (list, np.ndarray)):
            loss_history_plot = [l for l in loss_history_raw if np.isfinite(l)]
            if loss_history_plot:
                ax.plot(loss_history_plot, label=label, alpha=0.8, linewidth=1.5)
                all_finite_loss_values.extend(loss_history_plot)
                has_data_to_plot = True

    if has_data_to_plot and all_finite_loss_values:
        if ax.get_legend() is None or not ax.get_legend().get_texts(): # Nur Legende wenn nicht schon da und Inhalt hat
             ax.legend(fontsize='small')
        min_val_overall = np.min(all_finite_loss_values)
        max_val_overall = np.max(all_finite_loss_values)

        current_yscale = ax.get_yscale()
        if min_val_overall > 1e-9 and (max_val_overall / min_val_overall) > 500 and current_yscale != 'log':
            ax.set_yscale('log')
        elif any(v < 0 for v in all_finite_loss_values) and min_val_overall != max_val_overall and current_yscale != 'symlog':
            abs_values = np.abs([float(v) for v in all_finite_loss_values if v != 0])
            percentile_val = np.percentile(abs_values, 10) if len(abs_values) > 0 else 1e-9
            linthresh_val = max(1e-9, float(percentile_val))
            ax.set_yscale('symlog', linthresh=linthresh_val)
        elif current_yscale not in ['log', 'symlog']:
            ax.set_yscale('linear')

    elif has_data_to_plot and not all_finite_loss_values:
         ax.text(0.5, 0.5, "Loss-Werte nicht endlich.", ha='center', va='center', transform=ax.transAxes, fontsize=9, color='red')
    else:
        ax.text(0.5, 0.5, "Keine Loss-Daten", ha='center', va='center', transform=ax.transAxes, fontsize=9, color='gray')

    # Korrigierte Zeile: Prüfe ob canvas existiert
    if hasattr(ax.figure, 'canvas') and ax.figure.canvas: 
        ax.figure.canvas.draw_idle()
    return ax


def plot_3d_surface_and_paths(fig, ax, func_obj,
                              p1_range, p2_range,
                              paths_dict=None,
                              title="3D Landschaft",
                              p_names=('P1', 'P2', 'Wert'),
                              plot_resolution=50,
                              z_log_scale=False,
                              view=(30, -120)):
    """
    Zeichnet eine 3D-Oberfläche und Optimierungspfade.
    
    Args:
        fig: Matplotlib Figure
        ax: Matplotlib 3D-Achse
        func_obj: Funktion, die für Params ein Dict mit 'value' zurückgibt
        p1_range: Bereich für Parameter 1 (min, max)
        p2_range: Bereich für Parameter 2 (min, max)
        paths_dict: Dict mit Label als Key und Pfad-Info als Value
        title: Titel des Plots
        p_names: Namen der Parameter für Achsenbeschriftung (p1, p2, wert)
        plot_resolution: Anzahl der Punkte pro Dimension für das Gitter
        z_log_scale: Ob Z-Achse logarithmisch skaliert werden soll
        view: Blickwinkel (elevation, azimuth)
    
    Returns:
        ax: Die aktualisierte Achse
    """
    ax.clear() # Wichtig: Achse komplett leeren
    
    # Attribut für Colorbar-Handling hinzufügen, wenn nicht vorhanden
    if not hasattr(ax, '_my_current_3d_colorbar'):
        ax._my_current_3d_colorbar = None
    
    num_points_grid = int(plot_resolution)

    # Erweitere den Visualisierungsbereich für bessere Übersicht
    x_vals = np.linspace(p1_range[0], p1_range[1], num_points_grid)
    y_vals = np.linspace(p2_range[0], p2_range[1], num_points_grid)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    Z_raw = np.full_like(X_grid, np.nan)

    # Oberfläche berechnen
    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            params_ij = np.array([X_grid[i, j], Y_grid[i, j]])
            try:
                result_dict = func_obj(params_ij)
                Z_raw[i, j] = result_dict.get('value', np.nan)
            except Exception as e_func_call:
                if i < 1 and j < 1:
                    print(f"vs.py (3D) - FEHLER func_obj({params_ij}). E: {e_func_call}")
                Z_raw[i, j] = np.nan

    z_label_text = p_names[2] if len(p_names) > 2 else 'Wert'
    Z_plot_finite_for_lims = np.array([])

    # Log-Skalierung anwenden, wenn gewünscht
    if z_log_scale:
        Z_plot = np.full_like(Z_raw, np.nan)
        valid_Z_raw = np.isfinite(Z_raw) & (Z_raw > 1e-12)
        if np.any(valid_Z_raw):
            Z_plot[valid_Z_raw] = np.log10(Z_raw[valid_Z_raw])
        z_label_text = f"Log10({z_label_text})"
        if np.any(np.isfinite(Z_plot)): Z_plot_finite_for_lims = Z_plot[np.isfinite(Z_plot)]
    else:
        Z_plot = Z_raw
        if np.any(np.isfinite(Z_plot)): Z_plot_finite_for_lims = Z_plot[np.isfinite(Z_plot)]

    surf = None
    if Z_plot_finite_for_lims.size > 0 and np.any(np.isfinite(Z_plot)): # Nur plotten wenn valide Daten
        try:
            surf = ax.plot_surface(X_grid, Y_grid, Z_plot, cmap='viridis', edgecolor='k',
                                   linewidth=0.05, alpha=0.7, antialiased=True, rstride=1, cstride=1)
        except Exception as e:
            print(f"Fehler beim Plotten der 3D-Oberfläche: {e}")
            ax.text(0.5, 0.5, 0.5, "Fehler Plot Oberfläche", ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 0.5, "Keine validen Daten für 3D-Plot", ha='center', va='center', transform=ax.transAxes)

    # Achsenbeschriftungen
    ax.set_xlabel(p_names[0], fontsize=10)
    ax.set_ylabel(p_names[1], fontsize=10)
    ax.set_zlabel(z_label_text, fontsize=10)
    ax.set_title(title, fontsize=12)

    # Limits (um Teile der Oberfläche nicht abzuschneiden)
    # Explizite Limits nur auf X/Y, Z-Achse automatisch skalieren lassen
    ax.set_xlim(p1_range)
    ax.set_ylim(p2_range)

    # Verbesserte Kameraposition und Zoom
    ax.view_init(elev=view[0], azim=view[1])

    # Pfade nur hinzufügen, wenn Surface Plot existiert
    if surf is not None and paths_dict:
        for label, path_points in paths_dict.items():
            if isinstance(path_points, np.ndarray) and len(path_points.shape) == 2 and path_points.shape[1] >= 2:
                path_p1 = path_points[:, 0]
                path_p2 = path_points[:, 1]
                path_z = np.zeros_like(path_p1)
                
                # Berechne z-Werte an den gegebenen Punkten
                for k, (p1, p2) in enumerate(zip(path_p1, path_p2)):
                    try:
                        result = func_obj(np.array([p1, p2]))
                        raw_value = result.get('value', np.nan)
                        if z_log_scale and raw_value > 1e-12:
                            path_z[k] = np.log10(raw_value)
                        else:
                            path_z[k] = raw_value
                    except:
                        path_z[k] = np.nan  # Bei Fehlern NaN eintragen
                
                # Plotte den Optimierungspfad
                valid_idx = np.isfinite(path_z)
                if np.any(valid_idx):
                    ax.plot(path_p1[valid_idx], path_p2[valid_idx], path_z[valid_idx], 
                          '-o', linewidth=2, markersize=4, label=label)

        # Füge Legende hinzu (nur wenn Pfade existieren)
        if paths_dict:
            ax.legend(loc='upper left', fontsize=8)

    # Stelle sicher, dass die Colorbar bei jeder Änderung aktualisiert wird
    if hasattr(fig, '_current_surface_colorbar') and fig._current_surface_colorbar is not None:
        cb = fig._current_surface_colorbar
        cb.remove()
        fig._current_surface_colorbar = None

    if surf is not None:
        fig._current_surface_colorbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Plot rendern
    if hasattr(fig, 'canvas') and fig.canvas:
        fig.canvas.draw_idle()

    return ax

def create_contour_plot_matplotlib(func, x_range, y_range, n_points=200, ax=None, 
                                 title=None, cmap='viridis', add_gradients=True,
                                 optimization_paths=None, minima=None, levels=30):
    """Erstellt einen Konturplot mit Matplotlib."""
    X, Y = generate_2d_grid(x_range, y_range, n_points)
    Z = evaluate_function_on_grid(func, X, Y)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Konturplot
    contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.8)
    
    # Konturlinien hinzufügen
    contour_lines = ax.contour(X, Y, Z, levels=levels//2, colors='black', alpha=0.4, linewidths=0.5)
    
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    
    # Colorbar hinzufügen
    plt.colorbar(contour, ax=ax)
    
    # Gradientenfeld hinzufügen, wenn gewünscht
    if add_gradients:
        X_grad, Y_grad, U, V = compute_gradient_field(func, x_range, y_range, n_points=20)
        ax.quiver(X_grad, Y_grad, U, V, color='white', alpha=0.6, scale=30, width=0.003)
    
    # Optimierungspfade hinzufügen, wenn vorhanden
    if optimization_paths:
        for label, path in optimization_paths.items():
            path_points = np.array(path)
            ax.plot(path_points[:, 0], path_points[:, 1], 'o-', linewidth=2, markersize=4, label=label)
        ax.legend()
    
    # Minima markieren, wenn vorhanden
    if minima and len(minima) > 0:
        minima_x = [m[0] for m in minima]
        minima_y = [m[1] for m in minima]
        ax.scatter(minima_x, minima_y, c='red', s=100, marker='*', 
                  edgecolor='black', linewidth=1, label='Globale Minima')
    
    return ax

def create_interactive_surface_plotly(func, x_range, y_range, n_points=100, 
                                    title=None, colorscale='viridis', 
                                    optimization_paths=None, minima=None):
    """Erstellt einen interaktiven 3D-Oberflächenplot mit Plotly."""
    X, Y = generate_2d_grid(x_range, y_range, n_points)
    Z = evaluate_function_on_grid(func, X, Y)
    
    # Begrenzen der Z-Werte für eine bessere Visualisierung
    # Schutz vor Infinity-Werten oder NaNs
    Z = np.nan_to_num(Z, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Berechne 1. und 99. Perzentil für robusteres Clipping
    z_min = np.percentile(Z, 1)
    z_max = np.percentile(Z, 99)
    
    # Stelle sicher, dass z_max > z_min ist
    if z_max <= z_min:
        z_min = np.min(Z)
        z_max = np.max(Z)
        # Wenn immer noch gleich, füge einen Offset hinzu
        if z_max <= z_min:
            z_max = z_min + 1.0
    
    # Clip Werte für bessere Darstellung
    Z = np.clip(Z, z_min, z_max)
    
    # Erstellen der Figur
    fig = go.Figure()
    
    # Oberfläche hinzufügen
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale=colorscale, opacity=0.9))
    
    # Layout anpassen
    camera = dict(
        eye=dict(x=1.5, y=1.5, z=1.2)
    )
    
    fig.update_layout(
        title=title,
        autosize=True,
        width=800,
        height=600,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='f(X,Y)',
            aspectratio=dict(x=1, y=1, z=0.7),
            camera=camera,
            # Erweitere den sichtbaren Bereich für bessere Navigation
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range)
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    
    # Optimierungspfade hinzufügen, wenn vorhanden
    if optimization_paths:
        for label, path in optimization_paths.items():
            path_points = np.array(path)
            path_x = path_points[:, 0]
            path_y = path_points[:, 1]
            # Berechne Z-Werte für die Pfadpunkte
            path_z = np.zeros_like(path_x)
            for i, (x, y) in enumerate(zip(path_x, path_y)):
                path_z[i] = func(np.array([x, y]))['value']
            
            # Linienpfad hinzufügen
            # Pfad hinzufügen - mit besserer Visualisierung und Farbverlauf
            # Erstelle einen Farbverlauf basierend auf der Position im Pfad
            colors = [f'rgb({int(255*(1-i/len(path_x)))}, {int(50+200*i/len(path_x))}, {int(50*i/len(path_x))})' 
                     for i in range(len(path_x))]
            
            # Füge eine Linie für den gesamten Pfad hinzu
            fig.add_trace(go.Scatter3d(
                x=path_x, y=path_y, z=path_z,
                mode='lines',
                name=f"{label} (Pfad)",
                line=dict(width=6, color='rgba(255, 50, 50, 0.8)'),
                showlegend=True
            ))
            
            # Füge Marker für einzelne Punkte hinzu
            fig.add_trace(go.Scatter3d(
                x=path_x, y=path_y, z=path_z,
                mode='markers',
                name=f"{label} (Punkte)",
                marker=dict(
                    size=5,
                    color=colors,
                    colorscale='Viridis',
                    line=dict(color='black', width=1)
                ),
                showlegend=True
            ))
            
            # Füge eine Markierung für den Startpunkt hinzu
            fig.add_trace(go.Scatter3d(
                x=[path_x[0]], y=[path_y[0]], z=[path_z[0]],
                mode='markers',
                name='Startpunkt',
                marker=dict(
                    size=8,
                    color='blue',
                    symbol='circle',
                    line=dict(color='darkblue', width=1)
                ),
                showlegend=True
            ))
            
            # Füge eine Markierung für den Endpunkt hinzu
            fig.add_trace(go.Scatter3d(
                x=[path_x[-1]], y=[path_y[-1]], z=[path_z[-1]],
                mode='markers',
                name='Endpunkt',
                marker=dict(
                    size=8,
                    color='green',
                    symbol='diamond',
                    line=dict(color='darkgreen', width=1)
                ),
                showlegend=True
            ))
    
    # Minima markieren, wenn vorhanden
    if minima and len(minima) > 0:
        minima_x = [m[0] for m in minima]
        minima_y = [m[1] for m in minima]
        minima_z = [func(np.array([x, y]))['value'] for x, y in minima]
        
        fig.add_trace(go.Scatter3d(
            x=minima_x, y=minima_y, z=minima_z,
            mode='markers',
            name='Globale Minima',
            marker=dict(
                size=8,
                color='gold',
                symbol='diamond',
                line=dict(color='black', width=1)
            )
        ))
    
    return fig

def create_interactive_contour_plotly(func, x_range, y_range, n_points=200, 
                                    title=None, colorscale='viridis', 
                                    optimization_paths=None, minima=None,
                                    levels=30, add_gradients=True):
    """Erstellt einen interaktiven Konturplot mit Plotly."""
    X, Y = generate_2d_grid(x_range, y_range, n_points)
    Z = evaluate_function_on_grid(func, X, Y)
    
    # Erstellen der Figur
    fig = go.Figure()
    
    # Kontur hinzufügen
    fig.add_trace(go.Contour(
        z=Z,
        x=np.linspace(x_range[0], x_range[1], n_points),
        y=np.linspace(y_range[0], y_range[1], n_points),
        colorscale=colorscale,
        contours=dict(
            start=np.min(Z),
            end=np.percentile(Z, 95),  # Begrenzen für bessere Visualisierung
            size=(np.percentile(Z, 95) - np.min(Z)) / levels,
            coloring='fill',
            showlabels=True,
            labelfont=dict(size=10, color='white')
        ),
        colorbar=dict(
            title=dict(
                text='f(X,Y)',
                side='right',
                font=dict(size=12)
            ),
            tickfont=dict(size=10)
        ),
        hoverinfo='x+y+z'
    ))
    
    # Gradientenfeld hinzufügen, wenn gewünscht
    if add_gradients:
        X_grad, Y_grad, U, V = compute_gradient_field(func, x_range, y_range, n_points=15)
        
        # Quiver-Plot - verwende weniger Pfeile für bessere Übersichtlichkeit
        step = max(1, len(X_grad) // 15)  # Reduziere die Anzahl der Pfeile
        for i in range(0, len(X_grad), step):
            for j in range(0, len(X_grad[0]), step):
                fig.add_trace(go.Scatter(
                    x=[X_grad[i, j], X_grad[i, j] + U[i, j] * 0.3],
                    y=[Y_grad[i, j], Y_grad[i, j] + V[i, j] * 0.3],
                    mode='lines',
                    line=dict(color='rgba(255, 255, 255, 0.5)', width=1),
                    showlegend=False,
                    hoverinfo='none'
                ))
    
    # Optimierungspfade hinzufügen, wenn vorhanden
    if optimization_paths:
        for label, path in optimization_paths.items():
            if isinstance(path, np.ndarray) and len(path) > 0:
                path_points = np.array(path)
                
                # Erstelle einen Farbverlauf basierend auf der Position im Pfad
                colors = [f'rgb({int(255*(1-i/len(path_points)))}, {int(50+200*i/len(path_points))}, {int(50*i/len(path_points))})' 
                         for i in range(len(path_points))]
                
                # Hauptpfad mit schönerer Linie
                fig.add_trace(go.Scatter(
                    x=path_points[:, 0],
                    y=path_points[:, 1],
                    mode='lines',
                    name=f"{label} (Pfad)",
                    line=dict(width=4, color='rgba(255, 50, 50, 0.8)'),
                    hoverinfo='none',
                    showlegend=True
                ))
                
                # Pfadpunkte mit Farbverlauf
                fig.add_trace(go.Scatter(
                    x=path_points[:, 0],
                    y=path_points[:, 1],
                    mode='markers',
                    name=f"{label} (Punkte)",
                    marker=dict(
                        size=8, 
                        color=colors,
                        line=dict(color='black', width=1)
                    ),
                    hoverinfo='x+y+text',
                    hovertext=[f'Schritt {i}: ({x:.4f}, {y:.4f})' for i, (x, y) in enumerate(path_points)],
                    showlegend=True
                ))
                
                # Startpunkt markieren
                if len(path_points) > 0:
                    fig.add_trace(go.Scatter(
                        x=[path_points[0, 0]],
                        y=[path_points[0, 1]],
                        mode='markers',
                        name='Startpunkt',
                        marker=dict(
                            size=12,
                            color='blue',
                            symbol='circle',
                            line=dict(color='darkblue', width=1)
                        ),
                        showlegend=True
                    ))
                
                # Endpunkt markieren
                if len(path_points) > 1:
                    fig.add_trace(go.Scatter(
                        x=[path_points[-1, 0]],
                        y=[path_points[-1, 1]],
                        mode='markers',
                        name='Endpunkt',
                        marker=dict(
                            size=12,
                            color='green',
                            symbol='star',
                            line=dict(color='darkgreen', width=1)
                        ),
                        showlegend=True
                    ))
    
    # Minima markieren, wenn vorhanden
    if minima and len(minima) > 0:
        minima_x = [m[0] for m in minima]
        minima_y = [m[1] for m in minima]
        
        fig.add_trace(go.Scatter(
            x=minima_x,
            y=minima_y,
            mode='markers',
            name='Globale Minima',
            marker=dict(
                size=12,
                color='red',
                symbol='star',
                line=dict(color='black', width=1)
            )
        ))
    
    # Layout anpassen - erweiterte Grenzen für bessere Sichtbarkeit
    fig.update_layout(
        title=title,
        autosize=True,
        width=800,
        height=600,
        xaxis=dict(
            title='X', 
            tickfont=dict(size=10),
            range=x_range  # Explizite Grenzen setzen
        ),
        yaxis=dict(
            title='Y', 
            tickfont=dict(size=10), 
            scaleanchor="x", 
            scaleratio=1,
            range=y_range  # Explizite Grenzen setzen
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        hovermode='closest'
    )
    
    return fig

def plot_optimization_history(loss_history, algorithm_name, title=None, log_scale=True):
    """Erzeugt einen Plot der Verlaufshistorie einer Optimierung."""
    
    fig = go.Figure()
    iterations = list(range(len(loss_history)))
    
    # Füge die Verlaufskurve hinzu
    fig.add_trace(go.Scatter(
        x=iterations,
        y=loss_history,
        mode='lines+markers',
        name=algorithm_name,
        line=dict(width=2),
        marker=dict(size=5),
    ))
    
    # Layout anpassen
    fig.update_layout(
        title=title or f'Optimierungsverlauf: {algorithm_name}',
        xaxis=dict(title='Iteration', tickfont=dict(size=10)),
        yaxis=dict(
            title='Funktionswert',
            tickfont=dict(size=10),
            type='log' if log_scale else 'linear'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        hovermode='closest'
    )
    
    return fig

def animate_optimization_path(func, x_history, loss_history, x_range, y_range, 
                             n_points=100, title=None, cmap='viridis'):
    """
    Erstellt eine Animation des Optimierungspfades 
    und gibt eine Liste von Matplotlib-Figuren zurück.
    """
    X, Y = generate_2d_grid(x_range, y_range, n_points)
    Z = evaluate_function_on_grid(func, X, Y)
    
    figures = []
    
    for i in range(len(x_history)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Konturplot
        contour = ax1.contourf(X, Y, Z, 50, cmap=cmap, alpha=0.8)
        ax1.contour(X, Y, Z, 20, colors='black', alpha=0.4, linewidths=0.5)
        
        # Bisheriger Pfad
        path = np.array(x_history[:i+1])
        ax1.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=4)
        ax1.plot(path[-1, 0], path[-1, 1], 'bo', markersize=8)  # Aktueller Punkt
        
        ax1.set_xlim(x_range)
        ax1.set_ylim(y_range)
        ax1.set_title(f'Optimierungspfad (Iteration {i})', fontsize=14)
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        
        # Verlaufskurve
        ax2.plot(range(i+1), loss_history[:i+1], 'g-', linewidth=2)
        ax2.set_title('Funktionswert über die Iterationen', fontsize=14)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('f(x)', fontsize=12)
        if i > 0:  # Logskala nur aktivieren, wenn wir mehr als einen Punkt haben
            ax2.set_yscale('log')
        
        plt.tight_layout()
        figures.append(fig)
    
    return figures

def plot_multi_start_results(all_results, func, x_range, y_range, n_points=200):
    """
    Visualisiert die Ergebnisse mehrerer Optimierungsläufe mit verschiedenen Startpunkten.
    
    Args:
        all_results: Liste von (x_opt, x_history, loss_history, status) Tupeln
        func: Die zu optimierende Funktion
        x_range, y_range: Darstellungsbereich
    """
    # Erstelle den Konturplot
    X, Y = generate_2d_grid(x_range, y_range, n_points)
    Z = evaluate_function_on_grid(func, X, Y)
    
    # Erstellen der interaktiven Figur
    fig = go.Figure()
    
    # Kontur hinzufügen
    fig.add_trace(go.Contour(
        z=Z,
        x=np.linspace(x_range[0], x_range[1], n_points),
        y=np.linspace(y_range[0], y_range[1], n_points),
        colorscale='viridis',
        contours=dict(
            start=np.min(Z),
            end=np.percentile(Z, 95),
            size=(np.percentile(Z, 95) - np.min(Z)) / 30,
            coloring='fill',
            showlabels=True,
            labelfont=dict(size=10, color='white')
        ),
        colorbar=dict(
            title=dict(
                text='f(X,Y)',
                side='right',
                font=dict(size=12)
            ),
            tickfont=dict(size=10)
        ),
        hoverinfo='x+y+z'
    ))
    
    # Startpunkte markieren
    start_points = np.array([result[1][0] for result in all_results])
    fig.add_trace(go.Scatter(
        x=start_points[:, 0],
        y=start_points[:, 1],
        mode='markers',
        name='Startpunkte',
        marker=dict(
            size=10,
            color='blue',
            symbol='circle',
            line=dict(color='black', width=1)
        )
    ))
    
    # Endpunkte markieren
    end_points = np.array([result[0] for result in all_results])
    end_values = np.array([func(point)['value'] for point in end_points])
    
    # Finde den besten Endpunkt (niedrigster Funktionswert)
    best_idx = np.argmin(end_values)
    
    # Kategorisiere Endpunkte
    distinct_endpoints = []
    for i, point in enumerate(end_points):
        # Prüfe, ob dieser Punkt bereits in distinct_endpoints ist
        is_distinct = True
        for j, (p, idx, val) in enumerate(distinct_endpoints):
            if np.linalg.norm(point - p) < 1e-3:  # Nahe genug zum gleichen Punkt
                is_distinct = False
                # Behalte den besseren Wert
                if end_values[i] < val:
                    distinct_endpoints[j] = (point, i, end_values[i])
                break
        if is_distinct:
            distinct_endpoints.append((point, i, end_values[i]))
    
    # Für jeden Cluster von Endpunkten
    for j, (point, idx, val) in enumerate(distinct_endpoints):
        is_best = idx == best_idx
        
        # Optimierungspfad für diesen Punkt
        path = np.array(all_results[idx][1])
        
        # Pfad hinzufügen
        fig.add_trace(go.Scatter(
            x=path[:, 0],
            y=path[:, 1],
            mode='lines',
            name=f'Pfad {j+1} ({"Bester" if is_best else "Lokal"})',
            line=dict(
                width=2,
                color='red' if is_best else 'gray',
                dash='solid' if is_best else 'dot'
            )
        ))
        
        # Endpunkt hinzufügen
        fig.add_trace(go.Scatter(
            x=[point[0]],
            y=[point[1]],
            mode='markers',
            name=f'Endpunkt {j+1} (f={val:.4f})',
            marker=dict(
                size=12,
                color='gold' if is_best else 'silver',
                symbol='star' if is_best else 'diamond',
                line=dict(color='black', width=1)
            )
        ))
    
    # Layout anpassen
    fig.update_layout(
        title='Multi-Start Optimierungsergebnisse',
        autosize=True,
        width=800,
        height=600,
        xaxis=dict(
            title='X', 
            tickfont=dict(size=10),
            range=x_range  # Explizite Grenzen setzen
        ),
        yaxis=dict(
            title='Y', 
            tickfont=dict(size=10), 
            scaleanchor="x", 
            scaleratio=1,
            range=y_range  # Explizite Grenzen setzen
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        hovermode='closest'
    )
    
    return fig

def plot_loss_comparison(all_results, names):
    """
    Vergleicht die Verlaufskurven verschiedener Optimierungsläufe.
    
    Args:
        all_results: Liste von (x_opt, x_history, loss_history, status) Tupeln
        names: Liste von Namen für die Legenden
    """
    fig = go.Figure()
    
    # Farbpalette für verschiedene Algorithmen
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'teal', 'magenta', 'brown']
    
    for i, ((_, _, loss_history, _), name) in enumerate(zip(all_results, names)):
        # Verwende Modulo, um die Farbe zu zyklieren, wenn mehr Algorithmen als Farben
        color = colors[i % len(colors)]
        
        # Füge Verlaufskurve hinzu
        fig.add_trace(go.Scatter(
            y=loss_history,
            x=list(range(len(loss_history))),
            mode='lines',
            name=name,
            line=dict(width=2, color=color)
        ))
    
    # Layout anpassen
    fig.update_layout(
        title='Vergleich der Optimierungsverläufe',
        xaxis=dict(title='Iteration', tickfont=dict(size=10)),
        yaxis=dict(
            title='Funktionswert',
            tickfont=dict(size=10),
            type='log'  # Logarithmische Skala für bessere Visualisierung
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        hovermode='closest',
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='rgba(0, 0, 0, 0.5)',
            borderwidth=1
        )
    )
    
    return fig

def compute_convergence_metrics(all_results, func):
    """
    Berechnet Metriken zur Bewertung der Konvergenz verschiedener Optimierungsläufe.
    
    Args:
        all_results: Liste von (x_opt, x_history, loss_history, status) Tupeln
        func: Die zu optimierende Funktion
        
    Returns:
        DataFrame mit Konvergenzmetriken
    """
    # Wir verwenden Dictionary statt DataFrame für eine einfachere Implementierung
    metrics = {
        'Algorithm': [],
        'Final Value': [],
        'Final Gradient Norm': [],
        'Iterations': [],
        'Status': [],
        'Convergence Rate': []  # Durchschnittliche Verbesserung pro Iteration
    }
    
    for i, (x_opt, x_history, loss_history, status) in enumerate(all_results):
        # Algorithmus-Name (Index, wenn keiner gegeben)
        metrics['Algorithm'].append(f'Algorithm {i+1}')
        
        # Endwert
        final_value = loss_history[-1] if loss_history else float('nan')
        metrics['Final Value'].append(final_value)
        
        # Iterationen
        iterations = len(loss_history) - 1  # -1, weil der erste Wert der Initialwert ist
        metrics['Iterations'].append(iterations)
        
        # Status
        metrics['Status'].append(status)
        
        # Gradient-Norm am Ende
        try:
            final_grad = func(x_opt)['gradient']
            grad_norm = np.linalg.norm(final_grad)
        except:
            grad_norm = float('nan')
        metrics['Final Gradient Norm'].append(grad_norm)
        
        # Konvergenzrate (durchschnittliche Verbesserung pro Iteration)
        if len(loss_history) > 1:
            initial_value = loss_history[0]
            improvement = initial_value - final_value
            rate = improvement / iterations if iterations > 0 else 0
            metrics['Convergence Rate'].append(rate)
        else:
            metrics['Convergence Rate'].append(0)
    
    return metrics