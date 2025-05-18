#!/usr/bin/env python3
"""
IntelliScope Explorer
Interaktive Visualisierung von Optimierungslandschaften und -algorithmen
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sympy as sp
import re
import time
import os

# Importiere eigene Module
import problem_functions_v3 as pf
import optimization_algorithms_v3 as oa
import visualization_suite_v3 as vs
import improved_optimizer as iopt  # Renamed to avoid conflict with 'io'
import data_manager as dm

def create_visualization_tracker(func, x_range, y_range, contour_levels, minima):
    """
    Erstellt einen Tracker für den Optimierungspfad
    """
    path_history = []
    value_history = []
    
    # Callback-Funktion, die den Pfad aufzeichnet
    def callback(iteration, x, value, grad_norm, message):
        path_history.append(x.copy())
        value_history.append(value)
        
        # Status-Nachricht im Info-Bereich anzeigen
        info_text = f"""
        **Iteration:** {iteration+1}
        **Aktuelle Position:** [{x[0]:.4f}, {x[1]:.4f}]
        **Funktionswert:** {value:.6f}
        **Gradientennorm:** {grad_norm:.6f}
        """
        info_placeholder.markdown(info_text)
        
        # Nur alle 5 Iterationen visualisieren, um Performance zu verbessern
        if iteration % 5 == 0 or iteration < 5:
            # 2D Konturplot mit aktuellem Pfad
            fig_live = plt.figure(figsize=(8, 4))
            ax_live = fig_live.add_subplot(111)
            
            # Gitter für Konturplot
            X, Y = np.meshgrid(np.linspace(x_range[0], x_range[1], 50), 
                             np.linspace(y_range[0], y_range[1], 50))
            Z = np.zeros_like(X)
            
            # Berechne Funktionswerte auf dem Gitter
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        result = func(np.array([X[i, j], Y[i, j]]))
                        Z[i, j] = result.get('value', np.nan)
                    except:
                        Z[i, j] = np.nan

# Seitenkonfiguration mit verbesserten Einstellungen
st.set_page_config(
    page_title="IntelliScope Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# IntelliScope Explorer\nInteraktive Visualisierung von Optimierungslandschaften und -algorithmen. Entwickelt für die Analyse und das Verständnis verschiedener Optimierungsverfahren."
    }
)

# Stylesheet
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4d8bf0;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #4d8bf0;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .explanation-box {
        background-color: #f8f0ff;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        border-left: 4px solid #9061c2;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .explanation-box h3 {
        color: #6a2c91;
        margin-top: 0;
        border-bottom: 1px solid rgba(144, 97, 194, 0.3);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .explanation-box p {
        line-height: 1.6;
        color: #333;
    }
    
    .explanation-box ul, .explanation-box ol {
        padding-left: 1.5rem;
        margin: 0.8rem 0;
    }
    .tip-box {
        background-color: #f0fff8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #61c291;
    }
    .plot-container {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .loss-curve {
        height: 400px;
    }
    .custom-func {
        font-family: monospace;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 0.25rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f8ff;
        border-radius: 4px 4px 0 0;
        padding: 8px 16px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4d8bf0 !important;
        color: white !important;
    }
    
    /* Verbesserte Interaktivität und Animation */
    .plot-hover {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .plot-hover:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Bessere Hervorhebungen für wichtige Elemente */
    .highlight {
        background: linear-gradient(90deg, rgba(77,139,240,0.1) 0%, rgba(77,139,240,0) 100%);
        padding: 0.2rem 0.5rem;
        border-left: 3px solid #4d8bf0;
    }
    
    /* Animation für Ladezustände */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    .animate-pulse {
        animation: pulse 1.5s infinite ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# Initialisieren der Session-State-Variablen, falls sie noch nicht existieren
for key, default in [
    ('optimierungsergebnisse', {}),
    ('custom_funcs', {}),
    ('ausgewählte_funktion', "Rosenbrock"),
    ('custom_func_count', 0)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Header mit verbessertem Design
st.markdown("""
<div style="background: linear-gradient(90deg, #6a2c91, #4d8bf0); padding: 1.5rem; border-radius: 0.8rem; margin-bottom: 1.5rem; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
    <div class="main-header" style="color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">IntelliScope Explorer</div>
    <div class="sub-header" style="color: rgba(255,255,255,0.9); font-weight: 300;">Interaktive Visualisierung von Optimierungslandschaften und -algorithmen</div>
</div>
""", unsafe_allow_html=True)

# Sidebar für Einstellungen
with st.sidebar:
    st.header("Einstellungen")
    
    # Funktionsauswahl
    function_list = list(pf.MATH_FUNCTIONS_LIB.keys())
    custom_funcs_keys = list(st.session_state.custom_funcs.keys())
    
    all_functions = function_list.copy()
    if custom_funcs_keys:
        all_functions.append("----------")
        all_functions.extend(custom_funcs_keys)
    
    selected_function_name = st.selectbox(
        "Funktion auswählen",
        all_functions,
        index=all_functions.index(st.session_state.ausgewählte_funktion) if st.session_state.ausgewählte_funktion in all_functions else 0,
        key="sb_func_select"
    )
    
    if selected_function_name != "----------":
        st.session_state.ausgewählte_funktion = selected_function_name
    
    # Algorithmenwahl (Only one algorithm selectbox with unique key)
    algorithm_options = {
        "GD_Simple_LS": "Gradient Descent mit Liniensuche",
        "GD_Momentum":  "Gradient Descent mit Momentum",
        "Adam":         "Adam Optimizer"
    }
    selected_algorithm_key = st.selectbox(
        "Algorithmus auswählen",
        list(algorithm_options.keys()),
        format_func=lambda x: algorithm_options[x],
        key="sb_algo_select"
    )
    
    # Strategy selection
    strategy_options = {
        "single": "Einzelne Optimierung",
        "multi_start": "Multi-Start Optimierung",
        "adaptive": "Adaptive Multi-Start"
    }
    selected_strategy = st.selectbox(
        "Strategie auswählen",
        list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        key="sb_strategy_select"
    )
    
    # Algorithm parameters (fixed logic)
    st.subheader("Algorithmus-Parameter")
    
    optimizer_params = {}
    if selected_algorithm_key == "GD_Simple_LS":
        optimizer_params['max_iter'] = st.slider("Max. Iterationen", 100, 5000, 1000, step=100, key="gd_max_iter_slider")
        optimizer_params['step_norm_tol'] = st.slider("Schrittnorm Toleranz", 1e-12, 1e-3, 1e-7, format="%.0e", key="gd_step_tol_slider")
        optimizer_params['func_impr_tol'] = st.slider("Funktionsverbesserung Toleranz", 1e-12, 1e-3, 1e-9, format="%.0e", key="gd_func_tol_slider")
        optimizer_params['initial_t_ls'] = st.slider("Initialer Liniensuchschritt", 1e-6, 1.0, 0.1, format="%.0e", key="gd_init_t_ls_slider")

    elif selected_algorithm_key == "GD_Momentum":
        optimizer_params['max_iter'] = st.slider("Max. Iterationen", 100, 5000, 1000, step=100, key="gdm_max_iter_slider")
        optimizer_params['learning_rate'] = st.slider("Lernrate", 1e-5, 1.0, 0.05, format="%.4f", key="gdm_lr_slider")
        optimizer_params['momentum_beta'] = st.slider("Momentum Beta", 0.0, 0.999, 0.95, format="%.3f", key="gdm_beta_slider")
        optimizer_params['grad_norm_tol'] = st.slider("Gradientennorm Toleranz", 1e-12, 1e-3, 1e-7, format="%.0e", key="gdm_grad_tol_slider")

    elif selected_algorithm_key == "Adam":
        optimizer_params['max_iter'] = st.slider("Max. Iterationen", 100, 5000, 1000, step=100, key="adam_max_iter_slider")
        optimizer_params['learning_rate'] = st.slider("Lernrate", 1e-5, 1.0, 0.005, format="%.5f", key="adam_lr_slider")
        optimizer_params['beta1'] = st.slider("Beta1 (Momentum)", 0.0, 0.999, 0.95, format="%.3f", key="adam_beta1_slider")
        optimizer_params['beta2'] = st.slider("Beta2 (RMSProp)", 0.0, 0.9999, 0.9995, format="%.4f", key="adam_beta2_slider")
        optimizer_params['epsilon'] = st.slider("Epsilon (für Stabilität)", 1e-12, 1e-6, 1e-8, format="%.0e", key="adam_epsilon_slider")
        optimizer_params['grad_norm_tol'] = st.slider("Gradientennorm Toleranz", 1e-12, 1e-3, 1e-7, format="%.0e", key="adam_grad_tol_slider")
    
    # Strategy parameters
    multi_params = {}
    if selected_strategy != "single":
        st.subheader("Strategie-Parameter")
        if selected_strategy == "multi_start":
            multi_params['n_starts'] = st.slider("Anzahl der Starts", 2, 20, 5, key="ms_n_starts_slider")
            multi_params['use_challenging_starts'] = st.checkbox("Herausfordernde Startpunkte", value=True, key="ms_challenging_checkbox")
            multi_params['seed'] = st.slider("Seed", 0, 100, 42, key="ms_seed_slider")
        elif selected_strategy == "adaptive":
            multi_params['initial_starts'] = st.slider("Initiale Anzahl der Starts", 2, 10, 3, key="ad_initial_starts_slider")
            multi_params['max_starts'] = st.slider("Maximale Anzahl der Starts", multi_params['initial_starts'], 30, 10, key="ad_max_starts_slider")
            multi_params['min_improvement'] = st.slider("Min. Verbesserung für weitere Starts", 0.001, 0.1, 0.01, format="%.3f", key="ad_min_improvement_slider")
            multi_params['seed'] = st.slider("Seed", 0, 100, 42, key="ad_seed_slider")

    # Button zum Starten der Optimierung
    start_optimization = st.button("Optimierung starten", use_container_width=True)

    # Button zum Zurücksetzen aller Ergebnisse
    if st.button("Alle Ergebnisse zurücksetzen", use_container_width=True, key="reset_results_main"):
        st.session_state.optimierungsergebnisse = {}
        st.rerun()

# Hauptbereich für Visualisierung
# Tabs für verschiedene Visualisierungen und Interaktionen
tabs = st.tabs(["Optimierungsvisualisierung", "Funktionseditor", "Ergebnisvergleich", "Info & Hilfe"])

#Optimierungsvisualisierung
with tabs[0]:
    # Einheitliche Initialisierung für die aktuelle Funktion und Metainfos
    current_func_obj = None
    x_range = (-5, 5)
    y_range = (-5, 5)
    contour_levels = 30
    minima = None

    # 1. Standardfunktion aus der Funktionsbibliothek
    if st.session_state.ausgewählte_funktion in pf.MATH_FUNCTIONS_LIB:
        func_info = pf.MATH_FUNCTIONS_LIB[st.session_state.ausgewählte_funktion]
        current_func_obj = func_info.get("func")
        x_range = func_info.get("default_range", [(-5, 5), (-5, 5)])[0]
        y_range = func_info.get("default_range", [(-5, 5), (-5, 5)])[1]
        contour_levels = func_info.get("contour_levels", 40)
        try:
            # Metainfos holen (z.B. Minima, Tooltip)
            eval_point = np.array([(x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2])
            func_meta = current_func_obj(eval_point)
            if "tooltip" in func_meta:
                with st.expander("ℹ️ Über diese Funktion", expanded=False):
                    st.markdown(func_meta["tooltip"])
            minima = func_meta.get("minima", None)
        except Exception as e:
            st.warning(f"Metadaten für {st.session_state.ausgewählte_funktion} konnten nicht geladen werden: {e}")
            minima = None

    # 2. Benutzerdefinierte Funktion (Custom)
    elif st.session_state.ausgewählte_funktion in st.session_state.custom_funcs:
        current_func_obj = st.session_state.custom_funcs[st.session_state.ausgewählte_funktion]
        # Versuche Metadaten aus der Custom-Function zu extrahieren
        try:
            func_meta = current_func_obj(np.array([0.0, 0.0]))
            x_range = func_meta.get("x_range", (-5, 5))
            y_range = func_meta.get("y_range", (-5, 5))
            contour_levels = func_meta.get("contour_levels", 30)
            minima = func_meta.get("minima", None)
        except Exception:
            # Falls die Funktion keinen dict zurückgibt, Default-Werte
            x_range = (-5, 5)
            y_range = (-5, 5)
            contour_levels = 30
            minima = None

    # 3. Fehlerfall
    else:
        st.error("Die ausgewählte Funktion wurde nicht gefunden.")
        current_func_obj = None
        x_range = (-5, 5)
        y_range = (-5, 5)
        contour_levels = 30
        minima = None
   
    # Layout für die Visualisierung
    col1, col2 = st.columns([1, 1])
    
    with col1:
            # Erstelle 3D-Plot mit Matplotlib und füge Kontrollen hinzu
            if current_func_obj:
                # Erstelle Container für 3D Plot und Kontrollen
                plot3d_container = st.container()
                controls3d_container = st.container()
                
                # Parameter für Matplotlib Plot
                if 'elev_3d' not in st.session_state:
                    st.session_state.elev_3d = 30
                if 'azim_3d' not in st.session_state:
                    st.session_state.azim_3d = 45
                if 'dist_3d' not in st.session_state:
                    st.session_state.dist_3d = 10
                    
                # Steuerungsbereich mit ansprechendem Design
                with controls3d_container:
                    st.markdown("""
                    <div style="background-color: #4d8bf0; padding: 8px; border-radius: 8px; margin-bottom: 10px;">
                        <h4 style="color: white; margin: 0;">3D Ansicht Steuerung</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Buttons für Standardansichten in einer Zeile
                    btn_cols = st.columns(5)
                    if btn_cols[0].button("Von oben", key="top_view", type="primary", use_container_width=True):
                        st.session_state.elev_3d = 90
                        st.session_state.azim_3d = 0
                    
                    if btn_cols[1].button("Von vorne", key="front_view", type="primary", use_container_width=True):
                        st.session_state.elev_3d = 0
                        st.session_state.azim_3d = 0
                        
                    if btn_cols[2].button("Von rechts", key="right_view", type="primary", use_container_width=True):
                        st.session_state.elev_3d = 0
                        st.session_state.azim_3d = 90
                        
                    if btn_cols[3].button("Isometrisch", key="iso_view", type="primary", use_container_width=True):
                        st.session_state.elev_3d = 30
                        st.session_state.azim_3d = 45
                        
                    if btn_cols[4].button("Von links", key="left_view", type="primary", use_container_width=True):
                        st.session_state.elev_3d = 0
                        st.session_state.azim_3d = 270
                    
                    # Slider für feinere Kontrolle
                    cols = st.columns(3)
                    with cols[0]:
                        st.session_state.elev_3d = st.slider("Elevation", 0, 90, 
                                                           st.session_state.elev_3d, 
                                                           key="elev_slider")
                    with cols[1]:
                        st.session_state.azim_3d = st.slider("Azimuth", 0, 360, 
                                                           st.session_state.azim_3d, 
                                                           key="azim_slider")
                    with cols[2]:
                        st.session_state.dist_3d = st.slider("Zoom", 5, 20, 
                                                          st.session_state.dist_3d, 
                                                          key="dist_slider")
                
                # 3D Plot mit Matplotlib erzeugen
                with plot3d_container:
                    fig3d = plt.figure(figsize=(8, 6))
                    ax3d = fig3d.add_subplot(111, projection='3d')
                    
                    # Erzeuge Gitter für 3D-Plot
                    x = np.linspace(x_range[0], x_range[1], 50)
                    y = np.linspace(y_range[0], y_range[1], 50)
                    X, Y = np.meshgrid(x, y)
                    Z = np.zeros_like(X)
                    
                    # Berechne Funktionswerte auf dem Gitter
                    for i in range(X.shape[0]):
                        for j in range(X.shape[1]):
                            try:
                                params = np.array([X[i, j], Y[i, j]])
                                result = current_func_obj(params)
                                Z[i, j] = result['value']
                            except:
                                Z[i, j] = np.nan
                    
                    # Statistische Verarbeitung für bessere Visualisierung
                    Z_finite = Z[np.isfinite(Z)]
                    if len(Z_finite) > 0:
                        z_mean = np.mean(Z_finite)
                        z_std = np.std(Z_finite)
                        z_min = max(np.min(Z_finite), z_mean - 5*z_std)
                        z_max = min(np.max(Z_finite), z_mean + 5*z_std)
                        
                        # Extremwerte begrenzen
                        Z_plot = np.copy(Z)
                        Z_plot[(Z_plot < z_min) & np.isfinite(Z_plot)] = z_min
                        Z_plot[(Z_plot > z_max) & np.isfinite(Z_plot)] = z_max
                    else:
                        Z_plot = Z
                    
                    # Zeichne 3D-Oberfläche
                    surf = ax3d.plot_surface(X, Y, Z_plot, cmap='viridis', 
                                            linewidth=0, antialiased=True, alpha=0.8)
                    
                    # Achsenbeschriftungen
                    ax3d.set_xlabel('X')
                    ax3d.set_ylabel('Y')
                    ax3d.set_zlabel('Funktionswert')
                    ax3d.set_title(f"3D-Oberfläche: {st.session_state.ausgewählte_funktion}")
                    
                    # Zeige bekannte Minima, falls vorhanden
                    if minima is not None:
                        for m in minima:
                            try:
                                z_val = current_func_obj(np.array(m))['value']
                                ax3d.scatter([m[0]], [m[1]], [z_val], color='red', marker='+', s=120, 
                                            linewidths=2, label='Bekanntes Minimum')
                            except:
                                pass

                    # --- Code zum Plotten der Top 10 Pfade ---
                    
                    # Definieren, wie viele beste Pfade geplottet werden sollen
                    num_best_paths_to_plot = 10
                    
                    # Überprüfen, ob Optimierungsergebnisse vorhanden sind und die aktuelle Funktion ausgewählt ist
                    if 'optimierungsergebnisse' in st.session_state and st.session_state.optimierungsergebnisse \
                       and 'ausgewählte_funktion' in st.session_state and st.session_state.ausgewählte_funktion:
                    
                        # Ergebnisse für die aktuelle Funktion filtern, die eine Historie und einen endlichen 'value' haben
                        # Wir benötigen den algo_name für die Beschriftung, daher über items() iterieren
                        current_function_results_with_value = []
                        for algo_name, result_data in st.session_state.optimierungsergebnisse.items():
                            # Prüfen, ob result_data ein Dictionary ist
                            if isinstance(result_data, dict) and \
                               result_data.get("function") == st.session_state.ausgewählte_funktion and \
                               "history" in result_data and result_data["history"] and \
                               "value" in result_data and np.isfinite(result_data["value"]):
                                # Speichern als (algo_name, result_data) Tupel
                                current_function_results_with_value.append((algo_name, result_data))
                            # Optional: Fall behandeln, dass result_data eine Liste von Läufen für einen Multistart-Algo ist
                            # Das würde eine andere Logik erfordern, um die Liste zu "flachklopfen" und jeden Lauf zu verarbeiten.
                            # Der aktuelle Code geht davon aus, dass jedes Element in optimierungsergebnisse ein Einzellauf-Ergebnis ist.
                    
                    
                        # Ergebnisse nach ihrem finalen Zielfunktionswert ("value") sortieren - Annahme: Minimierung
                        # Einen grossen endlichen Wert oder np.inf für fehlende/NaN-Werte im Sortier-Key verwenden
                        sorted_results_by_value = sorted(
                            current_function_results_with_value,
                            key=lambda item: item[1].get("value", np.inf) # Sortieren nach dem 'value'-Schlüssel
                        )
                    
                        # Die Top N besten Ergebnisse auswählen
                        best_n_results = sorted_results_by_value[:num_best_paths_to_plot]
                    
                        # Überprüfen, ob es beste Pfade zum Plotten gibt
                        if best_n_results:
                            # Definieren einer Colormap für die Farben der anderen Pfade
                            # Die Colormap sollte N-1 Farben liefern, da der beste Pfad rot ist
                            cmap = plt.cm.get_cmap('viridis', max(1, len(best_n_results) -1)) # mind. 1 Farbe, auch wenn nur 1 Pfad da ist
                    
                            # Schleife durch die besten N Ergebnisse und plotten jedes Pfades
                            for rank, (algo_name, result_data) in enumerate(best_n_results):
                    
                                history = result_data.get("history")
                                if history:
                                    path_points = np.array(history)
                    
                                    # Sicherstellen, dass path_points mindestens 2 Dimensionen (für x und y) und mind. 1 Punkt hat
                                    if path_points.ndim < 2 or path_points.shape[1] < 2 or path_points.shape[0] < 1:
                                         print(f"Warnung: Historie für {algo_name} hat ungültige Dimensionen oder ist leer.")
                                         continue # Diesen Pfad überspringen
                    
                                    path_x = path_points[:, 0]
                                    path_y = path_points[:, 1]
                                    path_z = np.zeros(len(path_points)) # Z-Array initialisieren
                    
                                    # Z-Werte für jeden Punkt in der Pfad-Historie berechnen
                                    valid_path_indices = [] # Indices mit endlichen Z-Werten für die Linienzeichnung
                                    for i, point in enumerate(history): # Über originale Liste iterieren für Point-Format-Konsistenz
                                        try:
                                            params = np.array(point)
                                            # Robuste Methode zum Aufrufen der Zielfunktion verwenden
                                            if callable(current_func_obj):
                                                res = current_func_obj(params)
                                            elif callable(current_func):
                                                 res = current_func(params)
                                            else:
                                                # Fallback oder Fehlerbehandlung, wenn keine gültige Funktion gefunden
                                                print(f"Fehler: Optimierungsfunktion nicht gefunden oder nicht aufrufbar für {algo_name}.")
                                                path_z[i] = np.nan # NaN zuweisen, wenn Funktion fehlt
                                                continue # Berechnung für diesen Punkt überspringen
                    
                                            z_value = res.get('value', np.nan)
                                            path_z[i] = z_value
                    
                                            # Z-Clipping basierend auf dem Surface-Bereich anwenden
                                            # Sicherstellen, dass z_min und z_max endlich sind, bevor geclippt wird
                                            if np.isfinite(path_z[i]) and np.isfinite(z_min) and np.isfinite(z_max):
                                                 path_z[i] = min(max(path_z[i], z_min), z_max)
                    
                                            if np.isfinite(path_z[i]): # Nur endliche Z-Werte für die Linienzeichnung berücksichtigen
                                                valid_path_indices.append(i)
                    
                                        except Exception as e:
                                            # Potenzielle Fehler bei der Funktionsauswertung für Pfad-Punkte behandeln
                                            print(f"Fehler bei Z-Berechnung für Pfadpunkt {i} von {algo_name}: {e}")
                                            path_z[i] = np.nan # NaN bei Fehler zuweisen
                    
                                    # --- Plotting für den aktuellen Pfad ---
                    
                                    # Bestimmen der Farbe und des Stils basierend auf dem Rang
                                    if rank == 0: # Der beste Pfad (Rang 0)
                                        path_color = 'red'
                                        path_linewidth = 3
                                        path_markersize_line = 5
                                        start_marker_size = 120
                                        end_marker_size = 140
                                        path_alpha = 1.0
                                        label_suffix = " (Best)"
                                    else: # Die anderen Top 9 Pfade
                                        # Farbe aus der Colormap nehmen (rank-1, da der 0-te Index rot ist)
                                        path_color = cmap(rank -1)
                                        path_linewidth = 2
                                        path_markersize_line = 3
                                        start_marker_size = 80
                                        end_marker_size = 100
                                        path_alpha = 0.7
                                        label_suffix = ""
                    
                    
                                    # Startpunkt plotten (erster Punkt in der Historie)
                                    # path_points[0, 0], path_points[0, 1] für x, y verwenden
                                    # path_z[0] für z verwenden (könnte NaN sein, wenn Berechnung fehlschlug)
                                    # Einen Z-Wert für den Marker zuweisen, auch wenn die Berechnung fehlschlug, z.B. z_min
                                    start_x, start_y = path_points[0, 0], path_points[0, 1]
                                    start_z_plot = path_z[0] if np.isfinite(path_z[0]) else (z_min if np.isfinite(z_min) else 0) # Geclipptes z oder z_min/0 verwenden
                    
                                    ax3d.scatter([start_x], [start_y], [start_z_plot],
                                                 color=path_color, # Farbe je nach Rang
                                                 marker='o', s=start_marker_size, alpha=path_alpha,
                                                 label='_nolegend_') # Einzelne Startpunkte nicht in die Legende aufnehmen
                    
                                    # Endpunkt plotten (letzter Punkt in der Historie)
                                    end_x, end_y = path_points[-1, 0], path_points[-1, 1]
                                    end_z_plot = path_z[-1] if np.isfinite(path_z[-1]) else (z_min if np.isfinite(z_min) else 0) # Geclipptes z oder z_min/0 verwenden
                    
                                    ax3d.scatter([end_x], [end_y], [end_z_plot],
                                                 color=path_color, # Farbe je nach Rang
                                                 marker='*', s=end_marker_size, alpha=path_alpha + 0.1 if path_alpha <= 0.9 else 1.0, # Endpunkt ggf. etwas sichtbarer
                                                 label='_nolegend_') # Einzelne Endpunkte nicht in die Legende aufnehmen
                    
                                    # Pfad-Linie plotten (nur Punkte mit gültigen Z-Werten verbinden)
                                    if len(valid_path_indices) > 1: # Mindestens 2 gültige Punkte für eine Linie erforderlich
                                        valid_path_x = path_x[valid_path_indices]
                                        valid_path_y = path_y[valid_path_indices]
                                        valid_path_z_clipped = path_z[valid_path_indices] # Bereits geclippt
                    
                                        # Linie für diesen Pfad plotten
                                        ax3d.plot(valid_path_x, valid_path_y, valid_path_z_clipped,
                                                  color=path_color, # Farbe je nach Rang
                                                  linewidth=path_linewidth, markersize=path_markersize_line, alpha=path_alpha,
                                                  label=f'{algo_name} (Wert: {result_data["value"]:.4f}){label_suffix}') # Label für die Legende
                    
                    
                            # Eine einzige Legende für die Pfade hinzufügen
                            ax3d.legend()
                    
                        # else:
                            # Optional: Eine Nachricht anzeigen, wenn keine gültigen Ergebnisse für die aktuelle Funktion gefunden wurden
                            st.info("Keine Optimierungsergebnisse mit Historie und finalem Wert für die aktuelle Funktion gefunden.")
                            pass # Oder eine Nachricht hinzufügen
                    
                    # else:
                        # Optional: Eine Nachricht anzeigen, wenn session state leer ist oder Funktion nicht ausgewählt
                        st.info("Bitte führen Sie zuerst eine Optimierung durch.")
                        pass # Oder eine Nachricht hinzufügen
                    
                    # Legende & Colorbar
                    handles, labels = ax3d.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    if by_label:
                        ax3d.legend(by_label.values(), by_label.keys(), loc='upper right')
                    fig3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)
                    
                    # Kameraeinstellungen
                    ax3d.view_init(elev=st.session_state.elev_3d, azim=st.session_state.azim_3d)
                    try:
                        ax3d.dist = st.session_state.dist_3d / 10  # Skaliere für bessere Werte
                    except Exception:
                        pass
                    
                    # Zeige Plot
                    st.pyplot(fig3d)
                    plt.close(fig3d)
            
    with col2:
        # --- col2: Optimierungsvisualisierung, rechte Spalte ---
        if current_func_obj:
            plot2d_container = st.container()
            controls2d_container = st.container()
        
            # Parameter für Matplotlib Plot
            if 'contour_levels' not in st.session_state:
                st.session_state.contour_levels = contour_levels
            if 'zoom_factor' not in st.session_state:
                st.session_state.zoom_factor = 1.0
            if 'show_grid_2d' not in st.session_state:
                st.session_state.show_grid_2d = False
            if 'center_x' not in st.session_state:
                st.session_state.center_x = np.mean(x_range)
            if 'center_y' not in st.session_state:
                st.session_state.center_y = np.mean(y_range)
        
            with controls2d_container:
                st.markdown("""
                <div style="background-color: #6a2c91; padding: 8px; border-radius: 8px; margin-bottom: 10px;">
                    <h4 style="color: white; margin: 0;">2D Ansicht Steuerung</h4>
                </div>
                """, unsafe_allow_html=True)
        
                cols = st.columns(3)
                with cols[0]:
                    st.session_state.contour_levels = st.slider("Konturlinien", 10, 100, 
                                                              st.session_state.contour_levels, 
                                                              step=5,
                                                              key="contour_slider")
                with cols[1]:
                    st.session_state.zoom_factor = st.slider("Zoom", 0.5, 5.0, 
                                                           st.session_state.zoom_factor, 
                                                           step=0.1,
                                                           key="zoom_slider")
                with cols[2]:
                    st.session_state.show_grid_2d = st.checkbox("Gitter anzeigen", 
                                                              st.session_state.show_grid_2d, 
                                                              key="grid_checkbox")

        # --- 2D-Konturplot ---
        with st.container():
            fig2d = plt.figure(figsize=(8, 6))
            ax2d = fig2d.add_subplot(111)
            
            # Berechne zoomed-Bereich um das Zentrum
            x_half_range = (x_range[1] - x_range[0]) / (2 * st.session_state.zoom_factor)
            y_half_range = (y_range[1] - y_range[0]) / (2 * st.session_state.zoom_factor)
            x_zoom_range = (st.session_state.center_x - x_half_range, 
                           st.session_state.center_x + x_half_range)
            y_zoom_range = (st.session_state.center_y - y_half_range, 
                           st.session_state.center_y + y_half_range)
            
            # Erzeuge feines Gitter für Konturplot
            grid_size = int(100 * np.sqrt(st.session_state.zoom_factor))
            x = np.linspace(x_zoom_range[0], x_zoom_range[1], grid_size)
            y = np.linspace(y_zoom_range[0], y_zoom_range[1], grid_size)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            
            # Berechne Funktionswerte auf dem Gitter
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        params = np.array([X[i, j], Y[i, j]])
                        result = current_func_obj(params)
                        Z[i, j] = result['value']
                    except:
                        Z[i, j] = np.nan
            
            # Zeichne Konturplot
            cp = ax2d.contourf(X, Y, Z, levels=st.session_state.contour_levels, 
                             cmap='viridis', alpha=0.8)
            contour_lines = ax2d.contour(X, Y, Z, 
                                      levels=min(20, st.session_state.contour_levels//3), 
                                      colors='black', alpha=0.4, linewidths=0.5)
            ax2d.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
            
            # Farbskala hinzufügen
            colorbar = fig2d.colorbar(cp, ax=ax2d)
            colorbar.set_label('Funktionswert')
            
            # Achsenbeschriftungen und Titel
            ax2d.set_xlabel('X')
            ax2d.set_ylabel('Y')
            ax2d.set_title(f"Konturplot: {st.session_state.ausgewählte_funktion}")
            
            # Minima einzeichnen, falls vorhanden
            if minima is not None:
                for m in minima:
                    ax2d.plot(m[0], m[1], 'X', color='red', markersize=8, markeredgecolor='black')
            
            # Achsengrenzen setzen
            ax2d.set_xlim(x_zoom_range)
            ax2d.set_ylim(y_zoom_range)
            
            # Gitter zeichnen, falls gewünscht
            if st.session_state.show_grid_2d:
                ax2d.grid(True, linestyle='--', alpha=0.6)
            
            # Plot anzeigen
            st.pyplot(fig2d)
            plt.close(fig2d)
            
            # Callback und Parameter sollten zuvor erzeugt worden sein:
            
            # Resultate extrahieren
            best_x = result.x_best              # oder result.x, je nach Rückgabe
            best_history = result.history       # Pfad aller x_i
            best_loss_history = result.loss_history  # Funktionswerte
            status = result.status              # Status-String
            
            # Initialisiere Momentum-Variable
            velocity = np.zeros_like(x)
            
            # Initialisiere Parameter für adaptive Lernrate
            best_value = value
            patience = 5
            patience_counter = 0
            lr_reduce_factor = 0.5
            lr_increase_factor = 1.1
            min_lr = 1e-6
            current_lr = learning_rate
            
            # Metainformationen für Statusberichterstattung
            info_text = "Optimierung gestartet"
            
            # Zeichne Konturplot
            cp = ax_live.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
            
            # Zeichne Pfadverlauf
            if len(path_history) > 0:
                path_x = [p[0] for p in path_history]
                path_y = [p[1] for p in path_history]
                ax_live.plot(path_x, path_y, 'r-o', linewidth=2, markersize=4)
                ax_live.plot(path_x[0], path_y[0], 'bo', markersize=8, label='Start')
                ax_live.plot(path_x[-1], path_y[-1], 'g*', markersize=10, label='Aktuell')
            
            ax_live.set_xlim(x_range)
            ax_live.set_ylim(y_range)
            ax_live.set_title(f"Optimierungspfad (Iteration {iteration+1})")
            ax_live.legend()
            
            # Zeige Live-Plot
            live_plot_placeholder.pyplot(fig_live)
            
        #return callback, path_history, value_history
    
    # Bereich für Optimierungsergebnisse
    st.markdown("""
    <div style="background: linear-gradient(90deg, #4d8bf0, #6a2c91); padding: 12px; border-radius: 8px;">
        <h3 style="color: white; margin: 0;">Optimierungsergebnisse</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Live-Tracking & Info-Boxen-Container ---
    live_tracking_container = st.container()
    info_box_container = st.container()
    results_container = st.container()
    
    # Führe Optimierung aus, wenn der Button geklickt wurde
    if start_optimization and current_func_obj:
        with st.spinner("Optimierung läuft..."):
            # Räume die Live-Tracking-Container auf
            live_tracking_container.empty()
            info_box_container.empty()
            
            # Erstelle Callback-Funktion für Live-Verfolgung
            with live_tracking_container:
                st.markdown("""
                <div style="background-color: #f0f8ff; padding: 8px; border-radius: 8px; border-left: 4px solid #4d8bf0;">
                    <h4 style="color: #4d8bf0; margin: 0;">Live-Verfolgung der Optimierung</h4>
                </div>
                """, unsafe_allow_html=True)
                live_plot_placeholder = st.empty()
            
            with info_box_container:
                st.markdown("""
                <div style="background-color: #f8f0ff; padding: 8px; border-radius: 8px; border-left: 4px solid #6a2c91;">
                    <h4 style="color: #6a2c91; margin: 0;">Optimierungs-Status</h4>
                </div>
                """, unsafe_allow_html=True)
                info_placeholder = st.empty()
            
            # Erstelle Callback-Funktion
            visualization_callback, path_history, value_history = create_visualization_tracker(
                current_func_obj, x_range, y_range
            )
            
            # Wähle Startpunkt mit hohem Funktionswert
            # Grid-Suche für einen geeigneten Startpunkt
            start_x = np.linspace(x_range[0], x_range[1], 10)
            start_y = np.linspace(y_range[0], y_range[1], 10)
            highest_value = float('-inf')
            start_point = np.array([0.0, 0.0])
            
            for x in start_x:
                for y in start_y:
                    try:
                        point = np.array([x, y])
                        result = current_func_obj(point)
                        if 'value' in result and result['value'] > highest_value:
                            highest_value = result['value']
                            start_point = point.copy()
                    except:
                        continue
            
            st.write(f"Starte Optimierung von Punkt: [{start_point[0]:.4f}, {start_point[1]:.4f}]")
            
            # Führe Optimierung mit gewählten Parametern durch
            # Konfiguriere Optimierungsparameter basierend auf der ausgewählten Funktion und dem Algorithmus
            epsilon = 1e-8  # Standard-Epsilon für numerische Stabilität
            use_momentum = False
            use_adaptive_lr = True
            momentum_value = 0.9  # Standard-Momentum-Wert
            
            # Wähle Algorithmusparameter basierend auf der Funktion
            if selected_algorithm_key == "GD_Simple_LS":
                # Gradient Descent mit Liniensuche
                max_iter = optimizer_params.get("max_iter", 500)
                learning_rate = optimizer_params.get("initial_t_ls", 0.01)
                
                # Für schwierigere Funktionen wie Rosenbrock kleine Lernrate verwenden
                if st.session_state.ausgewählte_funktion == "Rosenbrock":
                    learning_rate = 0.005
                    use_adaptive_lr = True
                    
            elif selected_algorithm_key == "GD_Momentum":
                # Gradient Descent mit Momentum
                max_iter = optimizer_params.get("max_iter", 300)
                learning_rate = optimizer_params.get("learning_rate", 0.01)
                momentum_value = optimizer_params.get("momentum_beta", 0.9)
                use_momentum = True
                
                # Für schwierigere Funktionen wie Rosenbrock
                if st.session_state.ausgewählte_funktion == "Rosenbrock":
                    learning_rate = 0.005
                    momentum_value = 0.95
                    
            else:  # Adam
                # Adam Optimizer Konfiguration
                max_iter = optimizer_params.get("max_iter", 300)
                learning_rate = optimizer_params.get("learning_rate", 0.001)
                # Adam verwendet intern adaptives Momentum - wir verwenden hier die Basisimplementierung
                use_momentum = True
                momentum_value = 0.9  # Beta1 Parameter in Adam
                
                # Für multimodale Funktionen wie Rastrigin
                if st.session_state.ausgewählte_funktion in ["Rastrigin", "Ackley"]:
                    learning_rate = 0.002
                    max_iter = 500  # Mehr Iterationen für multimodale Funktionen
            
            # Status-Info anzeigen
            st.write(f"""
            **Optimierungseinstellungen:**
            - Algorithmus: {algorithm_options[selected_algorithm_key]}
            - Lernrate: {learning_rate}
            - Max. Iterationen: {max_iter}
            - Momentum: {'Ein' if use_momentum else 'Aus'} ({momentum_value if use_momentum else 'N/A'})
            - Adaptive Lernrate: {'Ein' if use_adaptive_lr else 'Aus'}
            """)
            
            # Direkte Optimierung ausführen via io.OPTIMIZERS
            optimizer_fn = iopt.OPTIMIZERS[selected_algorithm_key]
        
            # Visualization‑Tracker erzeugen (Callback + Speicher für Pfad & Werte)
            visualization_callback, path_hist, loss_hist = create_visualization_tracker(
                current_func_obj, x_range, y_range, contour_levels, minima
            )
        
            # Optimierung starten
            result = optimizer_fn(
                func=current_func_obj,
                x0=start_point,
                callback=visualization_callback,
                **optimizer_params
            )
        
            # Ergebnisse extrahieren
            best_x            = result.x_best
            best_history      = result.history
            best_loss_history = result.loss_history
            status            = result.status
            
            # Speichere Ergebnisse
            algorithm_display_name = f"{algorithm_options[selected_algorithm_key]}"
            
            st.session_state.optimierungsergebnisse[algorithm_display_name] = {
                "function": st.session_state.ausgewählte_funktion,
                "best_x": best_x,
                "history": best_history,
                "loss_history": best_loss_history,
                "status": status,
                "timestamp": time.time()
            }   
            
            # Zeige Zusammenfassung der Ergebnisse
            with results_container:
                st.markdown("""
                <div style="background-color: #f0fff8; padding: 12px; border-radius: 8px; border-left: 4px solid #15b371;">
                    <h3 style="color: #15b371; margin: 0;">Zusammenfassung der Optimierung</h3>
                </div>
                """, unsafe_allow_html=True)
                     
            # Zeige Details zu den Ergebnissen
            st.markdown("""
            <div style="background-color: #f0f8ff; padding: 8px; border-radius: 8px; margin-top: 15px;">
                <h4 style="color: #4d8bf0; margin: 0;">Optimierungs-Details</h4>
            </div>
            """, unsafe_allow_html=True)
                
            col1, col2, col3, col4 = st.columns(4)
                
            with col1:
                st.metric("Startpunkt", f"[{best_history[0][0]:.3f}, {best_history[0][1]:.3f}]" if best_history else "N/A")
                
            with col2:
                st.metric("Endpunkt", f"[{best_x[0]:.3f}, {best_x[1]:.3f}]" if best_x is not None else "N/A")
                
            with col3:
                st.metric("Funktionswert", f"{best_loss_history[-1]:.6f}" if best_loss_history else "N/A")
                
            with col4:
                st.metric("Iterationen", f"{len(best_loss_history)-1}" if best_loss_history else "N/A")
                
            st.markdown(f"**Status:** {status}")
                
            # Zeige Optimierungspfad als 3D-Visualisierung mit erweiterten Kontrollen
            if best_history:
                st.markdown("""
                <div style="background-color: #6a2c91; padding: 8px; border-radius: 8px; margin-bottom: 10px;">
                    <h3 style="color: white; margin: 0;">3D-Visualisierung des Optimierungspfades</h3>
                </div>
                """, unsafe_allow_html=True)
                    
            # 3D Plot Kontrollen
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                elev_3d_result = st.slider("Elevation", 0, 90, 30, key="elev_result")
            with col2:
                azim_3d_result = st.slider("Azimuth", 0, 360, 45, key="azim_result")
            with col3:
                dist_3d_result = st.slider("Zoom", 5, 20, 10, key="dist_result")
            with col4:
                resolution = st.slider("Auflösung", 30, 100, 50, key="resolution_result")
            
            # Buttons für Standardansichten
            btn_cols = st.columns(5)
            if btn_cols[0].button("Von oben", key="view_top", 
                               type="primary", 
                               use_container_width=True):
                elev_3d_result = 90
                azim_3d_result = 0
            if btn_cols[1].button("Von vorne", key="view_front", 
                               type="primary", 
                               use_container_width=True):
                elev_3d_result = 0
                azim_3d_result = 0
            if btn_cols[2].button("Von rechts", key="view_right", 
                               type="primary", 
                               use_container_width=True):
                elev_3d_result = 0
                azim_3d_result = 90
            if btn_cols[3].button("Isometrisch", key="view_iso", 
                               type="primary", 
                               use_container_width=True):
                elev_3d_result = 30
                azim_3d_result = 45
            if btn_cols[4].button("Von links", key="view_left", 
                               type="primary", 
                               use_container_width=True):
                elev_3d_result = 0
                azim_3d_result = 270
                
            # Zeichne 3D-Oberfläche mit Matplotlib
            fig3d_result = plt.figure(figsize=(10, 8))
            ax3d_result = fig3d_result.add_subplot(111, projection='3d')
            
            # Erzeuge Gitter für 3D-Plot
            x = np.linspace(x_range[0], x_range[1], resolution)
            y = np.linspace(y_range[0], y_range[1], resolution)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            
            # Berechne Funktionswerte auf dem Gitter
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        params = np.array([X[i, j], Y[i, j]])
                        result = current_func_obj(params)
                        Z[i, j] = result.get('value', np.nan)
                    except:
                        Z[i, j] = np.nan
            
            # Statistische Verarbeitung für bessere Visualisierung
            Z_finite = Z[np.isfinite(Z)]
            if len(Z_finite) > 0:
                z_mean = np.mean(Z_finite)
                z_std = np.std(Z_finite)
                z_min = max(np.min(Z_finite), z_mean - 5*z_std)
                z_max = min(np.max(Z_finite), z_mean + 5*z_std)
                
                # Extremwerte begrenzen
                Z_plot = np.copy(Z)
                Z_plot[(Z_plot < z_min) & np.isfinite(Z_plot)] = z_min
                Z_plot[(Z_plot > z_max) & np.isfinite(Z_plot)] = z_max
            else:
                Z_plot = Z
            
            # Zeichne 3D-Oberfläche
            surf = ax3d.plot_surface(X, Y, Z_plot, cmap='viridis',
                                     linewidth=0, antialiased=True, alpha=0.8)
            
            # Achsenbeschriftungen
            ax3d.set_xlabel('X')
            ax3d.set_ylabel('Y')
            ax3d.set_zlabel('Funktionswert')
            ax3d.set_title(f"3D-Oberfläche: {st.session_state.ausgewählte_funktion}")
            
            # Zeige bekannte Minima, falls vorhanden
            if minima is not None:
                for m in minima:
                    try:
                        z_val = current_func_obj(np.array(m))['value']
                        ax3d.scatter([m[0]], [m[1]], [z_val], color='red', marker='+', s=120,
                                     linewidths=2, label='Bekanntes Minimum')
                    except Exception:
                        pass
                    
            # Zeichne die Top-20 Optimierungspfade (aus der neuen App-Logik, falls gewünscht)
            paths_to_plot = []
            if "optimierungsergebnisse" in st.session_state and st.session_state.optimierungsergebnisse:
                # Nur Läufe für die aktuell ausgewählte Funktion filtern
                runs = [
                    r for r in st.session_state.optimierungsergebnisse.values()
                    if r.get('function') == st.session_state.ausgewählte_funktion and 'history' in r
                ]
                # Nach finalem Loss sortieren (kleinster Wert = beste Lösung)
                runs_sorted = sorted(
                    runs,
                    key=lambda r: r.get('loss_history', [float('inf')])[-1]
                )
                # Top 20 extrahieren
                for run in runs_sorted[:20]:
                    hist = run.get('history')
                    if hist:
                        paths_to_plot.append(np.array(hist))
            
            for idx, path in enumerate(paths_to_plot):
                xs, ys = path[:, 0], path[:, 1]
                zs = []
                for x, y in zip(xs, ys):
                    try:
                        val = current_func_obj(np.array([x, y]))['value']
                        val_clipped = np.clip(val, z_min, z_max) if np.isfinite(z_min) and np.isfinite(z_max) else val
                        zs.append(val_clipped)
                    except Exception:
                        zs.append(np.nan)
                ax3d.plot(xs, ys, zs, marker='o', linewidth=1, markersize=3,
                          alpha=0.6, label=f'Pfad {idx+1}' if idx < 1 else None)  # Nur 1x label für Legende
            
            # Zeige den neuesten Einzelpfad prominent wie in der alten App
            if st.session_state.optimierungsergebnisse:
                current_function_results = {
                    algo: result for algo, result in st.session_state.optimierungsergebnisse.items()
                    if result["function"] == st.session_state.ausgewählte_funktion and "history" in result
                }
                if current_function_results:
                    sorted_results = sorted(
                        current_function_results.items(),
                        key=lambda x: x[1].get("timestamp", 0),
                        reverse=True
                    )
                    # Neuestes Ergebnis nehmen
                    algo_name, result_data = sorted_results[0]
                    if "history" in result_data and result_data["history"]:
                        path_points = np.array(result_data["history"])
                        path_x = path_points[:, 0]
                        path_y = path_points[:, 1]
                        path_z = np.zeros(len(path_points))
                        for i, point in enumerate(result_data["history"]):
                            try:
                                params = np.array(point)
                                if callable(current_func_obj):
                                    res = current_func_obj(params)
                                else:
                                    print(f"Fehler: current_func_obj ist nicht aufrufbar an Punkt {algo_name} (Iteration {i}).")
                                    res = {'value': np.nan}
                                path_z[i] = res.get('value', np.nan)
                            except Exception as e:
                                print(f"Fehler bei Z-Berechnung für Pfadpunkt {i} von {algo_name}: {e}")
                                path_z[i] = np.nan
    
                        # Startpunkt
                        ax3d.scatter([path_x[0]], [path_y[0]], [path_z[0]],
                                     color='blue', marker='o', s=100, label='Start')
                        # Endpunkt
                        ax3d.scatter([path_x[-1]], [path_y[-1]], [path_z[-1]],
                                     color='green', marker='*', s=100, label='Ende')
                        # Pfad einzeichnen
                        ax3d.plot(path_x, path_y, path_z, 'r-o',
                                  linewidth=2, markersize=3, label='Optimierungspfad')
                    
            # Legende & Colorbar
            handles, labels = ax3d.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax3d.legend(by_label.values(), by_label.keys(), loc='upper right')
            fig3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)
            
            # Kameraeinstellungen
            ax3d.view_init(elev=st.session_state.elev_3d, azim=st.session_state.azim_3d)
            try:
                ax3d.dist = st.session_state.dist_3d / 10  # Skaliere für bessere Werte
            except Exception:
                pass
            
            # Zeige Plot
            st.pyplot(fig3d)
            plt.close(fig3d)
    
        # Zeige gespeicherte Ergebnisse
        if current_func_obj and st.session_state.optimierungsergebnisse:
            # Filtere Ergebnisse für die aktuelle Funktion
            current_function_results = {
                algo: result for algo, result in st.session_state.optimierungsergebnisse.items()
                if result["function"] == st.session_state.ausgewählte_funktion
            }
            
            if current_function_results:
                with results_container:
                    st.markdown("### Bisherige Optimierungsergebnisse")
                    
                    # Erstelle Auswahlbox für gespeicherte Ergebnisse
                    result_names = list(current_function_results.keys())
                    selected_result = st.selectbox("Ergebnis auswählen", result_names)
                    
                    if selected_result:
                        result_data = current_function_results[selected_result]
        
        # Zeige Details zu den Ergebnissen
        st.markdown("### Details")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            history = result_data.get("history", [])
            st.metric("Startpunkt", f"[{history[0][0]:.3f}, {history[0][1]:.3f}]" if history else "N/A")
        
        with col2:
            best_x = result_data.get("best_x", None)
            st.metric("Endpunkt", f"[{best_x[0]:.3f}, {best_x[1]:.3f}]" if best_x is not None else "N/A")
        
        with col3:
            loss_history = result_data.get("loss_history", [])
            st.metric("Funktionswert", f"{loss_history[-1]:.6f}" if loss_history else "N/A")
        
        with col4:
            loss_history = result_data.get("loss_history", [])
            st.metric("Iterationen", f"{len(loss_history)-1}" if loss_history else "N/A")
        
        st.markdown(f"**Status:** {result_data.get('status', 'Unbekannt')}")
        
        # Zeige Optimierungspfad als 3D-Visualisierung
        if "history" in result_data and result_data["history"]:
            st.markdown("""
            <div style="background-color: #6a2c91; padding: 8px; border-radius: 8px; margin-bottom: 10px;">
                <h3 style="color: white; margin: 0;">3D-Visualisierung des Optimierungspfades</h3>
            </div>
            """, unsafe_allow_html=True)
            
        # 3D Plot Kontrollen
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            elev_3d_prev = st.slider("Elevation", 0, 90, 30, key="elev_prev")
        with col2:
            azim_3d_prev = st.slider("Azimuth", 0, 360, 45, key="azim_prev")
        with col3:
            dist_3d_prev = st.slider("Zoom", 5, 20, 10, key="dist_prev")
        with col4:
            resolution_prev = st.slider("Auflösung", 30, 100, 50, key="resolution_prev")
        
        # Buttons für Standardansichten
        btn_cols = st.columns(5)
        if btn_cols[0].button("Von oben", key="prev_top", 
                           type="primary", 
                           use_container_width=True):
            elev_3d_prev = 90
            azim_3d_prev = 0
        if btn_cols[1].button("Von vorne", key="prev_front", 
                           type="primary", 
                           use_container_width=True):
            elev_3d_prev = 0
            azim_3d_prev = 0
        if btn_cols[2].button("Von rechts", key="prev_right", 
                           type="primary", 
                           use_container_width=True):
            elev_3d_prev = 0
            azim_3d_prev = 90
        if btn_cols[3].button("Isometrisch", key="prev_iso", 
                           type="primary", 
                           use_container_width=True):
            elev_3d_prev = 30
            azim_3d_prev = 45
        if btn_cols[4].button("Von links", key="prev_left", 
                           type="primary", 
                           use_container_width=True):
            elev_3d_prev = 0
            azim_3d_prev = 270
            
        # Zeichne 3D-Oberfläche mit Matplotlib
        fig3d_prev = plt.figure(figsize=(10, 8))
        ax3d_prev = fig3d_prev.add_subplot(111, projection='3d')
        
        # Erzeuge Gitter für 3D-Plot
        x = np.linspace(x_range[0], x_range[1], resolution_prev)
        y = np.linspace(y_range[0], y_range[1], resolution_prev)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # Berechne Funktionswerte auf dem Gitter
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    params = np.array([X[i, j], Y[i, j]])
                    result = current_func_obj(params)
                    Z[i, j] = result.get('value', np.nan)
                except:
                    Z[i, j] = np.nan
        
        # Statistische Verarbeitung für bessere Visualisierung
        Z_finite = Z[np.isfinite(Z)]
        if len(Z_finite) > 0:
            z_mean = np.mean(Z_finite)
            z_std = np.std(Z_finite)
            z_min = max(np.min(Z_finite), z_mean - 5*z_std)
            z_max = min(np.max(Z_finite), z_mean + 5*z_std)
            
            # Extremwerte begrenzen
            Z_plot = np.copy(Z)
            Z_plot[(Z_plot < z_min) & np.isfinite(Z_plot)] = z_min
            Z_plot[(Z_plot > z_max) & np.isfinite(Z_plot)] = z_max
        else:
            Z_plot = Z
            z_min = np.nan
            z_max = np.nan
        
        # Zeichne 3D-Oberfläche
        surf = ax3d_prev.plot_surface(X, Y, Z_plot, cmap='viridis', 
                                    linewidth=0, antialiased=True, alpha=0.7,
                                    rstride=1, cstride=1)
        
        # Zeichne Optimierungspfad in 3D
        if "history" in result_data:
            path_points = np.array(result_data["history"])
            path_x = path_points[:, 0]
            path_y = path_points[:, 1]
            path_z = np.zeros(len(path_points))
            
            # Berechne Z-Werte für den Pfad
            for i, point in enumerate(result_data["history"]):
                try:
                    params = np.array(point)
                    res = current_func_obj(params)
                    path_z[i] = res.get('value', np.nan)
                    
                    # Begrenze extreme Z-Werte falls statistisch verarbeitet
                    if np.isfinite(path_z[i]) and np.isfinite(z_min) and np.isfinite(z_max):
                        path_z[i] = min(max(path_z[i], z_min), z_max)
                except:
                    path_z[i] = np.nan
        
            # Startpunkt besonders hervorheben
            ax3d_prev.scatter([path_x[0]], [path_y[0]], [path_z[0]], 
                        color='blue', marker='o', s=100, label='Start')
            
            # Endpunkt besonders hervorheben
            ax3d_prev.scatter([path_x[-1]], [path_y[-1]], [path_z[-1]], 
                        color='red', marker='*', s=100, label='Ende')
            
            # Pfad einzeichnen
            ax3d_prev.plot(path_x, path_y, path_z, 'r-o', 
                      linewidth=2, markersize=4, label='Optimierungspfad')
        
        # Minima einzeichnen, falls vorhanden
        if minima is not None:
            for i, m in enumerate(minima):
                try:
                    params = np.array(m)
                    res = current_func_obj(params)
                    z_val = res.get('value', np.nan)
                    if np.isfinite(z_val):
                        ax3d_prev.scatter([m[0]], [m[1]], [z_val], 
                                    color='green', marker='+', s=120, 
                                    linewidths=2, label='Bekanntes Minimum' if i==0 else None)
                except:
                    pass
        
        # Achsenbeschriftungen und Titel
        ax3d_prev.set_xlabel('X')
        ax3d_prev.set_ylabel('Y')
        ax3d_prev.set_zlabel('Funktionswert')
        ax3d_prev.set_title(f"3D-Pfad: {selected_result}")
        
        # Blickwinkel setzen
        ax3d_prev.view_init(elev=elev_3d_prev, azim=azim_3d_prev)
        
        # Kameradistanz setzen (wenn möglich)
        try:
            ax3d_prev.dist = dist_3d_prev / 10
        except:
            pass  # Ältere matplotlib-Versionen unterstützen dies nicht
        
        # Legende anzeigen
        ax3d_prev.legend(loc='upper right')
        
        # Colorbar hinzufügen
        fig3d_prev.colorbar(surf, ax=ax3d_prev, shrink=0.5, aspect=5)
        
        # Plot anzeigen
        st.pyplot(fig3d_prev)
        plt.close(fig3d_prev)

with tabs[1]:
    st.markdown("## Funktionseditor")
    st.markdown("""
    Hier kannst du eigene mathematische Funktionen definieren und testen. Verwende `x` und `y` als Variablen.
    
    **Beispiele:**
    - `x**2 + y**2` (Parabel)
    - `sin(x) + cos(y)` (Sinuswelle)
    - `(x-2)**2 + (y-3)**2` (Verschobene Parabel)
    """)
    
    # Benutzerdefinierte Funktion erstellen
    col1, col2 = st.columns([3, 1])
    
    with col1:
        custom_func_expr = st.text_input(
            "Funktionsausdruck eingeben (mit x und y als Variablen)",
            value="x**2 + y**2",
            key="custom_func_input"
        )
    
    with col2:
        custom_func_name = st.text_input(
            "Name der Funktion",
            value=f"Custom_{st.session_state.custom_func_count + 1}",
            key="custom_func_name"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_min = st.number_input("X-Minimum", value=-5.0, step=0.5)
        y_min = st.number_input("Y-Minimum", value=-5.0, step=0.5)
    
    with col2:
        x_max = st.number_input("X-Maximum", value=5.0, step=0.5)
        y_max = st.number_input("Y-Maximum", value=5.0, step=0.5)
    
    if st.button("Funktion erstellen", use_container_width=True):
        # Füge Mathematische Operatoren hinzu, falls sie in simplen Form eingegeben wurden
        expr_to_parse = custom_func_expr
        expr_to_parse = re.sub(r'(?<![a-zA-Z])sin\(', 'sp.sin(', expr_to_parse)
        expr_to_parse = re.sub(r'(?<![a-zA-Z])cos\(', 'sp.cos(', expr_to_parse)
        expr_to_parse = re.sub(r'(?<![a-zA-Z])tan\(', 'sp.tan(', expr_to_parse)
        expr_to_parse = re.sub(r'(?<![a-zA-Z])exp\(', 'sp.exp(', expr_to_parse)
        expr_to_parse = re.sub(r'(?<![a-zA-Z])log\(', 'sp.log(', expr_to_parse)
        expr_to_parse = re.sub(r'(?<![a-zA-Z])sqrt\(', 'sp.sqrt(', expr_to_parse)
        expr_to_parse = re.sub(r'(?<![a-zA-Z])abs\(', 'sp.Abs(', expr_to_parse)
        
        try:
            # Erstelle benutzerdefinierte Funktion
            custom_func = pf.create_custom_function(
                expr_to_parse,
                name=custom_func_name,
                x_range=(x_min, x_max),
                y_range=(y_min, y_max)
            )
            
            # Teste die Funktion
            test_result = custom_func(np.array([1.0, 1.0]))
            if "value" in test_result and np.isfinite(test_result["value"]):
                # Funktion ist gültig
                st.session_state.custom_funcs[custom_func_name] = custom_func
                st.session_state.custom_func_count += 1
                st.session_state.ausgewählte_funktion = custom_func_name
                st.success(f"Funktion '{custom_func_name}' erfolgreich erstellt!")
                st.rerun()
            else:
                st.error(f"Die Funktion konnte nicht evaluiert werden. Überprüfe den Ausdruck auf Gültigkeit.")
        except Exception as e:
            st.error(f"Fehler beim Erstellen der Funktion: {e}")
    
    # Vorschau der aktuellen benutzerdefinierten Funktionen
    if st.session_state.custom_funcs:
        st.markdown("### Deine benutzerdefinierten Funktionen")
        
        for name, func in st.session_state.custom_funcs.items():
            with st.expander(name):
                # Zeige Vorschau der Funktion
                try:
                    # Direkte Implementierung des Konturplots
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111)
                    
                    # Parameterbereich
                    x_range = (-5, 5)
                    y_range = (-5, 5)
                    
                    # Erzeuge Gitter für Konturplot
                    X, Y = np.meshgrid(
                        np.linspace(x_range[0], x_range[1], 100),
                        np.linspace(y_range[0], y_range[1], 100)
                    )
                    Z = np.zeros_like(X)
                    
                    # Berechne Funktionswerte auf dem Gitter
                    for i in range(X.shape[0]):
                        for j in range(X.shape[1]):
                            try:
                                params = np.array([X[i, j], Y[i, j]])
                                result = func(params)
                                Z[i, j] = result.get('value', np.nan)
                            except:
                                Z[i, j] = np.nan
                    
                    # Statistische Verarbeitung für bessere Konturen
                    Z_finite = Z[np.isfinite(Z)]
                    if len(Z_finite) > 0:
                        z_mean = np.mean(Z_finite)
                        z_std = np.std(Z_finite)
                        
                        # Begrenze extreme Werte für bessere Visualisierung
                        lower_bound = max(np.min(Z_finite), z_mean - 5*z_std)
                        upper_bound = min(np.max(Z_finite), z_mean + 5*z_std)
                        
                        Z_clip = np.copy(Z)
                        Z_clip[(Z_clip < lower_bound) & np.isfinite(Z_clip)] = lower_bound
                        Z_clip[(Z_clip > upper_bound) & np.isfinite(Z_clip)] = upper_bound
                    else:
                        Z_clip = Z
                    
                    # Zeichne Konturplot
                    levels = 30
                    cp = ax.contourf(X, Y, Z_clip, levels=levels, cmap='viridis', alpha=0.8)
                    contour_lines = ax.contour(X, Y, Z_clip, levels=min(10, levels//3), 
                                            colors='black', alpha=0.3, linewidths=0.5)
                    
                    # Versuche Minima zu finden
                    try:
                        if "minima" in result and result["minima"] is not None:
                            for m in result["minima"]:
                                ax.plot(m[0], m[1], 'r*', markersize=10)
                    except:
                        pass
                    
                    # Achsenbeschriftungen
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_title(name)
                    
                    # Achsengrenzen setzen
                    ax.set_xlim(x_range)
                    ax.set_ylim(y_range)
                    
                    # Farbskala hinzufügen
                    fig.colorbar(cp, ax=ax, label='Funktionswert')
                    
                    # Zeichne Grid
                    ax.grid(True, linestyle='--', alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Füge Button zum Löschen hinzu
                    if st.button(f"Löschen: {name}"):
                        del st.session_state.custom_funcs[name]
                        if st.session_state.ausgewählte_funktion == name:
                            st.session_state.ausgewählte_funktion = "Rosenbrock"
                        st.rerun()
                except Exception as e:
                    st.error(f"Fehler beim Anzeigen der Funktion: {e}")

with tabs[2]:
    st.markdown("## Ergebnisvergleich")
    
    if not st.session_state.optimierungsergebnisse:
        st.info("Keine Optimierungsergebnisse verfügbar. Führe zuerst einige Optimierungen durch.")
    else:
        # Gruppiere Ergebnisse nach Funktionen
        function_groups = {}
        for algo, result in st.session_state.optimierungsergebnisse.items():
            func_name = result["function"]
            if func_name not in function_groups:
                function_groups[func_name] = []
            function_groups[func_name].append(algo)
        
        # Dropdown zur Auswahl der Funktion für den Vergleich
        selected_function_for_comparison = st.selectbox(
            "Funktion für Vergleich auswählen",
            list(function_groups.keys())
        )
        
        if selected_function_for_comparison:
            # Zeige Algorithmen für diese Funktion
            algos_for_function = function_groups[selected_function_for_comparison]
            
            # Multiselect für Algorithmen
            selected_algos = st.multiselect(
                "Algorithmen zum Vergleich auswählen",
                algos_for_function,
                default=algos_for_function[:min(3, len(algos_for_function))]
            )
            
            if selected_algos:
                # Extrahiere nur die Loss-Historie für jeden Algorithmus
                comparison_results = {}
                for algo in selected_algos:
                    if algo in st.session_state.optimierungsergebnisse:
                        algo_result = st.session_state.optimierungsergebnisse[algo]
                        if "loss_history" in algo_result:
                            comparison_results[algo] = algo_result["loss_history"]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Vergleiche Verlaufskurven (direkte Implementierung)
                    fig_comparison = plt.figure(figsize=(10, 6))
                    ax_comparison = fig_comparison.add_subplot(111)
                    
                    # Zeichne Verlaufskurven für alle Algorithmen
                    for algo_name, loss_hist in comparison_results.items():
                        if isinstance(loss_hist, list) and loss_hist:  # Sicherstellen dass es eine Liste ist
                            iterations = list(range(len(loss_hist)))  # Liste explizit erstellen
                            ax_comparison.plot(iterations, loss_hist, '-o', 
                                           label=algo_name, linewidth=2, markersize=3)
                    
                    # Logarithmische Y-Achse für bessere Sichtbarkeit
                    ax_comparison.set_yscale('log')
                    
                    # Achsenbeschriftungen
                    ax_comparison.set_xlabel('Iteration')
                    ax_comparison.set_ylabel('Funktionswert (log)')
                    ax_comparison.set_title('Vergleich der Optimierungsalgorithmen')
                    
                    # Grid und Legende
                    ax_comparison.grid(True, linestyle='--', alpha=0.7)
                    ax_comparison.legend(loc='best')
                    
                    # Layout verbessern
                    fig_comparison.tight_layout()
                    
                    # Zeige Plot
                    st.pyplot(fig_comparison)
                    plt.close(fig_comparison)
                
                with col2:
                    # Tabelle mit Ergebnissen
                    result_data = []
                    for algo in selected_algos:
                        result = st.session_state.optimierungsergebnisse[algo]
                        best_x = result.get("best_x", None)
                        loss_history = result.get("loss_history", [])
                        final_value = loss_history[-1] if loss_history else float('inf')
                        iterations = len(loss_history) - 1 if loss_history else 0
                        
                        result_data.append({
                            "Algorithmus": algo,
                            "Endwert": f"{final_value:.6f}",
                            "Iterationen": iterations,
                            "Endpunkt": f"[{best_x[0]:.3f}, {best_x[1]:.3f}]" if best_x is not None else "N/A"
                        })
                    
                    st.dataframe(result_data)
                
                # Zeige die Pfade aller ausgewählten Algorithmen in einem Plot
                if selected_function_for_comparison in pf.MATH_FUNCTIONS_LIB:
                    func_info = pf.MATH_FUNCTIONS_LIB[selected_function_for_comparison]
                    current_func_obj = func_info["func"]
                    x_range = func_info["default_range"][0]
                    y_range = func_info["default_range"][1]
                    contour_levels = func_info.get("contour_levels", 40)
                    try:
                        func_meta = current_func_obj(np.array([0, 0]))
                        minima = func_meta.get("minima", None)
                    except Exception:
                        minima = None
                
                elif selected_function_for_comparison in st.session_state.custom_funcs:
                    current_func_obj = st.session_state.custom_funcs[selected_function_for_comparison]
                    x_range = (-5, 5)
                    y_range = (-5, 5)
                    contour_levels = 30
                    minima = None
                else:
                    st.error("Die ausgewählte Funktion wurde nicht gefunden.")
                    current_func_obj = None
                
                if current_func_obj:
                    # Erstelle Figur und Achsen
                    fig, ax = plt.subplots(figsize=(10, 8))
                
                    # Erzeuge Mesh-Daten (direkte Implementierung)
                    n_points = 100  # Anzahl der Punkte pro Dimension
                    x = np.linspace(x_range[0], x_range[1], n_points)
                    y = np.linspace(y_range[0], y_range[1], n_points)
                    X_mesh, Y_mesh = np.meshgrid(x, y)
                    Z_mesh = np.zeros_like(X_mesh)
                
                    # Berechne Funktionswerte auf dem Gitter
                    for i in range(X_mesh.shape[0]):
                        for j in range(X_mesh.shape[1]):
                            try:
                                params = np.array([X_mesh[i, j], Y_mesh[i, j]])
                                result = current_func_obj(params)
                                Z_mesh[i, j] = result.get('value', np.nan)
                            except:
                                Z_mesh[i, j] = np.nan
                                    
                    # Statistische Verarbeitung für bessere Visualisierung
                    Z_finite = Z_mesh[np.isfinite(Z_mesh)]
                    if len(Z_finite) > 0:
                        z_mean = np.mean(Z_finite)
                        z_std = np.std(Z_finite)
                        z_min = max(np.min(Z_finite), z_mean - 5*z_std)
                        z_max = min(np.max(Z_finite), z_mean + 5*z_std)
                                        
                    # Extreme Werte begrenzen
                    Z_mesh_mod = np.copy(Z_mesh)
                    Z_mesh_mod[(Z_mesh_mod < z_min) & np.isfinite(Z_mesh_mod)] = z_min
                    Z_mesh_mod[(Z_mesh_mod > z_max) & np.isfinite(Z_mesh_mod)] = z_max
                    Z_mesh = Z_mesh_mod
                
                    # Zeichne Konturplot
                    contour = ax.contourf(X_mesh, Y_mesh, Z_mesh, contour_levels, cmap='viridis', alpha=0.8)
                                    
                    # Füge Farbbalken hinzu
                    cbar = fig.colorbar(contour, ax=ax)
                    cbar.set_label('Funktionswert')
                                    
                # Zeichne Pfade für jeden Algorithmus
                colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
                
                for i, algo in enumerate(selected_algos):
                    result = st.session_state.optimierungsergebnisse[algo]
                    path = result.get("history", [])
                
                    if path:
                        path_x = [p[0] for p in path]
                        path_y = [p[1] for p in path]
                        ax.plot(path_x, path_y, '-o', color=colors[i % len(colors)], linewidth=2, markersize=4, label=algo)
                
                        # Markiere Endpunkt
                        ax.plot(path_x[-1], path_y[-1], '*', color=colors[i % len(colors)], markersize=10)
                
                # Zeichne bekannte Minima, falls vorhanden
                if minima:
                    for minimum in minima:
                        ax.plot(minimum[0], minimum[1], 'X', color='white', markersize=8, markeredgecolor='black')
                
                # Füge Titel hinzu
                ax.set_title(f"Vergleich der Optimierungspfade: {selected_function_for_comparison}")
                
                # Achsenbeschriftungen
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                
                # Zeige Legende
                ax.legend()
                
                # Setze Achsengrenzen
                ax.set_xlim(x_range)
                ax.set_ylim(y_range)
                
                st.pyplot(fig)
                plt.close(fig)
                
                # 3D-Vergleich
                st.markdown("### 3D-Vergleich der Optimierungspfade")
                
                # Erstelle 3D-Plot mit Plotly
                fig3d = go.Figure()
                
                # Füge Oberfläche hinzu (direkte Implementierung)
                n_points = 50  # Reduzierte Auflösung für bessere Performance im 3D-Plot
                x = np.linspace(x_range[0], x_range[1], n_points)
                y = np.linspace(y_range[0], y_range[1], n_points)
                X_mesh, Y_mesh = np.meshgrid(x, y)
                Z_mesh = np.zeros_like(X_mesh)
                
                # Berechne Funktionswerte auf dem Gitter
                for i in range(X_mesh.shape[0]):
                    for j in range(X_mesh.shape[1]):
                        try:
                            params = np.array([X_mesh[i, j], Y_mesh[i, j]])
                            result = current_func_obj(params)
                            Z_mesh[i, j] = result.get('value', np.nan)
                        except:
                            Z_mesh[i, j] = np.nan
                
                # Statistische Verarbeitung für bessere Visualisierung
                Z_finite = Z_mesh[np.isfinite(Z_mesh)]
                if len(Z_finite) > 0:
                    z_mean = np.mean(Z_finite)
                    z_std = np.std(Z_finite)
                
                    # Begrenze extreme Werte für bessere Visualisierung
                    lower_bound = max(np.min(Z_finite), z_mean - 5*z_std)
                    upper_bound = min(np.max(Z_finite), z_mean + 5*z_std)
                
                    Z_mesh_mod = np.copy(Z_mesh)
                    Z_mesh_mod[(Z_mesh_mod < lower_bound) & np.isfinite(Z_mesh_mod)] = lower_bound
                    Z_mesh_mod[(Z_mesh_mod > upper_bound) & np.isfinite(Z_mesh_mod)] = upper_bound
                    Z_mesh = Z_mesh_mod
                
                fig3d.add_trace(go.Surface(
                    x=X_mesh, y=Y_mesh, z=Z_mesh,
                    colorscale='viridis',
                    opacity=0.8,
                    showscale=True
                ))
                
                # Füge Pfade hinzu
                for i, algo in enumerate(selected_algos):
                    result = st.session_state.optimierungsergebnisse[algo]
                    path = result.get("history", [])
                
                    if path:
                        path_x = [p[0] for p in path]
                        path_y = [p[1] for p in path]
                
                        # Berechne z-Werte für den Pfad
                        path_z = []
                        for p in path:
                            try:
                                result = current_func_obj(p)
                                if 'value' in result and np.isfinite(result['value']):
                                    path_z.append(result['value'])
                                else:
                                    path_z.append(None)
                            except:
                                path_z.append(None)
                
                        # Zeichne Pfad
                        fig3d.add_trace(go.Scatter3d(
                            x=path_x, y=path_y, z=path_z,
                            mode='lines+markers',
                            line=dict(color=colors[i % len(colors)], width=5),
                            marker=dict(size=4, color=colors[i % len(colors)]),
                            name=algo
                        ))
                
                # Füge bekannte Minima hinzu, falls vorhanden
                if minima:
                    min_x = [m[0] for m in minima]
                    min_y = [m[1] for m in minima]
                    min_z = []
                    for m in minima:
                        try:
                            z = current_func_obj(np.array(m))['value']
                            min_z.append(z)
                        except:
                            min_z.append(None)
                
                    fig3d.add_trace(go.Scatter3d(
                        x=min_x, y=min_y, z=min_z,
                        mode='markers',
                        marker=dict(size=8, color='white', symbol='x', line=dict(color='black', width=2)),
                        name='Bekannte Minima'
                    ))
                
                # Layout-Konfiguration
                fig3d.update_layout(
                    title=f"3D-Vergleich: {selected_function_for_comparison}",
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Funktionswert',
                        aspectratio=dict(x=1, y=1, z=0.8)
                    ),
                    margin=dict(l=0, r=0, b=0, t=30),
                    legend=dict(
                        x=0.02,
                        y=0.98,
                        bordercolor="Black",
                        borderwidth=1
                    )
                )
                
                st.plotly_chart(fig3d, use_container_width=True, height=600)


with tabs[3]: # Info & Hilfe
    st.header("Info & Hilfe")
    st.markdown("""
    ### Über IntelliScope Explorer
    
    Inteliscope Explorer ist ein interaktives Tool zur Erkundung von Optimierungsfunktionen und -algorithmen.
    Es ermöglicht Ihnen, verschiedene Algorithmen auf unterschiedlichen Testfunktionen zu visualisieren
    und deren Verhalten sowie Konvergenzeigenschaften zu analysieren.
    
    ### Kernfunktionen
    
    - **Testfunktionen**: Eine Bibliothek vordefinierter mathematischer Funktionen (z.B. Rosenbrock, Rastrigin) mit bekannten Eigenschaften und Herausforderungen für Optimierer.
    - **Benutzerdefinierte Funktionen**: Erstellen und visualisieren Sie eigene 2D-Funktionen direkt im Tool.
    - **Optimierungsalgorithmen**: Implementierungen gängiger Gradientenabstiegsverfahren:
        - **Gradient Descent mit Liniensuche**: Findet iterativ eine optimale Schrittweite.
        - **Gradient Descent mit Momentum**: Beschleunigt die Konvergenz durch Berücksichtigung vergangener Schritte.
        - **Adam Optimizer**: Kombiniert adaptive Lernraten mit Momentum für robuste Leistung.
    - **Visualisierungen**:
        - **3D-Oberflächenplots**: Darstellung der Funktionslandschaft.
        - **2D-Konturplots**: Visualisierung der Höhenlinien und Optimierungspfade.
        - **Live-Tracking**: Verfolgen Sie den Optimierungsprozess in Echtzeit.
        - **Konvergenzplots**: Zeigen den Verlauf des Funktionswertes über die Iterationen.
    - **Ergebnisvergleich**: Vergleichen Sie die Leistung und Pfade verschiedener Algorithmen auf derselben Funktion.
    - **Interaktive Steuerung**: Passen Sie Algorithmusparameter, Startpunkte und Visualisierungsdetails an.
    
    ### Wie funktioniert es?
    
    1.  **Funktion auswählen**: Wählen Sie eine Standardfunktion oder erstellen Sie eine eigene im Funktionseditor.
    2.  **Algorithmus wählen**: Entscheiden Sie sich für einen Optimierungsalgorithmus.
    3.  **Parameter anpassen**: Justieren Sie Parameter wie Lernrate, maximale Iterationen etc.
    4.  **Optimierung starten**: Beobachten Sie den Algorithmus bei der Arbeit.
    5.  **Ergebnisse analysieren**: Untersuchen Sie die Plots und Metriken, um das Verhalten des Algorithmus zu verstehen.
    
    ### Testfunktionen im Detail
    
    -   **Rosenbrock**: $f(x,y) = (a-x)^2 + b(y-x^2)^2$. Klassisch schwierig, mit einem langen, schmalen, gekrümmten Tal. Globales Minimum bei $(a, a^2)$.
    -   **Himmelblau**: $f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2$. Hat vier identische lokale Minima.
    -   **Rastrigin**: $f(x) = An + \sum_{i=1}^{n} [x_i^2 - A \cos(2\pi x_i)]$. Hochgradig multimodal mit vielen lokalen Minima. Globales Minimum bei $x_i=0$.
    -   **Ackley**: $f(x) = -a \exp(-b \sqrt{\frac{1}{n}\sum x_i^2}) - \exp(\frac{1}{n}\sum \cos(cx_i)) + a + \exp(1)$. Viele lokale Minima, globales Minimum bei $x_i=0$.
    -   **Schwefel**: $f(x) = 418.9829n - \sum x_i \sin(\sqrt{|x_i|})$. Viele lokale Minima, das globale Minimum ist weit von den nächstbesten entfernt.
    -   **Eggcrate**: $f(x,y) = x^2 + y^2 + 25(\sin^2(x) + \sin^2(y))$. Einfacheres Beispiel mit vielen regelmäßigen lokalen Minima.
        
    ### Tipps zur Nutzung
    
    -   Beginnen Sie mit einfacheren Funktionen (z.B. Eggcrate oder eine eigene quadratische Funktion), um ein Gefühl für die Algorithmen zu bekommen.
    -   Experimentieren Sie mit den Parametern der Algorithmen. Eine zu hohe Lernrate kann zur Divergenz führen, eine zu niedrige zu langsamer Konvergenz.
    -   Nutzen Sie die Visualisierungen, um zu verstehen, wie sich die Algorithmen in verschiedenen Regionen der Funktionslandschaft verhalten.
    -   Vergleichen Sie die Ergebnisse verschiedener Algorithmen auf derselben Funktion, um deren Stärken und Schwächen zu erkennen.
    
    Viel Spaß beim Erkunden der faszinierenden Welt der Optimierung!
    """)

# Der Footer bleibt danach unverändert am Ende der Datei:
st.markdown("---")
st.markdown("© 2024-2025 IntelliScope Explorer")

