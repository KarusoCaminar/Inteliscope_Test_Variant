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
import improved_optimizer as io
import data_manager as dm

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
if 'optimierungsergebnisse' not in st.session_state:
    st.session_state.optimierungsergebnisse = {}
if 'custom_funcs' not in st.session_state:
    st.session_state.custom_funcs = {}
if 'ausgewählte_funktion' not in st.session_state:
    st.session_state.ausgewählte_funktion = "Rosenbrock"
if 'custom_func_count' not in st.session_state:
    st.session_state.custom_func_count = 0

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
    custom_funcs = list(st.session_state.custom_funcs.keys())
    
    all_functions = function_list.copy()
    if custom_funcs:
        all_functions.append("----------")
        all_functions.extend(custom_funcs)
    
    selected_function = st.selectbox(
        "Funktion auswählen",
        all_functions,
        index=all_functions.index(st.session_state.ausgewählte_funktion) if st.session_state.ausgewählte_funktion in all_functions else 0
    )
    
    if selected_function != "----------":
        st.session_state.ausgewählte_funktion = selected_function
    
    # Algorithmenauswahl
    algorithm_options = {
        "GD_Simple_LS": "Gradient Descent mit Liniensuche",
        "GD_Momentum": "Gradient Descent mit Momentum",
        "Adam": "Adam Optimizer"
    }
    
    selected_algorithm = st.selectbox(
        "Algorithmus auswählen",
        list(algorithm_options.keys()),
        format_func=lambda x: algorithm_options[x]
    )
    
    # Optimierungsstrategie
    strategy_options = {
        "single": "Einzelne Optimierung",
        "multi_start": "Multi-Start Optimierung",
        "adaptive": "Adaptive Multi-Start"
    }
    
    selected_strategy = st.selectbox(
        "Strategie auswählen",
        list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x]
    )
    
    # Parameter für den ausgewählten Algorithmus
    st.subheader("Algorithmus-Parameter")
    
    optimizer_params = {}

    if selected_algorithm == "GD_Simple_LS":
        max_iter = st.slider("Max. Iterationen", 100, 5000, 1000, step=100)
        step_norm_tol = st.slider("Schrittnorm Toleranz", 1e-12, 1e-3, 1e-7, format="%.0e")
        func_impr_tol = st.slider("Funktionsverbesserung Toleranz", 1e-12, 1e-3, 1e-9, format="%.0e")
        initial_t_ls = st.slider("Initialer Liniensuchschritt", 1e-6, 1.0, 0.1, format="%.0e")

        optimizer_params = {
            "max_iter": max_iter,
            "step_norm_tol": step_norm_tol,
            "func_impr_tol": func_impr_tol,
            "initial_t_ls": initial_t_ls
        }

    elif selected_algorithm == "GD_Momentum":
        max_iter = st.slider("Max. Iterationen", 100, 5000, 1000, step=100)
        learning_rate = st.slider("Lernrate", 1e-5, 1.0, 0.05, format="%.4f")
        momentum_beta = st.slider("Momentum Beta", 0.0, 0.999, 0.95, format="%.3f")
        grad_norm_tol = st.slider("Gradientennorm Toleranz", 1e-12, 1e-3, 1e-7, format="%.0e")

        optimizer_params = {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "momentum_beta": momentum_beta,
            "grad_norm_tol": grad_norm_tol
        }

    elif selected_algorithm == "Adam":
        max_iter = st.slider("Max. Iterationen", 100, 5000, 1000, step=100)
        learning_rate = st.slider("Lernrate", 1e-5, 1.0, 0.005, format="%.5f")
        beta1 = st.slider("Beta1 (Momentum)", 0.0, 0.999, 0.95, format="%.3f")
        beta2 = st.slider("Beta2 (RMSProp)", 0.0, 0.9999, 0.9995, format="%.4f")
        epsilon = st.slider("Epsilon", 1e-12, 1e-6, 1e-8, format="%.0e")

        optimizer_params = {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon
        }
    
    # Parameter für die Optimierungsstrategie
    if selected_strategy != "single":
        st.subheader("Strategie-Parameter")
        
        multi_params = {}
        
        if selected_strategy == "multi_start":
            num_starts = st.slider("Anzahl der Starts", 2, 20, 5)
            use_challenging_starts = st.checkbox("Herausfordernde Startpunkte", value=True)
            multi_seed = st.slider("Seed", 0, 100, 42)
            
            multi_params = {
                "num_starts": num_starts,
                "use_challenging_starts": use_challenging_starts,
                "seed": multi_seed
            }
            
        elif selected_strategy == "adaptive":
            initial_starts = st.slider("Initiale Anzahl der Starts", 2, 10, 3)
            max_starts = st.slider("Maximale Anzahl der Starts", initial_starts, 30, 10)
            min_improvement = st.slider("Min. Verbesserung für weitere Starts", 0.001, 0.1, 0.01, format="%.3f")
            adaptive_seed = st.slider("Seed", 0, 100, 42)
            
            multi_params = {
                "initial_starts": initial_starts,
                "max_starts": max_starts,
                "min_improvement": min_improvement,
                "seed": adaptive_seed
            }
    else:
        multi_params = {}
    
    # Button zum Starten der Optimierung
    start_optimization = st.button("Optimierung starten", use_container_width=True)
    
    # Button zum Zurücksetzen aller Ergebnisse
    if st.button("Alle Ergebnisse zurücksetzen", use_container_width=True):
        st.session_state.optimierungsergebnisse = {}
        st.rerun()

# Hauptbereich für Visualisierung
# Tabs für verschiedene Visualisierungen und Interaktionen
tabs = st.tabs(["Optimierungsvisualisierung", "Funktionseditor", "Ergebnisvergleich"])

with tabs[0]:
    # Hole die aktuelle Funktion
    if st.session_state.ausgewählte_funktion in pf.MATH_FUNCTIONS_LIB:
        current_func_info = pf.MATH_FUNCTIONS_LIB[st.session_state.ausgewählte_funktion]
        current_func = current_func_info["func"]
        x_range = current_func_info["default_range"][0]
        y_range = current_func_info["default_range"][1]
        contour_levels = current_func_info.get("contour_levels", 40)
        
        # Zeige Tooltip für die Funktion, falls vorhanden
        func_result = current_func(np.array([0, 0]))
        if "tooltip" in func_result:
            with st.expander("ℹ️ Über diese Funktion", expanded=False):
                st.markdown(func_result["tooltip"])
        
        # Berechne bekannte Minima, falls vorhanden
        minima = func_result.get("minima", None)
        
    elif st.session_state.ausgewählte_funktion in st.session_state.custom_funcs:
        current_func = st.session_state.custom_funcs[st.session_state.ausgewählte_funktion]
        # Verwende Standardbereiche für benutzerdefinierte Funktionen
        x_range = (-5, 5)
        y_range = (-5, 5)
        contour_levels = 30
        minima = None
    else:
        st.error("Die ausgewählte Funktion wurde nicht gefunden.")
        current_func = None
        x_range = (-5, 5)
        y_range = (-5, 5)
        contour_levels = 30
        minima = None
    
    # Layout für die Visualisierung
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Erstelle 3D-Plot mit Matplotlib und füge Kontrollen hinzu
        if current_func:
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
                            result = current_func(params)
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
                            z_val = current_func(np.array(m))['value']
                            ax3d.scatter([m[0]], [m[1]], [z_val], color='red', marker='+', s=120, 
                                        linewidths=2, label='Bekanntes Minimum')
                        except:
                            pass
                
                # Zeichne Optimierungspfade aus vorherigen Optimierungen
                if st.session_state.optimierungsergebnisse:
                    # Filtere Ergebnisse für die aktuelle Funktion
                    current_function_results = {
                        algo: result for algo, result in st.session_state.optimierungsergebnisse.items()
                        if result["function"] == st.session_state.ausgewählte_funktion and "history" in result
                    }
                    
                    # Zeige den neuesten Pfad
                    if current_function_results:
                        # Sortiere nach Zeitstempel (neueste zuerst)
                        sorted_results = sorted(
                            current_function_results.items(),
                            key=lambda x: x[1].get("timestamp", 0),
                            reverse=True
                        )
                        
                        # Nimm die neueste Optimierung
                        algo_name, result_data = sorted_results[0]
                        
                        if "history" in result_data and result_data["history"]:
                            path_points = np.array(result_data["history"])
                            path_x = path_points[:, 0]
                            path_y = path_points[:, 1]
                            path_z = np.zeros(len(path_points))
                            
                            # Berechne Z-Werte für den Pfad
                            for i, point in enumerate(result_data["history"]):
                                try:
                                    params = np.array(point)
                                    res = current_func(params)
                                    path_z[i] = res.get('value', np.nan)
                                    
                                    # Begrenze extreme Z-Werte
                                    if np.isfinite(path_z[i]) and np.isfinite(z_min) and np.isfinite(z_max):
                                        path_z[i] = min(max(path_z[i], z_min), z_max)
                                except:
                                    path_z[i] = np.nan
                            
                            # Startpunkt besonders hervorheben
                            ax3d.scatter([path_x[0]], [path_y[0]], [path_z[0]], 
                                        color='blue', marker='o', s=100, label='Start')
                            
                            # Endpunkt besonders hervorheben
                            ax3d.scatter([path_x[-1]], [path_y[-1]], [path_z[-1]], 
                                        color='green', marker='*', s=100, label='Ende')
                            
                            # Pfad einzeichnen
                            ax3d.plot(path_x, path_y, path_z, 'r-o', 
                                    linewidth=2, markersize=3, label='Optimierungspfad')
                
                # Legende hinzufügen
                handles, labels = ax3d.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax3d.legend(by_label.values(), by_label.keys(), loc='upper right')
                
                # Füge Colorbar hinzu
                fig3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)
                
                # Wende Kameraeinstellungen an
                ax3d.view_init(elev=st.session_state.elev_3d, azim=st.session_state.azim_3d)
                
                # Versuche Distanz zu setzen (kann in älteren matplotlib Versionen fehlen)
                try:
                    ax3d.dist = st.session_state.dist_3d / 10  # Skaliere für bessere Werte
                except:
                    pass  # Distanz kann nicht gesetzt werden in älteren matplotlib Versionen
                
                # Zeige Plot
                st.pyplot(fig3d)
    
    with col2:
        # Erstelle 2D-Konturplot mit matplotlib und füge Kontrollen hinzu
        if current_func:
            # Erstelle Container für 2D Plot und Kontrollen
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
            
            # Steuerungsbereich mit farbigem Design
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
            
            # 2D Plot mit Matplotlib erzeugen
            with plot2d_container:
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
                            result = current_func(params)
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
    
    # Funktionen für die Optimierung direkt implementieren
    def run_simple_optimization(func, start_point, max_iter=500, learning_rate=0.01, 
                            epsilon=1e-8, momentum=0.9, use_momentum=False, 
                            use_adaptive_lr=True, callback=None):
        """
        Verbesserte Gradient Descent Implementierung für die Optimierung
        
        Parameter:
        - func: Die zu optimierende Funktion
        - start_point: Startpunkt
        - max_iter: Maximale Anzahl an Iterationen
        - learning_rate: Initiale Lernrate
        - epsilon: Numerische Stabilität und Abbruchtoleranz
        - momentum: Momentum-Faktor (falls use_momentum=True)
        - use_momentum: Ob Momentum verwendet werden soll
        - use_adaptive_lr: Ob die Lernrate adaptiv angepasst werden soll
        - callback: Callback-Funktion für Visualisierung
        """
        x = start_point.copy()
        x_history = [x.copy()]
        loss_history = []
        
        # Berechne initialen Funktionswert und Gradienten
        result = func(x)
        value = result.get('value', float('inf'))
        gradient = result.get('gradient', np.zeros_like(x))
        loss_history.append(value)
        
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
        
        for i in range(max_iter):
            # Berechne Gradientennorm für Reporting
            grad_norm = np.linalg.norm(gradient)
            
            # Momentum-basierte Aktualisierung
            if use_momentum:
                velocity = momentum * velocity - current_lr * gradient
                step = velocity
            else:
                step = -current_lr * gradient
            
            # Berechne neue Position
            x_new = x + step
            
            # Evaluiere an der neuen Position
            result_new = func(x_new)
            value_new = result_new.get('value', float('inf'))
            gradient_new = result_new.get('gradient', np.zeros_like(x))
            
            # Adaptive Lernratenanpassung
            if use_adaptive_lr:
                if value_new < value:  # Verbesserung
                    # Speichere den besten Wert
                    if value_new < best_value:
                        best_value = value_new
                        patience_counter = 0
                    
                    # Erhöhe Lernrate vorsichtig, wenn wir uns kontinuierlich verbessern
                    if patience_counter == 0:
                        current_lr = min(current_lr * lr_increase_factor, 0.1)
                    
                    # Akzeptiere den Schritt
                    x = x_new
                    value = value_new
                    gradient = gradient_new
                    
                    # Info-Text aktualisieren
                    info_text = f"Schritt akzeptiert, LR: {current_lr:.6f}"
                else:  # Verschlechterung
                    patience_counter += 1
                    
                    # Reduziere Lernrate, wenn wir uns mehrmals verschlechtern
                    if patience_counter >= patience:
                        current_lr = max(current_lr * lr_reduce_factor, min_lr)
                        patience_counter = 0
                        info_text = f"Lernrate reduziert auf: {current_lr:.6f}"
                    
                    # Verwerfe diesen Schritt und versuche es mit reduzierter Lernrate erneut
                    continue
            else:
                # Ohne adaptive Lernrate: Übernehme immer den neuen Zustand
                x = x_new
                value = value_new
                gradient = gradient_new
            
            # Speichere Verlauf
            x_history.append(x.copy())
            loss_history.append(value)
            
            # Rufe Callback auf, falls vorhanden
            if callback:
                callback(i, x, value, grad_norm, f"Iteration {i+1}/{max_iter}, {info_text}")
            
            # Abbruchkriterium: Wenn Gradient sehr klein wird
            if grad_norm < epsilon:
                return x, x_history, loss_history, f"Konvergenz erreicht (Gradientennorm < {epsilon})"
            
            # Abbruchkriterium: Wenn Lernrate zu klein wird
            if current_lr < min_lr:
                return x, x_history, loss_history, f"Minimale Lernrate erreicht ({min_lr})"
        
        return x, x_history, loss_history, f"Max. Iterationen erreicht ({max_iter})"
    
    def create_visualization_tracker(func, x_range, y_range):
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
        
        return callback, path_history, value_history
    
    # Bereich für Optimierungsergebnisse
    st.markdown("""
    <div style="background: linear-gradient(90deg, #4d8bf0, #6a2c91); padding: 12px; border-radius: 8px;">
        <h3 style="color: white; margin: 0;">Optimierungsergebnisse</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Container für den Live-Tracker
    live_tracking_container = st.container()
    
    # Container für die Info-Box
    info_box_container = st.container()
    
    # Bereich für die Optimierungspfade
    results_container = st.container()
    
    # Führe Optimierung aus, wenn der Button geklickt wurde
    if start_optimization and current_func:
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
                current_func, x_range, y_range
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
                        result = current_func(point)
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
            if selected_algorithm == "GD_Simple_LS":
                # Gradient Descent mit Liniensuche
                max_iter = optimizer_params.get("max_iter", 500)
                learning_rate = optimizer_params.get("initial_t_ls", 0.01)
                
                # Für schwierigere Funktionen wie Rosenbrock kleine Lernrate verwenden
                if st.session_state.ausgewählte_funktion == "Rosenbrock":
                    learning_rate = 0.005
                    use_adaptive_lr = True
                    
            elif selected_algorithm == "GD_Momentum":
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
            - Algorithmus: {algorithm_options[selected_algorithm]}
            - Lernrate: {learning_rate}
            - Max. Iterationen: {max_iter}
            - Momentum: {'Ein' if use_momentum else 'Aus'} ({momentum_value if use_momentum else 'N/A'})
            - Adaptive Lernrate: {'Ein' if use_adaptive_lr else 'Aus'}
            """)
            
            # Direkte Optimierung ausführen
            best_x, best_history, best_loss_history, status = run_simple_optimization(
                current_func, start_point,
                max_iter=max_iter,
                learning_rate=learning_rate,
                epsilon=epsilon,
                momentum=momentum_value,
                use_momentum=use_momentum,
                use_adaptive_lr=use_adaptive_lr,
                callback=visualization_callback
            )
            
            # Speichere Ergebnisse
            algorithm_display_name = f"{algorithm_options[selected_algorithm]}"
            
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
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Zeige Konturplot mit dem Optimierungspfad
                    if best_history:
                        # Erstelle Konturplot mit Matplotlib
                        fig_result = plt.figure(figsize=(8, 6))
                        ax_result = fig_result.add_subplot(111)
                        
                        # Erzeuge Gitter für Konturplot
                        X, Y = np.meshgrid(np.linspace(x_range[0], x_range[1], 100), 
                                         np.linspace(y_range[0], y_range[1], 100))
                        Z = np.zeros_like(X)
                        
                        # Berechne Funktionswerte
                        for i in range(X.shape[0]):
                            for j in range(X.shape[1]):
                                try:
                                    result = current_func(np.array([X[i, j], Y[i, j]]))
                                    Z[i, j] = result.get('value', np.nan)
                                except:
                                    Z[i, j] = np.nan
                        
                        # Zeichne Konturplot
                        cp = ax_result.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
                        
                        # Zeichne Pfadverlauf
                        path_x = [p[0] for p in best_history]
                        path_y = [p[1] for p in best_history]
                        ax_result.plot(path_x, path_y, 'r-o', linewidth=2, markersize=4)
                        ax_result.plot(path_x[0], path_y[0], 'bo', markersize=8, label='Start')
                        ax_result.plot(path_x[-1], path_y[-1], 'g*', markersize=10, label='Ende')
                        
                        # Zeichne bekannte Minima, falls vorhanden
                        if minima is not None:
                            for m in minima:
                                ax_result.plot(m[0], m[1], 'y+', markersize=10, markeredgewidth=2, 
                                              label='Bekanntes Minimum')
                        
                        ax_result.set_xlim(x_range)
                        ax_result.set_ylim(y_range)
                        ax_result.set_title(f"Optimierungspfad: {algorithm_display_name}")
                        ax_result.legend()
                        
                        st.pyplot(fig_result)
                    
                with col2:
                    # Zeige Verlauf des Funktionswertes
                    if best_loss_history:
                        # Erstelle Plot für Funktionswerte
                        fig_loss = plt.figure(figsize=(8, 4))
                        ax_loss = fig_loss.add_subplot(111)
                        
                        # Zeichne Verlaufskurve
                        iterations = range(len(best_loss_history))
                        ax_loss.plot(iterations, best_loss_history, '-o', color='blue', 
                                    linewidth=2, markersize=3)
                        
                        # Logarithmische Darstellung, falls sinnvoll
                        if min(best_loss_history) > 0 and max(best_loss_history) / min(best_loss_history) > 10:
                            ax_loss.set_yscale('log')
                        
                        # Füge Titel und Achsenbeschriftungen hinzu
                        ax_loss.set_title(f"Verlauf des Funktionswertes - {algorithm_display_name}")
                        ax_loss.set_xlabel('Iteration')
                        ax_loss.set_ylabel('Funktionswert')
                        
                        # Annotiere den finalen Wert
                        if len(best_loss_history) > 0:
                            final_value = best_loss_history[-1]
                            ax_loss.annotate(f"Final: {final_value:.4f}", 
                                          xy=(len(best_loss_history)-1, final_value),
                                          xytext=(len(best_loss_history)-5, final_value*1.1),
                                          arrowprops=dict(arrowstyle='->'),
                                          bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
                        
                        # Zeige horizontale Linie beim besten Wert
                        best_value = min(best_loss_history)
                        ax_loss.axhline(y=best_value, color='green', linestyle='--', alpha=0.5)
                        ax_loss.annotate(f"Best: {best_value:.4f}", 
                                      xy=(len(best_loss_history)//2, best_value),
                                      xytext=(len(best_loss_history)//2, best_value*0.9),
                                      arrowprops=dict(arrowstyle='->'),
                                      bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.3))
                        
                        # Verbessere das Erscheinungsbild
                        ax_loss.grid(True, alpha=0.3)
                        fig_loss.tight_layout()
                        
                        st.pyplot(fig_loss)
                
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
                                result = current_func(params)
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
                    surf = ax3d_result.plot_surface(X, Y, Z_plot, cmap='viridis', 
                                            linewidth=0, antialiased=True, alpha=0.7,
                                            rstride=1, cstride=1)
                    
                    # Zeichne Optimierungspfad in 3D
                    if best_history:
                        path_points = np.array(best_history)
                        path_x = path_points[:, 0]
                        path_y = path_points[:, 1]
                        path_z = np.zeros(len(path_points))
                        
                        # Berechne Z-Werte für den Pfad
                        for i, point in enumerate(best_history):
                            try:
                                params = np.array(point)
                                result = current_func(params)
                                path_z[i] = result.get('value', np.nan)
                                
                                # Begrenze extreme Z-Werte
                                if np.isfinite(path_z[i]):
                                    path_z[i] = min(max(path_z[i], z_min), z_max)
                            except:
                                path_z[i] = np.nan
                        
                        # Pfad einzeichnen mit Farbverlauf
                        points = np.array([path_x, path_y, path_z]).T.reshape(-1, 1, 3)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        
                        # Startpunkt besonders hervorheben
                        ax3d_result.scatter([path_x[0]], [path_y[0]], [path_z[0]], 
                                       color='blue', s=100, label='Start')
                        
                        # Endpunkt besonders hervorheben
                        ax3d_result.scatter([path_x[-1]], [path_y[-1]], [path_z[-1]], 
                                       color='red', s=100, marker='*', label='Ende')
                        
                        # Pfad einzeichnen
                        ax3d_result.plot(path_x, path_y, path_z, 'r-o', 
                                    linewidth=2, markersize=4, label='Optimierungspfad')
                    
                    # Minima einzeichnen, falls vorhanden
                    if minima is not None:
                        for i, m in enumerate(minima):
                            try:
                                params = np.array(m)
                                result = current_func(params)
                                z_val = result.get('value', np.nan)
                                if np.isfinite(z_val):
                                    ax3d_result.scatter([m[0]], [m[1]], [z_val], 
                                                color='green', s=120, marker='+', 
                                                linewidths=2, label='Bekanntes Minimum' if i==0 else None)
                            except:
                                pass
                    
                    # Achsenbeschriftungen und Titel
                    ax3d_result.set_xlabel('X')
                    ax3d_result.set_ylabel('Y')
                    ax3d_result.set_zlabel('Funktionswert')
                    ax3d_result.set_title(f"3D-Pfad: {algorithm_display_name}")
                    
                    # Blickwinkel setzen
                    ax3d_result.view_init(elev=elev_3d_result, azim=azim_3d_result)
                    
                    # Kameradistanz setzen
                    ax3d_result.dist = dist_3d_result / 10
                    
                    # Legende anzeigen
                    ax3d_result.legend(loc='upper right')
                    
                    # Colorbar hinzufügen
                    fig3d_result.colorbar(surf, ax=ax3d_result, shrink=0.5, aspect=5)
                    
                    # Plot anzeigen
                    st.pyplot(fig3d_result)
    
    # Zeige gespeicherte Ergebnisse
    elif current_func and st.session_state.optimierungsergebnisse:
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
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Zeige Konturplot mit dem Optimierungspfad direkt implementiert
                        if "history" in result_data and result_data["history"]:
                            # Erstelle direkten Konturplot mit Matplotlib
                            fig_result = plt.figure(figsize=(8, 6))
                            ax_result = fig_result.add_subplot(111)
                            
                            # Erzeuge Gitter für Konturplot
                            X, Y = np.meshgrid(np.linspace(x_range[0], x_range[1], 100), 
                                             np.linspace(y_range[0], y_range[1], 100))
                            Z = np.zeros_like(X)
                            
                            # Berechne Funktionswerte
                            for i in range(X.shape[0]):
                                for j in range(X.shape[1]):
                                    try:
                                        params = np.array([X[i, j], Y[i, j]])
                                        result = current_func(params)
                                        Z[i, j] = result.get('value', np.nan)
                                    except:
                                        Z[i, j] = np.nan
                            
                            # Zeichne Konturplot
                            cp = ax_result.contourf(X, Y, Z, levels=contour_levels, cmap='viridis', alpha=0.7)
                            
                            # Zeichne Pfadverlauf
                            path = result_data["history"]
                            path_x = [p[0] for p in path]
                            path_y = [p[1] for p in path]
                            ax_result.plot(path_x, path_y, 'r-o', linewidth=2, markersize=4)
                            ax_result.plot(path_x[0], path_y[0], 'bo', markersize=8, label='Start')
                            ax_result.plot(path_x[-1], path_y[-1], 'g*', markersize=10, label='Ende')
                            
                            # Zeichne bekannte Minima, falls vorhanden
                            if minima is not None:
                                for m in minima:
                                    ax_result.plot(m[0], m[1], 'y+', markersize=10, markeredgewidth=2, 
                                                  label='Bekanntes Minimum')
                            
                            # Achsenbeschriftungen
                            ax_result.set_xlabel('X')
                            ax_result.set_ylabel('Y')
                            ax_result.set_title(f"Optimierungspfad: {selected_result}")
                            
                            # Setze Grenzen
                            ax_result.set_xlim([x_range[0], x_range[1]])
                            ax_result.set_ylim([y_range[0], y_range[1]])
                            
                            # Legende
                            handles, labels = ax_result.get_legend_handles_labels()
                            by_label = dict(zip(labels, handles))
                            ax_result.legend(by_label.values(), by_label.keys())
                            
                            # Farbskala hinzufügen
                            fig_result.colorbar(cp, ax=ax_result, label='Funktionswert')
                            
                            # Zeichne Grid
                            ax_result.grid(True, linestyle='--', alpha=0.3)
                            
                            st.pyplot(fig_result)
                    
                    with col2:
                        # Zeige Verlauf des Funktionswertes direkt implementiert
                        if "loss_history" in result_data and result_data["loss_history"]:
                            # Erstelle Plot für Funktionswerte
                            fig_loss = plt.figure(figsize=(8, 4))
                            ax_loss = fig_loss.add_subplot(111)
                            
                            # Zeichne Verlaufskurve
                            loss_history = result_data["loss_history"]
                            iterations = range(len(loss_history))
                            ax_loss.plot(iterations, loss_history, '-o', color='blue', 
                                        linewidth=2, markersize=3)
                            
                            # Logarithmische Darstellung, falls sinnvoll
                            if min(loss_history) > 0 and max(loss_history) / min(loss_history) > 10:
                                ax_loss.set_yscale('log')
                            
                            # Füge Titel und Achsenbeschriftungen hinzu
                            ax_loss.set_title(f"Verlauf des Funktionswertes - {selected_result}")
                            ax_loss.set_xlabel('Iteration')
                            ax_loss.set_ylabel('Funktionswert')
                            
                            # Annotiere den finalen Wert
                            if len(loss_history) > 0:
                                final_value = loss_history[-1]
                                ax_loss.annotate(f"Final: {final_value:.4f}", 
                                              xy=(len(loss_history)-1, final_value),
                                              xytext=(len(loss_history)-5, final_value*1.1),
                                              arrowprops=dict(arrowstyle='->'),
                                              bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
                            
                            # Zeige horizontale Linie beim besten Wert
                            best_value = min(loss_history)
                            ax_loss.axhline(y=best_value, color='green', linestyle='--', alpha=0.5)
                            ax_loss.annotate(f"Best: {best_value:.4f}", 
                                          xy=(len(loss_history)//2, best_value),
                                          xytext=(len(loss_history)//2, best_value*0.9),
                                          arrowprops=dict(arrowstyle='->'),
                                          bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.3))
                            
                            # Verbessere das Erscheinungsbild
                            ax_loss.grid(True, alpha=0.3)
                            fig_loss.tight_layout()
                            
                            st.pyplot(fig_loss)
                    
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
                                    result = current_func(params)
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
                                    res = current_func(params)
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
                                    res = current_func(params)
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
                    current_func_info = pf.MATH_FUNCTIONS_LIB[selected_function_for_comparison]
                    current_func = current_func_info["func"]
                    x_range = current_func_info["default_range"][0]
                    y_range = current_func_info["default_range"][1]
                    contour_levels = current_func_info.get("contour_levels", 40)
                    
                    # Hole Minima, falls vorhanden
                    func_result = current_func(np.array([0, 0]))
                    minima = func_result.get("minima", None)
                    
                elif selected_function_for_comparison in st.session_state.custom_funcs:
                    current_func = st.session_state.custom_funcs[selected_function_for_comparison]
                    x_range = (-5, 5)
                    y_range = (-5, 5)
                    contour_levels = 30
                    minima = None
                else:
                    st.error("Die ausgewählte Funktion wurde nicht gefunden.")
                    current_func = None
                
                if current_func:
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
                                result = current_func(params)
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
                                result = current_func(params)
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
                                    result = current_func(p)
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
                                z = current_func(np.array(m))['value']
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
