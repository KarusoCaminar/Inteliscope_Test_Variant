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
# Dieser Code kommt in app.py in den Block "with tabs[0]:"
# direkt NACH der Zeile "with tabs[0]:" und VOR dem Block "if start_optimization and current_func:"

    # Hole die aktuelle Funktion (dieser Teil ist wichtig für die initiale Anzeige)
    # und stelle sicher, dass current_func initialisiert wird.
    current_func = None
    x_range = (-5, 5) # Default x_range
    y_range = (-5, 5) # Default y_range
    contour_levels = 30 # Default contour_levels
    minima = None # Default minima
    current_func_dim_initial = 2 # Default dimension

    if st.session_state.ausgewählte_funktion in pf.MATH_FUNCTIONS_LIB:
        current_func_info = pf.MATH_FUNCTIONS_LIB[st.session_state.ausgewählte_funktion]
        current_func = current_func_info["func"]
        x_range = current_func_info["default_range"][0]
        y_range = current_func_info["default_range"][1]
        contour_levels = current_func_info.get("contour_levels", 40)
        current_func_dim_initial = current_func_info.get("dimensions", 2)
        
        # Testaufruf für Tooltip und Minima nur wenn Funktion existiert
        if current_func:
            try:
                # Erzeuge einen Dummy-Input basierend auf der Dimension
                dummy_input_initial = np.array([0.0] * current_func_dim_initial)
                func_result_initial = current_func(dummy_input_initial) 
                if "tooltip" in func_result_initial:
                    with st.expander("ℹ️ Über diese Funktion", expanded=False):
                        st.markdown(func_result_initial["tooltip"])
                minima = func_result_initial.get("minima", None)
            except Exception as e_init_func_call:
                st.warning(f"Konnte initiale Funktionsdetails nicht laden: {e_init_func_call}")
        
    elif st.session_state.ausgewählte_funktion in st.session_state.custom_funcs:
        current_func = st.session_state.custom_funcs[st.session_state.ausgewählte_funktion]
        current_func_dim_initial = 2 # Custom functions are currently assumed to be 2D
        if current_func:
            try:
                test_eval_custom = current_func(np.array([0.0, 0.0])) # Testaufruf für 2D
                x_range = test_eval_custom.get('x_range', (-5,5))
                y_range = test_eval_custom.get('y_range', (-5,5))
                if "tooltip" in test_eval_custom:
                     with st.expander("ℹ️ Über diese Funktion", expanded=False):
                        st.markdown(test_eval_custom["tooltip"])
            except Exception as e_init_custom_func_call: # Fallback
                st.warning(f"Konnte initiale Details der benutzerdefinierten Funktion nicht laden: {e_init_custom_func_call}")
                x_range = (-5, 5)
                y_range = (-5, 5)
        contour_levels = 30
        minima = None 
    else:
        st.error("Die ausgewählte Funktion wurde nicht gefunden oder konnte nicht initialisiert werden.")
        # current_func bleibt None, die Default-Ranges werden verwendet

    # Layout für die initiale Visualisierung (bleibt auch während der Optimierung sichtbar)
    col1_display, col2_display = st.columns([1, 1]) 
    
    with col1_display:
        if current_func and current_func_dim_initial == 2: # 3D Plot nur für 2D Funktionen
            plot3d_container_initial = st.container() 
            controls3d_container_initial = st.container() 
            
            if 'elev_3d_initial' not in st.session_state: st.session_state.elev_3d_initial = 30
            if 'azim_3d_initial' not in st.session_state: st.session_state.azim_3d_initial = -60 
            if 'dist_3d_initial' not in st.session_state: st.session_state.dist_3d_initial = 10
            if 'res_3d_initial_plot' not in st.session_state: st.session_state.res_3d_initial_plot = 40 # Geringere Auflösung für initialen Plot
                
            with controls3d_container_initial:
                st.markdown("""<div style="background-color: #4d8bf0; padding: 8px; border-radius: 8px; margin-bottom: 10px;"><h4 style="color: white; margin: 0;">3D Ansicht Steuerung</h4></div>""", unsafe_allow_html=True)
                btn_cols_initial = st.columns(5)
                if btn_cols_initial[0].button("Von oben", key="top_view_initial", type="primary", use_container_width=True): st.session_state.elev_3d_initial = 90; st.session_state.azim_3d_initial = 0
                if btn_cols_initial[1].button("Von vorne", key="front_view_initial", type="primary", use_container_width=True): st.session_state.elev_3d_initial = 0; st.session_state.azim_3d_initial = 0
                if btn_cols_initial[2].button("Von rechts", key="right_view_initial", type="primary", use_container_width=True): st.session_state.elev_3d_initial = 0; st.session_state.azim_3d_initial = 90
                if btn_cols_initial[3].button("Isometrisch", key="iso_view_initial", type="primary", use_container_width=True): st.session_state.elev_3d_initial = 30; st.session_state.azim_3d_initial = 45
                if btn_cols_initial[4].button("Von links", key="left_view_initial", type="primary", use_container_width=True): st.session_state.elev_3d_initial = 0; st.session_state.azim_3d_initial = -90

                cols_slider_initial = st.columns(4) # Vierter Slider für Auflösung
                with cols_slider_initial[0]: st.session_state.elev_3d_initial = st.slider("Elevation", 0, 90, st.session_state.elev_3d_initial, key="elev_slider_initial")
                with cols_slider_initial[1]: st.session_state.azim_3d_initial = st.slider("Azimuth", -180, 180, st.session_state.azim_3d_initial, key="azim_slider_initial")
                with cols_slider_initial[2]: st.session_state.dist_3d_initial = st.slider("Distanz", 5, 20, st.session_state.dist_3d_initial, key="dist_slider_initial")
                with cols_slider_initial[3]: st.session_state.res_3d_initial_plot = st.slider("Auflösung (3D Initial)", 20, 60, st.session_state.res_3d_initial_plot, key="res_slider_initial_3d")
            
            with plot3d_container_initial:
                fig3d_initial = plt.figure(figsize=(8, 6))
                ax3d_initial = fig3d_initial.add_subplot(111, projection='3d')
                
                x_surf = np.linspace(x_range[0], x_range[1], st.session_state.res_3d_initial_plot) 
                y_surf = np.linspace(y_range[0], y_range[1], st.session_state.res_3d_initial_plot)
                X_surf, Y_surf = np.meshgrid(x_surf, y_surf)
                Z_surf = np.zeros_like(X_surf)
                
                for i_s in range(X_surf.shape[0]):
                    for j_s in range(X_surf.shape[1]):
                        try:
                            Z_surf[i_s, j_s] = current_func(np.array([X_surf[i_s, j_s], Y_surf[i_s, j_s]]))['value']
                        except: Z_surf[i_s, j_s] = np.nan
                
                Z_finite_surf = Z_surf[np.isfinite(Z_surf)]
                if len(Z_finite_surf) > 0:
                    z_min_s, z_max_s = np.percentile(Z_finite_surf, [1, 99]) 
                    Z_plot_surf = np.clip(Z_surf, z_min_s, z_max_s)
                else: Z_plot_surf = Z_surf
                
                surf_initial = ax3d_initial.plot_surface(X_surf, Y_surf, Z_plot_surf, cmap='viridis', edgecolor='none', alpha=0.8, rstride=1, cstride=1)
                ax3d_initial.set_xlabel('X'); ax3d_initial.set_ylabel('Y'); ax3d_initial.set_zlabel('Funktionswert')
                ax3d_initial.set_title(f"3D-Oberfläche: {st.session_state.ausgewählte_funktion}")
                if minima:
                    for m_init in minima:
                        if len(m_init) == 2: 
                            try:
                                z_m_init = current_func(np.array(m_init))['value']
                                if len(Z_finite_surf) > 0: z_m_init = np.clip(z_m_init, z_min_s, z_max_s)
                                ax3d_initial.scatter([m_init[0]], [m_init[1]], [z_m_init], color='red', marker='X', s=100, linewidths=1.5, label='Bek. Minimum' if 'Bek. Minimum' not in [l.get_label() for l in ax3d_initial.get_legend_handles_labels()] else "")
                            except: pass
                ax3d_initial.view_init(elev=st.session_state.elev_3d_initial, azim=st.session_state.azim_3d_initial)
                try: ax3d_initial.dist = st.session_state.dist_3d_initial
                except: pass
                if hasattr(surf_initial, 'colorbar') and surf_initial.colorbar: surf_initial.colorbar.remove() 
                fig3d_initial.colorbar(surf_initial, ax=ax3d_initial, shrink=0.6, aspect=10, pad=0.1)
                handles_i3d, labels_i3d = ax3d_initial.get_legend_handles_labels()
                if handles_i3d: ax3d_initial.legend(dict(zip(labels_i3d, handles_i3d)).values(), dict(zip(labels_i3d, handles_i3d)).keys(), loc='upper right')
                st.pyplot(fig3d_initial)
                plt.close(fig3d_initial) 
        elif not current_func:
            st.info("Wähle eine Funktion, um die 3D-Ansicht anzuzeigen.")
        elif current_func_dim_initial != 2:
            st.info("Die initiale 3D-Oberflächenansicht ist nur für 2D-Funktionen verfügbar.")


    with col2_display:
        if current_func and current_func_dim_initial == 2: # 2D Plot nur für 2D Funktionen
            plot2d_container_initial = st.container() 
            controls2d_container_initial = st.container() 

            if 'contour_levels_initial' not in st.session_state: st.session_state.contour_levels_initial = contour_levels
            if 'zoom_factor_initial' not in st.session_state: st.session_state.zoom_factor_initial = 1.0
            if 'show_grid_2d_initial' not in st.session_state: st.session_state.show_grid_2d_initial = True 
            if 'res_2d_initial_plot' not in st.session_state: st.session_state.res_2d_initial_plot = 60 # Auflösung für initialen 2D Plot
            
            with controls2d_container_initial:
                st.markdown("""<div style="background-color: #6a2c91; padding: 8px; border-radius: 8px; margin-bottom: 10px;"><h4 style="color: white; margin: 0;">2D Ansicht Steuerung</h4></div>""", unsafe_allow_html=True)
                cols_slider_2d_initial = st.columns(4) # Vierter Slider für Auflösung
                with cols_slider_2d_initial[0]: st.session_state.contour_levels_initial = st.slider("Konturlinien", 10, 100, st.session_state.contour_levels_initial, step=5, key="contour_slider_initial")
                with cols_slider_2d_initial[1]: st.session_state.zoom_factor_initial = st.slider("Zoom", 0.5, 5.0, st.session_state.zoom_factor_initial, step=0.1, key="zoom_slider_initial")
                with cols_slider_2d_initial[2]: st.session_state.show_grid_2d_initial = st.checkbox("Gitter anzeigen", st.session_state.show_grid_2d_initial, key="grid_checkbox_initial")
                with cols_slider_2d_initial[3]: st.session_state.res_2d_initial_plot = st.slider("Auflösung (2D Initial)", 30, 100, st.session_state.res_2d_initial_plot, key="res_slider_initial_2d")


            with plot2d_container_initial:
                fig2d_initial = plt.figure(figsize=(8, 6))
                ax2d_initial = fig2d_initial.add_subplot(111)
                
                center_x_initial = np.mean(x_range)
                center_y_initial = np.mean(y_range)
                x_half_range_initial = (x_range[1] - x_range[0]) / (2 * st.session_state.zoom_factor_initial)
                y_half_range_initial = (y_range[1] - y_range[0]) / (2 * st.session_state.zoom_factor_initial)
                x_zoom_range_initial = (center_x_initial - x_half_range_initial, center_x_initial + x_half_range_initial)
                y_zoom_range_initial = (center_y_initial - y_half_range_initial, center_y_initial + y_half_range_initial)
                
                grid_size_initial = int(st.session_state.res_2d_initial_plot * np.sqrt(st.session_state.zoom_factor_initial)) 
                x_contour = np.linspace(x_zoom_range_initial[0], x_zoom_range_initial[1], grid_size_initial)
                y_contour = np.linspace(y_zoom_range_initial[0], y_zoom_range_initial[1], grid_size_initial)
                X_contour, Y_contour = np.meshgrid(x_contour, y_contour)
                Z_contour = np.zeros_like(X_contour)

                for i_c in range(X_contour.shape[0]):
                    for j_c in range(X_contour.shape[1]):
                        try: Z_contour[i_c, j_c] = current_func(np.array([X_contour[i_c, j_c], Y_contour[i_c, j_c]]))['value']
                        except: Z_contour[i_c, j_c] = np.nan
                
                Z_finite_contour = Z_contour[np.isfinite(Z_contour)]
                cp_initial = None # Initialisieren
                if len(Z_finite_contour) > 0:
                    z_min_c, z_max_c = np.percentile(Z_finite_contour, [1,99])
                    Z_plot_contour = np.clip(Z_contour, z_min_c, z_max_c)
                    cp_initial = ax2d_initial.contourf(X_contour, Y_contour, Z_plot_contour, levels=st.session_state.contour_levels_initial, cmap='viridis', alpha=0.8)
                    contour_lines_initial = ax2d_initial.contour(X_contour, Y_contour, Z_plot_contour, levels=min(15, st.session_state.contour_levels_initial//3), colors='black', alpha=0.4, linewidths=0.5)
                    try: ax2d_initial.clabel(contour_lines_initial, inline=True, fontsize=8, fmt='%.1f')
                    except: pass 
                    if hasattr(cp_initial, 'colorbar') and cp_initial.colorbar: cp_initial.colorbar.remove()
                    fig2d_initial.colorbar(cp_initial, ax=ax2d_initial).set_label('Funktionswert')
                else: # Fallback if no finite Z values
                    ax2d_initial.text(0.5, 0.5, "Keine darstellbaren Funktionswerte im Bereich.", horizontalalignment='center', verticalalignment='center', transform=ax2d_initial.transAxes)


                ax2d_initial.set_xlabel('X'); ax2d_initial.set_ylabel('Y')
                ax2d_initial.set_title(f"Konturplot: {st.session_state.ausgewählte_funktion}")
                if minima:
                    for m_init_2d in minima:
                        if len(m_init_2d) == 2: ax2d_initial.plot(m_init_2d[0], m_init_2d[1], 'X', color='red', markersize=8, markeredgecolor='black', label='Bek. Minimum' if 'Bek. Minimum' not in [l.get_label() for l in ax2d_initial.get_legend().get_texts()] else "")
                ax2d_initial.set_xlim(x_zoom_range_initial); ax2d_initial.set_ylim(y_zoom_range_initial)
                if st.session_state.show_grid_2d_initial: ax2d_initial.grid(True, linestyle='--', alpha=0.6)
                handles_i2d, labels_i2d = ax2d_initial.get_legend_handles_labels()
                if handles_i2d: ax2d_initial.legend(dict(zip(labels_i2d, handles_i2d)).values(), dict(zip(labels_i2d, handles_i2d)).keys(), loc='best')
                st.pyplot(fig2d_initial)
                plt.close(fig2d_initial) 
        elif not current_func:
            st.info("Wähle eine Funktion, um die 2D-Ansicht anzuzeigen.")
        elif current_func_dim_initial != 2:
            st.info("Die initiale 2D-Konturansicht ist nur für 2D-Funktionen verfügbar.")


    # Bereich für Optimierungsergebnisse (Container-Definitionen hier, damit sie immer existieren)
    st.markdown("""
    <div style="background: linear-gradient(90deg, #4d8bf0, #6a2c91); padding: 12px; border-radius: 8px; margin-top: 20px;">
        <h3 style="color: white; margin: 0;">Optimierungsergebnisse & Live-Tracking</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Container für den Live-Tracker und Info (werden im Optimierungsblock gefüllt)
    live_tracking_container = st.container()
    info_box_container = st.container()
    
    # Container für die finalen Ergebnis-Plots und Details
    results_container = st.container()

    # ----- HIER BEGINNT DER GROSSE CODE-BLOCK, DEN ICH DIR ZULETZT GEGEBEN HABE -----
    # (also der Block, der mit "if start_optimization and current_func:" anfängt)
    # Dieser Block sollte jetzt hier folgen.

    
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
                
def create_visualization_tracker(func, x_range_vis, y_range_vis, live_plot_placeholder, info_placeholder): # Parameter hinzugefügt
    """
    Erstellt einen Tracker für den Optimierungspfad
    """
    path_history_cb = [] # Umbenannt, um Konflikt mit path_history_for_plot zu vermeiden
    value_history_cb = [] # Umbenannt
    
    # Callback-Funktion, die den Pfad aufzeichnet
    def callback(iteration, x_cb, value_cb, grad_norm_cb, message_cb): # Parameter umbenannt
        path_history_cb.append(x_cb.copy())
        value_history_cb.append(value_cb)
        
        # Status-Nachricht im Info-Bereich anzeigen
        info_text_cb = f"""
        **Iteration:** {iteration+1}
        **Aktuelle Position:** {np.round(x_cb, 4).tolist()}
        **Funktionswert:** {value_cb:.6f}
        **Gradientennorm:** {grad_norm_cb:.6f}
        **Status:** {message_cb}
        """
        # Verwende die übergebenen Platzhalter
        if info_placeholder:
            info_placeholder.markdown(info_text_cb)
        
        # Nur alle N Iterationen visualisieren, um Performance zu verbessern
        # oder die ersten paar und die letzte.
        # Die Logik für die Plot-Frequenz kann hier angepasst werden.
        # Fürs Erste belassen wir es bei jeder 5. Iteration oder den ersten 5.
        if live_plot_placeholder and (iteration % 5 == 0 or iteration < 5 or "erreicht" in message_cb.lower() or "fehler" in message_cb.lower()):
            try:
                # 2D Konturplot mit aktuellem Pfad
                fig_live_cb = plt.figure(figsize=(8, 4)) # Umbenannt
                ax_live_cb = fig_live_cb.add_subplot(111) # Umbenannt
                
                # Dimension der aktuellen Funktion bestimmen
                func_name_cb = st.session_state.ausgewählte_funktion
                func_details_cb = pf.MATH_FUNCTIONS_LIB.get(func_name_cb)
                dim_cb = 2
                if func_details_cb:
                    dim_cb = func_details_cb.get("dimensions", 2)
                
                if dim_cb == 2: # Nur plotten, wenn 2D
                    # Gitter für Konturplot
                    X_cb, Y_cb = np.meshgrid(np.linspace(x_range_vis[0], x_range_vis[1], 50), 
                                         np.linspace(y_range_vis[0], y_range_vis[1], 50))
                    Z_cb = np.zeros_like(X_cb)
                    
                    for i_cb in range(X_cb.shape[0]):
                        for j_cb in range(X_cb.shape[1]):
                            try:
                                result_cb = func(np.array([X_cb[i_cb, j_cb], Y_cb[i_cb, j_cb]]))
                                Z_cb[i_cb, j_cb] = result_cb.get('value', np.nan)
                            except:
                                Z_cb[i_cb, j_cb] = np.nan
                    
                    ax_live_cb.contourf(X_cb, Y_cb, Z_cb, levels=30, cmap='viridis', alpha=0.7)
                    
                    if len(path_history_cb) > 0:
                        path_points_cb = np.array(path_history_cb)
                        if path_points_cb.ndim == 2 and path_points_cb.shape[1] >= 2:
                            ax_live_cb.plot(path_points_cb[:, 0], path_points_cb[:, 1], 'r-o', linewidth=2, markersize=4)
                            ax_live_cb.plot(path_points_cb[0, 0], path_points_cb[0, 1], 'bo', markersize=8, label='Start')
                            ax_live_cb.plot(path_points_cb[-1, 0], path_points_cb[-1, 1], 'g*', markersize=10, label='Aktuell')
                    
                    ax_live_cb.set_xlim(x_range_vis)
                    ax_live_cb.set_ylim(y_range_vis)
                    ax_live_cb.set_title(f"Optimierungspfad (Iter. {iteration+1}) für {func_name_cb}")
                    ax_live_cb.legend(loc="upper right")
                    
                    live_plot_placeholder.pyplot(fig_live_cb)
                    plt.close(fig_live_cb) # Wichtig: Figur schließen, um Speicherprobleme zu vermeiden
                elif iteration == 0 : # Nur einmal für nD Funktionen eine Info ausgeben
                    live_plot_placeholder.info("Live Konturplot nur für 2D Funktionen verfügbar.")

            except Exception as e_cb_plot:
                if live_plot_placeholder: # Nur wenn Platzhalter existiert
                    live_plot_placeholder.warning(f"Fehler im Live-Plot: {e_cb_plot}")
    
    return callback, path_history_cb, value_history_cb
    
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
            # Räume die Live-Tracking-Container auf, falls sie existieren
            if 'live_plot_placeholder' in locals() or 'live_plot_placeholder' in globals():
                live_plot_placeholder.empty()
            if 'info_placeholder' in locals() or 'info_placeholder' in globals():
                info_placeholder.empty()
            
            # Erstelle Container für Live-Feedback, falls sie nicht schon existieren
            # Diese werden jetzt innerhalb des "if start_optimization" Blocks erstellt,
            # um sicherzustellen, dass sie nur bei Bedarf da sind.
            with live_tracking_container:
                st.markdown("""
                <div style="background-color: #f0f8ff; padding: 8px; border-radius: 8px; border-left: 4px solid #4d8bf0;">
                    <h4 style="color: #4d8bf0; margin: 0;">Live-Verfolgung der Optimierung</h4>
                </div>
                """, unsafe_allow_html=True)
                live_plot_placeholder = st.empty() # Platzhalter für den Live-Plot
            
            with info_box_container:
                st.markdown("""
                <div style="background-color: #f8f0ff; padding: 8px; border-radius: 8px; border-left: 4px solid #6a2c91;">
                    <h4 style="color: #6a2c91; margin: 0;">Optimierungs-Status</h4>
                </div>
                """, unsafe_allow_html=True)
                info_placeholder = st.empty() # Platzhalter für Live-Infos

            # Erstelle Callback-Funktion
            visualization_callback, path_history_for_plot, value_history_for_plot = create_visualization_tracker(
                current_func, x_range, y_range, live_plot_placeholder, info_placeholder
            )
            
            # Dimension der aktuellen Funktion bestimmen
            func_details = pf.MATH_FUNCTIONS_LIB.get(st.session_state.ausgewählte_funktion)
            current_func_dim = 2 # Default, falls nicht in LIB oder Custom
            if st.session_state.ausgewählte_funktion in pf.MATH_FUNCTIONS_LIB:
                current_func_dim = pf.MATH_FUNCTIONS_LIB[st.session_state.ausgewählte_funktion].get("dimensions", 2)
            elif st.session_state.ausgewählte_funktion in st.session_state.custom_funcs:
                # Für Custom Functions wird aktuell 2D angenommen, da die Eingabe darauf ausgelegt ist.
                # Man könnte dies erweitern, wenn Custom Functions nD unterstützen sollen.
                current_func_dim = 2


            # Startpunktauswahl
            if current_func_dim == 2:
                # Grid-Suche für einen "schwierigen" Startpunkt bei 2D-Funktionen
                start_x_coords = np.linspace(x_range[0], x_range[1], 5) 
                start_y_coords = np.linspace(y_range[0], y_range[1], 5)
                highest_value = float('-inf')
                start_point = np.array([np.mean(x_range), np.mean(y_range)]) # Fallback
                for sx in start_x_coords:
                    for sy in start_y_coords:
                        try:
                            point = np.array([sx, sy])
                            result = current_func(point)
                            if 'value' in result and np.isfinite(result['value']) and result['value'] > highest_value:
                                highest_value = result['value']
                                start_point = point.copy()
                        except Exception:
                            continue
            else: # Für nD > 2 oder unbekannte Dimension
                # Einfacher Startpunkt: Mitte des definierten Bereichs oder Zufallspunkt
                # Hier verwenden wir einen zufälligen Punkt innerhalb der Standard-Ranges
                # oder spezifische Ranges, falls für nD definiert.
                # Da x_range und y_range primär für 2D-Visualisierung sind,
                # generieren wir für höhere Dimensionen zufällige Werte um 0.
                low_bounds = [x_range[0]] + [y_range[0]] + [-2.0]*(current_func_dim-2)
                high_bounds = [x_range[1]] + [y_range[1]] + [2.0]*(current_func_dim-2)
                start_point = np.random.uniform(low=low_bounds[:current_func_dim], high=high_bounds[:current_func_dim], size=current_func_dim)

            st.write(f"Gewählte Funktion: {st.session_state.ausgewählte_funktion} (Dimension: {current_func_dim})")
            st.write(f"Starte Optimierung von Punkt: {np.round(start_point, 4).tolist()}")
            
            best_x = None
            best_history = []
            best_loss_history = []
            status = "Optimierung nicht ausgeführt"

            # Führe den ausgewählten Algorithmus direkt aus optimization_algorithms_v3.py aus
            try:
                # Die `optimizer_params` werden direkt aus der Sidebar übernommen
                algo_params_to_pass = optimizer_params.copy()

                if selected_algorithm == "GD_Simple_LS":
                    best_x, best_history, best_loss_history, status = oa.gradientDescent(
                        obj_fun=current_func,
                        initial_x=start_point,
                        callback=visualization_callback,
                        **algo_params_to_pass 
                    )
                elif selected_algorithm == "GD_Momentum":
                    best_x, best_history, best_loss_history, status = oa.gradientDescentWithMomentum(
                        obj_fun=current_func,
                        initial_x=start_point,
                        callback=visualization_callback,
                        **algo_params_to_pass
                    )
                elif selected_algorithm == "Adam":
                    # Adam benötigt `grad_norm_tol` nicht explizit in seiner Signatur,
                    # aber es ist gut, es in den `optimizer_params` zu haben, falls andere Adam-Varianten es nutzen.
                    # oa.adam_optimizer hat `grad_norm_tol` als Parameter.
                    if 'grad_norm_tol' not in algo_params_to_pass: # Sicherstellen, dass es einen Default gibt
                         algo_params_to_pass['grad_norm_tol'] = 1e-6 
                    best_x, best_history, best_loss_history, status = oa.adam_optimizer(
                        obj_fun=current_func,
                        initial_x=start_point,
                        callback=visualization_callback,
                        **algo_params_to_pass
                    )
                else:
                    st.error(f"Unbekannter Algorithmus: {selected_algorithm}")
                    status = f"Fehler: Unbekannter Algorithmus {selected_algorithm}"

            except Exception as e:
                st.error(f"Fehler während der Optimierung mit {selected_algorithm}: {e}")
                status = f"Laufzeitfehler: {e}"
                best_x = start_point # Fallback
                best_history = [start_point.copy()]
                try:
                    best_loss_history = [current_func(start_point)['value']]
                except:
                    best_loss_history = [np.nan]
            
            # Speichere Ergebnisse
            algorithm_display_name = f"{algorithm_options[selected_algorithm]}" # Name aus der UI
            
            # Verwende einen eindeutigeren Schlüssel, um Überschreibungen zu vermeiden,
            # falls derselbe Algorithmus mehrfach für dieselbe Funktion mit anderen Parametern läuft.
            # Hier: Algo-Name + kurzer Parameter-Hash oder Zeitstempel
            # Fürs Erste belassen wir es beim Algo-Namen für Einfachheit, aber das ist ein Punkt für Verfeinerung.
            
            st.session_state.optimierungsergebnisse[algorithm_display_name] = {
                "function": st.session_state.ausgewählte_funktion,
                "algorithm_name_code": selected_algorithm, # Interner Name
                "algorithm_name_ui": algorithm_display_name, # UI Name
                "best_x": best_x.tolist() if best_x is not None else None, # Als Liste speichern für JSON-Kompatibilität
                "history": [p.tolist() for p in best_history] if best_history else [], # Liste von Listen
                "loss_history": best_loss_history if best_loss_history else [],
                "status": status,
                "timestamp": time.time(),
                "params": optimizer_params.copy(), # Verwendete UI-Parameter
                "strategy": "single", # Vorerst nur "single"
                "multi_start_results": None # Für spätere Erweiterungen
            }
            
            # Leere die Platzhalter nach der Optimierung
            live_plot_placeholder.empty()
            info_placeholder.empty()

            # Zeige Zusammenfassung der Ergebnisse
            # (Dieser Teil ist umfangreich und beinhaltet die Plot-Logik)
            with results_container:
                st.markdown("""
                <div style="background-color: #f0fff8; padding: 12px; border-radius: 8px; border-left: 4px solid #15b371;">
                    <h3 style="color: #15b371; margin: 0;">Zusammenfassung der Optimierung</h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1_res, col2_res = st.columns([2, 1])
                
                with col1_res:
                    if best_history and len(best_history) > 0:
                        if current_func_dim == 2: # Nur für 2D plotten
                            fig_result_2d = plt.figure(figsize=(8, 6))
                            ax_result_2d = fig_result_2d.add_subplot(111)
                            
                            X_plot_grid, Y_plot_grid = np.meshgrid(np.linspace(x_range[0], x_range[1], 100), 
                                                                 np.linspace(y_range[0], y_range[1], 100))
                            Z_plot_grid = np.zeros_like(X_plot_grid)
                            
                            for i_plot in range(X_plot_grid.shape[0]):
                                for j_plot in range(X_plot_grid.shape[1]):
                                    try:
                                        res_plot = current_func(np.array([X_plot_grid[i_plot, j_plot], Y_plot_grid[i_plot, j_plot]]))
                                        Z_plot_grid[i_plot, j_plot] = res_plot.get('value', np.nan)
                                    except: Z_plot_grid[i_plot, j_plot] = np.nan
                            
                            # Clipping für bessere Visualisierung der Konturen
                            Z_finite_plot = Z_plot_grid[np.isfinite(Z_plot_grid)]
                            if len(Z_finite_plot) > 0:
                                z_min_plot, z_max_plot = np.percentile(Z_finite_plot, [1, 99])
                                Z_plot_grid_clipped = np.clip(Z_plot_grid, z_min_plot, z_max_plot)
                                cp = ax_result_2d.contourf(X_plot_grid, Y_plot_grid, Z_plot_grid_clipped, levels=contour_levels, cmap='viridis', alpha=0.7)
                            else: # Fallback, falls keine finiten Werte
                                cp = ax_result_2d.contourf(X_plot_grid, Y_plot_grid, Z_plot_grid, levels=contour_levels, cmap='viridis', alpha=0.7)

                            path_points_np = np.array(best_history)
                            if path_points_np.ndim == 2 and path_points_np.shape[1] >= 2:
                                ax_result_2d.plot(path_points_np[:, 0], path_points_np[:, 1], 'r-o', linewidth=2, markersize=4, label=f"{algorithm_display_name} Pfad")
                                ax_result_2d.plot(path_points_np[0, 0], path_points_np[0, 1], 'bo', markersize=8, label='Start')
                                ax_result_2d.plot(path_points_np[-1, 0], path_points_np[-1, 1], 'g*', markersize=10, label='Ende')
                            
                            if minima:
                                for m_idx, m_val in enumerate(minima):
                                    if len(m_val) == 2:
                                        ax_result_2d.plot(m_val[0], m_val[1], 'yX', markersize=10, markeredgewidth=1.5, markeredgecolor='black',
                                                          label='Bek. Minimum' if m_idx == 0 else None)
                            
                            ax_result_2d.set_xlim(x_range); ax_result_2d.set_ylim(y_range)
                            ax_result_2d.set_title(f"Optimierungspfad: {algorithm_display_name} auf {st.session_state.ausgewählte_funktion}")
                            ax_result_2d.legend(loc="best"); ax_result_2d.grid(True, alpha=0.3)
                            st.pyplot(fig_result_2d)
                            plt.close(fig_result_2d)
                        else:
                            st.info("2D Konturplot ist nur für 2D-Funktionen verfügbar.")
                    else:
                        st.info("Keine Pfadhistorie zum Anzeigen für den 2D-Plot.")
                    
                with col2_res:
                    if best_loss_history and len(best_loss_history) > 0:
                        fig_loss = plt.figure(figsize=(8, 4))
                        ax_loss = fig_loss.add_subplot(111)
                        ax_loss.plot(range(len(best_loss_history)), best_loss_history, '-o', color='blue', linewidth=2, markersize=3)
                        
                        positive_losses = [l for l in best_loss_history if l is not None and np.isfinite(l) and l > 0]
                        if positive_losses and len(positive_losses) > 1 and max(positive_losses) / min(positive_losses) > 100: # Stärkerer Trigger für Log
                            ax_loss.set_yscale('log')
                        
                        ax_loss.set_title(f"Verlauf Funktionswert - {algorithm_display_name}"); ax_loss.set_xlabel('Iteration'); ax_loss.set_ylabel('Funktionswert')
                        ax_loss.grid(True, alpha=0.3); fig_loss.tight_layout()
                        st.pyplot(fig_loss)
                        plt.close(fig_loss)
                    else:
                        st.info("Keine Loss-Historie zum Anzeigen.")
                
                st.markdown("""
                <div style="background-color: #f0f8ff; padding: 8px; border-radius: 8px; margin-top: 15px;">
                    <h4 style="color: #4d8bf0; margin: 0;">Optimierungs-Details</h4>
                </div>
                """, unsafe_allow_html=True)
                
                col1_det, col2_det, col3_det, col4_det = st.columns(4)
                start_p_disp = np.round(best_history[0], 3).tolist() if best_history and len(best_history) > 0 else "N/A"
                end_p_disp = np.round(best_x, 3).tolist() if best_x is not None else "N/A"
                final_loss_disp = f"{best_loss_history[-1]:.6f}" if best_loss_history and len(best_loss_history) > 0 and np.isfinite(best_loss_history[-1]) else "N/A"
                iter_disp = f"{len(best_loss_history)-1}" if best_loss_history and len(best_loss_history) > 0 else "0"
                
                with col1_det: st.metric("Startpunkt", f"{start_p_disp}")
                with col2_det: st.metric("Endpunkt", f"{end_p_disp}")
                with col3_det: st.metric("Funktionswert", final_loss_disp)
                with col4_det: st.metric("Iterationen", iter_disp)
                st.markdown(f"**Status:** {status}")

                if current_func_dim == 2 and best_history and len(best_history) > 0 : # 3D Plot nur für 2D Funktionen mit Pfad
                    st.markdown("""
                    <div style="background-color: #6a2c91; padding: 8px; border-radius: 8px; margin-bottom: 10px; margin-top: 15px;">
                        <h3 style="color: white; margin: 0;">3D-Visualisierung des Optimierungspfades</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Verwende eindeutige Keys für Slider im Ergebnisteil
                    elev_3d_res = st.session_state.get("elev_3d_finalplot", 30)
                    azim_3d_res = st.session_state.get("azim_3d_finalplot", -60) # Typischerer Blickwinkel
                    dist_3d_res = st.session_state.get("dist_3d_finalplot", 10)
                    res_3d_res = st.session_state.get("res_3d_finalplot", 50)

                    res_cols_3d_ctrl = st.columns(4)
                    with res_cols_3d_ctrl[0]: elev_3d_res = st.slider("Elevation (3D Result)", 0, 90, elev_3d_res, key="elev_3d_finalplot")
                    with res_cols_3d_ctrl[1]: azim_3d_res = st.slider("Azimuth (3D Result)", -180, 180, azim_3d_res, key="azim_3d_finalplot") # Angepasster Bereich
                    with res_cols_3d_ctrl[2]: dist_3d_res = st.slider("Distanz (3D Result)", 5, 20, dist_3d_res, key="dist_3d_finalplot")
                    with res_cols_3d_ctrl[3]: res_3d_res = st.slider("Auflösung (3D Result)", 30, 80, res_3d_res, key="res_3d_finalplot") # Max 80 für Performance

                    fig3d_final = plt.figure(figsize=(10, 8))
                    ax3d_final = fig3d_final.add_subplot(111, projection='3d')
                    
                    X_3d, Y_3d = np.meshgrid(np.linspace(x_range[0], x_range[1], res_3d_res), 
                                             np.linspace(y_range[0], y_range[1], res_3d_res))
                    Z_3d_surf = np.zeros_like(X_3d) # Neuer Name für Oberflächen-Z
                    for i_3d in range(X_3d.shape[0]):
                        for j_3d in range(X_3d.shape[1]):
                            try: Z_3d_surf[i_3d, j_3d] = current_func(np.array([X_3d[i_3d, j_3d], Y_3d[i_3d, j_3d]]))['value']
                            except: Z_3d_surf[i_3d, j_3d] = np.nan
                    
                    Z_3d_finite_surf = Z_3d_surf[np.isfinite(Z_3d_surf)]
                    z_min_3d_surf, z_max_3d_surf = (np.percentile(Z_3d_finite_surf, 1), np.percentile(Z_3d_finite_surf, 99)) if len(Z_3d_finite_surf) > 0 else (np.nan, np.nan)
                    Z_3d_plot_surf = np.clip(Z_3d_surf, z_min_3d_surf, z_max_3d_surf) if len(Z_3d_finite_surf) > 0 else Z_3d_surf

                    surf3d = ax3d_final.plot_surface(X_3d, Y_3d, Z_3d_plot_surf, cmap='viridis', edgecolor='none', alpha=0.7, rstride=1, cstride=1)
                    
                    path_points_3d = np.array(best_history)
                    path_x_3d, path_y_3d = path_points_3d[:, 0], path_points_3d[:, 1]
                    path_z_3d = np.array([current_func(p)['value'] if np.all(np.isfinite(p)) else np.nan for p in path_points_3d])
                    if len(Z_3d_finite_surf) > 0: path_z_3d = np.clip(path_z_3d, z_min_3d_surf, z_max_3d_surf)

                    ax3d_final.plot(path_x_3d, path_y_3d, path_z_3d, 'r-o', linewidth=2.5, markersize=5, label='Optimierungspfad', markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.5)
                    ax3d_final.scatter(path_x_3d[0], path_y_3d[0], path_z_3d[0], color='blue', s=100, label='Start', depthshade=True, edgecolor='black')
                    ax3d_final.scatter(path_x_3d[-1], path_y_3d[-1], path_z_3d[-1], color='lime', marker='*', s=150, label='Ende', depthshade=True, edgecolor='black')

                    if minima:
                        for m_val in minima:
                            if len(m_val) == 2:
                                try:
                                    z_m = current_func(np.array(m_val))['value']
                                    if len(Z_3d_finite_surf) > 0: z_m = np.clip(z_m, z_min_3d_surf, z_max_3d_surf)
                                    ax3d_final.scatter(m_val[0], m_val[1], z_m, color='gold', marker='X', s=150, linewidths=2, label='Bek. Minimum' if 'Bek. Minimum' not in [l.get_label() for l in ax3d_final.get_legend().get_texts()] else "", depthshade=True, edgecolor='black')
                                except: pass
                    
                    ax3d_final.set_xlabel('X'); ax3d_final.set_ylabel('Y'); ax3d_final.set_zlabel('Funktionswert')
                    ax3d_final.set_title(f"3D-Pfad: {algorithm_display_name} auf {st.session_state.ausgewählte_funktion}")
                    ax3d_final.view_init(elev=elev_3d_res, azim=azim_3d_res)
                    try: ax3d_final.dist = dist_3d_res
                    except: pass
                    
                    handles_3d, labels_3d = ax3d_final.get_legend_handles_labels()
                    by_label_3d = dict(zip(labels_3d, handles_3d))
                    ax3d_final.legend(by_label_3d.values(), by_label_3d.keys(), loc='upper left')
                    if surf3d.colorbar: surf3d.colorbar.remove() # Remove old if exists
                    fig3d_final.colorbar(surf3d, ax=ax3d_final, shrink=0.6, aspect=10, pad=0.1)
                    st.pyplot(fig3d_final)
                    plt.close(fig3d_final)
                elif current_func_dim != 2:
                    st.info("3D-Pfadvisualisierung ist nur für 2D-Funktionen verfügbar.")
    
    # ANZEIGE GESPEICHERTER ERGEBNISSE (wenn keine neue Optimierung gestartet wurde)
    # Dieser Block wird ausgeführt, wenn `start_optimization` False ist, aber Ergebnisse existieren.
    elif current_func and st.session_state.optimierungsergebnisse:
        # Filtere Ergebnisse für die aktuell ausgewählte Funktion
        # Der Schlüssel in optimierungsergebnisse ist algorithm_display_name
        results_for_current_func = {
            algo_name: data for algo_name, data in st.session_state.optimierungsergebnisse.items()
            if data.get("function") == st.session_state.ausgewählte_funktion
        }

        if results_for_current_func:
            with results_container: # Stellt sicher, dass dies im Hauptteil ist
                st.markdown("### Bisherige Optimierungsergebnisse (gespeichert)")
                
                sorted_result_keys = sorted(results_for_current_func.keys(), 
                                            key=lambda k: results_for_current_func[k].get("timestamp", 0), 
                                            reverse=True) # Neueste zuerst

                if not sorted_result_keys:
                    st.info("Noch keine Ergebnisse für diese Funktion und den gewählten Algorithmus-Typ gespeichert.")
                else:
                    # UI zur Auswahl eines gespeicherten Ergebnisses
                    # Standardmäßig das neueste Ergebnis für den aktuell ausgewählten Algorithmus anzeigen, falls vorhanden
                    default_selection = None
                    current_algo_ui_name = algorithm_options.get(selected_algorithm) # UI-Name des aktuell gewählten Algos
                    if current_algo_ui_name in sorted_result_keys:
                        default_selection = current_algo_ui_name
                    elif sorted_result_keys: # Sonst das neueste überhaupt
                        default_selection = sorted_result_keys[0]

                    selected_stored_result_key = st.selectbox(
                        "Gespeichertes Ergebnis zum Anzeigen auswählen:",
                        sorted_result_keys,
                        index=sorted_result_keys.index(default_selection) if default_selection and default_selection in sorted_result_keys else 0,
                        key="selectbox_select_stored_result"
                    )
                    
                    if selected_stored_result_key and selected_stored_result_key in results_for_current_func:
                        stored_data = results_for_current_func[selected_stored_result_key]
                        
                        # Dimension der Funktion für das gespeicherte Ergebnis
                        stored_func_name = stored_data["function"]
                        stored_func_details = pf.MATH_FUNCTIONS_LIB.get(stored_func_name)
                        stored_func_dim = 2 # Default
                        if stored_func_details:
                            stored_func_dim = stored_func_details.get("dimensions", 2)
                        elif stored_func_name in st.session_state.custom_funcs: # Custom func
                             stored_func_dim = 2


                        # Plotting für gespeicherte Ergebnisse (2D Kontur und Loss-Kurve)
                        col1_stored_disp, col2_stored_disp = st.columns([2,1])
                        with col1_stored_disp:
                            if "history" in stored_data and stored_data["history"] and stored_func_dim == 2:
                                fig_stored_2d = plt.figure(figsize=(8,6))
                                ax_stored_2d = fig_stored_2d.add_subplot(111)
                                # Konturplot-Logik (wie oben, aber mit stored_data)
                                X_s, Y_s = np.meshgrid(np.linspace(x_range[0], x_range[1], 100), np.linspace(y_range[0], y_range[1], 100))
                                Z_s_surf = np.zeros_like(X_s)
                                for i_s in range(X_s.shape[0]):
                                    for j_s in range(X_s.shape[1]):
                                        try: Z_s_surf[i_s, j_s] = current_func(np.array([X_s[i_s, j_s], Y_s[i_s, j_s]]))['value']
                                        except: Z_s_surf[i_s, j_s] = np.nan
                                Z_s_finite = Z_s_surf[np.isfinite(Z_s_surf)]
                                if len(Z_s_finite) > 0:
                                    z_min_s, z_max_s = np.percentile(Z_s_finite, [1,99])
                                    Z_s_surf_clipped = np.clip(Z_s_surf, z_min_s, z_max_s)
                                    ax_stored_2d.contourf(X_s, Y_s, Z_s_surf_clipped, levels=contour_levels, cmap='viridis', alpha=0.7)
                                else:
                                     ax_stored_2d.contourf(X_s, Y_s, Z_s_surf, levels=contour_levels, cmap='viridis', alpha=0.7)

                                path_s = np.array(stored_data["history"])
                                if path_s.ndim == 2 and path_s.shape[1] >=2:
                                    ax_stored_2d.plot(path_s[:,0], path_s[:,1], 'm-o', label=f'{selected_stored_result_key} Pfad (gesp.)') # Andere Farbe
                                    ax_stored_2d.plot(path_s[0,0], path_s[0,1], 'co', label='Start (gesp.)') # Andere Farbe
                                    ax_stored_2d.plot(path_s[-1,0], path_s[-1,1], 'k*', label='Ende (gesp.)') # Andere Farbe
                                if minima:
                                    for m_idx, m_val in enumerate(minima):
                                        if len(m_val) == 2: ax_stored_2d.plot(m_val[0], m_val[1], 'yX', markersize=10, markeredgewidth=1.5, markeredgecolor='black', label='Bek. Minimum' if m_idx==0 else None)
                                ax_stored_2d.set_xlim(x_range); ax_stored_2d.set_ylim(y_range)
                                ax_stored_2d.set_title(f"Gespeicherter Pfad: {selected_stored_result_key} auf {stored_data['function']}")
                                ax_stored_2d.legend(loc="best"); ax_stored_2d.grid(True, alpha=0.3)
                                st.pyplot(fig_stored_2d)
                                plt.close(fig_stored_2d)
                            elif stored_func_dim != 2: st.info("2D Konturplot nur für 2D Funktionen.")
                        
                        with col2_stored_disp:
                            if "loss_history" in stored_data and stored_data["loss_history"]:
                                fig_stored_loss = plt.figure(figsize=(8,4))
                                ax_stored_loss = fig_stored_loss.add_subplot(111)
                                losses_s = stored_data["loss_history"]
                                ax_stored_loss.plot(range(len(losses_s)), losses_s, '-o', color='purple')
                                positive_losses_s = [l for l in losses_s if l is not None and np.isfinite(l) and l > 0]
                                if positive_losses_s and len(positive_losses_s) > 1 and max(positive_losses_s) / min(positive_losses_s) > 100:
                                    ax_stored_loss.set_yscale('log')
                                ax_stored_loss.set_title(f"Gespeicherter Loss: {selected_stored_result_key}")
                                ax_stored_loss.set_xlabel("Iteration"); ax_stored_loss.set_ylabel("Funktionswert")
                                ax_stored_loss.grid(True, alpha=0.3); fig_stored_loss.tight_layout()
                                st.pyplot(fig_stored_loss)
                                plt.close(fig_stored_loss)

                        st.markdown("### Details (gespeichert)")
                        col1_s_det, col2_s_det, col3_s_det, col4_s_det = st.columns(4)
                        history_s_det = stored_data.get("history", [])
                        best_x_s_det = np.array(stored_data.get("best_x")) if stored_data.get("best_x") is not None else None
                        loss_s_det = stored_data.get("loss_history", [])
                        
                        with col1_s_det: st.metric("Startpunkt", f"{np.round(history_s_det[0],3).tolist()}" if history_s_det and len(history_s_det[0]) == stored_func_dim else "N/A")
                        with col2_s_det: st.metric("Endpunkt", f"{np.round(best_x_s_det,3).tolist()}" if best_x_s_det is not None and len(best_x_s_det) == stored_func_dim else "N/A")
                        with col3_s_det: st.metric("Funktionswert", f"{loss_s_det[-1]:.6f}" if loss_s_det and np.isfinite(loss_s_det[-1]) else "N/A")
                        with col4_s_det: st.metric("Iterationen", f"{len(loss_s_det)-1}" if loss_s_det else "0")
                        st.markdown(f"**Status:** {stored_data.get('status', 'Unbekannt')}")
                        
                        stored_params_disp = stored_data.get("params", {})
                        if stored_params_disp:
                            with st.expander("Verwendete Parameter für dieses gespeicherte Ergebnis"):
                                st.json(stored_params_disp)

                        # 3D Plot für gespeicherte Ergebnisse (falls 2D Funktion)
                        if "history" in stored_data and stored_data["history"] and stored_func_dim == 2:
                            st.markdown("### 3D-Visualisierung (gespeichert)")
                            # (Die 3D-Plot-Logik für gespeicherte Ergebnisse ist analog zur obigen,
                            #  aber mit Daten aus `stored_data` und eindeutigen Slider-Keys)
                            elev_3d_s = st.session_state.get("elev_3d_storedplot", 30)
                            azim_3d_s = st.session_state.get("azim_3d_storedplot", -60)
                            dist_3d_s = st.session_state.get("dist_3d_storedplot", 10)
                            res_3d_s = st.session_state.get("res_3d_storedplot", 50)

                            s_cols_3d_ctrl = st.columns(4)
                            with s_cols_3d_ctrl[0]: elev_3d_s = st.slider("Elevation (Stored 3D)", 0, 90, elev_3d_s, key="elev_3d_storedplot")
                            with s_cols_3d_ctrl[1]: azim_3d_s = st.slider("Azimuth (Stored 3D)", -180, 180, azim_3d_s, key="azim_3d_storedplot")
                            with s_cols_3d_ctrl[2]: dist_3d_s = st.slider("Distanz (Stored 3D)", 5, 20, dist_3d_s, key="dist_3d_storedplot")
                            with s_cols_3d_ctrl[3]: res_3d_s = st.slider("Auflösung (Stored 3D)", 30, 80, res_3d_s, key="res_3d_storedplot")
                            
                            fig3d_s = plt.figure(figsize=(10,8))
                            ax3d_s = fig3d_s.add_subplot(111, projection='3d')
                            # (Oberfläche plotten wie oben)
                            X_s3d, Y_s3d = np.meshgrid(np.linspace(x_range[0], x_range[1], res_3d_s), np.linspace(y_range[0], y_range[1], res_3d_s))
                            Z_s3d_surf = np.zeros_like(X_s3d)
                            for i_s3d in range(X_s3d.shape[0]):
                                for j_s3d in range(X_s3d.shape[1]):
                                    try: Z_s3d_surf[i_s3d, j_s3d] = current_func(np.array([X_s3d[i_s3d, j_s3d], Y_s3d[i_s3d, j_s3d]]))['value']
                                    except: Z_s3d_surf[i_s3d, j_s3d] = np.nan
                            Z_s3d_finite_surf = Z_s3d_surf[np.isfinite(Z_s3d_surf)]
                            z_min_s3d, z_max_s3d = (np.percentile(Z_s3d_finite_surf,1), np.percentile(Z_s3d_finite_surf,99)) if len(Z_s3d_finite_surf) > 0 else (np.nan,np.nan)
                            Z_s3d_plot_surf = np.clip(Z_s3d_surf, z_min_s3d, z_max_s3d) if len(Z_s3d_finite_surf) > 0 else Z_s3d_surf
                            surf3d_s = ax3d_s.plot_surface(X_s3d, Y_s3d, Z_s3d_plot_surf, cmap='viridis', edgecolor='none', alpha=0.7, rstride=1, cstride=1)

                            # (Pfad plotten wie oben, mit Daten aus stored_data)
                            path_s_3d_pts = np.array(stored_data["history"])
                            path_sx, path_sy = path_s_3d_pts[:,0], path_s_3d_pts[:,1]
                            path_sz = np.array([current_func(p)['value'] if np.all(np.isfinite(p)) else np.nan for p in path_s_3d_pts])
                            if len(Z_s3d_finite_surf) > 0: path_sz = np.clip(path_sz, z_min_s3d, z_max_s3d)
                            ax3d_s.plot(path_sx, path_sy, path_sz, 'm-o', linewidth=2.5, markersize=5, label='Gespeicherter Pfad', markerfacecolor='magenta', markeredgecolor='black', markeredgewidth=0.5)
                            ax3d_s.scatter(path_sx[0],path_sy[0],path_sz[0], color='cyan', s=100, label='Start (gesp.)', depthshade=True, edgecolor='black')
                            ax3d_s.scatter(path_sx[-1],path_sy[-1],path_sz[-1], color='black', marker='P', s=120, label='Ende (gesp.)', depthshade=True, edgecolor='white')

                            if minima: # Minima auch hier
                                for m_val in minima:
                                    if len(m_val) == 2:
                                        try:
                                            z_m_s = current_func(np.array(m_val))['value']
                                            if len(Z_s3d_finite_surf) > 0: z_m_s = np.clip(z_m_s, z_min_s3d, z_max_s3d)
                                            ax3d_s.scatter(m_val[0],m_val[1],z_m_s, color='gold', marker='X', s=150, linewidths=2, label='Bek. Minimum' if 'Bek. Minimum' not in [l.get_label() for l in ax3d_s.get_legend().get_texts()] else "", depthshade=True, edgecolor='black')
                                        except: pass
                            ax3d_s.set_xlabel('X'); ax3d_s.set_ylabel('Y'); ax3d_s.set_zlabel('Funktionswert')
                            ax3d_s.set_title(f"3D Gesp. Pfad: {selected_stored_result_key} auf {stored_data['function']}")
                            ax3d_s.view_init(elev=elev_3d_s, azim=azim_3d_s);
                            try: ax3d_s.dist = dist_3d_s
                            except: pass
                            handles_s3d, labels_s3d = ax3d_s.get_legend_handles_labels()
                            by_label_s3d = dict(zip(labels_s3d, handles_s3d))
                            ax3d_s.legend(by_label_s3d.values(), by_label_s3d.keys(), loc='upper left')
                            if surf3d_s.colorbar: surf3d_s.colorbar.remove()
                            fig3d_s.colorbar(surf3d_s, ax=ax3d_s, shrink=0.6, aspect=10, pad=0.1)
                            st.pyplot(fig3d_s)
                            plt.close(fig3d_s)
                        elif stored_func_dim != 2:
                            st.info("3D-Pfadvisualisierung ist nur für 2D-Funktionen verfügbar.")
        else:
            st.info("Keine gespeicherten Ergebnisse für die ausgewählte Funktion vorhanden.")
    elif not current_func:
        st.warning("Bitte zuerst eine Funktion im Sidebar oder im Funktionseditor auswählen/erstellen.")
        
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
