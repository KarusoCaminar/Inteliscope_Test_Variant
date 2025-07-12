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

# Importiere eigene Module
import problem_functions_v3 as pf
import optimization_algorithms_v3 as oa
import visualization_suite_v3 as vs
import improved_optimizer as iopt

def create_visualization_tracker(func, x_range, y_range, contour_levels, minima):
    """
    Erstellt einen Tracker für den Optimierungspfad
    """
    path_history = []
    value_history = []

    def callback(iteration, x, value, grad_norm, message):
        path_history.append(x.copy())
        value_history.append(value)

        info_text = f"""
        **Iteration:** {iteration+1}
        **Aktuelle Position:** [{x[0]:.4f}, {x[1]:.4f}]
        **Funktionswert:** {value:.6f}
        **Gradientennorm:** {grad_norm:.6f}
        """
        info_placeholder.markdown(info_text)

        if iteration % 5 == 0 or iteration < 5:
            fig_live = plt.figure(figsize=(8, 4))
            ax_live = fig_live.add_subplot(111)
            X, Y = np.meshgrid(np.linspace(x_range[0], x_range[1], 50), 
                               np.linspace(y_range[0], y_range[1], 50))
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        result = func(np.array([X[i, j], Y[i, j]]))
                        Z[i, j] = result.get('value', np.nan)
                    except:
                        Z[i, j] = np.nan
            if minima:
                for m in minima:
                    ax_live.plot(m[0], m[1], 'X', color='red', markersize=8, markeredgecolor='black')
            if len(path_history) > 0:
                path_arr = np.array(path_history)
                ax_live.plot(path_arr[:, 0], path_arr[:, 1], 'ro-', linewidth=2, markersize=3)
            plt.colorbar(ax_live.contourf(X, Y, Z, levels=contour_levels, cmap='viridis', alpha=0.8), ax=ax_live)
            ax_live.set_xlabel('X')
            ax_live.set_ylabel('Y')
            ax_live.set_title('Live-Optimierung')
            live_plot_placeholder.pyplot(fig_live)
            plt.close(fig_live)

    return callback, path_history, value_history
    
st.set_page_config(
    page_title="IntelliScope Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# IntelliScope Explorer\nInteraktive Visualisierung von Optimierungslandschaften und -algorithmen. Entwickelt für die Analyse und das Verständnis verschiedener Optimierungsverfahren."
    }
)

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
    
    .plot-hover {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .plot-hover:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .highlight {
        background: linear-gradient(90deg, rgba(77,139,240,0.1) 0%, rgba(77,139,240,0) 100%);
        padding: 0.2rem 0.5rem;
        border-left: 3px solid #4d8bf0;
    }
    
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

for key, default in [
    ('optimierungsergebnisse', {}),
    ('custom_funcs', {}),
    ('ausgewählte_funktion', "Rosenbrock"),
    ('custom_func_count', 0)
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown("""
<div style="background: linear-gradient(90deg, #6a2c91, #4d8bf0); padding: 1.5rem; border-radius: 0.8rem; margin-bottom: 1.5rem; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
    <div class="main-header" style="color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">IntelliScope Explorer</div>
    <div class="sub-header" style="color: rgba(255,255,255,0.9); font-weight: 300;">Interaktive Visualisierung von Optimierungslandschaften und -algorithmen</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Einstellungen")
    
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

    start_optimization = st.button("Optimierung starten", use_container_width=True)

    if st.button("Alle Ergebnisse zurücksetzen", use_container_width=True, key="reset_results_main"):
        st.session_state.optimierungsergebnisse = {}
        st.rerun()

tabs = st.tabs(["Optimierungsvisualisierung", "Funktionseditor", "Ergebnisvergleich", "Info & Hilfe"])

with tabs[0]:
    current_func_obj = None
    start_point = None
    x_range = (-5, 5)
    y_range = (-5, 5)
    contour_levels = 30
    minima = None

    if st.session_state.ausgewählte_funktion in pf.MATH_FUNCTIONS_LIB:
        func_info = pf.MATH_FUNCTIONS_LIB[st.session_state.ausgewählte_funktion]
        current_func_obj = func_info.get("func")
        x_range = func_info.get("default_range", [(-5, 5), (-5, 5)])[0]
        y_range = func_info.get("default_range", [(-5, 5), (-5, 5)])[1]
        contour_levels = func_info.get("contour_levels", 40)
        try:
            eval_point = np.array([(x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2])
            func_meta = current_func_obj(eval_point)
            if "tooltip" in func_meta:
                with st.expander("ℹ️ Über diese Funktion", expanded=False):
                    st.markdown(func_meta["tooltip"])
            minima = func_meta.get("minima", None)
        except Exception as e:
            st.warning(f"Metadaten für {st.session_state.ausgewählte_funktion} konnten nicht geladen werden: {e}")
            minima = None

    elif st.session_state.ausgewählte_funktion in st.session_state.custom_funcs:
        current_func_obj = st.session_state.custom_funcs[st.session_state.ausgewählte_funktion]
        try:
            func_meta = current_func_obj(np.array([0.0, 0.0]))
            x_range = func_meta.get("x_range", (-5, 5))
            y_range = func_meta.get("y_range", (-5, 5))
            contour_levels = func_meta.get("contour_levels", 30)
            minima = func_meta.get("minima", None)
        except Exception:
            x_range = (-5, 5)
            y_range = (-5, 5)
            contour_levels = 30
            minima = None

    else:
        st.error("Die ausgewählte Funktion wurde nicht gefunden.")
        current_func_obj = None
        x_range = (-5, 5)
        y_range = (-5, 5)
        contour_levels = 30
        minima = None
   
    col1, col2 = st.columns([1, 1])
    
    with col1:
            if current_func_obj:
                fig = vs.create_interactive_surface_plotly(
                    current_func_obj,
                    x_range,
                    y_range,
                    optimization_paths={
                        algo: result["history"]
                        for algo, result in st.session_state.optimierungsergebnisse.items()
                        if result["function"] == st.session_state.ausgewählte_funktion and "history" in result
                    },
                    minima=minima,
                    title=f"3D-Ansicht: {st.session_state.ausgewählte_funktion}"
                )
                st.plotly_chart(fig, use_container_width=True)
                    
    with col2:
        if current_func_obj:
            fig = vs.create_interactive_contour_plotly(
                current_func_obj,
                x_range,
                y_range,
                optimization_paths={
                    algo: result["history"]
                    for algo, result in st.session_state.optimierungsergebnisse.items()
                    if result["function"] == st.session_state.ausgewählte_funktion and "history" in result
                },
                minima=minima,
                title=f"2D-Ansicht: {st.session_state.ausgewählte_funktion}"
            )
            st.plotly_chart(fig, use_container_width=True)


    st.markdown("""
    <div style="background: linear-gradient(90deg, #4d8bf0, #6a2c91); padding: 12px; border-radius: 8px;">
        <h3 style="color: white; margin: 0;">Optimierungsergebnisse</h3>
    </div>
    """, unsafe_allow_html=True)
    
    live_tracking_container = st.container()
    info_box_container = st.container()
    results_container = st.container()
    
    if start_optimization and current_func_obj:
        with st.spinner("Optimierung läuft..."):
            live_tracking_container.empty()
            info_box_container.empty()
            
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

            visualization_callback, path_history, value_history = create_visualization_tracker(
                current_func_obj, x_range, y_range, contour_levels, minima
            )
            
            start_x = np.linspace(x_range[0], x_range[1], 10)
            start_y = np.linspace(y_range[0], y_range[1], 10)
            highest_value = float('-inf')
            best_point = None
            
            for x_val in start_x:
                for y_val in start_y:
                    try:
                        point = np.array([x_val, y_val])
                        result = current_func_obj(point)
                        if 'value' in result and np.isfinite(result['value']) and result['value'] > highest_value:
                            highest_value = result['value']
                            best_point = point.copy()
                    except Exception:
                        continue
            
            if best_point is not None:
                start_point = best_point
            else:
                start_point = np.array([(x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2])
                st.warning("Automatischer Startpunkt konnte nicht ermittelt werden, verwende Mittelpunkt des Bereichs.")
            
            st.write(f"Starte Optimierung von Punkt: [{start_point[0]:.4f}, {start_point[1]:.4f}]")
            
            try:
                result = current_func_obj(start_point)
                if not ("value" in result and np.isfinite(result["value"])):
                    st.error("Der Startpunkt liefert keinen gültigen Funktionswert – bitte Bereich oder Funktion überprüfen!")
            except Exception as e:
                st.error(f"Der Startpunkt ist ungültig für die Funktion: {e}")
            
            optimizer_fn = iopt.OPTIMIZERS_EXTENDED[selected_algorithm_key]
            
            result = optimizer_fn(current_func_obj, start_point, **optimizer_params, callback=visualization_callback)
            
            best_x, best_history, best_loss_history, status = result
            
            algorithm_display_name = f"{algorithm_options[selected_algorithm_key]}"
            
            st.session_state.optimierungsergebnisse[algorithm_display_name] = {
                "function": st.session_state.ausgewählte_funktion,
                "best_x": best_x,
                "history": best_history,
                "loss_history": best_loss_history,
                "status": status,
                "timestamp": time.time()
            }   
            
            with results_container:
                st.markdown("""
                <div style="background-color: #f0fff8; padding: 12px; border-radius: 8px; border-left: 4px solid #15b371;">
                    <h3 style="color: #15b371; margin: 0;">Zusammenfassung der Optimierung</h3>
                </div>
                """, unsafe_allow_html=True)
                     
            st.markdown("""
            <div style="background-color: #f0f8ff; padding: 8px; border-radius: 8px; margin-top: 15px;">
                <h4 style="color: #4d8bf0; margin: 0;">Optimierungs-Details</h4>
            </div>
            """, unsafe_allow_html=True)
                
            col1, col2, col3, col4 = st.columns(4)
                
            with col1:
                if best_history and len(best_history) > 0 and len(best_history[0]) > 1:
                    st.metric("Startpunkt", f"[{best_history[0][0]:.3f}, {best_history[0][1]:.3f}]")
                else:
                    st.metric("Startpunkt", "N/A")
            
            with col2:
                if best_x is not None and len(best_x) > 1:
                    st.metric("Endpunkt", f"[{best_x[0]:.3f}, {best_x[1]:.3f}]")
                else:
                    st.metric("Endpunkt", "N/A")
            
            with col3:
                if best_loss_history:
                    st.metric("Funktionswert", f"{best_loss_history[-1]:.6f}")
                else:
                    st.metric("Funktionswert", "N/A")
            
            with col4:
                if best_loss_history:
                    st.metric("Iterationen", f"{len(best_loss_history)-1}")
                else:
                    st.metric("Iterationen", "N/A")
            
            st.markdown(f"**Status:** {status}")
            st.rerun()
                
with tabs[1]:
    st.markdown("## Funktionseditor")
    st.markdown("""
    Hier kannst du eigene mathematische Funktionen definieren und testen. Verwende `x` und `y` als Variablen.
    
    **Beispiele:**
    - `x**2 + y**2` (Parabel)
    - `sin(x) + cos(y)` (Sinuswelle)
    - `(x-2)**2 + (y-3)**2` (Verschobene Parabel)
    """)
    
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
        expr_to_parse = custom_func_expr
        expr_to_parse = re.sub(r'(?<![a-zA-Z])sin\(', 'sp.sin(', expr_to_parse)
        expr_to_parse = re.sub(r'(?<![a-zA-Z])cos\(', 'sp.cos(', expr_to_parse)
        expr_to_parse = re.sub(r'(?<![a-zA-Z])tan\(', 'sp.tan(', expr_to_parse)
        expr_to_parse = re.sub(r'(?<![a-zA-Z])exp\(', 'sp.exp(', expr_to_parse)
        expr_to_parse = re.sub(r'(?<![a-zA-Z])log\(', 'sp.log(', expr_to_parse)
        expr_to_parse = re.sub(r'(?<![a-zA-Z])sqrt\(', 'sp.sqrt(', expr_to_parse)
        expr_to_parse = re.sub(r'(?<![a-zA-Z])abs\(', 'sp.Abs(', expr_to_parse)
        
        try:
            custom_func_obj = pf.create_custom_function(
                expr_to_parse,
                name=custom_func_name,
                x_range=(x_min, x_max),
                y_range=(y_min, y_max)
            )
            
            test_result = custom_func_obj(np.array([1.0, 1.0]))
            if "value" in test_result and np.isfinite(test_result["value"]):
                st.session_state.custom_funcs[custom_func_name] = custom_func_obj
                st.session_state.custom_func_count += 1
                st.session_state.ausgewählte_funktion = custom_func_name
                st.success(f"Funktion '{custom_func_name}' erfolgreich erstellt!")
                st.rerun()
            else:
                st.error(f"Die Funktion konnte nicht evaluiert werden. Überprüfe den Ausdruck auf Gültigkeit.")
        except Exception as e:
            st.error(f"Fehler beim Erstellen der Funktion: {e}")
    
    if st.session_state.custom_funcs:
        st.markdown("### Deine benutzerdefinierten Funktionen")
        
        for name, func in st.session_state.custom_funcs.items():
            with st.expander(name):
                try:
                    fig = vs.create_interactive_contour_plotly(
                        func,
                        (-5, 5),
                        (-5, 5),
                        title=name
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
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
        function_groups = {}
        for algo, result in st.session_state.optimierungsergebnisse.items():
            func_name = result["function"]
            if func_name not in function_groups:
                function_groups[func_name] = []
            function_groups[func_name].append(algo)
        
        selected_function_for_comparison = st.selectbox(
            "Funktion für Vergleich auswählen",
            list(function_groups.keys())
        )
        
        if selected_function_for_comparison:
            algos_for_function = function_groups[selected_function_for_comparison]
            
            selected_algos = st.multiselect(
                "Algorithmen zum Vergleich auswählen",
                algos_for_function,
                default=algos_for_function[:min(3, len(algos_for_function))]
            )
            
            if selected_algos:
                comparison_results = {}
                for algo in selected_algos:
                    if algo in st.session_state.optimierungsergebnisse:
                        algo_result = st.session_state.optimierungsergebnisse[algo]
                        if "loss_history" in algo_result:
                            comparison_results[algo] = algo_result["loss_history"]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig_comparison = vs.plot_loss_comparison(
                        [st.session_state.optimierungsergebnisse[algo] for algo in selected_algos],
                        selected_algos
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
                
                with col2:
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
                
                if selected_function_for_comparison in pf.MATH_FUNCTIONS_LIB:
                    func_info = pf.MATH_FUNCTIONS_LIB[selected_function_for_comparison]
                    current_func_obj = func_info["func"]
                    x_range = func_info["default_range"][0]
                    y_range = func_info["default_range"][1]
                    try:
                        func_meta = current_func_obj(np.array([0, 0]))
                        minima = func_meta.get("minima", None)
                    except Exception:
                        minima = None
                
                elif selected_function_for_comparison in st.session_state.custom_funcs:
                    current_func_obj = st.session_state.custom_funcs[selected_function_for_comparison]
                    x_range = (-5, 5)
                    y_range = (-5, 5)
                    minima = None
                else:
                    st.error("Die ausgewählte Funktion wurde nicht gefunden.")
                    current_func_obj = None
                
                if current_func_obj:
                    fig = vs.create_interactive_contour_plotly(
                        current_func_obj,
                        x_range,
                        y_range,
                        optimization_paths={
                            algo: st.session_state.optimierungsergebnisse[algo]["history"]
                            for algo in selected_algos
                        },
                        minima=minima,
                        title=f"Vergleich der Pfade: {selected_function_for_comparison}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    fig3d = vs.create_interactive_surface_plotly(
                        current_func_obj,
                        x_range,
                        y_range,
                        optimization_paths={
                            algo: st.session_state.optimierungsergebnisse[algo]["history"]
                            for algo in selected_algos
                        },
                        minima=minima,
                        title=f"3D-Vergleich der Pfade: {selected_function_for_comparison}"
                    )
                    st.plotly_chart(fig3d, use_container_width=True)

with tabs[3]:
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

st.markdown("---")
st.markdown("© 2024-2025 IntelliScope Explorer")
