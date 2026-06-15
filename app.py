from shiny import App, render, ui, reactive
import numpy as np
import pandas as pd
import neurokit2 as nk
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import io

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("file", "Upload SHIMMER CSV File", accept=[".csv"]),
        ui.input_numeric("sampling_rate", "Sampling Rate (Hz)", value=51.2, min=1),
        ui.input_numeric("trim_seconds", "Trim Seconds (start/end)", value=10, min=0),
        ui.input_action_button("analyze", "Analyze", class_="btn-primary"),
    ),
    ui.navset_tab(
        ui.nav_panel(
            "EDA Analysis",
            ui.card(
                ui.card_header("EDA Metrics"),
                ui.output_text_verbatim("eda_metrics")
            ),
            ui.card(
                ui.card_header("EDA Plots"),
                ui.output_plot("eda_plot", height="600px")
            )
        ),
        ui.nav_panel(
            "PPG & Heart Rate",
            ui.card(
                ui.card_header("Heart Rate Metrics"),
                ui.output_text_verbatim("hr_metrics")
            ),
            ui.card(
                ui.card_header("PPG Plots"),
                ui.output_plot("ppg_plot", height="600px")
            )
        ),
        ui.nav_panel(
            "HRV Analysis",
            ui.card(
                ui.card_header("HRV Metrics"),
                ui.output_text_verbatim("hrv_metrics")
            )
        ),
        ui.nav_panel(
            "Respiration",
            ui.card(
                ui.card_header("Respiratory Metrics"),
                ui.output_text_verbatim("resp_metrics")
            ),
            ui.card(
                ui.card_header("PPG-Derived Respiration"),
                ui.output_plot("resp_plot", height="400px")
            )
        )
    )
)

def server(input, output, session):
    # Reactive value to store analysis results
    analysis_results = reactive.value(None)
    
    @reactive.effect
    @reactive.event(input.analyze)
    def _():
        file_info = input.file()
        if file_info is None:
            return
        
        try:
            # Read the uploaded file
            df = pd.read_csv(file_info[0]["datapath"], sep='\t', skiprows=2)
            
            sampling_rate = input.sampling_rate()
            samples_to_remove = int(input.trim_seconds() * sampling_rate)
            
            # Process GSR/EDA signal
            gsr_signal = df.iloc[:, 3].astype(float).values
            processed_gsr_signal = gsr_signal[samples_to_remove : -samples_to_remove]
            
            eda_signals, eda_info = nk.eda_process(processed_gsr_signal, sampling_rate=sampling_rate)
            
            # EDA metrics
            n_peaks = len(eda_info.get("SCR_Peaks", []))
            mean_amplitude = np.nanmean(eda_info.get("SCR_Amplitude", [0]))
            mean_risetime = np.nanmean(eda_info.get("SCR_RiseTime", [0]))
            mean_recovery = np.nanmean(eda_info.get("SCR_RecoveryTime", [0]))
            mean_scl = np.mean(eda_signals['EDA_Tonic'])
            duration_minutes_eda = len(processed_gsr_signal) / sampling_rate / 60
            scr_per_min = n_peaks / duration_minutes_eda if duration_minutes_eda > 0 else 0
            
            # Process PPG signal
            ppg_signal = df.iloc[:, 5].astype(float).values
            processed_ppg_signal = ppg_signal[samples_to_remove : -samples_to_remove]

            # Clean obvious bad values
            processed_ppg_signal = np.nan_to_num(processed_ppg_signal, nan=0.0, posinf=0.0, neginf=0.0)

            # First detect peaks / process PPG (this populates ppg_info with peak locations)
            ppg_signals, ppg_info = nk.ppg_process(
                processed_ppg_signal,
                sampling_rate=sampling_rate,
                method='elgendi'
            )

            # Try to compute quality using detected peaks; otherwise fallback
            ppg_peaks = ppg_info.get("PPG_Peaks", None)

            try:
                # Normalize different possible formats from nk.ppg_process:
                # - sometimes PPG_Peaks is a dict {index: 1}, sometimes a list/array of indices
                peak_indices = None
                if ppg_peaks is None:
                    peak_indices = None
                elif isinstance(ppg_peaks, dict):
                    # keys are the sample indices
                    peak_indices = np.array(list(ppg_peaks.keys()), dtype=int)
                else:
                    # list / ndarray of indices
                    peak_indices = np.asarray(ppg_peaks, dtype=int)

                if peak_indices is None or peak_indices.size == 0:
                    # use a peak-free method like 'skewness'
                    ppg_quality = nk.ppg_quality(processed_ppg_signal, sampling_rate=sampling_rate, method='skewness')
                else:
                    # pass detected peak indices using the 'peaks' keyword
                    ppg_quality = nk.ppg_quality(processed_ppg_signal, sampling_rate=sampling_rate, peaks=peak_indices)

                mean_ppg_quality = float(np.mean(ppg_quality)) if np.size(ppg_quality) else 0.0
            except Exception:
                # Final fallback if neurokit still complains
                mean_ppg_quality = 0.0
    
            
            ppg_signal = df.iloc[:, 5].astype(float).values
            processed_ppg_signal = ppg_signal[samples_to_remove : -samples_to_remove]
            
            heart_rate = ppg_signals['PPG_Rate']
            
            # HRV analysis
            hrv_indices = nk.hrv(ppg_info, sampling_rate=sampling_rate, show=False)
            
            # Respiration from PPG
            edr = nk.ecg_rsp(ecg_rate=heart_rate.values, sampling_rate=sampling_rate)
            rsp_peaks, _ = find_peaks(edr, distance=int(sampling_rate * 2))
            
            if len(rsp_peaks) > 1:
                rsp_intervals = np.diff(rsp_peaks) / sampling_rate
                breathing_rate = 60 / np.mean(rsp_intervals)
            else:
                breathing_rate = 0.0
            
            # Store all results
            results = {
                'eda_signals': eda_signals,
                'eda_info': eda_info,
                'eda_metrics': {
                    'mean_scl': mean_scl,
                    'scr_per_min': scr_per_min,
                    'n_peaks': n_peaks,
                    'mean_amplitude': mean_amplitude,
                    'mean_risetime': mean_risetime,
                    'mean_recovery': mean_recovery
                },
                'ppg_signals': ppg_signals,
                'ppg_info': ppg_info,
                'mean_ppg_quality': mean_ppg_quality,
                'heart_rate': heart_rate,
                'hrv_indices': hrv_indices,
                'edr': edr,
                'breathing_rate': breathing_rate,
                'sampling_rate': sampling_rate
            }
            
            analysis_results.set(results)
            
        except Exception as e:
            analysis_results.set({'error': str(e)})
    
    @render.text
    def eda_metrics():
        results = analysis_results.get()
        if results is None:
            return "Upload a file and click 'Analyze' to see results."
        if 'error' in results:
            return f"Error: {results['error']}"
        
        metrics = results['eda_metrics']
        return (
            f"--- METRICHE EDA ---\n"
            f"Livello Tonico Medio (SCL): {metrics['mean_scl']:.3f} μS\n"
            f"Frequenza Picchi (SCR/min): {metrics['scr_per_min']:.2f} picchi/min\n"
            f"Numero totale picchi SCR: {metrics['n_peaks']}\n"
            f"Ampiezza media SCR: {metrics['mean_amplitude']:.3f} μS\n"
            f"Rise Time medio: {metrics['mean_risetime']:.3f} s\n"
            f"Half-Recovery Time medio: {metrics['mean_recovery']:.3f} s"
        )
    
    @render.plot
    def eda_plot():
        results = analysis_results.get()
        if results is None or 'error' in results:
            return
        
        fig = nk.eda_plot(results['eda_signals'], results['eda_info'])
        plt.subplots_adjust(hspace=0.5)
        return fig
    
    @render.text
    def hr_metrics():
        results = analysis_results.get()
        if results is None:
            return "Upload a file and click 'Analyze' to see results."
        if 'error' in results:
            return f"Error: {results['error']}"
        
        hr = results['heart_rate']
        return (
            f"--- METRICHE HEART RATE E QUALITÀ ---\n"
            f"Qualità media del segnale PPG (0-1): {results['mean_ppg_quality']:.3f}\n"
            f"HR Minimo: {np.min(hr):.2f} bpm\n"
            f"HR Massimo: {np.max(hr):.2f} bpm\n"
            f"HR Medio: {np.mean(hr):.2f} bpm"
        )
    
    @render.plot
    def ppg_plot():
        results = analysis_results.get()
        if results is None or 'error' in results:
            return
        
        fig = nk.ppg_plot(results['ppg_signals'], results['ppg_info'])
        plt.subplots_adjust(hspace=0.5)
        return fig
    
    @render.text
    def hrv_metrics():
        results = analysis_results.get()
        if results is None:
            return "Upload a file and click 'Analyze' to see results."
        if 'error' in results:
            return f"Error: {results['error']}"
        
        hrv = results['hrv_indices']
        metrics_to_print = ['HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_LFn', 'HRV_HFn', 'HRV_LFHF', 'HRV_SampEn']
        
        output = "--- METRICHE HRV PRINCIPALI ---\n"
        for metric in metrics_to_print:
            if metric in hrv.columns:
                output += f"{metric}: {hrv[metric].values[0]:.4f}\n"
        
        return output
    
    @render.text
    def resp_metrics():
        results = analysis_results.get()
        if results is None:
            return "Upload a file and click 'Analyze' to see results."
        if 'error' in results:
            return f"Error: {results['error']}"
        
        return (
            f"--- METRICHE RESPIRATORIE ---\n"
            f"Breathing Rate Stimato: {results['breathing_rate']:.2f} bpm (atti al minuto)"
        )
    
    @render.plot
    def resp_plot():
        results = analysis_results.get()
        if results is None or 'error' in results:
            return
        
        edr = results['edr']
        sampling_rate = results['sampling_rate']
        time_normalized = np.arange(len(edr)) / sampling_rate
        
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(time_normalized, edr)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Respiration")
        ax.set_title("PPG-derived respiration over time")
        ax.grid(True)
        
        return fig

app = App(app_ui, server)
