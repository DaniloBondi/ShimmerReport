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
        ui.input_numeric("peak_rel_height", "EDA Peak Rel. Height Min", value=4, min=0),
        ui.input_action_button("analyze", "Analyze", class_="btn-primary"),
    ),
    ui.navset_tab(
        ui.nav_panel(
            "EDA Analysis",
            ui.card(ui.card_header("EDA Full Report & Metrics"), ui.output_text_verbatim("eda_metrics")),
            ui.card(ui.card_header("Custom Component Plot (Total/Tonic/Phasic)"), ui.output_plot("eda_custom_plot", height="400px")),
            ui.card(ui.card_header("EDA Standard Diagnostic Plot"), ui.output_plot("eda_plot", height="500px"))
        ),
        ui.nav_panel(
            "PPG & Heart Rate",
            ui.card(ui.card_header("Heart Rate Metrics"), ui.output_text_verbatim("hr_metrics")),
            ui.card(ui.card_header("Heart Rate Trend"), ui.output_plot("hr_custom_plot", height="350px")),
            ui.card(ui.card_header("PPG Standard Plot"), ui.output_plot("ppg_plot", height="500px"))
        ),
        ui.nav_panel(
            "HRV Analysis",
            ui.card(ui.card_header("Full HRV Indices"), ui.output_text_verbatim("hrv_metrics")),
            ui.card(ui.card_header("HRV Poincaré Plot"), ui.output_plot("hrv_plot", height="500px"))
        ),
        ui.nav_panel(
            "Respiration",
            ui.card(ui.card_header("Respiratory Metrics"), ui.output_text_verbatim("resp_metrics")),
            ui.card(ui.card_header("PPG-Derived Respiration (PDR)"), ui.output_plot("resp_plot", height="400px"))
        )
    )
)

def server(input, output, session):
    @reactive.Calc
    @reactive.event(input.analyze)
    def processed_data():
        file_info = input.file()
        if file_info is None: return None
        with ui.Progress(min=1, max=5) as p:
            try:
                p.set(message="Reading CSV...", value=1)
                df = pd.read_csv(file_info[0]["datapath"], sep=None, skiprows=2, engine='python')
                sr = input.sampling_rate()
                trim = int(input.trim_seconds() * sr)
                if len(df) <= (2 * trim) + 20: return {'error': 'File too short.'}
                
                gsr = pd.to_numeric(df.iloc[trim:-trim, 3], errors='coerce').dropna().values
                ppg = pd.to_numeric(df.iloc[trim:-trim, 5], errors='coerce').dropna().values

                p.set(message="EDA Processing...", value=2)
                eda_sigs, eda_info = nk.eda_process(gsr, sampling_rate=sr)
                eda_pks = nk.signal_findpeaks(eda_sigs['EDA_Phasic'], relative_height_min=input.peak_rel_height())
                
                p.set(message="PPG Processing...", value=3)
                ppg_sigs, ppg_info = nk.ppg_process(ppg, sampling_rate=sr, method='elgendi')

                p.set(message="HRV & Resp Analysis...", value=4)
                hrv = nk.hrv(ppg_info, sampling_rate=sr) if len(ppg_info.get('PPG_Peaks', [])) > 3 else None
                edr = np.array([])
                if 'PPG_Rate' in ppg_sigs.columns:
                    hr_v = ppg_sigs['PPG_Rate'].values
                    if hr_v.size > 0: edr = nk.ecg_rsp(ecg_rate=hr_v, sampling_rate=sr)

                p.set(message="Finalizing...", value=5)
                return {'eda_sigs': eda_sigs, 'eda_info': eda_info, 'eda_pks': eda_pks, 'ppg_sigs': ppg_sigs, 'ppg_info': ppg_info, 'hrv': hrv, 'edr': edr, 'sr': sr, 'raw_gsr': gsr}
            except Exception as e: return {'error': str(e)}

    @render.text
    def eda_metrics():
        res = processed_data()
        if res is None or 'error' in res: return res.get('error', "In attesa di dati...")
        sigs = res['eda_sigs']
        duration_min = len(res['raw_gsr']) / res['sr'] / 60
        n_peaks = len(res['eda_pks']['Peaks'])
        out = f"--- EDA FULL REPORT ---\n"
        out += f"Mean Total GSR: {np.mean(res['raw_gsr']):.3f} ̆S [Min: {np.min(res['raw_gsr']):.3f}, Max: {np.max(res['raw_gsr']):.3f}]\n"
        out += f"Mean Tonic (SCL): {np.mean(sigs['EDA_Tonic']):.3f} ̆S [Min: {np.min(sigs['EDA_Tonic']):.3f}, Max: {np.max(sigs['EDA_Tonic']):.3f}]\n"
        out += f"Mean Phasic (SCR): {np.mean(np.abs(sigs['EDA_Phasic'])):.3f} ̆S\n"
        out += f"Frequency: {n_peaks/duration_min:.2f} SCR/min (Total Peaks: {n_peaks})\n\n"
        if n_peaks > 0:
            top_idx = np.argmax(res['eda_pks']['Height'])
            pk_time = res['eda_pks']['Peaks'][top_idx] / res['sr']
            out += f"Highest Peak: Timestamp {pk_time:.2f}s, Height {res['eda_pks']['Height'][top_idx]:.3f} ̆S\n"
        return out

    @render.plot
    def eda_custom_plot():
        res = processed_data()
        if not res or 'eda_sigs' not in res: return
        time = np.arange(len(res['raw_gsr'])) / res['sr']
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time, res['raw_gsr'], label='Total EDA (Raw)', alpha=0.4, color='gray')
        ax.plot(time, res['eda_sigs']['EDA_Tonic'], label='Tonic Component', linewidth=2, color='blue')
        ax.plot(time, res['eda_sigs']['EDA_Phasic'], label='Phasic Component', alpha=0.8, color='red')
        ax.set_title("EDA Component Decomposition"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Conductance (̆S)")
        ax.legend(); ax.grid(True)
        return fig

    @render.plot
    def eda_plot():
        res = processed_data()
        if not res or 'eda_sigs' not in res: return
        return nk.eda_plot(res['eda_sigs'], res['eda_info'])

    @render.text
    def hr_metrics():
        res = processed_data()
        if not res or 'ppg_sigs' not in res: return ""
        hr = res['ppg_sigs']['PPG_Rate'].values
        return f"Median HR: {np.nanmedian(hr):.2f} BPM\nMin/Max: {np.nanmin(hr):.1f} / {np.nanmax(hr):.1f}"

    @render.plot
    def hr_custom_plot():
        res = processed_data()
        if not res or 'ppg_sigs' not in res: return
        hr = res['ppg_sigs']['PPG_Rate'].values
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(len(hr))/res['sr'], hr)
        ax.set_ylim(max(30, np.nanmin(hr)-5), min(200, np.nanmax(hr)+5))
        ax.set_title("Heart Rate Trend"); ax.grid(True)
        return fig

    @render.plot
    def ppg_plot():
        res = processed_data()
        if not res or 'ppg_sigs' not in res: return
        return nk.ppg_plot(res['ppg_sigs'], res['ppg_info'])

    @render.text
    def hrv_metrics():
        res = processed_data()
        if not res or res.get('hrv') is None: return "Nessun dato HRV."
        h = res['hrv']
        metrics = ['HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_LFn', 'HRV_HFn', 'HRV_LFHF', 'HRV_SampEn']
        return "\n".join([f"{m}: {h[m].iloc[0]:.4f}" for m in metrics if m in h.columns])

    @render.plot
    def hrv_plot():
        res = processed_data()
        if not res or res.get('hrv') is None: return
        # Fix: Capture the figure object specifically
        nk.hrv_nonlinear(res['ppg_info'], sampling_rate=res['sr'], show=True)
        fig = plt.gcf()
        return fig

    @render.text
    def resp_metrics():
        res = processed_data()
        if not res or res['edr'].size == 0: return ""
        peaks, _ = find_peaks(res['edr'], distance=int(res['sr'] * 2))
        br = (len(peaks) / (len(res['edr']) / res['sr'])) * 60 if len(peaks) > 1 else 0
        return f"Estimated Breathing Rate: {br:.2f} breaths/min"

    @render.plot
    def resp_plot():
        res = processed_data()
        if not res or res['edr'].size == 0: return
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(len(res['edr']))/res['sr'], res['edr'])
        ax.set_title("PPG-Derived Respiration (PDR)"); ax.grid(True)
        return fig

app = App(app_ui, server)
