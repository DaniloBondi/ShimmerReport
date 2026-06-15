import numpy as np
import pandas as pd
import neurokit2 as nk
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

# ==========================================
# 1. CARICAMENTO E PREPARAZIONE DATI
# ==========================================

# Assumendo che il file SHIMMER si chiami "FILE.csv"
df = pd.read_csv("FILE.csv", sep='\t', skiprows=2)

sampling_rate = 51.2
# Rimuovi i primi e gli ultimi 10 secondi per evitare artefatti di stabilizzazione
samples_to_remove = int(10 * sampling_rate)


# ==========================================
# 2. ANALISI EDA (Electrodermal Activity)
# ==========================================

gsr_signal = df.iloc[:, 3].astype(float).values
processed_gsr_signal = gsr_signal[samples_to_remove : -samples_to_remove]

# Processa l'EDA con NeuroKit2
eda_signals, eda_info = nk.eda_process(processed_gsr_signal, sampling_rate=sampling_rate)

# Estrazione delle metriche EDA
n_peaks = len(eda_info.get("SCR_Peaks", []))
mean_amplitude = np.nanmean(eda_info.get("SCR_Amplitude", [0]))
mean_risetime = np.nanmean(eda_info.get("SCR_RiseTime", [0]))
mean_recovery = np.nanmean(eda_info.get("SCR_RecoveryTime", [0]))

# NUOVO: Calcolo SCL (Skin Conductance Level - Media della componente Tonica)
mean_scl = np.mean(eda_signals['EDA_Tonic'])

# NUOVO: Calcolo SCR/min (Frequenza dei picchi fasici al minuto)
duration_minutes_eda = len(processed_gsr_signal) / sampling_rate / 60
scr_per_min = n_peaks / duration_minutes_eda if duration_minutes_eda > 0 else 0

print("--- METRICHE EDA ---")
print(f"Livello Tonico Medio (SCL): {mean_scl:.3f} μS")
print(f"Frequenza Picchi (SCR/min): {scr_per_min:.2f} picchi/min")
print(f"Numero totale picchi SCR: {n_peaks}")
print(f"Ampiezza media SCR: {mean_amplitude:.3f} μS")
print(f"Rise Time medio: {mean_risetime:.3f} s")
print(f"Half-Recovery Time medio: {mean_recovery:.3f} s\n")

# Plot nativo EDA di NeuroKit2
nk.eda_plot(eda_signals, eda_info)
plt.subplots_adjust(hspace=0.5)
plt.show()


# ==========================================
# 3. ANALISI PPG E FREQUENZA CARDIACA
# ==========================================

ppg_signal = df.iloc[:, 5].astype(float).values
processed_ppg_signal = ppg_signal[samples_to_remove : -samples_to_remove]

# NUOVO: Verifica della qualità del segnale PPG
ppg_quality = nk.ppg_quality(processed_ppg_signal, sampling_rate=sampling_rate)
mean_ppg_quality = np.mean(ppg_quality)

# Processa il PPG
ppg_signals, ppg_info = nk.ppg_process(
    processed_ppg_signal, 
    sampling_rate=sampling_rate, 
    method='elgendi'
)

# Plot nativo PPG di NeuroKit2
nk.ppg_plot(ppg_signals, ppg_info)
plt.subplots_adjust(hspace=0.5)
plt.show()

# Statistiche Heart Rate (HR)
heart_rate = ppg_signals['PPG_Rate']
print("--- METRICHE HEART RATE E QUALITÀ ---")
print(f"Qualità media del segnale PPG (0-1): {mean_ppg_quality:.3f}")
print(f"HR Minimo: {np.min(heart_rate):.2f} bpm")
print(f"HR Massimo: {np.max(heart_rate):.2f} bpm")
print(f"HR Medio: {np.mean(heart_rate):.2f} bpm\n")


# ==========================================
# 4. ANALISI HRV (Heart Rate Variability)
# ==========================================

# Calcola l'HRV usando i picchi validati
hrv_indices = nk.hrv(ppg_info, sampling_rate=sampling_rate, show=False)

print("--- METRICHE HRV PRINCIPALI ---")
metrics_to_print = ['HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_LFn', 'HRV_HFn', 'HRV_LFHF', 'HRV_SampEn']
for metric in metrics_to_print:
    if metric in hrv_indices.columns:
        print(f"{metric}: {hrv_indices[metric].values[0]:.4f}")


# ==========================================
# 5. RESPIRAZIONE DERIVATA DA PPG (PDR / EDR)
# ==========================================

# Estrae il pattern respiratorio basandosi sull'Aritmia Sinusale Respiratoria (RSA)
edr = nk.ecg_rsp(ecg_rate=heart_rate.values, sampling_rate=sampling_rate)

# NUOVO: Trova i picchi respiratori imponendo una distanza minima di 2 secondi
# (int(sampling_rate * 2) previene artefatti ad alta frequenza riconosciuti come respiri)
rsp_peaks, _ = find_peaks(edr, distance=int(sampling_rate * 2))

# NUOVO: Calcolo del Breathing Rate basato sugli intervalli Breath-to-Breath
if len(rsp_peaks) > 1:
    rsp_intervals = np.diff(rsp_peaks) / sampling_rate
    breathing_rate = 60 / np.mean(rsp_intervals)
else:
    breathing_rate = 0.0

print(f"\n--- METRICHE RESPIRATORIE ---")
print(f"Breathing Rate Stimato: {breathing_rate:.2f} bpm (atti al minuto)")

plt.figure(figsize=(15, 5))
plt.plot(time_normalized, edr)
plt.xlabel("Time (s)")
plt.ylabel("Respiration")
plt.title("PPG-derived respiration over time")
plt.grid(True)
plt.show()
