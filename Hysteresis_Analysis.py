#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solo plot essenziali per barometri:
1) **Smoothing**: raw vs smoothed + picchi rilevati (valle→picco→valle).
2) **Loading vs Unloading** per canale: metà ascendente (tempo invertito) e metà discendente
   (tempo dal picco), così entrambe partono da x=0 al picco.

Niente CSV/metriche: solo figure.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== CONFIG (edit) ==================
CSV_PATH = r"C:\\Users\\aurir\\OneDrive\\Desktop\\Thesis- Biorobotics Lab\\Sensor-Logs\\datalog_2025-11-06_14-35-45.csv"
PRESSURE_REGEX = r"(p\d+|pressure|baro|barometer)"
SMOOTH_WINDOW = 3          # finestra media mobile (dispari ≥3)
MIN_PROM_PCT = 2.0         # prominenza minima del picco (% dell'intervallo del segnale)
MIN_GAP = 3                # campioni minimi tra valle/ picco/ valle
# ===================================================

# --------- helper ---------
def get_time_seconds(df):
    if 'Time(ms)' in df.columns:
        return df['Time(ms)'].astype(float).to_numpy() / 1000.0
    time_cols = [c for c in df.columns if re.search(r"(time|timestamp|ts)", str(c), re.I)]
    if not time_cols:
        raise ValueError('Colonna tempo non trovata')
    col = time_cols[0]
    s = df[col].astype(str)
    if ':' in str(s.iloc[0]):
        def to_s(x):
            h,m,sx = str(x).split(':'); return float(h)*3600 + float(m)*60 + float(sx)
        return s.map(to_s).to_numpy(float)
    return pd.to_numeric(s, errors='coerce').to_numpy(float)


def smooth_ma(y, w):
    w = max(3, int(w)|1)
    pad = w//2
    return np.convolve(np.pad(y, (pad,pad), mode='edge'), np.ones(w)/w, mode='valid')


def find_cycles(y):
    dy = np.gradient(y)
    sgn = np.sign(dy); sgn[sgn==0]=1
    
    # Find both maxima and minima
    maxima = np.where((sgn[:-1] > 0) & (sgn[1:] < 0))[0] + 1
    minima = np.where((sgn[:-1] < 0) & (sgn[1:] > 0))[0] + 1
    
    # Determine which type of peaks to use (maxima or minima) based on prominence
    rng = np.ptp(y)
    min_prom = rng * (MIN_PROM_PCT/100.0)
    
    # Calculate prominence for both types
    max_prominences = []
    for ip in maxima:
        lv = minima[minima < ip]; rv = minima[minima > ip]
        if len(lv) > 0 and len(rv) > 0:
            base = min(y[lv[-1]], y[rv[0]])
            max_prominences.append(y[ip] - base)
            
    min_prominences = []
    for ip in minima:
        lv = maxima[maxima < ip]; rv = maxima[maxima > ip]
        if len(lv) > 0 and len(rv) > 0:
            base = max(y[lv[-1]], y[rv[0]])
            min_prominences.append(base - y[ip])
    
    # Choose peaks based on which type has larger average prominence
    max_prom_avg = np.mean(max_prominences) if max_prominences else 0
    min_prom_avg = np.mean(min_prominences) if min_prominences else 0
    
    peaks = maxima if max_prom_avg >= min_prom_avg else minima
    valleys = minima if max_prom_avg >= min_prom_avg else maxima
    is_maximum = max_prom_avg >= min_prom_avg
    
    cycles = []
    for ip in peaks:
        lv = valleys[valleys < ip]; rv = valleys[valleys > ip]
        if len(lv)==0 or len(rv)==0: continue
        il = lv[-1]; ir = rv[0]
        if ip-il < MIN_GAP or ir-ip < MIN_GAP: continue
        base = min(y[il], y[ir]) if is_maximum else max(y[il], y[ir])
        prominence = y[ip] - base if is_maximum else base - y[ip]
        if prominence < min_prom: continue
        cycles.append((il, ip, ir))
    return cycles

# ---------------- main ----------------
def main():
    df = pd.read_csv(CSV_PATH)
    t = get_time_seconds(df)
    pres_cols = [c for c in df.columns if re.search(PRESSURE_REGEX, str(c), re.I)]
    if not pres_cols:
        raise ValueError('Nessuna colonna di pressione trovata')

    for col in pres_cols:
        y_raw = pd.to_numeric(df[col], errors='coerce').to_numpy(float)
        y = smooth_ma(y_raw, SMOOTH_WINDOW)
        cycles = find_cycles(y)
        if len(cycles)==0:
            print(f"{col}: nessun ciclo rilevato")
            continue

        # --- Plot 1: smoothing + picchi ---
        fig1, ax1 = plt.subplots(figsize=(11,4))
        ax1.plot(t, y_raw, alpha=0.35, label='raw')
        ax1.plot(t[:len(y)], y, label='smoothed')
        for (il,ip,ir) in cycles:
            ax1.axvline(t[ip], ls='--', alpha=0.25)
            ax1.plot(t[ip], y[ip], '^')
        ax1.set_title(f'{col} — smoothing & peaks')
        ax1.set_xlabel('tempo [s]'); ax1.set_ylabel('pressione')
        ax1.grid(True, alpha=0.3); ax1.legend(); fig1.tight_layout()

        # --- Plot 2: loading vs unloading (overlapped at peak, both start x=0) ---
        fig2, ax2 = plt.subplots(figsize=(7,5))
        for k,(il,ip,ir) in enumerate(cycles, start=1):
            # Loading: valley -> peak. Build time-from-peak so it starts at 0 and increases.
            t_up, y_up = t[il:ip+1], y[il:ip+1]
            x_up = (t_up[-1] - t_up)           # 0 at peak, T_up at valley
            x_up = x_up[::-1]; y_up = y_up[::-1]  # make it 0..T_up (increasing)

            # Unloading: peak -> valley. Time-from-peak naturally starts at 0.
            t_dn, y_dn = t[ip:ir+1], y[ip:ir+1]
            x_dn = (t_dn - t_dn[0])            # 0..T_dn

            # Calculate hysteresis metrics for this cycle
            # Resample both curves to same number of points for accurate area calculation
            n_points = 100
            x_grid = np.linspace(0, min(x_up.max(), x_dn.max()), n_points)
            y_up_resampled = np.interp(x_grid, x_up, y_up)
            y_dn_resampled = np.interp(x_grid, x_dn, y_dn)
            
            # Calculate areas
            area_between = np.trapz(np.abs(y_up_resampled - y_dn_resampled), x_grid)
            area_loading = np.trapz(np.abs(y_up_resampled), x_grid)
            
            # Calculate hysteresis percentage
            hysteresis_pct = 100 * area_between / area_loading if area_loading > 0 else 0
            
            ax2.plot(x_up, y_up, alpha=0.9, 
                    label=f'loading (cycle {k})' if k==1 else None)
            ax2.plot(x_dn, y_dn, '--', alpha=0.9, 
                    label=f'unloading (cycle {k})' if k==1 else None)
            
            print(f"{col} - Cycle {k} Hysteresis: {hysteresis_pct:.1f}%")

        ax2.set_title(f'{col} — overlapped at peak (x=0)')
        ax2.set_xlabel('time from peak [s]')
        ax2.set_ylabel('pressure (smoothed)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        fig2.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()
