#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Relationships Between Barometer Readings and Forces/Positions

This script explores:
1. Correlation between barometers and forces/positions
2. Pressure patterns for different contact locations
3. Pressure magnitude vs force magnitude
4. Feature importance from Random Forest
5. Mutual information (non-linear relationships)

Author: Analysis Script
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def load_and_prepare_data(data_path):
    """Load data and prepare for analysis."""
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    
    print(f"Loaded {len(df):,} samples")
    print(f"Columns: {df.columns.tolist()}\n")
    
    # Define variables
    barometer_cols = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6']
    position_cols = ['x', 'y']
    force_cols = ['fx', 'fy', 'fz']
    torque_cols = ['tx', 'ty', 'tz']
    
    return df, barometer_cols, position_cols, force_cols, torque_cols


def plot_correlation_heatmap(df, barometer_cols, target_cols, target_type="Forces"):
    """
    Create correlation heatmap between barometers and targets.
    """
    print(f"\n{'='*70}")
    print(f"CORRELATION ANALYSIS: Barometers vs {target_type}")
    print("="*70)
    
    # Calculate correlations
    corr_data = []
    for baro in barometer_cols:
        for target in target_cols:
            pearson_r, _ = pearsonr(df[baro], df[target])
            spearman_r, _ = spearmanr(df[baro], df[target])
            corr_data.append({
                'Barometer': baro,
                'Target': target,
                'Pearson': pearson_r,
                'Spearman': spearman_r
            })
    
    corr_df = pd.DataFrame(corr_data)
    
    # Print strongest correlations
    print(f"\nStrongest Pearson Correlations (|r| > 0.3):")
    strong = corr_df[abs(corr_df['Pearson']) > 0.3].sort_values('Pearson', ascending=False)
    if len(strong) > 0:
        for _, row in strong.iterrows():
            print(f"  {row['Barometer']} ↔ {row['Target']}: r = {row['Pearson']:.3f}")
    else:
        print("  ⚠️  No strong linear correlations found (all |r| < 0.3)")
    
    # Create pivot tables for heatmaps
    pearson_pivot = corr_df.pivot(index='Barometer', columns='Target', values='Pearson')
    spearman_pivot = corr_df.pivot(index='Barometer', columns='Target', values='Spearman')
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pearson
    sns.heatmap(pearson_pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax1, cbar_kws={'label': 'Correlation'})
    ax1.set_title(f'Pearson Correlation\n(Linear Relationship)', fontweight='bold')
    ax1.set_xlabel('Target Variable')
    ax1.set_ylabel('Barometer')
    
    # Spearman
    sns.heatmap(spearman_pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax2, cbar_kws={'label': 'Correlation'})
    ax2.set_title(f'Spearman Correlation\n(Monotonic Relationship)', fontweight='bold')
    ax2.set_xlabel('Target Variable')
    ax2.set_ylabel('Barometer')
    
    plt.tight_layout()
    
    return fig, corr_df


def plot_pressure_by_location(df, barometer_cols, position_cols, n_bins=4):
    """
    Visualize how pressure patterns change with contact location.
    This shows if position is encoded in pressure distribution.
    """
    print(f"\n{'='*70}")
    print("PRESSURE PATTERNS BY CONTACT LOCATION")
    print("="*70)
    
    # Create spatial bins
    df_temp = df.copy()
    df_temp['x_bin'] = pd.cut(df_temp[position_cols[0]], bins=n_bins, labels=False)
    df_temp['y_bin'] = pd.cut(df_temp[position_cols[1]], bins=n_bins, labels=False)
    
    # Get bin edges for labeling
    x_edges = pd.cut(df_temp[position_cols[0]], bins=n_bins, retbins=True)[1]
    y_edges = pd.cut(df_temp[position_cols[1]], bins=n_bins, retbins=True)[1]
    
    fig, axes = plt.subplots(n_bins, n_bins, figsize=(16, 16))
    
    max_pressure = df[barometer_cols].max().max()
    min_pressure = df[barometer_cols].min().min()
    
    for xi in range(n_bins):
        for yi in range(n_bins):
            ax = axes[yi, xi]
            
            # Get samples in this spatial bin
            mask = (df_temp['x_bin'] == xi) & (df_temp['y_bin'] == yi)
            samples = df_temp[mask]
            
            if len(samples) >= 10:
                # Average pressure pattern
                avg_pressure = samples[barometer_cols].mean()
                std_pressure = samples[barometer_cols].std()
                
                # Bar plot
                x_pos = np.arange(1, 7)
                ax.bar(x_pos, avg_pressure, yerr=std_pressure, alpha=0.7, 
                      color='steelblue', capsize=5)
                
                # Labels
                x_range = f"[{x_edges[xi]:.1f}, {x_edges[xi+1]:.1f}]"
                y_range = f"[{y_edges[yi]:.1f}, {y_edges[yi+1]:.1f}]"
                ax.set_title(f'x={x_range}mm\ny={y_range}mm\nn={len(samples)}',
                           fontsize=9)
                ax.set_ylim([min_pressure - 0.1, max_pressure + 0.1])
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f'b{i}' for i in range(1, 7)], fontsize=8)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add mean total pressure as text
                total_p = avg_pressure.sum()
                ax.text(0.95, 0.95, f'Σ={total_p:.2f}', 
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, f'No data\n(n={len(samples)})', 
                       ha='center', va='center', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            
            if yi == n_bins - 1:
                ax.set_xlabel('Barometer', fontsize=9)
            if xi == 0:
                ax.set_ylabel('Pressure (hPa)', fontsize=9)
    
    fig.suptitle('Average Pressure Patterns by Contact Location\n' + 
                 'Different patterns → Position IS encoded in pressure ✓',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Analysis
    print("\nInterpretation Guide:")
    print("  ✓ If you see DIFFERENT bar heights/patterns in different subplots:")
    print("    → Position IS encoded in pressure distribution")
    print("    → ML models CAN learn contact location from barometers")
    print("\n  ✗ If all patterns look SIMILAR:")
    print("    → Position NOT well encoded")
    print("    → Need better sensor placement or more sensors")
    
    return fig


def plot_pressure_vs_force_magnitude(df, barometer_cols, force_cols):
    """
    Analyze relationship between total pressure and force magnitude.
    """
    print(f"\n{'='*70}")
    print("PRESSURE MAGNITUDE vs FORCE MAGNITUDE")
    print("="*70)
    
    # Calculate total pressure and force magnitude
    df_temp = df.copy()
    df_temp['total_pressure'] = df[barometer_cols].sum(axis=1)
    df_temp['force_magnitude'] = np.sqrt((df[force_cols]**2).sum(axis=1))
    
    # Calculate correlation
    corr, _ = pearsonr(df_temp['total_pressure'], df_temp['force_magnitude'])
    print(f"\nCorrelation between total pressure and force magnitude: r = {corr:.3f}")
    
    if corr > 0.7:
        print("  ✓ STRONG positive correlation - Forces can be predicted from pressure!")
    elif corr > 0.4:
        print("  ⚠️  MODERATE correlation - Some relationship exists")
    else:
        print("  ✗ WEAK correlation - Difficult to predict forces from pressure alone")
    
    # Create figure
    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, figure=fig)
    
    # Scatter plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(df_temp['total_pressure'], df_temp['force_magnitude'], 
               alpha=0.3, s=10, c=df['fz'], cmap='viridis')
    ax1.set_xlabel('Total Pressure (sum of b1-b6) [hPa]', fontsize=10)
    ax1.set_ylabel('Force Magnitude (√(fx²+fy²+fz²)) [N]', fontsize=10)
    ax1.set_title(f'Pressure vs Force Magnitude\nPearson r = {corr:.3f}', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df_temp['total_pressure'], df_temp['force_magnitude'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_temp['total_pressure'].min(), df_temp['total_pressure'].max(), 100)
    ax1.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Linear fit: y={z[0]:.2f}x+{z[1]:.2f}')
    ax1.legend()
    
    # Hexbin plot (for density)
    ax2 = fig.add_subplot(gs[0, 1])
    hb = ax2.hexbin(df_temp['total_pressure'], df_temp['force_magnitude'], 
                    gridsize=30, cmap='YlOrRd', mincnt=1)
    ax2.set_xlabel('Total Pressure [hPa]', fontsize=10)
    ax2.set_ylabel('Force Magnitude [N]', fontsize=10)
    ax2.set_title('Density Plot', fontweight='bold')
    plt.colorbar(hb, ax=ax2, label='Count')
    
    # Individual force components vs total pressure
    ax3 = fig.add_subplot(gs[0, 2])
    for force in force_cols:
        corr_component, _ = pearsonr(df_temp['total_pressure'], df[force])
        ax3.scatter(df_temp['total_pressure'], df[force], 
                   alpha=0.3, s=10, label=f'{force} (r={corr_component:.2f})')
    ax3.set_xlabel('Total Pressure [hPa]', fontsize=10)
    ax3.set_ylabel('Force Components [N]', fontsize=10)
    ax3.set_title('Individual Force Components', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def analyze_mutual_information(df, barometer_cols, target_cols, target_type="Target"):
    """
    Calculate mutual information (captures non-linear relationships).
    """
    print(f"\n{'='*70}")
    print(f"MUTUAL INFORMATION ANALYSIS: Barometers vs {target_type}")
    print("="*70)
    print("(Captures both linear AND non-linear relationships)\n")
    
    X = df[barometer_cols].values
    
    results = []
    for target in target_cols:
        y = df[target].values
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        for baro, mi in zip(barometer_cols, mi_scores):
            results.append({
                'Barometer': baro,
                'Target': target,
                'MI_Score': mi
            })
        
        # Print summary for this target
        total_mi = mi_scores.sum()
        print(f"{target}:")
        print(f"  Total MI: {total_mi:.4f}")
        top_baros = sorted(zip(barometer_cols, mi_scores), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top contributors: {', '.join([f'{b}({mi:.3f})' for b, mi in top_baros])}")
    
    # Create pivot and plot
    results_df = pd.DataFrame(results)
    pivot = results_df.pivot(index='Barometer', columns='Target', values='MI_Score')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, 
                cbar_kws={'label': 'Mutual Information'})
    ax.set_title(f'Mutual Information: Barometers vs {target_type}\n' + 
                 '(Higher = Stronger Relationship, includes non-linear)',
                 fontweight='bold')
    ax.set_xlabel('Target Variable')
    ax.set_ylabel('Barometer')
    
    plt.tight_layout()
    
    print("\nInterpretation:")
    print("  • MI > 0.1: Strong relationship (useful for prediction)")
    print("  • MI < 0.05: Weak relationship (limited predictive power)")
    
    return fig, results_df


def analyze_feature_importance(df, barometer_cols, target_cols, target_type="Target"):
    """
    Use Random Forest to determine feature importance.
    """
    print(f"\n{'='*70}")
    print(f"RANDOM FOREST FEATURE IMPORTANCE: Barometers vs {target_type}")
    print("="*70)
    print("(Shows which barometers are most useful for prediction)\n")
    
    X = df[barometer_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    importance_data = []
    
    for target in target_cols:
        y = df[target].values
        
        # Train RF
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
        rf.fit(X_scaled, y)
        
        # Get importances
        importances = rf.feature_importances_
        
        for baro, imp in zip(barometer_cols, importances):
            importance_data.append({
                'Barometer': baro,
                'Target': target,
                'Importance': imp
            })
        
        # Print summary
        r2 = rf.score(X_scaled, y)
        print(f"{target}:")
        print(f"  R² score: {r2:.4f}")
        top_features = sorted(zip(barometer_cols, importances), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Most important: {', '.join([f'{b}({imp:.3f})' for b, imp in top_features])}")
    
    # Create pivot and plot
    importance_df = pd.DataFrame(importance_data)
    pivot = importance_df.pivot(index='Barometer', columns='Target', values='Importance')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='Greens', ax=ax,
                cbar_kws={'label': 'Feature Importance'})
    ax.set_title(f'Random Forest Feature Importance: Barometers vs {target_type}\n' + 
                 '(Higher = More useful for prediction)',
                 fontweight='bold')
    ax.set_xlabel('Target Variable')
    ax.set_ylabel('Barometer')
    
    plt.tight_layout()
    
    return fig, importance_df


def plot_pairwise_relationships(df, barometer_cols, target_cols):
    """
    Create pairplot showing relationships between key variables.
    """
    print(f"\n{'='*70}")
    print("GENERATING PAIRWISE RELATIONSHIP PLOTS")
    print("="*70)
    
    # Select subset for visualization (sample if too large)
    if len(df) > 2000:
        df_sample = df.sample(n=2000, random_state=42)
        print(f"Sampling 2000 points for visualization (from {len(df)} total)")
    else:
        df_sample = df
    
    # Select key variables
    plot_cols = ['b1', 'b3', 'b6', target_cols[0], target_cols[1]]  # Representative barometers + 2 targets
    
    # Create pairplot
    fig = plt.figure(figsize=(14, 14))
    
    # Use seaborn pairplot
    g = sns.pairplot(df_sample[plot_cols], diag_kind='hist', plot_kws={'alpha': 0.3, 's': 10},
                     diag_kws={'bins': 30})
    g.fig.suptitle('Pairwise Relationships: Barometers and Targets', 
                   y=1.01, fontsize=14, fontweight='bold')
    
    return g.fig


def generate_summary_report(corr_df, mi_df, importance_df, target_type="Forces/Positions"):
    """
    Generate text summary of findings.
    """
    print(f"\n{'='*70}")
    print(f"SUMMARY REPORT: Predictability of {target_type} from Barometers")
    print("="*70)
    
    print("\n📊 KEY FINDINGS:\n")
    
    # 1. Correlation analysis
    strong_corr = corr_df[abs(corr_df['Pearson']) > 0.5]
    moderate_corr = corr_df[(abs(corr_df['Pearson']) > 0.3) & (abs(corr_df['Pearson']) <= 0.5)]
    
    print("1. LINEAR CORRELATIONS:")
    if len(strong_corr) > 0:
        print(f"   ✓ {len(strong_corr)} STRONG correlations found (|r| > 0.5)")
        print("   → Linear models (Linear Regression) should work well")
    elif len(moderate_corr) > 0:
        print(f"   ⚠️  {len(moderate_corr)} MODERATE correlations found (0.3 < |r| < 0.5)")
        print("   → Non-linear models (RF, NN) recommended")
    else:
        print("   ✗ NO strong linear correlations found")
        print("   → Non-linear relationships may exist - try RF or Neural Networks")
    
    # 2. Mutual Information
    print("\n2. MUTUAL INFORMATION (Non-linear relationships):")
    avg_mi_by_target = mi_df.groupby('Target')['MI_Score'].mean().sort_values(ascending=False)
    best_target = avg_mi_by_target.index[0]
    best_mi = avg_mi_by_target.values[0]
    
    if best_mi > 0.15:
        print(f"   ✓ STRONG predictive signal detected")
        print(f"   → Best target: {best_target} (avg MI = {best_mi:.3f})")
        print("   → Machine learning models should perform well")
    elif best_mi > 0.08:
        print(f"   ⚠️  MODERATE predictive signal")
        print(f"   → Best target: {best_target} (avg MI = {best_mi:.3f})")
        print("   → ML possible but may need feature engineering")
    else:
        print(f"   ✗ WEAK predictive signal")
        print("   → Target prediction will be challenging")
        print("   → Consider: 1) More sensors, 2) Better placement, 3) Feature engineering")
    
    # 3. Feature Importance
    print("\n3. FEATURE IMPORTANCE:")
    avg_importance = importance_df.groupby('Barometer')['Importance'].mean().sort_values(ascending=False)
    print(f"   Most informative barometers:")
    for baro, imp in avg_importance.head(3).items():
        print(f"     • {baro}: {imp:.3f}")
    
    # 4. Overall recommendation
    print("\n4. OVERALL RECOMMENDATION:")
    
    avg_corr = abs(corr_df['Pearson']).mean()
    avg_mi = mi_df['MI_Score'].mean()
    avg_imp = importance_df['Importance'].mean()
    
    score = (avg_corr * 0.3 + avg_mi * 0.4 + avg_imp * 0.3)
    
    if score > 0.15:
        print("   ✓✓✓ EXCELLENT - Prediction should work well!")
        print("   Recommended models: Random Forest, Neural Network, XGBoost")
        print("   Expected accuracy: Good to Excellent")
    elif score > 0.10:
        print("   ✓✓ GOOD - Prediction is feasible")
        print("   Recommended models: Random Forest with feature engineering, Neural Network")
        print("   Expected accuracy: Moderate to Good")
    elif score > 0.05:
        print("   ✓ FAIR - Prediction possible but challenging")
        print("   Recommended: Feature engineering + Neural Network")
        print("   Expected accuracy: Fair")
    else:
        print("   ✗ POOR - Prediction will be very difficult")
        print("   Recommended: 1) Add more sensors, 2) Improve sensor placement")
        print("   Expected accuracy: Poor")
    
    print(f"\n   Overall predictability score: {score:.3f} / 0.30")
    print("="*70)


def main():
    """
    Main analysis pipeline.
    """
    # Configuration
    DATA_DIRECTORY = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data"
    CSV_FILENAME = r"test 404 - sensor v4\synchronized_events_404.csv"
    OUTPUT_DIR = os.path.join(DATA_DIRECTORY, "relationship_analysis")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    data_path = os.path.join(DATA_DIRECTORY, CSV_FILENAME)
    df, barometer_cols, position_cols, force_cols, torque_cols = load_and_prepare_data(data_path)
    
    # Store all figures
    figures = {}
    
    # 1. POSITION ANALYSIS
    print("\n" + "🎯"*35)
    print("PART 1: BAROMETERS vs POSITION (x, y)")
    print("🎯"*35)
    
    fig1, corr_pos = plot_correlation_heatmap(df, barometer_cols, position_cols, "Position")
    figures['correlation_position'] = fig1
    
    fig2 = plot_pressure_by_location(df, barometer_cols, position_cols, n_bins=4)
    figures['pressure_patterns'] = fig2
    
    fig3, mi_pos = analyze_mutual_information(df, barometer_cols, position_cols, "Position")
    figures['mutual_info_position'] = fig3
    
    fig4, imp_pos = analyze_feature_importance(df, barometer_cols, position_cols, "Position")
    figures['feature_importance_position'] = fig4
    
    # 2. FORCE ANALYSIS
    print("\n" + "💪"*35)
    print("PART 2: BAROMETERS vs FORCES (fx, fy, fz)")
    print("💪"*35)
    
    fig5, corr_force = plot_correlation_heatmap(df, barometer_cols, force_cols, "Forces")
    figures['correlation_forces'] = fig5
    
    fig6 = plot_pressure_vs_force_magnitude(df, barometer_cols, force_cols)
    figures['pressure_vs_force_magnitude'] = fig6
    
    fig7, mi_force = analyze_mutual_information(df, barometer_cols, force_cols, "Forces")
    figures['mutual_info_forces'] = fig7
    
    fig8, imp_force = analyze_feature_importance(df, barometer_cols, force_cols, "Forces")
    figures['feature_importance_forces'] = fig8
    
    # 3. COMBINED ANALYSIS
    print("\n" + "📈"*35)
    print("PART 3: COMBINED POSITION + FORCE ANALYSIS")
    print("📈"*35)
    
    all_targets = position_cols + force_cols
    
    fig9, corr_all = plot_correlation_heatmap(df, barometer_cols, all_targets, "All Targets")
    figures['correlation_all'] = fig9
    
    # 4. PAIRWISE RELATIONSHIPS
    fig10 = plot_pairwise_relationships(df, barometer_cols, all_targets[:2])
    figures['pairplot'] = fig10
    
    # 5. GENERATE SUMMARY REPORTS
    print("\n" + "📋"*35)
    print("GENERATING SUMMARY REPORTS")
    print("📋"*35)
    
    # Combine all analysis data
    all_corr = pd.concat([corr_pos, corr_force])
    all_mi = pd.concat([mi_pos, mi_force])
    all_imp = pd.concat([imp_pos, imp_force])
    
    generate_summary_report(all_corr, all_mi, all_imp, "Position and Forces")
    
    # 6. SAVE ALL FIGURES
    print(f"\n{'='*70}")
    print("SAVING FIGURES")
    print("="*70)
    
    for name, fig in figures.items():
        save_path = os.path.join(OUTPUT_DIR, f'{name}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {name}.png")
    
    plt.close('all')
    
    # 7. SAVE DATA SUMMARY
    summary_path = os.path.join(OUTPUT_DIR, 'analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BAROMETER-TARGET RELATIONSHIP ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("POSITION CORRELATIONS (Top 5):\n")
        top_pos = corr_pos.nlargest(5, 'Pearson')
        for _, row in top_pos.iterrows():
            f.write(f"  {row['Barometer']} -> {row['Target']}: r={row['Pearson']:.3f}\n")
        
        f.write("\nFORCE CORRELATIONS (Top 5):\n")
        top_force = corr_force.nlargest(5, 'Pearson')
        for _, row in top_force.iterrows():
            f.write(f"  {row['Barometer']} -> {row['Target']}: r={row['Pearson']:.3f}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"  ✓ Saved: analysis_summary.txt")
    print(f"\n✅ Analysis complete! All results saved to:\n   {OUTPUT_DIR}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
    print("\n🎉 Done! Review the generated plots and summary to understand your data relationships.\n")