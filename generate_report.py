import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json
import os
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ttest_ind, wilcoxon, ranksums, mannwhitneyu, normaltest, levene, kruskal, friedmanchisquare, spearmanr, pearsonr, ks_2samp
import warnings
warnings.filterwarnings("ignore")

# HF Incidence
time_windows = [3, 5, 10]
measures = ['both', 'ecg', 'eeg']
disease = 'stroke'
disease1 = 'stk'
date = True

# Create PDF
pdf = PdfPages(f'age_estimation_report_disease_{disease}.pdf')

# Dictionary to store results for all time windows
all_time_results = {}
all_time_stage_results = {}

for time in time_windows:
    timeLimit = 365 * time
    measure_data = {}

    for measure in measures:
        loaded = np.load(os.path.join(
            'outputs', f'test_outputs_remaining_{measure}.npz'))
        predictions = loaded['predictions']
        ground_truths = loaded['ground_truths']
        subjects = loaded['subjects']
        labels = loaded['labels']
        sleep_stages = loaded['sleep_stages']

        subs = np.unique(subjects)

        cvd_summary = pd.read_csv(os.path.join(
            'preprocess', 'shhs-cvd-summary-dataset-0.19.0.csv'))

        healthy_at_baseline = cvd_summary[cvd_summary['prev_mi'] ==
                                          0][cvd_summary['prev_stk'] == 0][cvd_summary['prev_chf'] == 0]['nsrrid'].values

        healthy_after_baseline = cvd_summary[cvd_summary['nsrrid'].isin(
            healthy_at_baseline)][(cvd_summary['stroke'] == 0)][(cvd_summary['mi'] == 0)][(cvd_summary['chf'] == 0)]['nsrrid'].values

        if date:
            disease_after_baseline = cvd_summary[cvd_summary['nsrrid'].isin(healthy_at_baseline)][(
                cvd_summary[disease] > 0)][(cvd_summary[disease1+'_date'] < timeLimit)]['nsrrid'].values
        else:
            disease_after_baseline = cvd_summary[cvd_summary['nsrrid'].isin(healthy_at_baseline)][(
                cvd_summary[disease] > 0)]['nsrrid'].values

        not_av = [x for x in subs if x not in disease_after_baseline.tolist() +
                  healthy_after_baseline.tolist()]

        remove_idx = [idx for x in not_av for idx in np.where(subjects == x)[
            0].tolist()]

        predictions = np.delete(predictions, remove_idx, axis=0)
        ground_truths = np.delete(ground_truths, remove_idx, axis=0)
        subjects = np.delete(subjects, remove_idx, axis=0)
        labels = np.delete(labels, remove_idx, axis=0)
        sleep_stages = np.delete(sleep_stages, remove_idx, axis=0)

        labels = np.array(
            [0 if x in healthy_after_baseline else 1 if x in disease_after_baseline else 2 for x in subjects])

        subs = np.unique(subjects)
        unique_stages = np.unique(sleep_stages)

        # Calculate differences for each subject
        differences = []
        outcomes = []
        for sub in subs:
            sub_idx = np.where(subjects == sub)[0]
            differences.append(
                ground_truths[sub_idx].mean() - predictions[sub_idx].mean())
            outcomes.append(np.unique(labels[sub_idx])[0])

        differences = np.array(differences)
        outcomes = np.array(outcomes)

        # Calculate statistics
        ctrl_diffs = differences[outcomes == 0]
        pat_diffs = differences[outcomes == 1]

        # Test for normality only if enough samples
        is_normal = False
        if len(ctrl_diffs) >= 8 and len(pat_diffs) >= 8:
            _, ctrl_norm_p = normaltest(ctrl_diffs)
            _, pat_norm_p = normaltest(pat_diffs)
            is_normal = ctrl_norm_p > 0.05 and pat_norm_p > 0.05

        # Statistical tests
        min_len = min(len(ctrl_diffs), len(pat_diffs))
        wilcoxon_stat, wilcoxon_p = wilcoxon(
            ctrl_diffs[:min_len], pat_diffs[:min_len])
        ranksum_stat, ranksum_p = ranksums(ctrl_diffs, pat_diffs)
        mannwhitney_stat, mannwhitney_p = mannwhitneyu(ctrl_diffs, pat_diffs)
        t_stat, t_p = ttest_ind(ctrl_diffs, pat_diffs)

        # Additional statistical tests
        levene_stat, levene_p = levene(
            ctrl_diffs, pat_diffs)  # Test for equal variances
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = ks_2samp(ctrl_diffs, pat_diffs)
        kruskal_stat, kruskal_p = kruskal(
            ctrl_diffs, pat_diffs)  # Kruskal-Wallis H-test
        spearman_corr, spearman_p = spearmanr(
            differences, outcomes)  # Spearman correlation
        pearson_corr, pearson_p = pearsonr(
            differences, outcomes)  # Pearson correlation

        # Store overall results
        if time not in all_time_results:
            all_time_results[time] = {}
            all_time_stage_results[time] = {}

        mean_diff = np.mean(pat_diffs) - np.mean(ctrl_diffs)
        pooled_std = np.sqrt(
            (np.std(ctrl_diffs)**2 + np.std(pat_diffs)**2) / 2)

        # Calculate effect sizes
        cohens_d = mean_diff / pooled_std
        hedges_g = cohens_d * \
            (1 - (3 / (4 * (len(ctrl_diffs) + len(pat_diffs) - 2) - 1)))
        glass_delta = mean_diff / np.std(ctrl_diffs)

        all_time_results[time][measure] = {
            'mean_diff': mean_diff,
            'pooled_std': pooled_std,
            'wilcoxon_p': wilcoxon_p,
            'ranksum_p': ranksum_p,
            'mannwhitney_p': mannwhitney_p,
            't_p': t_p,
            'is_normal': is_normal,
            'levene_p': levene_p,
            'ks_p': ks_p,
            'kruskal_p': kruskal_p,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p,
            'pearson_corr': pearson_corr,
            'pearson_p': pearson_p,
            'cohens_d': cohens_d,
            'hedges_g': hedges_g,
            'glass_delta': glass_delta
        }

        # Calculate statistics by sleep stage
        all_time_stage_results[time][measure] = {}
        for stage in unique_stages:
            stage_mask = sleep_stages == stage
            stage_differences = []
            stage_outcomes = []

            for sub in subs:
                sub_idx = np.where((subjects == sub) & stage_mask)[0]
                if len(sub_idx) > 0:
                    stage_differences.append(
                        ground_truths[sub_idx].mean() - predictions[sub_idx].mean())
                    stage_outcomes.append(np.unique(labels[sub_idx])[0])

            stage_differences = np.array(stage_differences)
            stage_outcomes = np.array(stage_outcomes)

            stage_ctrl_diffs = stage_differences[stage_outcomes == 0]
            stage_pat_diffs = stage_differences[stage_outcomes == 1]

            if len(stage_ctrl_diffs) >= 8 and len(stage_pat_diffs) >= 8:
                _, stage_ctrl_norm_p = normaltest(stage_ctrl_diffs)
                _, stage_pat_norm_p = normaltest(stage_pat_diffs)
                stage_is_normal = stage_ctrl_norm_p > 0.05 and stage_pat_norm_p > 0.05

                stage_min_len = min(len(stage_ctrl_diffs),
                                    len(stage_pat_diffs))
                stage_wilcoxon_stat, stage_wilcoxon_p = wilcoxon(
                    stage_ctrl_diffs[:stage_min_len], stage_pat_diffs[:stage_min_len])
                stage_ranksum_stat, stage_ranksum_p = ranksums(
                    stage_ctrl_diffs, stage_pat_diffs)
                stage_mannwhitney_stat, stage_mannwhitney_p = mannwhitneyu(
                    stage_ctrl_diffs, stage_pat_diffs)
                stage_t_stat, stage_t_p = ttest_ind(
                    stage_ctrl_diffs, stage_pat_diffs)

                # Additional statistical tests for stages
                stage_levene_stat, stage_levene_p = levene(
                    stage_ctrl_diffs, stage_pat_diffs)
                stage_ks_stat, stage_ks_p = ks_2samp(
                    stage_ctrl_diffs, stage_pat_diffs)
                stage_kruskal_stat, stage_kruskal_p = kruskal(
                    stage_ctrl_diffs, stage_pat_diffs)
                stage_spearman_corr, stage_spearman_p = spearmanr(
                    stage_differences, stage_outcomes)
                stage_pearson_corr, stage_pearson_p = pearsonr(
                    stage_differences, stage_outcomes)

                stage_mean_diff = np.mean(
                    stage_pat_diffs) - np.mean(stage_ctrl_diffs)
                stage_pooled_std = np.sqrt(
                    (np.std(stage_ctrl_diffs)**2 + np.std(stage_pat_diffs)**2) / 2)

                # Calculate effect sizes for stages
                stage_cohens_d = stage_mean_diff / stage_pooled_std
                stage_hedges_g = stage_cohens_d * \
                    (1 - (3 / (4 * (len(stage_ctrl_diffs) + len(stage_pat_diffs) - 2) - 1)))
                stage_glass_delta = stage_mean_diff / np.std(stage_ctrl_diffs)

                all_time_stage_results[time][measure][stage] = {
                    'mean_diff': stage_mean_diff,
                    'pooled_std': stage_pooled_std,
                    'wilcoxon_p': stage_wilcoxon_p,
                    'ranksum_p': stage_ranksum_p,
                    'mannwhitney_p': stage_mannwhitney_p,
                    't_p': stage_t_p,
                    'is_normal': stage_is_normal,
                    'levene_p': stage_levene_p,
                    'ks_p': stage_ks_p,
                    'kruskal_p': stage_kruskal_p,
                    'spearman_corr': stage_spearman_corr,
                    'spearman_p': stage_spearman_p,
                    'pearson_corr': stage_pearson_corr,
                    'pearson_p': stage_pearson_p,
                    'cohens_d': stage_cohens_d,
                    'hedges_g': stage_hedges_g,
                    'glass_delta': stage_glass_delta
                }

# Create comparison table
comparison_data = []
for time in time_windows:
    for measure in measures:
        stats = all_time_results[time][measure]
        comparison_data.append({
            'Time Window (years)': time,
            'Measure': measure.upper(),
            'Mean Difference ± SD': f"{stats['mean_diff']:.2f} ± {stats['pooled_std']:.2f}",
            't-test p': f"{stats['t_p']:.4f}",
            'Wilcoxon p': f"{stats['wilcoxon_p']:.4f}",
            'Mann-Whitney p': f"{stats['mannwhitney_p']:.4f}",
            'KS p': f"{stats['ks_p']:.4f}",
            'Kruskal p': f"{stats['kruskal_p']:.4f}",
            "Cohen's d": f"{stats['cohens_d']:.4f}",
            "Hedges' g": f"{stats['hedges_g']:.4f}",
            "Glass' Δ": f"{stats['glass_delta']:.4f}"
        })

comparison_df = pd.DataFrame(comparison_data)

# Create table figure
fig, ax = plt.subplots(figsize=(15, len(comparison_data)*0.4 + 1))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=comparison_df.values,
                 colLabels=comparison_df.columns,
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
plt.title(f"Statistical Comparison Across Time Windows and Measures\nDisease: {disease.upper()}",
          pad=20, fontsize=14)
plt.tight_layout()
pdf.savefig()
plt.close()

# Create heatmap of p-values
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
test_names = ['t-test', 'Wilcoxon',
              'Mann-Whitney', 'KS-test', 'Kruskal', 'Levene']
p_value_keys = ['t_p', 'wilcoxon_p',
                'mannwhitney_p', 'ks_p', 'kruskal_p', 'levene_p']

for idx, (test_name, p_key) in enumerate(zip(test_names, p_value_keys)):
    row = idx // 3
    col = idx % 3
    p_values = np.zeros((len(measures), len(time_windows)))
    for i, measure in enumerate(measures):
        for j, time in enumerate(time_windows):
            p_values[i, j] = all_time_results[time][measure][p_key]

    im = axes[row, col].imshow(np.log10(p_values), cmap='RdYlBu')
    axes[row, col].set_xticks(range(len(time_windows)))
    axes[row, col].set_yticks(range(len(measures)))
    axes[row, col].set_xticklabels(time_windows)
    axes[row, col].set_yticklabels([m.upper() for m in measures])
    axes[row, col].set_title(f'{test_name} log10(p-value)')
    axes[row, col].set_xlabel('Time Window (years)')
    if col == 0:
        axes[row, col].set_ylabel('Measure')

    plt.colorbar(im, ax=axes[row, col])

    for i in range(len(measures)):
        for j in range(len(time_windows)):
            text = f'{p_values[i, j]:.4f}'
            axes[row, col].text(j, i, text, ha='center', va='center')

plt.suptitle(
    f'Statistical Significance Heatmap - {disease.upper()}', fontsize=14)
plt.tight_layout()
pdf.savefig()
plt.close()

# Create sleep stage comparison tables
for time in time_windows:
    stage_comparison_data = []
    for measure in measures:
        for stage in unique_stages:
            if stage in all_time_stage_results[time][measure]:
                stats = all_time_stage_results[time][measure][stage]
                stage_comparison_data.append({
                    'Sleep Stage': stage,
                    'Measure': measure.upper(),
                    'Mean Difference ± SD': f"{stats['mean_diff']:.2f} ± {stats['pooled_std']:.2f}",
                    't-test p': f"{stats['t_p']:.4f}",
                    'Wilcoxon p': f"{stats['wilcoxon_p']:.4f}",
                    'Mann-Whitney p': f"{stats['mannwhitney_p']:.4f}",
                    'KS p': f"{stats['ks_p']:.4f}",
                    'Kruskal p': f"{stats['kruskal_p']:.4f}",
                    "Cohen's d": f"{stats['cohens_d']:.4f}",
                    "Hedges' g": f"{stats['hedges_g']:.4f}",
                    "Glass' Δ": f"{stats['glass_delta']:.4f}"
                })

    if stage_comparison_data:
        stage_comparison_df = pd.DataFrame(stage_comparison_data)

        fig, ax = plt.subplots(
            figsize=(15, len(stage_comparison_data)*0.4 + 1))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=stage_comparison_df.values,
                         colLabels=stage_comparison_df.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        plt.title(f"Sleep Stage Statistical Comparison - Time Window: {time} years\nDisease: {disease.upper()}",
                  pad=20, fontsize=14)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

pdf.close()
