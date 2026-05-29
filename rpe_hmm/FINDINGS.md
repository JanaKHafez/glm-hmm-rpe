# Summary of Findings

## Question

> Are latent-state transitions better predicted by reward prediction error than by raw reward-history structure, and what do the fitted state-transition summaries imply about motivational switching?

See `rpe_hmm/README.md` for details on the analysis pipeline and models used.

## Dataset and inclusion

The final run included **36 of 38 mice**. Two mice were excluded because their RL fits hit parameter boundaries, indicating unreliable Q-learning parameter estimates.

## Main findings

## Model performance relative to the `L0` baseline

All fitted transition models improved strongly over the `L0` training-set choice-frequency baseline.

- `M1` Raw Reward: **$+0.3450$ bits/trial**, 95% CI **$[+0.3316, +0.3579]$**
- `M2-Q` RPE: **$+0.5588$ bits/trial**, 95% CI **$[+0.5487, +0.5671]$**
- `M2-Belief`: **$+0.5951$ bits/trial**, 95% CI **$[+0.5891, +0.6009]$**
- `M3-Q` Combined: **$+0.5702$ bits/trial**, 95% CI **$[+0.5616, +0.5776]$**
- `M3-Belief` Combined: **$+0.6008$ bits/trial**, 95% CI **$[+0.5951, +0.6067]$**

Interpretation: all structured transition models beat the simple baseline, and the best-performing models are those that include RPE signals, especially belief-state RPEs.

### RPE-based transitions outperform raw reward-history transitions

The central result is strong and highly consistent across animals.

- `M2-Q` beat `M1` in **every included mouse**.
- Mean held-out improvement: **$+0.2138$ bits/trial** for $LL(M2\text{-}Q) - LL(M1)$.
- 95% bootstrap CI: **$[+0.2027, +0.2264]$**.
- Wilcoxon signed-rank test: **$p = 2.91 \times 10^{-11}$**.
- Rank-biserial effect size: **$r = 1.0$**.

Interpretation: latent-state transitions are predicted substantially better by reward prediction error than by the original raw reward-history transition baseline.

### Belief-state RPE outperforms Q-learning RPE

Among the RPE-based transition models, the belief-state version performs best.

- Mean held-out improvement: **$+0.0363$ bits/trial** for $LL(M2\text{-}Belief) - LL(M2\text{-}Q)$.
- 95% bootstrap CI: **$[+0.0293, +0.0437]$**.
- Wilcoxon signed-rank test: **$p = 2.91 \times 10^{-11}$**.
- Rank-biserial effect size: **$r = 1.0$**.

Interpretation: belief-like reward prediction errors provide an additional predictive gain beyond a simpler Q-learning RPE account.

## State-transition signatures

The event-based transition analysis shows a clear directional asymmetry around motivational state switches.

Across **1990 sessions**:

### Engaged $\rightarrow$ Disengaged

- events: **52,244**
- `cumpos`: **$-0.1364$**, 95% CI **$[-0.1439, -0.1286]$**
- `cumneg`: **$-1.1920$**, 95% CI **$[-1.1982, -1.1850]$**
- `abs_running`: **$+0.4507$**, 95% CI **$[+0.4429, +0.4582]$**
- `current`: **$-1.5907$**, 95% CI **$[-1.5980, -1.5837]$**

### Disengaged $\rightarrow$ Engaged

- events: **52,342**
- `cumpos`: **$+0.3947$**, 95% CI **$[+0.3861, +0.4019]$**
- `cumneg`: **$-0.6805$**, 95% CI **$[-0.6863, -0.6737]$**
- `abs_running`: **$+0.4989$**, 95% CI **$[+0.4914, +0.5063]$**
- `current`: **$+0.9927$**, 95% CI **$[+0.9848, +1.0003]$**

Switches out of engagement are associated with strongly negative current RPE and more negative cumulative recent RPE, whereas switches back into engagement are associated with positive current RPE and more positive cumulative recent RPE. Unsigned RPE magnitude is elevated in both directions, suggesting that state changes are generally associated with surprising outcomes, while the sign of RPE helps determine the direction of the switch.

## Individual differences

The learning-rate analysis revealed a modest but significant relationship between positive learning rate and engaged-state stability.

- Spearman correlation between `alpha_pos` and mean engaged-state dwell time: **$\rho = -0.370$**
- **$p = 0.0263$**
- Variance explained by the rank correlation: **$\rho^2 = 0.137$**
- Mean engaged-state dwell time across mice: **$5.5$ trials**

Mice with larger positive learning rates tend to show shorter engaged-state dwell times, consistent with faster updating being associated with less stable engagement.