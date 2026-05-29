# Summary of Findings (of Analysis 1)

## Question

> Are latent-state transitions better predicted by reward prediction error than by raw reward-history structure?

See `rpe_hmm/README.md` for details on the analysis pipeline and models used.

## Results

The RPE-based transition model (`M2-Q`) outperforms the raw reward-history transition model (`M1`) in held-out data.

Across the **36** (out of 38) mice that passed the RL-fit inclusion criteria (not hitting parameter boundaries):

- `M2-Q` beat `M1` in **every mouse**.
- Mean improvement: **$+0.2142$ bits/trial** (`M2-Q - M1`): the RPE-based transition model predicts held-out choices better than the raw reward-history transition model.
- 95% bootstrap CI: **$[+0.2018, 0.2305]$** --> entirely above 0: the improvement is consistently positive.
- Wilcoxon signed-rank test: **$p = 2.91 \times 10^{-11}$**.
- Rank-biserial effect size: **$r = 1.0$**: the effect is perfectly consistent in direction.

## Further analyses under development

- **Analysis 2:** transition-weight analysis. This is meant to identify which RPE regressors are associated with switches between engaged and disengaged states.
- **Analysis 3:** individual-differences analysis. This is meant to test whether animal-level learning parameters, especially $\alpha^+$, relate to state stability such as engaged-state dwell time.
- **Analysis 4:** block-boundary analysis. This is meant to test whether block switches produce systematic changes in RPE and in the probability of occupying the engaged state around the boundary.

