# ML Odds Module Study v1

Compared site-facing moneyline probability modules on the walk-forward archive only.

Primary variants:
- cap14_mu_sigma
- beta_cap14
- meta_small_v1
- phase_logistic_cap14
- phase_beta_cap14
- raw_mu_sigma

Best practical result: meta_small_v1

Why:
- best pooled log loss and Brier
- best Dec 15+ proper scores
- uses only mu, sigma_cap14, z_cap14, abs(mu), and season phase
- beats cap14_mu_sigma consistently across evaluation seasons
