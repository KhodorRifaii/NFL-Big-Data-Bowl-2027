# Do Combine route movements carry over to NFL receiver routes?

## Summary

We compared movement recorded during wide receiver Combine route drills with movement and separation on similarly named NFL routes. The data provide a useful exploratory look, but they do not support a reliable claim that Combine movement predicts NFL route performance. One raw association between speed out drill path efficiency and NFL out route separation weakened substantially when players needed more route observations. A raw association between go drill speed and NFL go route speed remained visible, but its adjusted uncertainty includes no relationship. The findings should be treated as leads for further study, not as a scouting rule.

## Question and approach

The NFL Big Data Bowl 2027 asks participants to connect Combine sensor tracking with regular season performance. We focused on drafted wide receivers and paired six route families with NFL route labels, combining the two speed-out drill variants into one group. The primary player outcome was mean separation at pass release on the matching route family. We also compared movement speed measured during Combine drills with speed during matched NFL routes.

Combine movement metrics were summarized across available attempts. For NFL movement, we used player tracking from ball snap through pass forward, summarized within each player-play, then aggregated by player and route. Player associations required at least 10 NFL observations on a route. Adjusted linear models controlled for official 40 yard time, overall draft pick, and draft class; 95% confidence intervals use HC3 robust standard errors. We report Spearman rank correlations and Benjamini-Hochberg q values to show the effect of examining many combinations.

The analysis covered 106 Combine wide receivers. 82 had at least one eligible route record after restricting to the selected regular seasons and rows with measured separation. It used 2023, 2024, and 2025 regular seasons, limited to a player's rookie year and the following season when available. The 2026 partial season was excluded. We matched Combine drill names to NFL routes by route family; this is an approximate mapping, especially where a drill or route can have different geometry.

## Findings

**Speed out path efficiency and separation:** Among 56 receivers, greater path efficiency on the Combine speed out drill had a positive rank association with mean NFL out route separation (Spearman rho = 0.36, unadjusted p = 0.0068). In the adjusted model, a one standard deviation increase in efficiency corresponded to 0.21 yards greater separation (95% CI 0.05 to 0.37). However, when we required at least 25 NFL out route observations per player, the sample fell to 34 and the relationship weakened (rho = 0.13, p = 0.48; adjusted estimate 0.06 yards, 95% CI -0.08 to 0.20). The all-comparison adjusted q value was 0.33. This instability means the apparent relationship is not robust evidence.

**Go drill speed and NFL go route speed:** Among 60 receivers, Combine go route 90th percentile speed had a positive raw rank association with NFL go route speed (rho = 0.35, unadjusted p = 0.0055). After adjustment, the estimate was 0.13 yards per second per standard deviation (95% CI -0.03 to 0.28). The confidence interval includes zero, and the association did not survive correction across the broader exploratory search (q = 0.33). Requiring at least 25 route observations retained a similar raw rank correlation, but the adjusted confidence interval still included zero. The data therefore do not show that this Combine measure adds dependable information beyond the controls.

Across other tested route and movement combinations, no relationship provided consistent evidence after accounting for uncertainty and the number of comparisons. The figure below shows selected raw patterns, which should be read alongside the uncertainty and sensitivity results above.

![Selected exploratory associations](outputs/summary_findings.png)

## Limitations

The player samples are small, and many players have different numbers of route observations. Averaging reduces noise but does not make the averages equally reliable. Players from the 2025 draft class have one observed season, while earlier classes may have two. Missing draft selections were assigned a late-pick value for adjustment, which is a practical modeling choice rather than observed draft information. The Combine metrics are reconstructed from sensor tracks and are not official timings. Route labels do not guarantee identical paths, coverage, or game situations. The design is observational and cannot establish causation. This work has not been validated for player evaluation.

## Next step

For a stronger follow-up, define route geometry from tracking rather than route names alone, quantify each player's uncertainty using play-level resampling, and validate candidate relationships on a later draft class held out from feature selection. Until then, the most defensible result is that this dataset and sample do not yet establish a stable Combine-to-NFL movement relationship.

## Reproducibility

Run `python src/analyze_wr.py` after attaching the Kaggle competition data. The accompanying notebook contains the same analysis pipeline. Generated tables include raw correlations, adjusted estimates, confidence intervals, multiple comparison summaries, and sensitivity checks. Data files are not included in this project.
