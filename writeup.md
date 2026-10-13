# Does Combine go-drill speed carry over to NFL routes?

*A rookie route-speed baseline, and a limit on what it can tell us about separation*

## The football question

The 40-yard dash measures straight-line speed. The Combine go-route drill measures speed while a receiver runs a football route. If the added tracking information matters, it should first help estimate how fast that receiver runs NFL go routes. Whether he gets open is a separate question: separation also depends on coverage, route design, quarterback timing, and other players.

We tested both steps for wide receivers in the [NFL Big Data Bowl 2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027). The result is a possible **movement baseline**, with no meaningful evidence that this one sensor measure predicts separation.

## Data and method

We linked 10 Hz Combine sensor tracking from the `GO_ROUTE_RIGHT` drill to regular-season NFL tracking on `GO` routes in each player's **rookie season**. Of 106 wide receivers with Combine records, 54 had the drill and at least 10 rookie go routes with a valid tracking window and 10 of those routes with recorded separation. The sample includes 22 players drafted in 2023, 18 in 2024, and 14 in 2025.

For each Combine attempt, we measured the 90th percentile sensor speed and took the median across attempts. For each NFL go route, we measured the receiver's 90th percentile speed from ball snap to pass release. We then averaged each player's measured route speed and separation at pass release. All speeds are in yards per second; separation is in yards.

We compared two simple linear models. The benchmark used official 40-yard time and overall draft pick. The second model added Combine go-drill speed. We held out one entire draft class at a time, fit on the other two, and predicted the omitted class. Training-class data alone determined the 40-time imputation and feature scaling. Four players lacked a 40 time, and five lacked a draft pick; the latter received a late-pick value of 260. Draft pick makes this a **post-draft development** comparison, not a tool ready for use before the draft.

## What transferred

The raw player-level correlation between Combine drill speed and rookie NFL go-route speed was **ρ = 0.45**. In the 2025 class, which was predicted using the 2023 and 2024 classes, adding sensor speed lowered route-speed root mean squared error (RMSE) from **0.433 to 0.354 yd/s** across 14 players. Across all three held-out classes, RMSE fell from **0.408 to 0.358 yd/s**, an improvement of about 12%. The 2024 class did not improve: its RMSE rose slightly from 0.353 to 0.358 yd/s.

That improvement remains uncertain. A paired bootstrap of the held-class predictions placed the pooled RMSE change from **-0.113 to +0.015 yd/s**; a negative change means the sensor model performed better. Because this interval includes zero, these data do not establish a dependable gain. The bootstrap keeps the fitted models fixed, so it does not capture all uncertainty from selecting and fitting a model on small draft classes.

For **separation**, the same sensor feature added almost nothing: pooled RMSE changed from **0.436 to 0.434 yd**, with a paired interval for the change from **-0.018 to +0.014 yd**. The Combine drill may reflect usable running speed, but this sensor measure did not meaningfully improve our prediction of separation at pass release.

![Combine go-drill speed, held-class rookie go-route speed errors, and separation errors](figures/go_route_transfer.png)

## Football context and practical use

The raw movement relationship appears in both coverage groups: among primary-sample players with at least five routes in that group, Spearman ρ was **0.45 for man coverage** (47 players) and **0.40 for zone coverage** (54 players). Those are descriptive checks, not proof that the relation is independent of coverage or play design.

A team could pilot this as a weekly **speed translation check** after drafting a receiver. Once enough go routes have been tracked, compare his cumulative observed route speed with the model's rough expectation from his Combine go-drill speed, 40 time, and draft position. The notebook produces a retrospective table for the 2025 class after each player's tenth tracked go route; those players were selected using full-season data. A notable gap would prompt video review of route depth, coverage, alignment, target timing, and assignment. It would not by itself imply a coaching problem, a player limitation, or poor separation skill. The current model is too uncertain to set a player-specific pass/fail threshold.

## Limits and next validation

The sample is small and includes only rookies with enough recorded go routes, so it may exclude players who seldom get opportunities. A `GO` label does not guarantee identical route geometry, and pass release may occur before a route is complete. Coverage, offensive scheme, game situation, and quarterback behavior can change route speed and separation. The feature was selected after a broader exploratory search, so the draft-class checks are useful stress tests but not untouched external validation. None of the associations establishes causation.

The next test is to lock this one feature and model, then evaluate it on a later draft class without revising the definition. A stronger version would also compare matched route depth and coverage at the play level, and attach uncertainty to each player's weekly speed estimate.

The full [Jupyter notebook](https://github.com/KhodorRifaii/NFL-Big-Data-Bowl-2027/blob/main/notebooks/go_route_transfer.ipynb) contains the feature construction, tracking joins, class-held-out predictions, audit counts, and figure code. The competition data are attached at runtime and are not redistributed here.
