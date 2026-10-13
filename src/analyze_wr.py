"""Study whether Combine movement transfers to NFL receiver routes.

The primary route family is the slant: it has broad Combine coverage and an
unambiguous player-play label. Other position-specific drills are secondary,
exploratory comparisons. Game tracking is streamed from the Kaggle ZIP so the
three large tracking CSVs do not need to be unpacked to disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[1]
else:
    ROOT = Path.cwd()
    if not (ROOT / "data/raw").is_dir() and (ROOT.parent / "data/raw").is_dir():
        ROOT = ROOT.parent


def find_data_dir() -> Path:
    configured = os.environ.get("BDB_DATA_DIR")
    candidates = [Path(configured)] if configured else []
    candidates += [ROOT / "data/raw", Path("/kaggle/input/nfl-big-data-bowl-2027"),
                   Path("/kaggle/input/competitions/nfl-big-data-bowl-2027")]
    for candidate in candidates:
        if (candidate / "players.csv").is_file() and (candidate / "combine_tracking.csv").is_file():
            return candidate
        nested = candidate / "nfl-big-data-bowl-2027"
        if (nested / "players.csv").is_file() and (nested / "combine_tracking.csv").is_file():
            return nested
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        for player_file in kaggle_root.rglob("players.csv"):
            if (player_file.parent / "combine_tracking.csv").is_file():
                return player_file.parent
    return ROOT / "data/raw"


DATA = find_data_dir()
ARCHIVE = Path(os.environ.get("BDB_ARCHIVE", Path.home() / "Downloads/nfl-big-data-bowl-2027.zip"))
OUT = Path(os.environ.get("BDB_OUTPUT_DIR", ROOT / "outputs"))
ROUTE_PAIRS = {
    "SLANT_ROUTE_LEFT": ["SLANT"],
    "GO_ROUTE_RIGHT": ["GO"],
    "CURL_ROUTE_RIGHT": ["HITCH"],
    "COMEBACK_ROUTE_RIGHT": ["HITCH"],
    "POST_CORNER_ROUTE_RIGHT": ["POST", "CORNER"],
    "SPEED_OUT": ["OUT"],
}
DRILL_ALIASES = {"SPEED_OUT": ["SPEED_OUT_ROUTE_LEFT", "SPEED_OUT_LEFT"]}
TRACK_COLS = ["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "a", "event"]


def standardize(s: pd.Series) -> pd.Series:
    scale = s.std(ddof=0)
    return (s - s.mean()) / (scale if scale else 1)


def ols_hc3(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """OLS with HC3 standard errors; the reported coefficient is the last x."""
    design = np.column_stack([np.ones(len(x)), x])
    inv_xtx = np.linalg.pinv(design.T @ design)
    beta = inv_xtx @ design.T @ y
    residual = y - design @ beta
    leverage = np.sum(design * (design @ inv_xtx), axis=1)
    scaled = residual / np.maximum(1 - leverage, 1e-8)
    meat = design.T @ (design * (scaled**2)[:, None])
    cov = inv_xtx @ meat @ inv_xtx
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    return {
        "coef": float(beta[-1]), "se_hc3": float(se[-1]),
        "ci95_low": float(beta[-1] - 1.96 * se[-1]),
        "ci95_high": float(beta[-1] + 1.96 * se[-1]),
        "r_squared": float(1 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2)),
    }


def build_combine_features(wr: pd.DataFrame) -> pd.DataFrame:
    tracked_drills = [d for vals in DRILL_ALIASES.values() for d in vals]
    tracked_drills += [d for d in ROUTE_PAIRS if d not in DRILL_ALIASES]
    tracking = pd.read_csv(DATA / "combine_tracking.csv")
    tracking = tracking[
        tracking.nfl_id.isin(wr.nfl_id)
        & tracking.drill_name.isin(tracked_drills)
        & tracking.entity_type.eq("PLAYER")
    ].copy()
    tracking["timestamp"] = pd.to_datetime(tracking.time, errors="coerce", utc=True)
    tracking = tracking.sort_values(["event_id", "timestamp"])
    group_cols = ["draft_year", "nfl_id", "drill_name", "attempt", "event_id"]
    rows = []
    for key, g in tracking.groupby(group_cols, sort=False):
        xy = g[["x", "y"]].to_numpy(dtype=float)
        steps = np.sqrt(np.square(np.diff(xy, axis=0)).sum(axis=1))
        dt = g.timestamp.diff().dt.total_seconds().to_numpy()[1:]
        valid = np.isfinite(steps) & np.isfinite(dt) & (dt > 0)
        path = float(steps[valid].sum())
        net = float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) > 1 else np.nan
        speed = g.s.to_numpy(dtype=float)
        accel = g.a.to_numpy(dtype=float)
        rows.append((*key[:4],
                     float(g.timestamp.max().timestamp() - g.timestamp.min().timestamp()),
                     float(np.nanquantile(speed, 0.9)), float(np.nanmax(speed)),
                     float(np.nanmax(accel)), path,
                     net / path if path > 0 else np.nan))
    reps = pd.DataFrame(rows, columns=[
        "draft_year", "nfl_id", "drill_name", "attempt", "duration_s",
        "speed_p90", "speed_max", "accel_max", "path_length", "path_efficiency",
    ])
    features = reps.groupby(["nfl_id", "drill_name"], as_index=False).median(numeric_only=True)
    for canonical, aliases in DRILL_ALIASES.items():
        part = features[features.drill_name.isin(aliases)].copy()
        if not part.empty:
            numeric = [c for c in part if c not in ("nfl_id", "drill_name")]
            collapsed = part.groupby("nfl_id", as_index=False)[numeric].median()
            collapsed["drill_name"] = canonical
            features = pd.concat([features[~features.drill_name.isin(aliases)], collapsed], ignore_index=True)
    return features


def game_tracking_features(target_plays: pd.DataFrame, wr_ids: set[int]) -> tuple[pd.DataFrame, dict]:
    """Read tracking only for listed WR-play keys; retain snap-to-pass-forward."""
    keys = target_plays[["game_id", "play_id", "nfl_id"]].drop_duplicates()
    key_tuples = set(map(tuple, keys.itertuples(index=False, name=None)))
    key_play_tuples = set(map(tuple, keys[["game_id", "play_id"]].drop_duplicates().itertuples(index=False, name=None)))
    needed_seasons = set(target_plays.season.astype(int).unique())
    selected_chunks = []
    markers: dict[tuple[int, int], dict[str, pd.Timestamp]] = {}
    season_counts = {}
    archive = ZipFile(ARCHIVE) if ARCHIVE.exists() else None
    for season in sorted(needed_seasons):
            tracking_file = DATA / f"game_tracking_{season}.csv"
            if tracking_file.is_file():
                stream_context = tracking_file.open("rb")
            elif archive is not None:
                member = f"nfl-big-data-bowl-2027/game_tracking_{season}.csv"
                if member not in archive.namelist():
                    continue
                stream_context = archive.open(member)
            else:
                raise FileNotFoundError(
                    f"Missing {tracking_file} and no archive at {ARCHIVE}. "
                    "Set BDB_DATA_DIR or BDB_ARCHIVE."
                )
            before = sum(len(x) for x in selected_chunks)
            with stream_context as stream:
                for chunk in pd.read_csv(stream, usecols=TRACK_COLS, chunksize=750_000):
                    chunk["game_id"] = pd.to_numeric(chunk.game_id, errors="coerce").astype("Int64")
                    chunk["play_id"] = pd.to_numeric(chunk.play_id, errors="coerce").astype("Int64")
                    event_rows = chunk[chunk.event.isin(["ball_snap", "pass_forward"])]
                    if not event_rows.empty:
                        event_key_tuples = zip(event_rows.game_id, event_rows.play_id)
                        event_rows = event_rows.loc[[tuple(map(int, k)) in key_play_tuples for k in event_key_tuples]]
                    if not event_rows.empty:
                        event_rows = event_rows.assign(
                            timestamp=pd.to_datetime(event_rows.time, errors="coerce", utc=True))
                        for (gid, pid, event), group in event_rows.groupby(["game_id", "play_id", "event"], sort=False):
                            k = (int(gid), int(pid))
                            current = group.timestamp.min()
                            existing = markers.setdefault(k, {}).get(event)
                            if pd.notna(current) and (existing is None or current < existing):
                                markers[k][event] = current
                    # Filter by player first, then exact game/play/player keys.
                    candidate = chunk[chunk.nfl_id.isin(wr_ids)]
                    if candidate.empty:
                        continue
                    tuples = zip(candidate.game_id.astype("int64"), candidate.play_id.astype("int64"),
                                 candidate.nfl_id.astype("int64"))
                    keep = [k in key_tuples for k in tuples]
                    if any(keep):
                        selected_chunks.append(candidate.loc[keep].copy())
            after = sum(len(x) for x in selected_chunks)
            season_counts[str(season)] = after - before
    if archive is not None:
        archive.close()

    if not selected_chunks:
        raise ValueError("No matching WR tracking rows were found for the selected plays")
    tracks = pd.concat(selected_chunks, ignore_index=True)
    tracks["timestamp"] = pd.to_datetime(tracks.time, errors="coerce", utc=True)
    grouped = tracks.groupby(["game_id", "play_id", "nfl_id"], sort=False)
    rows = []
    for key, g in grouped:
        global_markers = markers.get((int(key[0]), int(key[1])), {})
        snap, release = global_markers.get("ball_snap"), global_markers.get("pass_forward")
        if snap is None or release is None or release <= snap:
            continue
        g = g[(g.timestamp >= snap) & (g.timestamp <= release)].sort_values("timestamp")
        if len(g) < 3:
            continue
        xy = g[["x", "y"]].to_numpy(dtype=float)
        path = float(np.sqrt(np.square(np.diff(xy, axis=0)).sum(axis=1)).sum())
        net = float(np.linalg.norm(xy[-1] - xy[0]))
        speed = g.s.to_numpy(dtype=float)
        accel = g.a.to_numpy(dtype=float)
        rows.append((*key, (release - snap).total_seconds(),
                     float(np.nanquantile(speed, .9)), float(np.nanmax(speed)),
                     float(np.nanmax(accel)), path,
                     net / path if path > 0 else np.nan, len(g)))
    cols = ["game_id", "play_id", "nfl_id", "route_window_s", "game_speed_p90",
            "game_speed_max", "game_accel_max", "game_path_length", "game_path_efficiency", "frames"]
    return pd.DataFrame(rows, columns=cols), {
        "tracked_wr_play_keys": int(len(keys)),
        "tracked_rows_retained": int(len(tracks)),
        "play_windows_with_snap_and_release": int(len(rows)),
        "tracked_rows_by_season": season_counts,
    }


def analysis_model(frame: pd.DataFrame, feature: str, outcome: str = "mean_separation") -> dict:
    required = [feature, outcome, "forty", "draft_overall_pick", "draft_year"]
    frame = frame[required].replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[feature, outcome, "draft_year"]).copy()
    if len(frame) < 15:
        return {"n_complete": int(len(frame))}
    frame["forty"] = frame.forty.fillna(frame.forty.median())
    frame["draft_overall_pick"] = frame.draft_overall_pick.fillna(260)
    x = frame[feature].astype(float)
    y = frame[outcome].astype(float)
    rho, pval = spearmanr(x, y)
    controls = frame[["forty", "draft_overall_pick", "draft_year"]].copy()
    controls["draft_overall_pick"] = controls.draft_overall_pick.fillna(260)
    controls["forty"] = controls.forty.fillna(controls.forty.median())
    numeric_controls = controls[["forty", "draft_overall_pick"]].apply(standardize)
    class_controls = pd.get_dummies(controls.draft_year.astype(str), prefix="draft_class", drop_first=True, dtype=float)
    controls = pd.concat([numeric_controls, class_controls], axis=1)
    z = standardize(x)
    y_arr = y.to_numpy()
    base = ols_hc3(y_arr, controls.to_numpy())
    adjusted = ols_hc3(y_arr, np.column_stack([controls.to_numpy(), z.to_numpy()]))
    return {
        "n_complete": int(len(frame)),
        "spearman_rho": float(rho), "spearman_p_unadjusted": float(pval),
        "adjusted_coef_per_sd": adjusted["coef"],
        "adjusted_ci95_low": adjusted["ci95_low"],
        "adjusted_ci95_high": adjusted["ci95_high"],
        "r2_controls": base["r_squared"],
        "r2_controls_plus_feature": adjusted["r_squared"],
    }


def scatter(frame: pd.DataFrame, x: str, y: str, title: str, xlabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 5.2))
    ax.scatter(frame[x], frame[y], alpha=.76, color="#176B87", edgecolor="white", linewidth=.4)
    if len(frame) >= 3:
        slope, intercept = np.polyfit(frame[x].to_numpy(), frame[y].to_numpy(), 1)
        values = np.linspace(frame[x].min(), frame[x].max(), 100)
        ax.plot(values, intercept + slope * values, color="#C55A11", linewidth=2)
    ax.set(title=title, xlabel=xlabel, ylabel="Mean NFL separation at pass forward (ft)")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def summary_figure(player_outputs: list[pd.DataFrame]) -> None:
    data = pd.concat(player_outputs, ignore_index=True)
    panels = [
        ("SPEED_OUT", "path_efficiency__SPEED_OUT", "mean_separation",
         "Combine speed-out efficiency", "Mean separation at pass release (yd)"),
        ("GO_ROUTE_RIGHT", "speed_p90__GO_ROUTE_RIGHT", "mean_game_speed_p90",
         "Combine go-drill speed vs NFL go-route speed", "Mean NFL speed (yd/s)"),
        ("POST_CORNER_ROUTE_RIGHT", "mean_game_speed_p90", "mean_separation",
         "NFL post/corner route speed vs separation", "Mean separation at pass release (yd)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (drill, x, y, title, ylabel) in zip(axes, panels):
        frame = data[data.combine_drill.eq(drill)].dropna(subset=[x, y])
        ax.scatter(frame[x], frame[y], alpha=.76, color="#176B87",
                   edgecolor="white", linewidth=.4)
        if len(frame) >= 3:
            slope, intercept = np.polyfit(frame[x].to_numpy(), frame[y].to_numpy(), 1)
            xx = np.linspace(frame[x].min(), frame[x].max(), 100)
            ax.plot(xx, intercept + slope * xx, color="#C55A11", linewidth=1.8)
            rho, _ = spearmanr(frame[x], frame[y])
            note = f"n={len(frame)}  Spearman ρ={rho:.2f}"
        else:
            note = f"n={len(frame)}"
        ax.set_title(title, fontsize=11)
        xlabel = {
            "path_efficiency__SPEED_OUT": "Combine speed-out path efficiency (ratio)",
            "speed_p90__GO_ROUTE_RIGHT": "Combine go-drill 90th percentile speed (yd/s)",
            "mean_game_speed_p90": "Mean NFL route 90th percentile speed (yd/s)",
        }.get(x, x.replace("__", " ").replace("_", " ").title())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(.04, .96, note, transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": .9, "edgecolor": "#cccccc"})
        ax.grid(alpha=.2)
    fig.suptitle("Selected exploratory relationships (one point per player)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "summary_findings.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    players = pd.read_csv(DATA / "players.csv")
    combine = pd.read_csv(DATA / "combine_results.csv")
    wr = players.merge(combine, on=["nfl_id", "draft_year"], how="inner")
    wr = wr[wr.combine_position.eq("WR") & wr.nfl_position.eq("WR")].copy()
    movement = build_combine_features(wr)

    games = pd.read_csv(DATA / "games.csv", usecols=["game_id", "season", "season_type"])
    play_cols = ["game_id", "play_id", "nfl_id", "route_ran", "separation_at_pass_forward", "target"]
    play = pd.read_csv(DATA / "player_play.csv", usecols=play_cols)
    play["game_id"] = pd.to_numeric(play.game_id, errors="coerce").astype("Int64")
    play["play_id"] = pd.to_numeric(play.play_id, errors="coerce").astype("Int64")
    games["game_id"] = pd.to_numeric(games.game_id, errors="coerce").astype("Int64")
    play = play.merge(games, on="game_id", how="left", validate="many_to_one")
    play = play.merge(wr[["nfl_id", "draft_year"]], on="nfl_id", how="inner", validate="many_to_one")
    play["season"] = pd.to_numeric(play.season, errors="coerce")
    play = play[(play.season >= play.draft_year) & (play.season <= play.draft_year + 1)]
    play = play[play.season_type.eq("REG") & play.separation_at_pass_forward.notna()]
    play = play[play.route_ran.isin({r for routes in ROUTE_PAIRS.values() for r in routes})].copy()

    game_tracking, tracking_audit = game_tracking_features(play, set(wr.nfl_id.astype(int)))
    play = play.merge(game_tracking, on=["game_id", "play_id", "nfl_id"], how="left", validate="one_to_one")

    movement_wide = movement.pivot(index="nfl_id", columns="drill_name",
                                   values=["duration_s", "speed_p90", "speed_max", "accel_max",
                                           "path_length", "path_efficiency"])
    movement_wide.columns = [f"{metric}__{drill}" for metric, drill in movement_wide.columns]
    movement_wide = movement_wide.reset_index()
    results = []
    movement_associations = []
    sensitivity_associations = []
    player_outputs = []
    for drill, routes in ROUTE_PAIRS.items():
        sub = play[play.route_ran.isin(routes)].copy()
        if sub.empty:
            continue
        agg = sub.groupby("nfl_id").agg(
            mean_separation=("separation_at_pass_forward", "mean"),
            route_observations=("separation_at_pass_forward", "size"),
            mean_game_speed_p90=("game_speed_p90", "mean"),
            mean_game_speed_max=("game_speed_max", "mean"),
            mean_game_accel_max=("game_accel_max", "mean"),
            mean_game_path_efficiency=("game_path_efficiency", "mean"),
            tracked_route_observations=("game_speed_p90", "count"),
            target_count=("target", "sum"),
        ).reset_index()
        canonical_drill = "SPEED_OUT" if drill.startswith("SPEED_OUT") else drill
        frame = agg.merge(movement_wide, on="nfl_id", how="inner").merge(
            wr[["nfl_id", "draft_year", "draft_overall_pick", "forty"]],
            on="nfl_id", how="inner", validate="one_to_one")
        speed_col = f"speed_p90__{canonical_drill}"
        eff_col = f"path_efficiency__{canonical_drill}"
        frame = frame[frame[speed_col].notna() & frame.route_observations.ge(10)].copy()
        for c in ["mean_game_speed_p90", "mean_game_speed_max", "mean_game_accel_max", "mean_game_path_efficiency"]:
            frame[c] = frame[c].replace([np.inf, -np.inf], np.nan)
        result = {"combine_drill": drill, "game_routes": ",".join(routes), "n_players": int(len(frame))}
        if len(frame) >= 15:
            result.update({f"combine_speed_{k}": v for k, v in analysis_model(frame, speed_col).items()})
            # Does a Combine track-speed signal transfer to actual tracked speed?
            combine_to_game = analysis_model(frame, speed_col, outcome="mean_game_speed_p90")
            result.update({f"combine_to_game_adjusted_{k}": v for k, v in combine_to_game.items()})
            if frame[eff_col].notna().sum() >= 15:
                result.update({f"combine_efficiency_{k}": v for k, v in analysis_model(frame.dropna(subset=[eff_col]), eff_col).items()})
                eff_to_game = analysis_model(frame.dropna(subset=[eff_col]), eff_col,
                                             outcome="mean_game_speed_p90")
                result.update({f"efficiency_to_game_adjusted_{k}": v for k, v in eff_to_game.items()})
            game_feature = "mean_game_speed_p90"
            if frame[game_feature].notna().sum() >= 15:
                result.update({f"game_speed_{k}": v for k, v in analysis_model(frame.dropna(subset=[game_feature]), game_feature).items()})
                combine_game = frame.dropna(subset=[game_feature, speed_col])
                if len(combine_game) >= 15:
                    rho, p = spearmanr(combine_game[speed_col], combine_game[game_feature])
                    result["combine_to_game_speed_rho"] = float(rho)
                    result["combine_to_game_speed_p_unadjusted"] = float(p)
            frame["combine_drill"] = drill
            player_outputs.append(frame)
            scatter(frame, speed_col, "mean_separation",
                    f"{drill.replace('_', ' ').title()} vs NFL route separation",
                    "Combine drill 90th-percentile tracked speed",
                    OUT / f"{drill.lower()}_separation.png")

            # Full movement-profile sweep. Treat these as exploratory; q-values
            # below account for searching across routes within each metric family.
            combine_metrics = [
                f"{name}__{canonical_drill}" for name in
                ["duration_s", "speed_p90", "speed_max", "accel_max", "path_length", "path_efficiency"]
                if f"{name}__{canonical_drill}" in frame
            ]
            game_metrics = ["mean_game_speed_p90", "mean_game_speed_max",
                            "mean_game_accel_max", "mean_game_path_efficiency"]
            relation_specs = [
                ("combine_to_separation", combine_metrics, "mean_separation"),
                ("combine_to_game_speed", combine_metrics, "mean_game_speed_p90"),
                ("game_movement_to_separation", game_metrics, "mean_separation"),
            ]
            for relation, candidate_metrics, outcome_col in relation_specs:
                for metric in candidate_metrics:
                    usable = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric, outcome_col])
                    if len(usable) < 15:
                        continue
                    stats = analysis_model(usable, metric, outcome=outcome_col)
                    movement_associations.append({
                        "combine_drill": drill,
                        "game_routes": ",".join(routes),
                        "relation": relation,
                        "movement_metric": metric,
                        "n_players": int(stats.get("n_complete", len(usable))),
                        **stats,
                    })
            sensitivity_frames = {
                "primary_min_10_routes": frame,
                "min_25_route_observations": frame[frame.route_observations.ge(25)],
                "two_season_classes_2023_2024": frame[frame.draft_year.lt(2025)],
            }
            for sensitivity, sens_frame in sensitivity_frames.items():
                for relation, feature, outcome_col in [
                    ("combine_to_separation", eff_col, "mean_separation"),
                    ("combine_to_game_speed", speed_col, "mean_game_speed_p90"),
                    ("game_movement_to_separation", "mean_game_speed_p90", "mean_separation"),
                ]:
                    if feature not in sens_frame:
                        continue
                    stats = analysis_model(sens_frame, feature, outcome=outcome_col)
                    sensitivity_associations.append({
                        "combine_drill": drill,
                        "relation": relation,
                        "movement_metric": feature,
                        "sensitivity": sensitivity,
                        **stats,
                    })
        results.append(result)

    summary = pd.DataFrame(results)
    # Correct rank-correlation p-values for the six route families in each
    # relation family, and also report a conservative across-analysis q-value.
    p_columns = [c for c in summary if c.endswith("spearman_p_unadjusted") or c == "combine_to_game_speed_p_unadjusted"]
    all_p = []
    for col in p_columns:
        valid = summary[col].dropna()
        if valid.empty:
            continue
        order = np.argsort(valid.to_numpy())
        ranked = valid.to_numpy()[order]
        q = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
        q = np.minimum(q, 1.0)
        q_original = np.empty_like(q)
        q_original[order] = q
        summary.loc[valid.index, col.replace("p_unadjusted", "q_bh_route_family")] = q_original
        all_p.extend((float(p), int(idx), col) for idx, p in valid.items())
    if all_p:
        all_p.sort(key=lambda v: v[0])
        raw_q = [min(1.0, p * len(all_p) / (i + 1)) for i, (p, _, _) in enumerate(all_p)]
        for i in range(len(raw_q) - 2, -1, -1):
            raw_q[i] = min(raw_q[i], raw_q[i + 1])
        for (_, idx, col), q in zip(all_p, raw_q):
            summary.loc[idx, col.replace("p_unadjusted", "q_bh_all_exploratory")] = q
    assoc = pd.DataFrame(movement_associations)
    if not assoc.empty:
        assoc["q_bh_route_family"] = np.nan
        assoc["q_bh_all_exploratory"] = np.nan
        for _, indices in assoc.groupby(["relation", "movement_metric"]).groups.items():
            valid = assoc.loc[indices, "spearman_p_unadjusted"].dropna()
            if valid.empty:
                continue
            order = np.argsort(valid.to_numpy())
            ranked = valid.to_numpy()[order]
            q = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
            q = np.minimum(q, 1.0)
            restored = np.empty_like(q)
            restored[order] = q
            assoc.loc[valid.index, "q_bh_route_family"] = restored
        valid_all = assoc["spearman_p_unadjusted"].notna()
        order = np.argsort(assoc.loc[valid_all, "spearman_p_unadjusted"].to_numpy())
        p_sorted = assoc.loc[valid_all, "spearman_p_unadjusted"].to_numpy()[order]
        q_sorted = np.minimum.accumulate((p_sorted * len(p_sorted) / np.arange(1, len(p_sorted) + 1))[::-1])[::-1]
        q_sorted = np.minimum(q_sorted, 1.0)
        q_all = np.empty_like(q_sorted)
        q_all[order] = q_sorted
        assoc.loc[valid_all, "q_bh_all_exploratory"] = q_all
        assoc.to_csv(OUT / "movement_associations.csv", index=False)
    pd.DataFrame(sensitivity_associations).to_csv(OUT / "sensitivity_associations.csv", index=False)
    summary.to_csv(OUT / "route_associations.csv", index=False)
    movement.to_csv(OUT / "combine_route_movement.csv", index=False)
    if player_outputs:
        pd.concat(player_outputs, ignore_index=True).to_csv(OUT / "player_route_analysis.csv", index=False)
        summary_figure(player_outputs)

    metadata = {
        "n_wr_combine_players": int(wr.nfl_id.nunique()),
        "n_wr_players_with_eligible_route_rows": int(play.nfl_id.nunique()),
        "data_directory": str(DATA),
        "game_seasons_used": sorted(play.season.dropna().astype(int).unique().tolist()),
        "tracking_audit": tracking_audit,
        "game_window": "regular-season rookie year and following season when available; partial 2026 excluded",
        "game_tracking_window": "ball_snap through pass_forward; positions, speed and acceleration are summarized per player-play",
        "separation_outcome": "mean separation_at_pass_forward on matched NFL route types; require >=10 observations per player",
        "units": {
            "separation_at_pass_forward": "yards, as provided in the competition player_play.csv",
            "tracking_speed_s": "yards per second, as provided in the competition tracking CSVs",
            "path_efficiency": "net displacement divided by summed frame-to-frame path length; unitless",
        },
        "adjustment_variables": ["official 40-yard time", "draft overall pick (missing treated as 260)", "draft year"],
        "multiple_testing": "Benjamini-Hochberg q-values reported by route family and across all exploratory movement/outcome tests.",
        "caveats": [
            "Observational player-level associations; do not infer causation.",
            "Small and uneven player/route samples; several route comparisons are exploratory.",
            "The 2025 class has one observed season, earlier classes have up to two.",
            "Combine clip speed is tracking-derived and should not be treated as an official timing.",
            "The analysis does not establish a validated scouting tool.",
        ],
    }
    (OUT / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print("\nRoute results:\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
