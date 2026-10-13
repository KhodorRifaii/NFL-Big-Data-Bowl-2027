# NFL Big Data Bowl 2027: Combine movement and receiver routes

This project examines whether wide receiver movement traits measured in route style Combine drills carry over to matched NFL routes. The work uses the Kaggle [NFL Big Data Bowl 2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027) data and processes game tracking without extracting all large tracking CSVs.

## Main conclusion

The available sample does not establish a reliable relationship between Combine movement and NFL route outcomes. A promising unadjusted relationship between speed out drill path efficiency and NFL out route separation is sensitive to how many NFL route observations are required per player. Combine go route speed has a modest raw relationship with go route game speed, but its adjusted confidence interval includes zero. These are exploratory associations, not a validated scouting tool or causal results.

## Run it

Install dependencies from `requirements.txt`, then run:

```bash
python src/analyze_wr.py
```

The script looks for the competition CSVs in `data/raw/`. You can also set `BDB_DATA_DIR` to the directory containing them, or put the original archive at `~/Downloads/nfl-big-data-bowl-2027.zip`. On Kaggle, it searches `/kaggle/input` automatically. Tracking rows are streamed from the ZIP when full season tracking CSVs are not already present in the data directory.

## Deliverables

- `notebooks/nfl_big_data_bowl_2027.ipynb`: self-contained notebook version, designed to run with competition data attached on Kaggle.
- `writeup.md`: short project write-up and findings.
- `outputs/summary_findings.png`: figure summarizing selected exploratory relationships.
- `outputs/route_associations.csv`, `outputs/movement_associations.csv`: association tables with uncertainty and multiple comparison summaries.
- `outputs/sensitivity_associations.csv`: checks using higher route-count thresholds and earlier draft cohorts.
- `outputs/analysis_metadata.json`: coverage and methods audit.

Data files and generated analysis tables are excluded from version control. Do not publish or redistribute the competition data. To submit to Kaggle, upload the notebook and write-up through the competition page and follow its current rules and deadlines. Completing this analysis locally does not create a Kaggle submission or award competition points or medals.
