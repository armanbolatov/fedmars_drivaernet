Oct 15 2026: stage-3 records from the greedy local-epoch walk, which stopped one rung
early when neighbouring rungs scored within the score's noise (the non-spectral methods
stopped at E=1 because E=2 scored 2% worse). Superseded by the exhaustive ladder
(tune_full.py --epochs, which now evaluates every rung); kept for the record.

Oct 30 2026, schedule_ablation/: the figure, table and local-epoch table that still
listed the constant-rate runs of the non-spectral methods (fedavg, scaffold, fedlion),
the schedule ablation of the paper's appendix. The figures and tables of record now
hold the cosine-schedule methods only; schedule_heterogeneity became heterogeneity30.
