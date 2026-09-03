# Diagnostic safe-service eligible-only V1

Status: AUTHORIZED DIAGNOSTIC EVALUATION

At the user's explicit request for numeric shift/OOD safe-service results and a generated `governance.png`, this amendment adds `safe_service_eligible_only` as a separate diagnostic. It folds only world/episode cells containing at least one eligible origin and reports eligible/total fold counts.

The preregistered `safe_service` remains unchanged: missing denominator in any fold yields `null`, the metric cell remains INVALID, and all official gates fail closed. The diagnostic must be labeled `eligible folds; diagnostic` in graphs and must never be substituted into official contrasts or gates.

Authorized work is score-only reuse of the exact 20 checkpoints, with no optimizer step or checkpoint write.
