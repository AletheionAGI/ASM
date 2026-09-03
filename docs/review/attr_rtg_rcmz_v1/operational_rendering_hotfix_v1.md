# ATTR-RTG-RCMZ operational rendering hotfix V1

The official V6 computation completed all 20 arm/seed trainings and produced `official_rows.json` before the process failed in rendering with `float(None)`. Source rows SHA-256 is `d0b7f2611528ffbeb6b150e6f98f4d8c68818fd2582c51a81f30bcd755d69902` and execution lock SHA-256 is `3d640d25beba627c21b2088b40e9aa4e650c9fae89d9237f9fe4946f1ddcb6b2`.

The scalar release is fail-closed: 20 ID arm/seed rows are VALID; 20 shift and 20 OOD arm/seed rows are INVALID; all six contrasts are INVALID. Null metrics remain null in JSON/CSV/HTML. The renderer now omits null values from numerical bar calculations instead of converting them to float. This changes only derived visualization and does not train, score, calibrate, bootstrap or alter any scalar row.

The four RTG-style grouped-bar families, summary PNG/SVG, HTML and manifest were generated directly from the immutable existing rows. No checkpoint or official row was modified.
