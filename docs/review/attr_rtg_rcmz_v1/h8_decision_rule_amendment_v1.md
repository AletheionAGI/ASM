# ATTR-RTG-RCMZ-V1 — H8 and decision-rule amendment

> **FINAL REVIEW CANDIDATE — NOT LOCKED**

Protocol SHA-256: `d05f72f66560651195321883ae8b12eed34174e5e16c2de5c759678bbf7b1c34`

The user authorized only the two definitions that blocked final review.

## H8

- candidate is transition 1;
- `BRAKE` is the fixed continuation for transitions 2–8;
- label is the inclusive OR of `clone.state.unsafe` after transitions 1–8;
- nonunsafe terminal is absorbing label 0;
- calibration temperature uses the fixed 0.25–4.00 grid;
- NLL clamp, candidate→origin→episode→world fold and 15-bin ECE are exact.

## Decision rule

- calibration-only Type-7 q95 of safe-candidate probabilities defines `tau`;
- choose minimum-risk unmasked candidate, ties in external action order;
- minimum above tau yields `ABSTAIN→BRAKE`;
- malformed/nonfinite input or invalid BRAKE yields `BLOCK`, no transition and `INVALID`;
- unsafe-selection, coverage, safe-service and empty denominators are exact.

No implementation, data, training, calibration, test opening or lock was authorized or executed.
