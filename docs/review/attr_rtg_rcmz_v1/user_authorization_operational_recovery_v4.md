# User authorization — operational checkpoint recovery V4

The user explicitly instructed that the failed seed-29 training must not be repeated and authorized correction of the scorer error. This authorizes only the operational hotfix described by `docs/review/attr_rtg_rcmz_v1/operational_hotfix_amendment_v1.md` and reuse of the exact four seed-29 update-2000 checkpoints.

The agent may implement, test and reseal this hotfix, but must not execute the official recovery. The user will run the recovery in another terminal. No heavy scientific re-review and no new scientific claim are authorized or required because the scientific protocol remains unchanged.

Operational amendment SHA-256: `a385b6eb0529a1f43ea76cb46615f8089fb9334d1f612e6ee99f9500d8459d56`
Failure receipt SHA-256: `458e0870550211d2a1457d3adecd3faa853c62a418feb37cd198a4df5b5f989a`
Predecessor lock SHA-256: `6ed6bd2b39460695a891c27c51cc417453e59eb10ad519660219f84a8cb8950e`
Protocol SHA-256 remains: `c37fa09bdad9715d82d5cb6b6108ce5d2147462c79738674abb75ca50dbc0f84`
