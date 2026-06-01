You are working inside my retail demand forecasting + price optimisation repo.



Task: fix the MILP/promo-selection no-op price-change problem accurately across the whole codebase.



Context:

The project has a pricing/promo optimisation layer using PuLP MILP/greedy fallback. The problem is that promo selection currently allows `0.0` in the promo discount grid and often defaults `require\_price\_change=False`. This can cause misleading outputs where the optimiser “selects” actions that are actually no price change. In promo/action-selection mode, selected actions should normally be real price changes.



Important:

\- This is MILP, not MLP. Do not add neural networks.

\- Do not rewrite the whole pricing system.

\- Do not change forecasting logic.

\- Do not remove PuLP/MILP logic.

\- Make the smallest clean production-style fix.

\- Preserve the ability to run no-change candidates only when explicitly requested, but default promo selection should require real price changes.



Files to inspect and update where relevant:

\- m5\_pipeline/m5\_promo\_selection.py

\- api/schemas.py

\- api/services.py

\- api/main.py if needed

\- agents/agent\_tools.py

\- m5\_pipeline/pipeline.py if promo-selection CLI defaults are inconsistent

\- README.md or relevant docs if they mention promo\_discount\_grid / require\_price\_change

\- tests/



Required behaviour changes:



1\. In the promo-selection config, make the default promo mode require real price changes:

&#x20;  - require\_price\_change should default to True.

&#x20;  - promo\_discount\_grid should default to discounts only, e.g. (-0.20, -0.10, -0.05).

&#x20;  - Remove 0.0 from the default promo\_discount\_grid.



2\. Fix candidate-delta construction:

&#x20;  Current/old behaviour may append 0.0 even when require\_price\_change=True.

&#x20;  Change it so:

&#x20;  - if require\_price\_change=True, do not add 0.0 automatically.

&#x20;  - if require\_price\_change=False, 0.0 may be included as an explicit no-change baseline candidate.

&#x20;  - Deduplicate and sort deltas safely.



&#x20;  The intended logic should look like this conceptually:



&#x20;  deltas = list(cfg.promo\_discount\_grid)



&#x20;  if not cfg.require\_price\_change and 0.0 not in deltas:

&#x20;      deltas.append(0.0)



&#x20;  deltas = sorted(set(float(x) for x in deltas))



3\. Fix eligibility/selection semantics:

&#x20;  When require\_price\_change=True:

&#x20;  - selected rows must have a real non-zero applied price change.

&#x20;  - use a small epsilon such as 1e-9 if comparing floats.

&#x20;  - do not count no-op rows as selected promo actions.

&#x20;  - output summaries should not report no-op selected rows as real actions.



4\. Align all API/service/agent defaults:

&#x20;  Search the entire repo for:

&#x20;  - require\_price\_change

&#x20;  - promo\_discount\_grid

&#x20;  - 0.0

&#x20;  - forbid\_price\_increase



&#x20;  Update defaults consistently:

&#x20;  - api/schemas.py should default require\_price\_change=True.

&#x20;  - api/schemas.py promo\_discount\_grid should default to \[-0.20, -0.10, -0.05].

&#x20;  - api/services.py should default require\_price\_change=True.

&#x20;  - api/services.py should default promo\_discount\_grid to (-0.20, -0.10, -0.05).

&#x20;  - agents/agent\_tools.py should default require\_price\_change=True.

&#x20;  - agents/agent\_tools.py should default promo\_discount\_grid to \[-0.2, -0.1, -0.05].

&#x20;  - If m5\_pipeline/pipeline.py has CLI defaults that allow price increases or no-op promo actions by default, make CLI behaviour consistent with promo mode: discount-only / require real change by default. If adding flags, prefer an explicit opt-in flag like --allow-price-increase rather than requiring users to remember --forbid-price-increase.



5\. Add or update tests:

&#x20;  Add a focused test file, for example:

&#x20;  tests/test\_promo\_selection\_milp.py



&#x20;  Tests should verify:

&#x20;  - default PromoSelectionConfig does not contain 0.0 in promo\_discount\_grid.

&#x20;  - default require\_price\_change is True.

&#x20;  - when require\_price\_change=True, selected rows have non-zero price changes.

&#x20;  - max\_price\_changes\_total is respected.

&#x20;  - if PuLP is available, MILP path still works.

&#x20;  - if the code has greedy fallback, the same no-op protection applies to fallback outputs too.



&#x20;  Use small synthetic data; do not require the real M5 dataset in CI.



6\. Update documentation:

&#x20;  Update README or relevant docs to say:

&#x20;  - promo selection defaults to discount-only real price changes.

&#x20;  - 0.0/no-change candidates are only allowed when require\_price\_change=False.

&#x20;  - MILP chooses executable promo/price actions under constraints; it should not count no-op rows as selected business actions.



7\. Acceptance criteria:

&#x20;  After changes, these should pass:

&#x20;  - pytest -q

&#x20;  - pytest -q tests/test\_api\_smoke.py

&#x20;  - pytest -q tests/test\_promo\_selection\_milp.py



&#x20;  Also run a grep/check to confirm no default promo grid still contains 0.0:

&#x20;  - search for promo\_discount\_grid defaults

&#x20;  - search for require\_price\_change defaults

&#x20;  - search for code that appends 0.0 unconditionally



Expected final behaviour:

\- Default promo/MILP selection recommends real discount actions only.

\- No-op 0.0 actions are not selected by default.

\- If a user explicitly sets require\_price\_change=False, the system may include 0.0 as a baseline/no-change candidate.

\- API, CLI, agents, tests, and docs all agree.

\- The pricing system remains explainable elasticity + guardrails + MILP/greedy constrained selection.

