"""Step 8 — organize output.

A pure, safe-to-re-run step that makes no model calls. It turns what step 7 produced
(the per-step run records plus the evidence packets handed across the PHI boundary)
into the final operator-facing files. It owns the output writes so step 7 can stay
focused on running the recipe's steps. Re-runnable without spending model tokens.
"""
