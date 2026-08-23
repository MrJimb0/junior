"""Junior's interfaces: the CLI, the interactive prompt, and the Shiny review app.

Everything here is an operator surface over the ``jr_pipeline`` package in src/.
Interfaces call the engine; the engine never imports back. Deployment (SLURM,
the cloud) sits on top of the CLI, not beside it.
"""
