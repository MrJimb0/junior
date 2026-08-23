"""The built-in step types a recipe can use, and the lookup that finds them.

A recipe is built from "steps", and each step has a kind (for example
``retrieve_and_prompt`` or ``direct_parquet``). This file collects all the
built-in step kinds and registers them so a recipe can refer to them by name.

To add a new kind, create
``src/jr_pipeline/pipeline_steps/step_7_extract_variables/recipe_steps/<kind>_step.py``
and import it here. Importing it runs its ``@register_step`` line, which is what
adds the new kind to the lookup table.
"""
# Importing each module runs its @register_step line, which adds that step kind
# to the registry so recipes can use it by name.
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps import (  # noqa: F401,E402
    direct_parquet_step,
    llm_only_step,
    map_table_rows_and_prompt_step,
    python_step,
    retrieve_and_prompt_step,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (  # noqa: F401
    StepContext,
    StepHandler,
    StepResult,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_type_lookup import (  # noqa: F401
    build_step,
    register_step,
)
