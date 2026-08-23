"""Step 6: take the chart passages chosen for a question and package them for the
language model. The model can only read so much text at once (its "context
window"), so this step keeps the best passages that fit within that limit and
labels each one with where it came from, so the model can cite its source."""
