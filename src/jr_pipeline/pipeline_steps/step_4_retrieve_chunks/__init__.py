"""Retrieve relevant text chunks for a clinical query.

Step 4 takes a question (e.g. "what is the AJCC stage?") and pulls the handful
of source-text chunks most likely to answer it, so the extract step reads only
relevant evidence instead of the whole chart.
"""
