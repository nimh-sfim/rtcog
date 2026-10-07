# Processing orchestration

Processors connect incoming scanner data to preprocessing and downstream
experiment behavior. `BasicProcessor` manages the real-time volume stream and
pipeline; `ESAMProcessor` adds template matching, hit detection, shared state,
and live visualization for experience-sampling runs.
