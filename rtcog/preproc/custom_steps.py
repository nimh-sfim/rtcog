"""Project-specific preprocessing steps.

Add subclasses of ``PreprocStep`` here to register them for use in the
pipeline configuration.
"""

from rtcog.preproc.preproc_steps import PreprocStep


# Example:
#
# class CustomStep(PreprocStep):
#     def _start(self, pipeline):
#         """Optional setup before the first processed TR."""
#         pass
#
#     def _run(self, pipeline):
#         return pipeline.processed_tr
#
#     def _save(self, pipeline):
#         """Optional saving after processing is complete."""
#         pass
