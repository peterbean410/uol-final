# deepqnetwork - Deep Q-Network Trading Agent
#
# DQNAdvisor and ActionResult are intentionally NOT re-exported here,
# importing the package should be a no-op so lightweight consumers (pipeline
# DSL compilation, config inspection) don't pay the cost of pulling in
# numpy/torch via advisor.py. Use the explicit module path instead:
#
#     from deepqnetwork.advisor import DQNAdvisor, ActionResult
