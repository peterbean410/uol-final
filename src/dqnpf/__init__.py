"""Intraday trading agent: DQN + ProbabilisticForecaster integration layer.

Symbols are intentionally NOT re-exported at the package root so that
lightweight consumers (pipeline DSL compilation in CI, config inspection)
don't pay the cost of importing numpy/torch via forecaster_bridge etc.
Use the explicit module paths:

    from tradingmodel.intraday.dqnpf.action_mapper import map_action, ACTION_MAP
    from tradingmodel.intraday.dqnpf.config import IntegrationConfig, load_config
    from tradingmodel.intraday.dqnpf.forecaster_bridge import ForecasterBridge
    from tradingmodel.intraday.dqnpf.integration import IntegrationLayer
    from tradingmodel.intraday.dqnpf.signal_cache import SignalCache
"""
