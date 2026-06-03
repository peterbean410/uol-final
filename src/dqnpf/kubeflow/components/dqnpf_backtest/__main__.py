"""CLI entry point for the dqnpf_backtest KFP component.

Wraps :func:`run_dqnpf_backtest` with argparse so the container ENTRYPOINT
can invoke it with KFP-injected inputValue and outputPath arguments.
"""

from __future__ import annotations

import argparse
import logging
import sys

from tradingmodel.intraday.dqnpf.kubeflow.components.dqnpf_backtest.component import (
    run_dqnpf_backtest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="dqnpf-intraday backtest component")
    parser.add_argument("--integration-config-yaml", required=True)
    parser.add_argument("--dqn-model-registry-name", required=True)
    parser.add_argument("--forecaster-model-registry-name", required=True)
    parser.add_argument("--output-artifact-path", required=True)
    parser.add_argument(
        "--no-sidecar",
        action="store_true",
        help="Skip launching the modelenv subprocess (used when the sidecar "
        "is already running in the pod).",
    )
    parser.add_argument(
        "--trade-log-output-path",
        default=None,
        help="Optional destination for modelenv's JSONL trade log (every fill + "
        "position close), exported after the run so it can be downloaded. "
        "Accepts a local path or a minio:// / s3:// URI.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    run_dqnpf_backtest(
        integration_config_yaml=args.integration_config_yaml,
        dqn_model_registry_name=args.dqn_model_registry_name,
        forecaster_model_registry_name=args.forecaster_model_registry_name,
        output_artifact_path=args.output_artifact_path,
        start_sidecar=not args.no_sidecar,
        trade_log_output_path=args.trade_log_output_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
