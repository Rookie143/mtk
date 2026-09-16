"""Example CLI for running an MTK adaptive attack."""

from __future__ import annotations

import argparse

from adaptive_attack import load_hidden_state_library, run_mtk_attack

from .common import (
    add_common_arguments,
    build_config,
    generate_text,
    load_model_and_tokenizer,
    print_result,
)


def parse_args() -> argparse.Namespace:
    """Parse MTK example arguments."""
    parser = argparse.ArgumentParser(
        description="Run an MTK adaptive attack"
    )
    add_common_arguments(parser)
    parser.add_argument("--feature-library", required=True)
    parser.add_argument("--loss-type", choices=("l1", "l2", "l3"), default="l3")
    parser.add_argument("--loss-scale", type=float, default=1.0)
    parser.add_argument("--benign-label", type=int, default=1)
    parser.add_argument("--malicious-label", type=int, default=0)
    parser.add_argument(
        "--lambda",
        dest="lambda_value",
        type=float,
        default=None,
        help="Paper lambda; overrides sequence_weight and feature_weight when set",
    )
    return parser.parse_args()


def main() -> None:
    """Load the reference bank and run an L1/L2/L3 surrogate attack."""
    args = parse_args()
    if args.lambda_value is not None:
        if not 0.0 <= args.lambda_value <= 1.0:
            raise ValueError("lambda must be in [0, 1]")

        # The paper objective uses J=(1-lambda)*L_adv+lambda*L_evasion. Override
        # both weights here so CLI users cannot create an ambiguous configuration.
        args.sequence_weight = 1.0 - args.lambda_value
        args.feature_weight = args.lambda_value

    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.dtype)
    library = load_hidden_state_library(
        args.feature_library,
        map_location=args.device,
    )
    result = run_mtk_attack(
        model=model,
        tokenizer=tokenizer,
        messages=args.prompt,
        target=args.target,
        feature_library=library,
        loss_type=args.loss_type,
        config=build_config(args),
        scale=args.loss_scale,
        benign_label=args.benign_label,
        malicious_label=args.malicious_label,
    )
    generation = generate_text(
        model,
        tokenizer,
        args.prompt,
        " " + result.best_string,
        args.device,
    )
    print_result(result, generation)


if __name__ == "__main__":
    main()
