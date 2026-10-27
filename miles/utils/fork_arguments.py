"""Command-line flags the fork adds to Miles.  [fork hook-014]

Every fork flag is registered here, through the one call in arguments.add_miles_arguments, so an
upstream sync meets one line in arguments.py instead of a block in each argument group. Validation
that reads these flags stays in arguments.py next to the upstream checks it extends.

Before adding a flag, read "Flags, recipe keys and env vars" in fork/README.md: a fix has no
switch, and a new flag needs a run that uses its other value.
"""


def add_fork_arguments(parser):
    # Rollout (fully async)
    # [fork fix-004] begin
    parser.add_argument(
        "--async-single-pass",
        action="store_true",
        default=False,
        help=(
            "Fully async mode hands each prompt of --prompt-data out once instead of starting the "
            "next epoch when the data runs out, still keeping --async-max-concurrent-samples in "
            "flight. Once every prompt is out and finished, a rollout step that cannot fill "
            "--rollout-batch-size takes the groups that are left. For an eval run as one rollout "
            "(--num-rollout 1, --rollout-batch-size = number of prompts): every prompt gets exactly "
            "--n-samples-per-prompt samples and the step waits for all of them."
        ),
    )
    # [fork fix-004] end
    # [fork fix-048] begin
    parser.add_argument(
        "--async-retry-aborted-tries",
        type=int,
        default=1,  # [fork fix-059] on by default; 0 keeps upstream's whole-group handling
        help=(
            "Fully async mode regenerates an aborted try at once, up to this many times, and keeps "
            "its group's other tries, instead of handing the whole group to "
            "--async-unused-samples-handler. A group with a try still aborted after the last retry "
            "is handled as before. Default 1; 0 turns it off (needed with --group-rm or "
            "deterministic inference)."
        ),
    )
    # [fork fix-048] end
    # [fork fix-054] begin
    parser.add_argument(
        "--max-token-weighted-staleness",
        type=float,
        default=None,
        help=(
            "Maximum allowed gap between the current engine weight version and the mean weight version "
            "of a group's generated tokens. A group must also pass --max-weight-staleness (its oldest "
            "token), so the two bound the average and the worst token. Groups exceeding it go to "
            "--async-unused-samples-handler like other stale groups. None (default) disables it."
        ),
    )
    # [fork fix-054] end

    # Training
    # [fork fix-006] begin: ported from upstream branch jiajun/mimo26-mxfp4-qat
    parser.add_argument(
        "--mxfp4-qat-routed-experts",
        action="store_true",
        help=(
            "For an MXFP4 rollout engine: after every optimizer step, write the routed-expert weights "
            "as the values their MXFP4 encoding decodes to (the weight-sync quantizer), so the trainer "
            "computes with exactly the experts the engine serves; the fp32 master keeps the full "
            "update. Without it the engine's experts round back to their previous grid points while "
            "the trainer's move, and the train/rollout mismatch grows with the expert updates. "
            "Requires --stream-optimizer-state-to-disk with Adam and expert data parallel size 1."
        ),
    )
    # [fork fix-006] end

    # Hooks the fork's extension code plugs into
    # [fork hook-001] begin
    parser.add_argument(
        "--custom-weight-quantizer-path",
        type=str,
        default=None,
        help=(
            "Path to a function that turns each converted HF tensor list into the rollout "
            "checkpoint's format during a weight sync, replacing the quantizer chosen from "
            "quantization_config. Signature: ``def quantize(args, megatron_name, "
            "converted_named_params, quantization_config) -> list[tuple[str, torch.Tensor]]``."
        ),
    )
    # [fork hook-001] end
    # [fork hook-013] begin
    parser.add_argument(
        "--custom-weight-transfer-protocol-path",
        type=str,
        default=None,
        help=(
            "Import path of a WeightTransferProtocol subclass (or a callable(args) returning one) that "
            "moves weights to the rollout engines instead of the one --update-weight-transfer-mode picks."
        ),
    )
    # [fork hook-013] end
    # [fork hook-002] begin
    parser.add_argument(
        "--custom-placement-hook-path",
        type=str,
        default=None,
        help=(
            "Path to a function the driver calls once after creating the GPU placement group, e.g. to "
            "start per-node observers that know which GPUs train and which serve rollouts. Signature: "
            "``def hook(args, pgs: dict[str, PlacementGroupInfo]) -> None``."
        ),
    )
    # [fork hook-002] end

    return parser
