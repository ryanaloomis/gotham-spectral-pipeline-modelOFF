"""Shared options for the native and exported-data entry points."""

from ..reference import SmoothOffReference


def add_model_options(parser):
    parser.add_argument("--normalization", choices=["off", "on", "none"], default="off")
    parser.add_argument("--knot-spacing", type=int, default=512, help="Fixed knot spacing in native channels")
    parser.add_argument("--fit-bin", type=int, default=32)
    parser.add_argument("--model-edge-channels", type=int, default=1024)
    parser.add_argument("--model-clip-sigma", type=float, default=5.0)
    parser.add_argument("--model-iterations", type=int, default=4)


def make_builder(args):
    return SmoothOffReference(
        knot_spacing=args.knot_spacing, fit_bin=args.fit_bin,
        edge_channels=args.model_edge_channels, clip_sigma=args.model_clip_sigma,
        iterations=args.model_iterations, normalization=args.normalization,
    )
