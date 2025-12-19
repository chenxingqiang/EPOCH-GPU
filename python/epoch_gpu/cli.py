"""
EPOCH-GPU command-line interface.

Provides CLI entry points for common operations.
"""

import argparse
import sys
import json

from .config import GPUConfig, get_gpu_info
from .builder import build_epoch, clean_build, get_build_info


def gpu_info():
    """CLI entry point: Display GPU information."""
    parser = argparse.ArgumentParser(
        description="Display EPOCH-GPU configuration and GPU information"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    args = parser.parse_args()
    
    config = GPUConfig()
    
    if args.json:
        print(json.dumps(config.to_dict(), indent=2))
    else:
        config.print_info()


def run_tests():
    """CLI entry point: Run EPOCH-GPU tests."""
    parser = argparse.ArgumentParser(
        description="Run EPOCH-GPU tests"
    )
    parser.add_argument(
        "--dimension", "-d",
        choices=["1d", "2d", "3d", "all"],
        default="all",
        help="Dimension to test (default: all)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Run GPU tests (default)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run CPU tests"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()
    
    # Import test runner
    try:
        from .testing import run_epoch_tests
        success = run_epoch_tests(
            dimension=args.dimension,
            gpu=args.gpu and not args.cpu,
            verbose=args.verbose
        )
        sys.exit(0 if success else 1)
    except ImportError:
        print("Test framework not available. Run tests manually:")
        print("  python scripts/run_gpu_tests.py")
        sys.exit(1)


def build_epoch_cli():
    """CLI entry point: Build EPOCH-GPU."""
    parser = argparse.ArgumentParser(
        description="Build EPOCH-GPU"
    )
    parser.add_argument(
        "--dimension", "-d",
        choices=["1d", "2d", "3d", "all"],
        default="2d",
        help="Dimension to build (default: 2d)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Build CPU version (no GPU)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean before building"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        help="Number of parallel jobs (default: 4)"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show build information and exit"
    )
    args = parser.parse_args()
    
    if args.info:
        info = get_build_info()
        print(json.dumps(info, indent=2))
        return
    
    dimensions = ["1d", "2d", "3d"] if args.dimension == "all" else [args.dimension]
    
    success = True
    for dim in dimensions:
        result = build_epoch(
            dimension=dim,
            gpu=not args.cpu,
            clean_first=args.clean,
            parallel_jobs=args.jobs
        )
        if not result:
            success = False
    
    sys.exit(0 if success else 1)


# Alias for pyproject.toml
build_epoch = build_epoch_cli


if __name__ == "__main__":
    # Default to gpu_info when run directly
    gpu_info()

