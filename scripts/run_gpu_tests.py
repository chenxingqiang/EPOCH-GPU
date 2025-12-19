#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EPOCH-GPU Test Runner

Runs all GPU tests for 1D, 2D, and 3D versions of EPOCH.

Usage:
    python run_gpu_tests.py              # Run all tests
    python run_gpu_tests.py 1d           # Run only 1D tests
    python run_gpu_tests.py 2d           # Run only 2D tests
    python run_gpu_tests.py 3d           # Run only 3D tests
    python run_gpu_tests.py --verbose    # Verbose output
"""

import sys
import os
import unittest
import time
import argparse


def run_tests(dimensions=None, verbose=2):
    """Run GPU tests for specified dimensions."""
    
    # Get project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Define test directories
    test_dirs = {
        '1d': os.path.join(project_root, 'epoch1d', 'tests_gpu'),
        '2d': os.path.join(project_root, 'epoch2d', 'tests_gpu'),
        '3d': os.path.join(project_root, 'epoch3d', 'tests_gpu'),
    }
    
    # Filter by requested dimensions
    if dimensions:
        test_dirs = {k: v for k, v in test_dirs.items() if k in dimensions}
    
    # Results summary
    results = {}
    
    print("=" * 60)
    print("           EPOCH-GPU Test Suite")
    print("=" * 60)
    print(f"Running tests for: {', '.join(test_dirs.keys())}")
    print("=" * 60)
    
    overall_start = time.time()
    
    for dim, test_dir in test_dirs.items():
        print(f"\n{'='*60}")
        print(f"  Running {dim.upper()} GPU Tests")
        print(f"{'='*60}")
        
        if not os.path.exists(test_dir):
            print(f"  Warning: Test directory not found: {test_dir}")
            continue
        
        # Add test directory to path
        sys.path.insert(0, test_dir)
        
        # Discover and run tests
        loader = unittest.TestLoader()
        suite = loader.discover(test_dir, pattern='test_*.py')
        
        start_time = time.time()
        runner = unittest.TextTestRunner(verbosity=verbose)
        result = runner.run(suite)
        elapsed = time.time() - start_time
        
        results[dim] = {
            'tests': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'time': elapsed
        }
        
        # Remove from path
        sys.path.remove(test_dir)
    
    overall_time = time.time() - overall_start
    
    # Print summary
    print("\n" + "=" * 60)
    print("                 Test Summary")
    print("=" * 60)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for dim, res in results.items():
        status = "PASS" if res['failures'] == 0 and res['errors'] == 0 else "FAIL"
        print(f"  {dim.upper()}: {res['tests']} tests, "
              f"{res['failures']} failures, {res['errors']} errors - "
              f"{res['time']:.2f}s [{status}]")
        total_tests += res['tests']
        total_failures += res['failures']
        total_errors += res['errors']
    
    print("-" * 60)
    print(f"  Total: {total_tests} tests, "
          f"{total_failures} failures, {total_errors} errors")
    print(f"  Time: {overall_time:.2f} seconds")
    print("=" * 60)
    
    # Exit code
    return 0 if (total_failures == 0 and total_errors == 0) else 1


def main():
    parser = argparse.ArgumentParser(description='Run EPOCH-GPU tests')
    parser.add_argument('dimensions', nargs='*', 
                        help='Dimensions to test (1d, 2d, 3d)')
    parser.add_argument('-v', '--verbose', action='count', default=2,
                        help='Increase verbosity')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Quiet mode')
    
    args = parser.parse_args()
    
    verbosity = 0 if args.quiet else args.verbose
    dimensions = [d.lower() for d in args.dimensions] if args.dimensions else None
    
    exit_code = run_tests(dimensions, verbosity)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

