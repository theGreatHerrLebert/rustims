"""
DIA clustering pipeline.

This module provides CLI tools for running the DIA clustering pipeline.
"""
from __future__ import annotations
import argparse


def main():
    """Main entry point for the cluster pipeline CLI."""
    parser = argparse.ArgumentParser(description="DIA clustering pipeline")
    parser.add_argument("--data", required=True, help="Path to timsTOF DIA dataset (.d folder)")
    parser.add_argument("--out", required=True, help="Output directory for cluster results")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads to use")

    args = parser.parse_args()

    print(f"DIA clustering pipeline")
    print(f"  Data: {args.data}")
    print(f"  Output: {args.out}")
    print(f"  Threads: {args.threads}")
    print()
    print("Note: Full pipeline implementation is in development.")
    print("For now, use the Python API directly:")
    print()
    print("  from imspy_dia import TimsDatasetDIAClustering")
    print("  ds = TimsDatasetDIAClustering(data_path)")
    print("  # ... run clustering methods")


if __name__ == "__main__":
    main()
