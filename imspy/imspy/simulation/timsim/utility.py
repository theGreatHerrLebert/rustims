import os
import argparse


# check if the path exists
def check_path(p: str) -> str:
    if not os.path.exists(p):
        raise argparse.ArgumentTypeError(f"Invalid path: {p}")
    return p
