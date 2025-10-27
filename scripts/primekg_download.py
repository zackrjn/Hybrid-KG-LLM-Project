#!/usr/bin/env python3
import os
import subprocess


PRIMEKG_GIT = "https://github.com/mims-harvard/PrimeKG"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, default="third_party/PrimeKG")
    args = parser.parse_args()

    os.makedirs("third_party", exist_ok=True)

    if not os.path.exists(args.target_dir):
        print(f"Cloning PrimeKG into {args.target_dir}...")
        subprocess.check_call(["git", "clone", "--depth", "1", PRIMEKG_GIT, args.target_dir])
    else:
        print(f"PrimeKG already exists at {args.target_dir}")


