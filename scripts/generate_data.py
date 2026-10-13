"""Generate the mock customer dataset: python scripts/generate_data.py [n_customers]"""

import sys

from segmentation.data import generate_customers, save_customers

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    path = save_customers(generate_customers(n=n))
    print(f"Wrote {n:,} customers to {path}")
