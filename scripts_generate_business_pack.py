from __future__ import annotations

import argparse
from pprint import pprint

from m5_pipeline.business_outputs import BusinessPackConfig, generate_business_pack


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate executive KPI, scenario, reason-code, and dashboard-ready outputs."
    )
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    out = generate_business_pack(BusinessPackConfig(data_dir=args.data_dir))
    pprint(out)


if __name__ == "__main__":
    main()
