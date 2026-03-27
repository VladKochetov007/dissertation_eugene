"""Feature construction pipeline: reads customer JSON records, builds features, writes output."""

import json
import logging
import sys

logger = logging.getLogger(__name__)

CUSTOMER_VALUE_MAP = {"High": 1, "Medium": 2, "Low": 3}
CUSTOMER_COUNTRY_MAP = {"UK": 1, "France": 2, "Australia": 3}


def encode_feature(mapping, value):
    """Look up a categorical value in a mapping dict; raise ValueError if unknown."""
    if value not in mapping:
        raise ValueError(f"Unknown feature value: {value!r}")
    return mapping[value]


def build_features(customer_data):
    """Construct the feature dict for a single customer record."""
    cid = customer_data.get("customer_id")
    customer_features = {"customer_id": cid if isinstance(cid, int) else -1}

    customer_features["categorical_features"] = [
        encode_feature(CUSTOMER_VALUE_MAP, customer_data["customer_value"]),
        encode_feature(CUSTOMER_COUNTRY_MAP, customer_data["customer_country"]),
    ]

    order_counts = customer_data["global_order_count"]
    visit_counts = customer_data["global_visit_count"]

    if len(order_counts) != len(visit_counts):
        raise ValueError("global_order_count and global_visit_count must have equal length")

    try:
        customer_features["numerical_averages"] = [
            o / v for o, v in zip(order_counts, visit_counts)
        ]
    except ZeroDivisionError:
        logger.error("Zero in global_visit_count for customer %s", cid)
        customer_features["numerical_averages"] = []

    return customer_features


def process_data_file(input_filename, output_filename):
    """Read JSON-lines input, compute features, write JSON-lines output."""
    with open(input_filename) as infile, open(output_filename, "w") as outfile:
        for line in infile:
            outfile.write(json.dumps(build_features(json.loads(line))) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_file> <output_file>")
        sys.exit(1)

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    logger.info("Processing features for file: %s", input_filename)
    process_data_file(input_filename, output_filename)
