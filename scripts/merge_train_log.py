import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser("merge log when train over multiple day")
    parser.add_argument("--input_file", nargs='+', type=str, help="list of input file, please put in order")
    parser.add_argument("--output_file", type=str, help="output file")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    input_files = args.input_file
    output_file = args.output_file
    data = []
    for input_file in input_files:
        with open(input_file, "r") as handle:
            lines = handle.readlines()
            lines = [l.rstrip() for l in lines]
            for idx, line in enumerate(lines):
                if idx < 40:
                    continue
                if not line.startswith("2023"):
                    continue
                data.append(line)
    with open(output_file, "w") as handle:
        for line in data:
            handle.write(line)
            handle.write("\n")