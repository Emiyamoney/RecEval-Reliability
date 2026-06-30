"""
报告生成脚本
用法: python scripts/generate_report.py --results_dir results/
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.report_generator import generate_report

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results/smoke")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    output = args.output or os.path.join(args.results_dir, "report.md")
    generate_report(args.results_dir, output)
    print(f"Report: {output}")

if __name__ == "__main__":
    main()
