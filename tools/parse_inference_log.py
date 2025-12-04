#!/usr/bin/env python3
"""
Log Parser for Inference Pipeline
Extracts configuration, performance metrics, and durations from inference log files.

Usage:
  ./parse_inference_log.py <inference.log> [output.csv]

This file is intentionally placed under `tools/` as a lightweight utility used by
the bash wrapper `tools/parse_inference_log.sh`.
"""

import re
import csv
import sys
from pathlib import Path


def parse_log_file(log_path):
    """Parse inference log file and extract job information."""

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Find all job starts and create sections
    job_starts = list(re.finditer(r"\[(\d+:\d+:\d+)\] Starting: (\S+)", content))

    if not job_starts:
        print("Warning: No jobs found in log file")
        return []

    job_sections = []
    for i, start_match in enumerate(job_starts):
        start_pos = start_match.start()
        if i + 1 < len(job_starts):
            end_pos = job_starts[i + 1].start()
        else:
            end_pos = len(content)

        job_sections.append(
            {
                "name": start_match.group(2),
                "start_time": start_match.group(1),
                "content": content[start_pos:end_pos],
            }
        )

    # Parse each job section
    all_jobs = []

    for section in job_sections:
        job_data = {
            "name": section["name"],
            "start_time": section["start_time"],
        }

        text = section["content"]

        # Extract end time
        done_match = re.search(
            r"\[(\d+:\d+:\d+)\] DONE: " + re.escape(section["name"]), text
        )
        if done_match:
            job_data["end_time"] = done_match.group(1)

        # Extract configuration
        config_patterns = {
            "total_files": r"│ Total Files\s+│\s+([0-9,]+)",
            "to_process": r"│ To Process\s+│\s+([0-9,]+)",
            "avg_chunks_per_file": r"│ Avg Chunks/File\s+│\s+([0-9.]+)",
            "ram_budget_gb": r"│ RAM Budget\s+│\s+([0-9.]+) GB",
            "files_per_group": r"│ Files per Group\s+│\s+([0-9,]+)",
            "total_groups": r"│ Total Groups\s+│\s+([0-9,]+)",
            "system_cpus": r"│ System CPUs\s+│\s+([0-9,]+)",
            "max_cpus": r"│ Max CPUs \(limit\)\s+│\s+([0-9,]+)",
            "inference_workers": r"│ Inference Workers\s+│\s+([0-9,]+)",
            "threads_per_worker": r"│ Threads per Worker\s+│\s+([0-9,]+)",
            "loader_threads": r"│ Loader Threads\s+│\s+([0-9,]+)",
            "batch_size": r"│ Batch Size\s+│\s+([0-9,]+)",
            "embedding_format": r"│ Embedding Format\s+│\s+(\S+)",
        }

        for key, pattern in config_patterns.items():
            match = re.search(pattern, text)
            if match:
                val = match.group(1).replace(",", "")
                job_data[key] = val

        # Extract performance metrics
        perf_patterns = {
            "files_processed": r"║ Files Processed\s+│\s+([0-9,]+)",
            "running_time_s": r"║ Total Running Time\s+│\s+([0-9.]+)s",
            "speed_realtime": r"║ Speed\s+│\s+([0-9.]+)x realtime",
            "throughput_files_sec": r"║ Throughput\s+│\s+([0-9.]+) files/sec",
            "cpu_efficiency_pct": r"║ CPU Efficiency\s+│\s+([0-9.]+)%",
        }

        for key, pattern in perf_patterns.items():
            match = re.search(pattern, text)
            if match:
                val = match.group(1).replace(",", "")
                job_data[key] = val

        # Extract output sizes
        emb_match = re.search(r"Embeddings size: ([0-9.]+) GB", text)
        pred_match = re.search(r"Predictions size: ([0-9.]+) MB", text)

        if emb_match:
            job_data["embeddings_gb"] = emb_match.group(1)
        if pred_match:
            job_data["predictions_mb"] = pred_match.group(1)

        all_jobs.append(job_data)

    return all_jobs


def print_summary(jobs):
    """Print summary statistics from parsed jobs."""

    if not jobs:
        print("No jobs to summarize")
        return

    # Calculate statistics
    total_files = sum(
        int(j.get("files_processed", 0)) for j in jobs if j.get("files_processed")
    )
    total_time = sum(
        float(j.get("running_time_s", 0)) for j in jobs if j.get("running_time_s")
    )

    jobs_with_throughput = [j for j in jobs if j.get("throughput_files_sec")]
    jobs_with_speed = [j for j in jobs if j.get("speed_realtime")]

    avg_throughput = (
        (
            sum(float(j["throughput_files_sec"]) for j in jobs_with_throughput)
            / len(jobs_with_throughput)
        )
        if jobs_with_throughput
        else 0
    )
    avg_speed = (
        (
            sum(float(j["speed_realtime"]) for j in jobs_with_speed)
            / len(jobs_with_speed)
        )
        if jobs_with_speed
        else 0
    )

    # Get common configuration from first job
    sample_job = jobs[0]

    print("=" * 80)
    print("INFERENCE LOG SUMMARY")
    print("=" * 80)
    print(f"\n## Overview")
    print(f"- Total job runs: {len(jobs)}")
    print(f"- Total files processed: {total_files:,}")
    print(
        f"- Total running time: {total_time / 3600:.2f} hours ({total_time:.0f} seconds)"
    )
    if avg_throughput > 0:
        print(f"- Average throughput: {avg_throughput:.2f} files/second")
    if avg_speed > 0:
        print(f"- Average speed: {avg_speed:.1f}x realtime")

    print(f"\n## System Configuration")
    print(f"- Inference workers: {sample_job.get('inference_workers', 'N/A')}")
    print(f"- Threads per worker: {sample_job.get('threads_per_worker', 'N/A')}")
    print(f"- Batch size: {sample_job.get('batch_size', 'N/A')}")
    print(f"- RAM budget: {sample_job.get('ram_budget_gb', 'N/A')} GB")
    print(f"- Embedding format: {sample_job.get('embedding_format', 'N/A')}")
    if sample_job.get("cpu_efficiency_pct"):
        print(f"- CPU efficiency: ~{sample_job.get('cpu_efficiency_pct')}%")

    # Performance range
    throughputs = [float(j["throughput_files_sec"]) for j in jobs_with_throughput]
    speeds = [float(j["speed_realtime"]) for j in jobs_with_speed]

    if throughputs or speeds:
        print(f"\n## Performance Range")
        if throughputs:
            print(
                f"- Throughput: {min(throughputs):.2f} - {max(throughputs):.2f} files/sec"
            )
        if speeds:
            print(f"- Speed: {min(speeds):.1f}x - {max(speeds):.1f}x realtime")

    # Output sizes
    total_embeddings = sum(
        float(j.get("embeddings_gb", 0)) for j in jobs if j.get("embeddings_gb")
    )
    total_predictions = sum(
        float(j.get("predictions_mb", 0)) for j in jobs if j.get("predictions_mb")
    )

    if total_embeddings > 0 or total_predictions > 0:
        print(f"\n## Output Sizes")
        if total_embeddings > 0:
            print(f"- Total embeddings: {total_embeddings:.2f} GB")
        if total_predictions > 0:
            print(
                f"- Total predictions: {total_predictions:.2f} MB ({total_predictions / 1024:.2f} GB)"
            )

    print("=" * 80)


def export_csv(jobs, output_path):
    """Export job data to CSV file."""

    fieldnames = [
        "name",
        "start_time",
        "end_time",
        "total_files",
        "files_processed",
        "running_time_s",
        "throughput_files_sec",
        "speed_realtime",
        "cpu_efficiency_pct",
        "embeddings_gb",
        "predictions_mb",
        "inference_workers",
        "threads_per_worker",
        "batch_size",
        "ram_budget_gb",
        "total_groups",
        "embedding_format",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for job in jobs:
            writer.writerow({k: job.get(k, "") for k in fieldnames})

    print(f"\n✓ Detailed results exported to: {output_path}")


def main():
    """Main entry point."""

    if len(sys.argv) < 2:
        print("Usage: python parse_inference_log.py <log_file> [output.csv]")
        print("\nExample:")
        print("  python parse_inference_log.py inference.log")
        print("  python parse_inference_log.py inference.log results.csv")
        sys.exit(1)

    log_file = Path(sys.argv[1])

    if not log_file.exists():
        print(f"Error: Log file '{log_file}' not found")
        sys.exit(1)

    # Parse the log
    print(f"Parsing log file: {log_file}...")
    jobs = parse_log_file(log_file)
    print(f"Successfully parsed {len(jobs)} jobs\n")

    # Print summary
    print_summary(jobs)

    # Export to CSV
    if len(sys.argv) >= 3:
        csv_output = Path(sys.argv[2])
    else:
        csv_output = log_file.with_suffix(".csv")

    export_csv(jobs, csv_output)


if __name__ == "__main__":
    main()
