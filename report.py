from pathlib import Path
import json
from datetime import datetime
import numpy
from collections import defaultdict
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

''' metrics
concurrency
elapsed_time (second)
total prompt tokens
total completion tokens
token throughput (completion tokens)
token throughput (prompt+completion tokens)
RPS (request per second)
TTFT (second) (time to first token)
TBT (second) (time between token)
P50 (second)(50% TTFT time)
P99 (second)(99% TTFT time)
'''

def mean_without_zero(arr):
    masked_arr = numpy.ma.masked_equal(arr, 0)
    return float(masked_arr.mean()) if masked_arr.count() > 0 else 0


def main(
    output_dir: str = None,
    report_dir: str = "./report",
    tokenizer_dir: str = None,
):
    # assume jsonl file is api_monitor result
    # assume json files are input and output

    output_dir = Path(output_dir)
    if output_dir is None or not output_dir.exists():
        raise ValueError(f"Output directory {output_dir} does not exist")
    tokenizer_dir = Path(tokenizer_dir)
    if tokenizer_dir is None or not tokenizer_dir.exists():
        raise ValueError(f"Tokenizer directory {tokenizer_dir} does not exist")
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"{output_dir.name}.json"

    # Initialize variables
    concurrency = 0
    elapsed_time = 0
    RPS = 0
    TTFT = 0
    P50 = 0
    P99 = 0
    TBT = 0

    # Read data
    as_list = []
    ttft_list = []
    ts_list1 = []
    ts_list2 = []
    total_requests = 0
    tbt_dict = defaultdict(list)

    for file in output_dir.glob("*.jsonl"):
        # print(f"monitor file: {file}")
        with open(file, "r", encoding="utf-8") as file:
            for line in file:
                status = json.loads(line)

                as_list.append(status["active_sessions"])
                ts = datetime.fromisoformat(status["timestamp"])
                t1 = ts.timestamp()
                t2 = t1 + float(status["elapsed_seconds"])
                ts_list1.append(t1)
                ts_list2.append(t2)
                concurrency = int(max(concurrency, status["concurrency"]))
                total_requests = int(max(total_requests, status["total_sessions"]))
                for idx, tokens_latency in enumerate(status["tokens_latency"]):
                    if not tokens_latency:
                        if tbt_dict[idx] and type(tbt_dict[idx]) != float:
                            tbt_dict[idx] = mean_without_zero(tbt_dict[idx][1:]) # skip first token latency in each request
                    else:
                        if tokens_latency[0] == status["first_token_latency"][idx]:
                            tbt_dict[idx].extend(tokens_latency[1:]) # skip first token latency in each request
                        else:
                            tbt_dict[idx].extend(tokens_latency)
            # end file

        for idx, tbt in tbt_dict.items():
            if type(tbt) != float:
                tbt_dict[idx] = mean_without_zero(tbt)
        tbt_list = list(tbt_dict.values())

        for ttft in status["first_token_latency"]:
            ttft = float(ttft)
            if ttft != -1:
                ttft_list.append(ttft)

    max_as = max(as_list)
    min_as = min((x for x in as_list if x > 0), default=None)
    mean_as = mean_without_zero(as_list)
    elapsed_time = abs(min(ts_list1) - max(ts_list2))
    RPS = total_requests / elapsed_time
    TTFT = mean_without_zero(ttft_list)
    P50 = numpy.percentile(ttft_list, 50)
    P99 = numpy.percentile(ttft_list, 99)
    TBT = mean_without_zero(tbt_list)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    token_completion_throughput = 0
    token_throughput = 0
    for file in output_dir.glob("in_*.json"):
        with open(file, "r", encoding="utf-8") as file:
            for line in file:
                prompt = json.loads(line)
                if prompt is None:
                    continue
                try:
                    embedding = tokenizer(prompt["messages"][0]["content"])
                    total_prompt_tokens += len(embedding["input_ids"])
                except Exception as e:
                    print(file)
                    print(e)
                    print(prompt)
                    print(embedding)
                    exit(1)
    for file in output_dir.glob("out_*.json"):
        with open(file, "r", encoding="utf-8") as file:
            for line in file:
                response = json.loads(line)
                if response is None:
                    continue
                try:
                    embedding = tokenizer(response["data"]["choices"][0]["delta"]["content"])
                    total_completion_tokens += len(embedding["input_ids"])
                except Exception as e:
                    print(file)
                    print(e)
                    print(response)
                    print(embedding)
                    exit(1)
    token_completion_throughput = total_completion_tokens / elapsed_time
    token_throughput = (total_prompt_tokens + total_completion_tokens) / elapsed_time

    report_data = {
        "concurrency": concurrency,
        "active_sessions (max,min,avg)": [max_as, min_as, round(mean_as, 5)],
        "total_requests": len(ttft_list),
        "elapsed_time_seconds": round(elapsed_time, 5),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "completion_throughput": round(token_completion_throughput, 5),
        "token_throughput": round(token_throughput, 5),
        "RPS": round(RPS, 5),
        "TTFT_seconds": round(TTFT, 5),
        "P99_seconds": round(P99, 5),
        "P50_seconds": round(P50, 5),
        "token_latency_seconds": round(TBT, 5),
    }
    with open(report_file, "a+") as f:
        json.dump(report_data, f, indent=4)
        f.write("\n")

    print(f"Results saved to {report_file}:")
    print(f"concurrency: {concurrency}")
    print(f"active_sessions (max,min,avg): {max_as}, {min_as}, {mean_as:.5f}")
    print(f"total requests: {len(ttft_list)}")
    print(f"elapsed time (second): {elapsed_time:.5f}")
    print(f"total prompt tokens: {total_prompt_tokens}")
    print(f"total completion tokens: {total_completion_tokens}")
    print(f"token throughput (completion tokens / second): {token_completion_throughput:.5f}")
    print(f"token throughput (prompt+completion tokens / second): {token_throughput:.5f}")
    print(f"RPS: {RPS:.5f}")
    print(f"TTFT (second): {TTFT:.5f}")
    print(f"P99 (second): {P99:.5f}")
    print(f"P50 (second): {P50:.5f}")
    print(f"token latency (second): {TBT:.5f}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
