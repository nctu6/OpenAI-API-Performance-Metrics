# SPDX-License-Identifier: Apache-2.0
"""
Offline benchmark to test the long document QA throughput.

Example usage:
    # This workload samples 8 different prompts with a default input
    # length of 20000 tokens, then replicates each prompt 2 times
    # in random order.
    python benchmark_long_document_qa_throughput.py \
        --api-url http://localhost:8910/v1/chat/completions \
        --model demo

Commandline arguments:
    --num-documents: The number of documents to sample prompts from.

    --document-length: The length of each document in tokens.
                       (Optional, default: 20000)

    --output-len: The number of tokens to generate for each prompt.
                  (Optional, default: 10)

    --repeat-count: The number of times to repeat each prompt.
                    (Optional, default: 2)

    --repeat-mode: The mode to repeat prompts. The supported modes are:
        - 'random': shuffle the prompts randomly. (Default)
        - 'tile': the entire prompt list is repeated in sequence. (Potentially
                  lowest cache hit)
        - 'interleave': each prompt is repeated consecutively before
                        moving to the next element. (Highest cache hit)

    --shuffle-seed: Random seed when the repeat mode is "random".
                    (Optional, default: 0)

"""

import os
import random
import time
import asyncio
import openai
import argparse
from transformers import AutoTokenizer

openai_client = None

async def fetch_remote(model, prompt, max_new_tokens):
    # response = await openai_client.completions.create(
    # model=model,
    # prompt=prompt,
    # max_tokens=max_new_tokens,
    # n=1,
    # stop=None,
    # temperature=0.0,
    # )
    # return response.choices[0].text

    message = [{"role": "user", "content": prompt}]
    response = await openai_client.chat.completions.create(
    model=model,
    messages=message,
    max_tokens=max_new_tokens,
    n=1,
    stop=None,
    temperature=0.0,
    )
    return response.choices[0].message.content



async def process_prompts(model, prompts, max_tokens):
    results = []
    tasks = [asyncio.create_task(fetch_remote(model, prompt, max_tokens)) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    return results


def test_long_document_qa(model=None, prompts=None, max_tokens=None, tokenizer=None):
    """
    Test long document QA with the given prompts and sampling parameters.
    Print the time spent in processing all the prompts.

    Args:
        llm: The language model used for generating responses.
        sampling_params: Sampling parameter used to generate the response.
        prompts: A list of prompt strings to be processed by the LLM.
    """
    start_time = time.time()
    results = asyncio.run(process_prompts(model, prompts, max_tokens))
    end_time = time.time()
    elipsed_time = end_time - start_time
    print(f"Time to execute all requests: {elipsed_time:.4f} secs")
    prompt_tokens = 0
    response_tokens = 0
    for p in prompts:
        prompt_tokens += len(tokenizer.encode(p))
    for r in results:
        response_tokens += len(tokenizer.encode(r))
    print("prompt tokens: ", prompt_tokens)
    print("response tokens: ", response_tokens)
    throughput = (prompt_tokens + response_tokens) / elipsed_time
    print(f"throughput: {throughput:.4f} toks/s")


def repeat_prompts(prompts, repeat_count, mode: str):
    """
    Repeat each prompt in the list for a specified number of times.
    The order of prompts in the output list depends on the mode.

    Args:
        prompts: A list of prompts to be repeated.
        repeat_count: The number of times each prompt is repeated.
        mode: The mode of repetition. Supported modes are:
            - 'random': Shuffle the prompts randomly after repetition.
            - 'tile': Repeat the entire prompt list in sequence.
              Example: [1, 2, 3] -> [1, 2, 3, 1, 2, 3].
            - 'interleave': Repeat each prompt consecutively before moving to
              the next. Example: [1, 2, 3] -> [1, 1, 2, 2, 3, 3].

    Returns:
        A list of repeated prompts in the specified order.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    print("Repeat mode: ", mode)
    if mode == 'random':
        repeated_prompts = prompts * repeat_count
        random.shuffle(repeated_prompts)
        return repeated_prompts
    elif mode == 'tile':
        return prompts * repeat_count
    elif mode == 'interleave':
        repeated_prompts = []
        for prompt in prompts:
            repeated_prompts.extend([prompt] * repeat_count)
        return repeated_prompts
    else:
        raise ValueError(f"Invalid mode: {mode}, only support "
                         "'random', 'tile', 'interleave'")


def main(args):
    global openai_client
    api_key = os.getenv("OPENAI_API_KEY", "test")
    base_url = os.getenv("OPENAI_BASE_URL", f"{args.api_url}")
    openai_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=86400)

    random.seed(args.shuffle_seed)

    # Prepare the prompts:
    # we append the document id at the beginning to avoid any of the document
    # being the prefix of other documents
    prompts = [
        str(i) + ' '.join(['hi'] * args.document_length)
        for i in range(args.num_documents)
    ]

    prompts = repeat_prompts(prompts, args.repeat_count, mode=args.repeat_mode)

    warmup_prompts = [
        "This is warm up request " + str(i) + \
                ' '.join(['hi'] * args.document_length)
        for i in range(args.num_documents)]

    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_path}")

    print("------warm up------")
    test_long_document_qa(
        model=args.model,
        prompts=warmup_prompts,
        max_tokens=args.output_len,
        tokenizer=tokenizer,
    )

    print("------start generating------")
    test_long_document_qa(
        model=args.model,
        prompts=prompts,
        max_tokens=args.output_len,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--document-length',
        type=int,
        # Roughly the number of tokens for a system paper,
        # excluding images
        default=20000,
        help='Range of input lengths for sampling prompts,'
        'specified as "min:max" (e.g., "128:256").')

    parser.add_argument('--num-documents',
                        type=int,
                        default=8,
                        help='Range of input lengths for sampling prompts,'
                        'specified as "min:max" (e.g., "128:256").')

    parser.add_argument('--output-len', type=int, default=10)

    parser.add_argument('--repeat-count',
                        type=int,
                        default=2,
                        help='Number of times to repeat each prompt')

    parser.add_argument("--repeat-mode",
                        type=str,
                        default='random',
                        help='The mode to repeat prompts. The supported '
                        'modes are "random", "tile", and "interleave". '
                        'See repeat_prompts() in the source code for details.')

    parser.add_argument("--shuffle-seed",
                        type=int,
                        default=0,
                        help='Random seed when the repeat mode is "random"')

    parser.add_argument("--api-url",
                        type=str,
                        default='http://localhost:8910/v1/chat/completions',
                        help='The api to test')

    parser.add_argument('--model',
                        type=str,
                        default='demo',
                        help='The model to test')

    parser.add_argument('--model-path',
                        type=str,
                        default=None,
                        help='The model path for loading the tokenizer')

    args = parser.parse_args()
    main(args)
