import requests
import threading
import time
import json
from datetime import datetime
import time
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich import box
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import math
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import os
from dotenv import load_dotenv
import uuid

from datasets import load_dataset

import logging

load_dotenv()

# Configure logging level from environment variable
# (Console/File) Log levels
console_log_level = os.getenv("CLL", "info").upper()
file_log_level = os.getenv("FLL", "").upper()

logger = logging.getLogger("Metrics")
logger.setLevel(getattr(logging, console_log_level, logging.DEBUG))

runtime_uuid = str(uuid.uuid4()).replace("-", "")

if "" != console_log_level:
    # Add a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)

    # Define a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

if "" != file_log_level:
    # Add a file handler
    file_handler = logging.FileHandler("metrics.log")
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Suppress SSL warnings
urllib3.disable_warnings(InsecureRequestWarning)

questions = []
count_id = 0

class FileHandler:
    def __init__(self, filename: str, mode: str, virtual: bool = False):
        self.filename = filename
        self.file = open(filename, mode) if not virtual else None

    def write(self, data):
        if self.file:
            self.file.write(data)

    def close(self):
        if self.file:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

class APIThroughputMonitor:
    def __init__(self, model: str, api_url: str, api_key: str, max_concurrent: int = 5, columns: int = 3, log_file: str = "api_monitor.jsonl", output_dir: str = None):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.columns = columns
        self.log_file = log_file
        self.sessions = {}
        self.lock = threading.Lock()
        self.console = Console()
        self.active_sessions = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.prev_total_chars = 0
        self.last_update_time = self.start_time
        self.update_interval = 0.25  # Screen update interval in seconds
        self.output_dir = output_dir

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write('')

    def get_session_status(self, session_id, info):
        status_style = {
            "Starting": "yellow",
            "Processing": "blue",
            "Completed": "green",
            "Failed": "red"
        }.get(info["status"], "white")

        return (
            f"{session_id:3d} | "
            f"[{status_style}]{info['status']:10}[/{status_style}] | "
            f"Time: {info['response_time'] or '-':8} | "
            f"Chars: {info['total_chars']:5} | "
            f"Chunks: {info['chunks_received']:3}"
        )

    def generate_status_table(self):
        table = Table(
            title="API Throughput Monitor",
            box=box.ROUNDED,
            title_style="bold magenta",
            header_style="bold cyan",
        )

        for i in range(self.columns):
            table.add_column(f"Session Group {i+1}", justify="left")

        # with self.lock: # Locking is not necessary here, since it will cause the hanging condition while make_request and log_status
        if True:
            sorted_sessions = sorted(self.sessions.items(), key=lambda x: int(x[0]))
            num_sessions = len(sorted_sessions)
            num_rows = math.ceil(num_sessions / self.columns)

            for row_idx in range(num_rows):
                row_data = []
                for col_idx in range(self.columns):
                    session_idx = row_idx * self.columns + col_idx
                    if session_idx < len(sorted_sessions):
                        session_id, info = sorted_sessions[session_idx]
                        row_data.append(self.get_session_status(session_id, info))
                    else:
                        row_data.append("")
                table.add_row(*row_data)

            elapsed_time = time.time() - self.start_time
            total_chars = sum(s["total_chars"] for s in self.sessions.values())
            total_chunks = sum(s["chunks_received"] for s in self.sessions.values())
            chars_per_sec = total_chars / elapsed_time if elapsed_time > 0 else 0

            table.add_section()
            stats_summary = (
                f"[bold cyan]Summary Stats:[/bold cyan]\n"
                f"Time: {elapsed_time:.1f}s \n"
                f"Active: {self.active_sessions} | "
                f"Total: {self.total_requests} | "
                f"Success: {self.successful_requests} | "
                f"Failed: {self.failed_requests}\n"
                f"Chars/s: {chars_per_sec:.1f} | "
                f"Total Chars: {total_chars} | "
                f"Total Chunks: {total_chunks}"
            )
            table.add_row(stats_summary)

        return table

    def log_status(self):
        current_time = time.time()
        elapsed = current_time - self.start_time

        with self.lock:
            total_chars = sum(session["total_chars"] for session in self.sessions.values())
            chars_per_second = (total_chars - self.prev_total_chars) / (current_time - self.last_log_time)
            active_sessions = len([s for s in self.sessions.values() if s["status"] in ["Starting", "Processing"]])
            completed_sessions = len([s for s in self.sessions.values() if s["status"] == "Completed"])

            ttft = [ self.sessions[id]['ttft'] for id in self.sessions ]
            tokens_latency = [ self.sessions[id]['tokens_latency'] for id in self.sessions ]
            tokens_amount = [ self.sessions[id]['tokens_amount'] for id in self.sessions ]

            for id in self.sessions:
                self.sessions[id]['ttft'] = -1
                self.sessions[id]['tokens_latency'] = []
                self.sessions[id]['tokens_amount'] = []

            status = {
                "timestamp": datetime.now().isoformat(),
                "elapsed_seconds": elapsed,
                "total_chars": total_chars,
                "chars_per_second": round(chars_per_second, 2),
                "active_sessions": active_sessions,
                "completed_sessions": completed_sessions,
                "total_sessions": len(self.sessions),
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "first_token_latency": ttft,
                "tokens_latency": tokens_latency,
                "tokens_amount": tokens_amount,
            }

            with open(self.log_file, 'a') as f:
                f.write(json.dumps(status) + '\n')

            self.prev_total_chars = total_chars
            self.last_log_time = current_time

    def process_stream_line(self, line):
        try:
            # Decode the line from bytes to string if necessary
            if isinstance(line, bytes):
                line = line.decode('utf-8')

            # Remove the "data: " prefix if it exists
            if line.startswith('data: '):
                line = line[6:]

            # Handle stream completion marker
            if line.strip() == '[DONE]':
                return None

            # Parse the JSON content
            data = json.loads(line)

            # Extract the content from the response structure
            if 'choices' in data and len(data['choices']) > 0:
                if 'delta' in data['choices'][0] and 'content' in data['choices'][0]['delta']:
                    return data['choices'][0]['delta']['content']

            return None
        except json.JSONDecodeError:
            logger.error("<<< JSON pasring error >>")
            return None
        except Exception as e:
            logger.error(f"Error processing line: {str(e)}")
            return None

    def process_stream_info(self, line):
        try:
            # Decode the line from bytes to string if necessary
            if isinstance(line, bytes):
                line = line.decode('utf-8')

            # Remove the "data: " prefix if it exists
            data_key = 'data: '
            if line.startswith(data_key):
                line = line[len(data_key):]

            if line.strip() == '[DONE]':
                return None

            data = json.loads(line)
            elapsed_time = time.time() - self.start_time
            return {"data": data, "timestamp": time.time(), "in-time": self.duration > elapsed_time}
        except json.JSONDecodeError:
            logger.error("<<< JSON pasring error >>")
            logger.debug(f"Error processing line: {line}")
            return None
        except Exception as e:
            logger.error(f"Error processing line: {str(e)}")
            logger.debug(f"Error processing line: {line}")
            return None

    def make_request(self, session_id):
        logger.debug("SESSION ID", session_id)
        global count_id
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "stream": True,
            "messages": [
                {"role": "user", "content": questions[count_id % len(questions)]}
            ]
        }
        count_id += 1

        try:
            with self.lock:
                self.sessions[session_id] = {
                    "status": "Starting",
                    "start_time": time.time(),
                    "response_time": None,
                    "error": None,
                    "total_chars": 0,
                    "chunks_received": 0,
                    "ttft": -1,
                    "tokens_latency": [],
                    "tokens_amount": [],
                }

            start_time = time.time()
            next_token_time = start_time

            # Make request with SSL verification disabled
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                stream=True,
                verify=False,
                timeout=180
            )
            # Record the payload and response to files
            payload_record = FileHandler(f"{self.output_dir}/in_{runtime_uuid}_{session_id}.json", "w", self.output_dir is None)
            output_record = FileHandler(f"{self.output_dir}/out_{runtime_uuid}_{session_id}.json", "w", self.output_dir is None)

            # Write the payload to the file
            payload_record.write(json.dumps(payload))
            payload_record.close()

            for line in response.iter_lines():
                if line:
                    data = self.process_stream_info(line)
                    output_record.write(json.dumps(data) + "\n")
                    if data is None:
                        break
                    content = data["data"]["choices"][0]["delta"]["content"]
                    with self.lock:
                        latency = round(time.time() - next_token_time, 5)
                        self.sessions[session_id]["status"] = "Processing"
                        self.sessions[session_id]["chunks_received"] += 1
                        self.sessions[session_id]["total_chars"] += len(content)
                        self.sessions[session_id]["tokens_amount"].append(len(content))
                        self.sessions[session_id]["tokens_latency"].append(latency)
                        if self.sessions[session_id]["ttft"] == -1:
                            self.sessions[session_id]["ttft"] = latency
                        next_token_time = time.time()

            output_record.close()

            response_time = time.time() - start_time

            with self.lock:
                self.sessions[session_id].update({
                    "status": "Completed",
                    "response_time": f"{response_time:.2f}s",
                    "error": None
                })
                self.successful_requests += 1

        except Exception as e:
            with self.lock:
                logger.error(f"Error in session {session_id}: {str(e)}")
                self.sessions[session_id].update({
                    "status": "Failed",
                    "error": str(e),
                    "response_time": "N/A"
                })
                self.failed_requests += 1

        finally:
            with self.lock:
                self.total_requests += 1
                self.active_sessions -= 1

    def should_update_display(self):
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False

    def run(self, duration=60):
        console = Console()
        console_height = console.height - 10
        self.duration = duration

        with Live(
            self.generate_status_table(),
            refresh_per_second=4,
            vertical_overflow="visible",
            auto_refresh=True
        ) as live:
            with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                end_time = time.time() + self.duration
                session_id = 0
                last_display_update = time.time()

                while time.time() < end_time:
                    current_time = time.time()

                    if current_time - self.last_log_time >= 1.0:
                        self.log_status()

                    if self.active_sessions < self.max_concurrent:
                        with self.lock:
                            self.active_sessions += 1
                        session_id += 1
                        executor.submit(self.make_request, session_id)

                    if self.should_update_display():
                        live.update(self.generate_status_table())

                    time.sleep(0.1)

                # Ensure all threads finish before exiting
                executor.shutdown(wait=True)
                # Force a final log update
                self.log_status()
                # Force a final console update
                live.update(self.generate_status_table())

def load_dataset_as_questions(dataset_name: str, template: str):
    # I think user might want to implement a custom data loader
    dataset = load_dataset(dataset_name)['train']
    return [template.format(**data) for data in dataset]

def main(
    model: str = "gpt-3.5-turbo",
    api_url: str = None,
    max_concurrent: int = 5,
    columns: int = 3,
    log_file: str = None,
    output_dir: str = None,
    env: str = None,
    dataset: str = "tatsu-lab/alpaca",
    template: str = "{input}\nQuestion: {instruction}",
    time_limit: int = 120
):
    global questions
    if env is not None:
        load_dotenv(env)

    questions = load_dataset_as_questions(dataset, template)

    # Set default values
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    api_url = api_url if api_url is not None else os.environ.get('API_URL')
    api_key = os.environ.get('API_KEY')
    model = model if model is not None else os.environ.get('MODEL')

    log_file = log_file if log_file is not None else f"{output_dir}/api_monitor.jsonl" if output_dir is not None else "api_monitor.jsonl"

    # Display configuration
    logger.info(f"API URL: {api_url}")
    logger.info(f"Model: {model}")
    logger.info(f"Log File: {log_file}")
    logger.info(f"Max Concurrent Requests: {max_concurrent}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Time Limit: {time_limit} seconds")

    monitor = APIThroughputMonitor(
        model="gpt-3.5-turbo" if model is None else model,
        api_url=api_url,
        api_key=api_key,
        max_concurrent=max_concurrent,
        columns=columns,
        log_file=log_file,
        output_dir=output_dir,
    )

    logger.info("🚀 Starting API Throughput Monitor...")
    logger.info("Press Ctrl+C to stop the monitor\n")

    try:
        monitor.run(duration=time_limit)
    except KeyboardInterrupt:
        logger.info("\n\n👋 Shutting down monitor...")
    finally:
        logger.info("\n✨ Monitor stopped. Final statistics displayed above.")
        logger.info(f"Log file saved as: {monitor.log_file}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)