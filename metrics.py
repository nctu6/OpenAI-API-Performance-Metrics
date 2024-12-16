import requests
import threading
import time
import json
from datetime import datetime
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich import box
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import math
from urllib3.exceptions import InsecureRequestWarning
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(InsecureRequestWarning)

count_id = 0

class APIThroughputMonitor:
    def __init__(self, api_url, api_key, max_concurrent=5, columns=3, log_file="api_monitor.jsonl"):
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

        with self.lock:
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
            
            status = {
                "timestamp": datetime.now().isoformat(),
                "elapsed_seconds": elapsed,
                "total_chars": total_chars,
                "chars_per_second": round(chars_per_second, 2),
                "active_sessions": active_sessions,
                "completed_sessions": completed_sessions,
                "total_sessions": len(self.sessions),
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests
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
            return None
        except Exception as e:
            print(f"Error processing line: {str(e)}")
            return None

    def make_request(self, session_id):
        global count_id
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "$model...",
            "stream": True,
            "messages": [
                {"role": "user", "content": "店鋪停電應對措施"}
            ]
        }
        count_id += 1

        try:
            with self.lock:
                self.sessions[session_id] = {
                    "status": "Starting",
                    "start_time": datetime.now(),
                    "response_time": None,
                    "error": None,
                    "total_chars": 0,
                    "chunks_received": 0
                }

            start_time = time.time()
            
            # Make request with SSL verification disabled
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                stream=True,
                verify=False,
                timeout=180
            )

            # Open file for real-time writing
            with open(f"output_{session_id}.txt", "a") as f:
                for line in response.iter_lines():
                    if line:
                        content = self.process_stream_line(line)
                        if content is not None:
                            with self.lock:
                                self.sessions[session_id]["status"] = "Processing"
                                self.sessions[session_id]["chunks_received"] += 1
                                self.sessions[session_id]["total_chars"] += len(content)
                                f.write(content)
                                f.flush()

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

        with Live(
            self.generate_status_table(),
            refresh_per_second=4,
            vertical_overflow="visible",
            auto_refresh=True
        ) as live:
            with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                end_time = time.time() + duration
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

def main():
    api_url = "..."
    api_key = "sk-..."
    
    monitor = APIThroughputMonitor(
        api_url=api_url,
        api_key=api_key,
        max_concurrent=32,
        columns=3,
        log_file="api_monitor.jsonl"
    )
    
    print("\n🚀 Starting API Throughput Monitor...")
    print("Press Ctrl+C to stop the monitor\n")
    
    try:
        monitor.run(duration=60)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down monitor...")
    finally:
        print("\n✨ Monitor stopped. Final statistics displayed above.")
        print(f"Log file saved as: {monitor.log_file}")

if __name__ == "__main__":
    main()