import os
import time
import random
import json
import uuid
import statistics
from locust import HttpUser, task, between, events, LoadTestShape, run_single_user
from transformers import AutoTokenizer 
from datasets import load_dataset 
import tempfile

# Load default environment variables
model_name = str(os.getenv("LOCUST_MODEL", "deepseek-ai/DeepSeek-R1-70B"))
tokenizer_name = str(os.getenv("LOCUST_TOKENIZER", "deepseek-ai/DeepSeek-R1"))  
host_url = str(os.getenv("LOCUST_HOST", "https://dekallm.cloudeka.ai"))   #edited
locust_users = int(os.getenv("LOCUST_USERS", 100))
locust_spawn_rate = int(os.getenv("LOCUST_SPAWN_RATE", 100))
locust_duration = int(os.getenv("LOCUST_DURATION", 60))

# Initialize tokenizer with user-specified tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
    token="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"
)

# Load Banking77 dataset prompts
dataset = load_dataset("mteb/banking77", split="test")
prompts = dataset["text"]

# Global variables for metrics collection
ttft_times = []
end_to_end_latencies = []
inter_token_latencies = []
tokens_per_second_list = []
start_benchmark_time = None
total_input_tokens = 0
total_output_tokens = 0

# Statistics variables
ttft_stats_avg = 10
ttft_stats_max = 10
ttft_stats_min = 10
ttft_stats_median = 10

e2e_stats_avg = 10
e2e_stats_max = 10
e2e_stats_min = 10
e2e_stats_median = 10

token_speed_stats_avg = 10
token_speed_stats_max = 10
token_speed_stats_min = 10
token_speed_stats_median = 10

inter_token_stats_avg = 10
inter_token_stats_max = 10
inter_token_stats_min = 10
inter_token_stats_median = 10

input_token_throughput = 0
output_token_throughput = 0

# Create a temporary file path for metrics
METRICS_FILE = os.path.join(tempfile.gettempdir(), 'locust_metrics.json')

def calculate_stats(data):
    if not data:
        return {
            "average": 0,
            "maximum": 0,
            "minimum": 0,
            "median": 0,
        }
    return {
        "average": round(sum(data) / len(data), 2),
        "maximum": round(max(data), 2),
        "minimum": round(min(data), 2),
        "median": round(statistics.median(data), 2),
    }

class nemotron(HttpUser):
    # Set wait time between tasks to 0.5 to 5 seconds
    wait_time = between(0.5, 5)

    # Set default host from environment variable
    host = host_url

    @task()
    def generate_response(self):
        global total_input_tokens, total_output_tokens, start_benchmark_time
        global ttft_times, end_to_end_latencies, inter_token_latencies, tokens_per_second_list

        if start_benchmark_time is None:
            start_benchmark_time = time.time()

        # Track request start time
        start_time = time.time()
        first_token_time = None
        tokens = []
        
        try:
            # Select a random prompt and append UUID
            input_text = f"{random.choice(prompts)} {uuid.uuid4()}"
            input_length = len(tokenizer(input_text)['input_ids'])
            total_input_tokens += input_length

            # Send request
            # print("------------data-----------------")
            # print(f"model used: {model_name}")
            # print(f"tokenizer used: {tokenizer_name}")
            # print(f"url used: {host_url}")
            # print(f"duration used: {locust_duration}")
            # print(f"user used: {locust_users}")
            # print(f"spawnrate used: {locust_spawn_rate}")
            # print("--------------------------------")

            response = self.client.post(
                url="/v1/chat/completions",
                headers={
                    'Content-type': 'application/json',
                    'Accept': 'application/json',
                    'Authorization': 'Bearer sk-G1_wkZ37sEmY4eqnGdcNig'
                },

                data=json.dumps({
                    "model": model_name,
                    "messages": [{"role": "user", "content": input_text}],
                    "stream": True,
                    "temperature": 0.9,
                    "top_p": 0.9,
                    "max_tokens": 128,
                    "min_tokens": 20
                }),
                stream=True
            )

            # Process streamed response
            for line in response.iter_lines():
                if line:
                    token_time = time.time()
                    if first_token_time is None:
                        first_token_time = token_time
                        ttft = (first_token_time - start_time) * 1000
                        ttft_times.append(ttft)
                    tokens.append(line)

            # Calculate metrics
            end_time = time.time()
            e2e_latency = (end_time - start_time) * 1000
            end_to_end_latencies.append(e2e_latency)

            output_length = len(tokens)
            total_output_tokens += output_length

            if len(tokens) > 1:
                itl = ((end_time - first_token_time) / (len(tokens) - 1)) * 1000
                inter_token_latencies.append(itl)

            token_speed = output_length / (end_time - start_time)
            tokens_per_second_list.append(token_speed)

            # Debug print
            print(f"Request completed - TTFT: {ttft_times[-1]:.2f}ms, E2E: {e2e_latency:.2f}ms")

        except Exception as e:
            print(f"Error in request: {str(e)}")

@events.quitting.add_listener
def display_metrics_summary(environment, **kwargs):
    global input_token_throughput, output_token_throughput
    global ttft_stats_avg, ttft_stats_max, ttft_stats_min, ttft_stats_median
    global e2e_stats_avg, e2e_stats_max, e2e_stats_min, e2e_stats_median
    global inter_token_stats_avg, inter_token_stats_max, inter_token_stats_min, inter_token_stats_median
    global token_speed_stats_avg, token_speed_stats_max, token_speed_stats_min, token_speed_stats_median

    # Debug print for data collection
    print(f"\nCollected data points:")
    print(f"TTFT times: {len(ttft_times)} points")
    print(f"E2E latencies: {len(end_to_end_latencies)} points")
    print(f"Inter-token latencies: {len(inter_token_latencies)} points")
    print(f"Token speeds: {len(tokens_per_second_list)} points")

    # Calculate stats only if we have data
    if ttft_times:  # Check if we have any data
        # Calculate benchmark duration
        benchmark_duration = time.time() - start_benchmark_time if start_benchmark_time else 0

        # Calculate throughput
        input_token_throughput = total_input_tokens / benchmark_duration if benchmark_duration > 0 else 0
        output_token_throughput = total_output_tokens / benchmark_duration if benchmark_duration > 0 else 0

        # Calculate stats
        ttft_stats = calculate_stats(ttft_times)
        e2e_stats = calculate_stats(end_to_end_latencies)
        inter_token_stats = calculate_stats(inter_token_latencies)
        token_speed_stats = calculate_stats(tokens_per_second_list)

        # Store metrics in a temporary file
        metrics = {
            "metrics": {
                "time_to_first_token": ttft_stats,
                "end_to_end_latency": e2e_stats,
                "inter_token_latency": inter_token_stats,
                "token_speed": token_speed_stats,
                "throughput": {
                    "input_tokens_per_second": round(input_token_throughput, 2),
                    "output_tokens_per_second": round(output_token_throughput, 2)
                }
            }
        }
        
        # Save metrics to temporary file
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics, f)

        # Update global statistics variables with actual values
        ttft_stats_avg, ttft_stats_max = ttft_stats['average'], ttft_stats['maximum']
        ttft_stats_min, ttft_stats_median = ttft_stats['minimum'], ttft_stats['median']
        
        e2e_stats_avg, e2e_stats_max = e2e_stats['average'], e2e_stats['maximum']
        e2e_stats_min, e2e_stats_median = e2e_stats['minimum'], e2e_stats['median']
        
        inter_token_stats_avg, inter_token_stats_max = inter_token_stats['average'], inter_token_stats['maximum']
        inter_token_stats_min, inter_token_stats_median = inter_token_stats['minimum'], inter_token_stats['median']
        
        token_speed_stats_avg, token_speed_stats_max = token_speed_stats['average'], token_speed_stats['maximum']
        token_speed_stats_min, token_speed_stats_median = token_speed_stats['minimum'], token_speed_stats['median']

        # # Debug print for calculated values
        # print("\nCalculated values:")
        # print(f"TTFT avg: {ttft_stats_avg}")
        # print(f"E2E avg: {e2e_stats_avg}")
        # print(f"Throughput: {input_token_throughput:.2f} input, {output_token_throughput:.2f} output")

        # # Print metrics summary
        # print("\n--- Metrics Summary ---")
        # print(f"{'Metric':<40} {'Average':<10} {'Max':<10} {'Min':<10} {'Median':<10}")
        # print("-" * 80)
        # print(f"{'Time to First Token (ms)':<40} {ttft_stats_avg:<10.2f} {ttft_stats_max:<10.2f} {ttft_stats_min:<10.2f} {ttft_stats_median:<10.2f}")
        # print(f"{'End-to-End Latency (ms)':<40} {e2e_stats_avg:<10.2f} {e2e_stats_max:<10.2f} {e2e_stats_min:<10.2f} {e2e_stats_median:<10.2f}")
        # print(f"{'Inter-Token Latency (ms)':<40} {inter_token_stats_avg:<10.2f} {inter_token_stats_max:<10.2f} {inter_token_stats_min:<10.2f} {inter_token_stats_median:<10.2f}")
        # print(f"{'Individual User Token Speed (tokens/sec)':<40} {token_speed_stats_avg:<10.2f} {token_speed_stats_max:<10.2f} {token_speed_stats_min:<10.2f} {token_speed_stats_median:<10.2f}")
        # print(f"{'Input Token Throughput (tokens/sec)':<40} {input_token_throughput:<10.2f}")
        # print(f"{'Output Token Throughput (tokens/sec)':<40} {output_token_throughput:<10.2f}")
        # print("-" * 80)

def get_current_metrics():
    print("------------get_current_metrics-----------------")
    try:
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "metrics": {
                "time_to_first_token": calculate_stats(ttft_times),
                "end_to_end_latency": calculate_stats(end_to_end_latencies),
                "inter_token_latency": calculate_stats(inter_token_latencies),
                "token_speed": calculate_stats(tokens_per_second_list),
                "throughput": {
                    "input_tokens_per_second": round(input_token_throughput, 2),
                    "output_tokens_per_second": round(output_token_throughput, 2)
                }
            }
        }
    


# Define the load test shape
class StagesShape(LoadTestShape):
    """
    Fixed staged load pattern that runs through predefined stages sequentially
    """

    def tick(self):
        run_time = self.get_run_time()
        
        if run_time < locust_duration:
            return (locust_users, locust_spawn_rate)
        return None

# Start the benchmark
start_benchmark_time = time.time()