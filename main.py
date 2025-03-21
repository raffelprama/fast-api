from gevent import monkey
monkey.patch_all()

from fastapi import FastAPI, Query, HTTPException
from typing import Union
import uvicorn
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import subprocess
import logging
import os
from locustfile import get_current_metrics, events, HttpUser, run_single_user, task
import uuid
from datasets import load_dataset
import random
import json
import requests
import pickle

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='\n\n\n%(asctime)s - %(name)s - %(levelname)s - %(message)s ')

app = FastAPI()

# Global variables for test configuration
user_global = 100
spawnrate_global = 100
model_global = "deepseek-ai/DeepSeek-R1-70B"
duration_global = 60
status_url=""
global model_data
global global_data

# Global variables for metrics storage
metrics_data = {
    "time_to_first_token": {"average": 0, "maximum": 0, "minimum": 0, "median": 0},
    "end_to_end_latency": {"average": 0, "maximum": 0, "minimum": 0, "median": 0},
    "inter_token_latency": {"average": 0, "maximum": 0, "minimum": 0, "median": 0},
    "token_speed": {"average": 0, "maximum": 0, "minimum": 0, "median": 0},
    "throughput": {
        "input_tokens_per_second": 0,
        "output_tokens_per_second": 0
    }
}

@app.post("/run-load-test")
async def run_load_test(
        user: Union[int, None] = Query(default=100), 
        spawnrate: Union[int, None] = Query(default=100),
        model: Union[str, None] = Query(None), #default="deepseek-ai/DeepSeek-R1-70B"
        tokenizer: Union[str, None] = Query(None), #default="deepseek-ai/DeepSeek-R1"
        url: Union[str, None] = Query(default="https://dekallm.cloudeka.ai"),
        duration: Union[int, None] = Query(default=60),
        ):
    try:
        def testing2(url_test):
            testingurl = requests.get(url_test)
            print (testingurl.status_code)

        # MAIN PROGRAM
    # try:
        # Set environment variables for Locust
        os.environ["LOCUST_USERS"] = str(user)
        os.environ["LOCUST_SPAWN_RATE"] = str(spawnrate)
        os.environ["LOCUST_DURATION"] = str(duration)
        os.environ["LOCUST_HOST"] = str(url)

        if model==None:
            print("MODEL NONE")
            model_name = f"{url}/v1/models"
            models_response = requests.get(model_name, headers={"Authorization": "Bearer sk-G1_wkZ37sEmY4eqnGdcNig"})
            if models_response.status_code == 200:
                model_data = models_response.json()
                print(model_data)
                model_data = model_data.get("data", [{}])[0].get("id", "meta-llama/Llama-3.2-90B-Vision-Instruct")
            else:
                print(f"Failed to fetch models, status: {models_response.status_code}")
                model_data = "meta-llama/Llama-3.2-90B-Vision-Instruct"
        else:
            model_data = str(model)

        print(f"Using model:", model_data)
        os.environ["LOCUST_MODEL"] = model_data
        
        if tokenizer==None:
            print("TOKEN NONE")
            token_data = model_data
        else: 
            token_data = str(tokenizer)

        print(f"Using token:", token_data)
        print(f"end token")
        
        os.environ["LOCUST_TOKENIZER"] = token_data

        logger.info("Environment variables set")
        # Start the Locust test
        locust_command = [
            "locust",
            "-f", "locustfile.py",
            "--headless",
            "--users", str(user),
            "--spawn-rate", str(spawnrate),
            "--run-time", f"{duration}s",
            "--host", url
        ]

        try:
            print(f"Testing url = {url}")
            testing2(url)
        except Exception as e:
            return HTTPException(status_code=400, detail=str(e))

        logger.info("tester4")
        # Start the Locust process and capture output
        process = subprocess.Popen(
            locust_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Wait for the process to complete
        stdout, stderr = process.communicate()
        # print("Locust process output:", stdout)
        print("Locust process error:", stderr)

        metrics={}
        # Now get the metrics
        metrics = get_current_metrics()
        logger.info("Metrics:", metrics)
        print(f"Metrics: {metrics}")

        # # Check if the process completed successfully
        # if process.returncode != 0:
        #     # logger.info("Error: Test completed with some failure", process.returncode)
        #     logger.info("tester4")
        #     return {
        #             "ps": process.returncode,
        #             "status": "Test completed 1", 
        #             "metrics": metrics["metrics"] if isinstance(metrics, dict) and "metrics" in metrics else metrics,
        #             "configuration": {
        #                     "user": user,
        #                     "spawnrate": spawnrate,
        #                     "model": model_data,
        #                     "tokenizer": token_data,
        #                     "url": url,
        #                     "duration": duration
        #                 },
        #             # "error": stderr
        #             }
        # # return HTTPException(status_code=401, detail=str(e))
        return {
            # "errror": e,
            # "ps": process.returncode,
            "status": "Test completed", 
            "metrics": metrics["metrics"] if isinstance(metrics, dict) and "metrics" in metrics else metrics,
            "configuration": {
                "user": user,
                "spawnrate": spawnrate,
                "model": model_data,
                "tokenizer": token_data,
                "url": url,
                "duration": duration
            }
        }
    # except Exception as e:
            
    except Exception as e:
        logger.error(f"Error in run_load_test: 4 {e}")
        return HTTPException(status_code=402, detail=str(e))
        

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)