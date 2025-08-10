import sys
import json
import time

def log_event(event_data):
    with open("session_log.jsonl", "a") as f:
        f.write(json.dumps(event_data) + "\n")

    return {
        "status": "success"
    }

if __name__ == "__main__":
    event_data = json.loads(sys.argv[1])
    event_data["timestamp"] = time.time()
    result = log_event(event_data)
    print(json.dumps(result))