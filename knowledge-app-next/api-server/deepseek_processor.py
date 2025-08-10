import sys
import json
import time

def process_document(file_path):
    stages = [
        {"stage": "initialization", "message": "Initializing training environment...", "progress": 10},
        {"stage": "data_loading", "message": "Loading training data...", "progress": 30},
        {"stage": "model_setup", "message": "Setting up model architecture...", "progress": 50},
        {"stage": "training", "message": "Training in progress...", "progress": 80},
        {"stage": "validation", "message": "Validating model performance...", "progress": 95},
        {"stage": "completion", "message": "Training completed successfully!", "progress": 100}
    ]

    for stage in stages:
        print(json.dumps(stage))
        sys.stdout.flush()
        time.sleep(1)

if __name__ == "__main__":
    file_path = sys.argv[1]
    process_document(file_path)