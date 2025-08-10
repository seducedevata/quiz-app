
import sys
import json

CONFIG_FILE = "config.json"

def get_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_config(config_data):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_data = json.loads(sys.argv[1])
        save_config(config_data)
        print(json.dumps({"status": "success"}))
    else:
        config = get_config()
        print(json.dumps(config))
