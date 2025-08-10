
import sys
import json
import time

def generate_streaming_mcq(config):
    # Simulate a delay to represent the complex generation process
    time.sleep(1)

    for i in range(config.get("numQuestions", 1)):
        question = f"This is a streaming {config.get('difficulty', 'medium')} question about {config.get('topic', 'a topic')}."
        for char in question:
            print(json.dumps({"type": "token", "data": char}))
            sys.stdout.flush()
            time.sleep(0.01)

        print(json.dumps({"type": "question_end"}))
        sys.stdout.flush()

    print(json.dumps({"type": "stream_end"}))
    sys.stdout.flush()

if __name__ == "__main__":
    config = json.loads(sys.argv[1])
    generate_streaming_mcq(config)
