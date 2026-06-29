import json
import os

c_id = 'a3e0f475-08e0-4cb9-bbc2-4bd173be458c'
log_path = f"C:\\Users\\Acer\\.gemini\\antigravity\\brain\\{c_id}\\.system_generated\\logs\\transcript.jsonl"

if os.path.exists(log_path):
    print("Reading subagent log...")
    with open(log_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if 'send_message' in line:
                try:
                    data = json.loads(line)
                    for tc in data.get('tool_calls', []):
                        if 'send_message' in tc.get('name', '') or 'send_message' in tc.get('tool', ''):
                            msg = tc.get('args', {}).get('Message', '')
                            print(f"Message length in json: {len(msg)}")
                            print(f"Ends with: {msg[-100:]}")
                            # Save to scratch
                            with open(f"C:\\Users\\Acer\\.gemini\\antigravity\\brain\\de435c1e-f126-4fcd-9522-4f9fd2a18fd3\\scratch\\raw_{c_id}.txt", 'w', encoding='utf-8') as out:
                                out.write(msg)
                            print("Saved raw message.")
                except Exception as e:
                    print(f"Error parsing line {i}: {e}")
else:
    print("Log not found.")
