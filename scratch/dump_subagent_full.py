import json
import os

c_id = 'a3e0f475-08e0-4cb9-bbc2-4bd173be458c'
log_path = f"C:\\Users\\Acer\\.gemini\\antigravity\\brain\\{c_id}\\.system_generated\\logs\\transcript.jsonl"

if os.path.exists(log_path):
    print("Reading subagent log...")
    with open(log_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                print(f"\nStep {data.get('step_index')}: {data.get('type')} ({data.get('source')})")
                if data.get('type') == 'PLANNER_RESPONSE':
                    content = data.get('content', '')
                    print(f"  PLANNER RESPONSE: {content[:200]}...")
                    if 'tool_calls' in data:
                        for tc in data['tool_calls']:
                            print(f"    Tool: {tc.get('name') or tc.get('tool')}")
                            # If it is send_message, let's print the Message arg keys
                            if 'send_message' in str(tc):
                                print(f"      Message keys: {list(tc.get('args', {}).keys())}")
                                print(f"      Message arg type: {type(tc.get('args', {}).get('Message'))}")
                                # Print first 200 chars of message
                                m = tc.get('args', {}).get('Message', '')
                                print(f"      Message snippet: {m[:200]}...")
                                print(f"      Message total length: {len(str(m))}")
            except Exception as e:
                print(f"  Error on line {i}: {e}")
else:
    print("Log not found.")
