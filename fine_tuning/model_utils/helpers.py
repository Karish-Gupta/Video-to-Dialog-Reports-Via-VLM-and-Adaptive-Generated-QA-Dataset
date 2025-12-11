import json

# Helper to load JSONL
def load_jsonl(filename):
   data = []
   with open(filename, 'r') as f:
      for line in f:
         data.append(json.loads(line))
   return data

# Helper to save JSONL
def save_jsonl(data, filename):
   with open(filename, 'w') as f:
      for entry in data:
         f.write(json.dumps(entry) + '\n')