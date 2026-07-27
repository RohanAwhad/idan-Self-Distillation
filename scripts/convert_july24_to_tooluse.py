import json
from datasets import Dataset

src = 'data/maas_data/eshwars_data/july_24/train_msgs.jsonl'
dst = 'data/tooluse_data/july_24_train'

rows = []
with open(src) as f:
    for line in f:
        d = json.loads(line)
        msgs = d['messages']
        system_content = msgs[0]['content']
        user_content = msgs[1]['content']
        assistant_content = msgs[2]['content']
        rows.append({
            'prompt': system_content + '\n' + user_content,
            'name': '',
            'description': '',
            'nl_documentation': '',
            'instruction': '',
            'golden_answer': {'Action': [], 'Action_Input': []},
            'golden_response': [assistant_content],
        })

ds = Dataset.from_list(rows)
ds.save_to_disk(dst)
print(f"Saved {len(ds)} samples to {dst}")
