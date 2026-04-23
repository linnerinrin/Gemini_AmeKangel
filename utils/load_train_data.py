import json


def txt_to_sharegpt(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    conversations = []
    for line in lines[:200]:
        line = line.strip()
        if not line:
            continue

        conv = {
            "conversations": [
                {"from": "human", "value": ""},
                {"from": "gpt", "value": line}
            ]
        }
        conversations.append(conv)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

txt_to_sharegpt("../data/ame.txt", "../data/ame.json")
txt_to_sharegpt("../data/kangel.txt", "../data/kangel.json")