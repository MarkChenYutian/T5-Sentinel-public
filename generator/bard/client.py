import os
import time
import json
import yaml
from tqdm import tqdm
from bardapi import Bard
from pathlib import Path
from typing import List


def run_client():
    """
    Run agent with BardAPI.
        - Read config and secrets
        - Cache work already done
        - Iterate over lines to get answers
        - Avoid rate limit if needed
    """

    with open("./generator/bard/generator-bard.yaml") as f:
        config = yaml.safe_load(f)
    with open("./generator/bard/secrets.yaml") as f:
        secrets = yaml.safe_load(f)
    
    for filename in config['required-files']:
        chains = secrets["bard-agent"]["api-keys"]
        import_path = Path(config['import-path'], filename)
        export_path = Path(config['export-path'], filename)
        question_prompt = config['question-prompt']
        
        # avoid duplicate work
        export_work = set()
        if export_path.exists():
            with open(export_path, 'r') as df2:
                for line in df2:
                    data = json.loads(line)
                    export_work.add(data['uid'])

        requests_count = 0
        df1 = open(import_path, 'r')
        df2 = open(export_path, 'a')

        # create chains of agents
        agents: List[Bard] = []
        for chain in chains:
            os.environ['_BARD_API_KEY'] = chain
            agents.append(Bard(config['timeout-seconds']))

        # iterate over lines        
        for i, line1 in tqdm(enumerate(df1), desc=f"Rephrasing"):
            try:
                # read from original dataset
                data1 = json.loads(line1)
                uid, text1 = data1['uid'], data1['text']
                if uid not in export_work:
                    # get answer from agent
                    agent = agents[i % len(agents)]
                    text2 = agent.get_answer(question_prompt + '\n' + text1)['content']
                    if text2.startswith('Response Error'):
                        continue
                    df2.write(json.dumps({'uid': uid, 'text': text2}) + '\n')
                    # avoid rate limit
                    requests_count += 1
                    if requests_count == config['requests-limit']:
                        time.sleep(config['silence-minutes'] * 60)
                        requests_count = 0
            except:
                continue
        
        # release file descriptors
        df1.close()
        df2.close()


if __name__ == "__main__":
    run_client()
