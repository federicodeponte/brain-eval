import json,re
from pathlib import Path
H=Path.home()/'work/vals-hyp'
out=H/'controls/sensor';out.mkdir(parents=True,exist_ok=True)
rows=[json.loads(l) for l in open(H/'human-neurons/tasks.jsonl')][:20]
HEAD=("Below are counts from 40 sensors on an industrial machine. Each number is how many pulses a sensor logged "
      "during a one-second window. Each row is one reading; columns are the same sensors in every row.")
with open(out/'tasks.jsonl','w') as f:
    for t in rows:
        p=t['prompt']
        body=p[p.index('\n\nneuron:')+2:]
        body=body.replace('neuron:','sensor:',1)
        body=re.sub(r'\nregion:[^\n]*','',body)
        names=re.findall(r'Class ([A-D]) = photo of (\w+):',body)
        for L,n in names: body=body.replace(f'Class {L} = photo of {n}:',f'Class {L} = operating mode {L}:')
        body=re.sub(r'Query pseudo-trial \([^)]*\):','Query reading (a new reading, not among the example rows above):',body)
        body=body.replace('  query:','  query:')
        q=body[body.index('Which category'):]
        body=body[:body.index('Which category')]+"Which operating mode was the machine in during the query reading? A) mode A B) mode B C) mode C D) mode D\nAnswer with a single letter."
        assert 'neuron' not in body.lower() and 'photo' not in body.lower() and 'brain' not in body.lower(), t['id']
        n=dict(t);n['prompt']=HEAD+'\n\n'+body;n['id']='sensor-'+t['id'].split('-')[-1];n['src_id']=t['id']
        f.write(json.dumps(n)+'\n')
print(n['prompt'][:500]);print('...');print(n['prompt'][-300:])
