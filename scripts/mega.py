"""Official Mega inputs with locked Nex transport; --dry-run is offline."""
import argparse
import hashlib
import html
import json
import math
import re
from pathlib import Path

import json5
import pandas as pd
import yaml
from timed_nex import MODEL, PRIMARY_EFFORT, FALLBACK_EFFORT, THINK_SECONDS, ANSWER_SECONDS, answer

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'upstream/Twin-2K-500-Mega-Study-main'


def config(study):
    return yaml.safe_load((UPSTREAM / f'configs/{study}/{study}.yaml').read_text(encoding='utf-8'))['text_simulation']


def prompt_for(study, survey, persona):
    cfg = config(study)
    join = cfg['create_text_simulation_input']
    instruction = cfg['LLM_simulation']['system_instruction'].strip()
    header = join.get('persona_prompt_header', "## Persona Profile (This individual's past survey responses):").strip()
    separator = join.get('persona_question_prompt_separator', '## New Survey Question & Instructions (Please respond as the persona described above):').strip()
    # Explicit official instruction in one user message, as required by the user.
    # Published survey_text already includes the entire output template.
    return '\n\n'.join([instruction, header, persona, separator, survey])


def schema_for(survey):
    marker = re.search(r'#+\s*Format Instructions:', survey)
    if not marker:
        raise ValueError('Published survey has no format instructions')
    tail = survey[marker.end():]
    start = tail.find('{')
    if start < 0:
        raise ValueError('No published response template')
    # Published templates contain Python-style array repetition and occasional
    # missing commas. Read their declared fields without rewriting model input.
    template = {}
    blocks = list(re.finditer(r'"(Q\d+)"\s*:\s*\{', tail[start:]))
    body = tail[start:]
    for i, match in enumerate(blocks):
        block = body[match.end():blocks[i + 1].start() if i + 1 < len(blocks) else len(body)]
        kind = re.search(r'"Question Type"\s*:\s*"([^"]+)"', block)
        fields = re.findall(r'"?(SelectedByPosition|SelectedText|Values|Text)"\s*:', block)
        if not kind or not fields:
            raise ValueError('Unrecognized published response shape')
        template[match[1]] = {'Question Type': kind[1], 'Answers': dict.fromkeys(fields)}
    if not template or any(not re.fullmatch(r'Q\d+', k) for k in template):
        raise ValueError('Unexpected official question numbering')
    return template


def questions(document):
    """Visit randomized, conditional and looped blocks in published order."""
    if isinstance(document, dict):
        if document.get('QuestionID'):
            yield document
        else:
            for value in document.values():
                yield from questions(value)
    elif isinstance(document, list):
        for value in document:
            yield from questions(value)


def routed_questions(document, schema):
    items = [q for q in questions(document) if q.get('QuestionType') != 'DB' and not q.get('is_descriptive')]
    if len(items) != len(schema):
        raise ValueError(f'Q-label/source mapping mismatch: {len(items)} != {len(schema)}')
    return dict(zip(schema, items))


def parse_prediction(text, survey, document):
    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    parsed = json5.loads(cleaned)
    schema = schema_for(survey)
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError('Question keys differ from published template')
    # Author converters handle missing answers per question/variable. Do not
    # impose a whole-survey semantic consistency or range rejection here.
    for key, value in parsed.items():
        if key in schema and not isinstance(value, dict):
            raise ValueError(f'{key}: answer must be an object')
    return parsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', default='context_effects', help='study name or all')
    p.add_argument('--limit', type=int)
    p.add_argument('--max-tokens', type=int, choices=[16384])
    p.add_argument('--seed', type=int, help='Random subset seed; used with --limit')
    p.add_argument('--persona', choices=['full', 'empty'], default='full')
    p.add_argument('--output', type=Path)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not a.dry_run and a.output is None:
        p.error('New experiments require an explicit --output directory')
    if a.limit is not None and a.limit < 1:
        p.error('--limit must be positive')
    personas = {}
    for f in sorted((ROOT / 'data/twin/full_persona/chunks').glob('*.parquet')):
        for row in pd.read_parquet(f, columns=['pid', 'persona_text']).to_dict('records'):
            personas['pid_' + str(row['pid'])] = row['persona_text']
    studies = sorted((ROOT / 'data/mega/data').glob('*.parquet'))
    if a.study != 'all':
        studies = [ROOT / f'data/mega/data/{a.study}-00000-of-00001.parquet']
    for f in studies:
        study = f.name.split('-00000')[0]
        data = pd.read_parquet(f)
        if a.limit:
            data = data.sample(n=min(a.limit, len(data)), random_state=a.seed) if a.seed is not None else data.head(a.limit)
        dst = a.output / study / a.persona / 'predictions.jsonl' if a.output else None
        temperature = config(study)['LLM_simulation']['LLM_config']['temperature']
        if dst and not a.dry_run:
            settings = {'model': MODEL, 'primary_effort': PRIMARY_EFFORT, 'fallback_effort': None,
                        'task_token_budget': 16384, 'budget_scope': 'per_questionnaire_completion','track_retry_limit':1,
                        'temperature': temperature, 'max_tokens': 16384, 'persona': a.persona, 'seed': a.seed, 'limit': a.limit,
                        'input_protocol': 'official-per-study-config-one-user',
                        'official_config_sha256': hashlib.sha256((UPSTREAM / f'configs/{study}/{study}.yaml').read_bytes()).hexdigest(),
                        'survey_sha256': hashlib.sha256(f.read_bytes()).hexdigest()}
            settings_path = dst.parent / 'config.json'
            if settings_path.exists() and json.loads(settings_path.read_text(encoding='utf-8')) != settings:
                raise ValueError('Experiment configuration changed; select a new output directory')
            dst.parent.mkdir(parents=True, exist_ok=True)
            settings_path.write_text(json.dumps(settings, indent=2), encoding='utf-8')
        from track_retry import retryable
        attempts = {}
        if dst and dst.exists():
            for line in dst.read_text(encoding='utf-8').splitlines():
                rec=json.loads(line);attempts.setdefault(rec['id'],[]).append(rec)
        for row in data.to_dict('records'):
            pid = row['PID']
            prior=attempts.get(pid,[])
            if prior and not retryable(prior[-1],len(prior)):
                continue
            if pid not in personas:
                raise ValueError(f'Missing official persona: {pid}')
            schema = schema_for(row['survey_text'])
            document = json.loads(row['survey_json_with_human_response'])
            routed_questions(document, schema)
            prompt = prompt_for(study, row['survey_text'], personas[pid] if a.persona == 'full' else '[Empty Persona Profile]')
            if a.dry_run:
                print(study, pid, 'questions', len(schema), 'prompt_chars', len(prompt))
                continue
            options = {'max_tokens': a.max_tokens} if a.max_tokens is not None else {}
            while not prior or retryable(prior[-1],len(prior)):
                result = answer(prompt, temperature=temperature, **options)
                record = {'id': pid, 'model': MODEL, 'persona': a.persona, 'temperature': temperature,
                          'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'status': result['status'], 'raw': result,'attempt':len(prior)+1}
                if result['status'] == 'complete':
                    try:
                        record['prediction'] = parse_prediction(result['answer'], row['survey_text'], document)
                    except (ValueError, TypeError, KeyError) as exc:
                        record.update(status='invalid', error=str(exc))
                dst.parent.mkdir(parents=True, exist_ok=True)
                with dst.open('a', encoding='utf-8') as out:
                    out.write(json.dumps(record, ensure_ascii=False) + '\n')
                prior.append(record)
                print(study, pid, record['status'], 'attempt',record['attempt'],flush=True)


if __name__ == '__main__':
    main()
