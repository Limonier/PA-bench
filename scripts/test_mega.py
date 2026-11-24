"""Offline checks only: official answer replay is NEVER a Nex experiment."""
import contextlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import mega
import mega_scoring


def replay_records(study, limit=60):
    data = pd.read_parquet(mega.ROOT / f'data/mega/data/{study}-00000-of-00001.parquet')
    records = {}
    for row in data.head(limit).to_dict('records'):
        schema = mega.schema_for(row['survey_text'])
        source = mega.routed_questions(json.loads(row['survey_json_with_human_response']), schema)
        records[row['PID']] = {'prediction': {k: {'Question Type': schema[k]['Question Type'], 'Answers': q.get('Answers', {})} for k, q in source.items()}}
    return records




def test_prompt_and_validation():
    study = 'context_effects'
    row = pd.read_parquet(mega.ROOT / f'data/mega/data/{study}-00000-of-00001.parquet').iloc[0]
    prompt = mega.prompt_for(study, row.survey_text, 'HISTORY_SENTINEL')
    assert prompt.count(row.survey_text) == 1 and prompt.count('HISTORY_SENTINEL') == 1
    assert prompt.startswith(mega.config(study)['LLM_simulation']['system_instruction'].strip())
    source = json.loads(row.survey_json_with_human_response)
    pred = replay_records(study, 1)[row.PID]['prediction']
    mega.parse_prediction(json.dumps(pred), row.survey_text, source)
    pred['Q1']['Answers'] = {'SelectedByPosition': 1, 'SelectedText': 'different model text'}
    mega.parse_prediction(json.dumps(pred), row.survey_text, source)
    q = mega.routed_questions(source, mega.schema_for(row.survey_text))['Q1']
    numeric = list(mega_scoring.response_cells(q, pred['Q1']['Answers']))
    labels = list(mega_scoring.response_cells(q, pred['Q1']['Answers'], labels=True))
    assert numeric[0][2] == 1 and labels[0][2] == 'different model text'
    mega.parse_prediction('{"Q1":{"Answers":{}}}', row.survey_text, source)
    print('Official position/text precedence and per-question missing-answer handling passed')


def test_judge_adapter():
    calls = []
    def fake(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return {'status': 'complete', 'answer': '{"classification":"synthesis","confidence":95}'}
    with patch.object(mega_scoring, 'answer', fake):
        result = mega_scoring.NexJudge().create(messages=[{'role': 'system', 'content': 'OFFICIAL INSTRUCTION'}, {'role': 'user', 'content': 'OFFICIAL TASK'}], model='gpt-4o-mini')
    assert calls == [('OFFICIAL INSTRUCTION\n\nOFFICIAL TASK', {'temperature': 0})]
    assert json.loads(result.choices[0].message.content)['classification'] == 'synthesis'
    print('Nex judge adaptation passed without network access')


def test_runner():
    fixture = next(iter(replay_records('context_effects', 1).values()))['prediction']
    fake = {'status': 'complete', 'answer': json.dumps(fixture), 'phases': [], 'fallback': False}
    with tempfile.TemporaryDirectory(prefix='mega-runner-') as tmp, contextlib.redirect_stdout(io.StringIO()):
        args = ['mega.py', '--study', 'context_effects', '--limit', '1', '--output', tmp]
        with patch.object(sys, 'argv', args), patch.object(mega, 'answer', return_value=fake) as call:
            mega.main()
            mega.main()  # Resume must not spend another model call.
            assert call.call_count == 1
            assert call.call_args.kwargs == {'temperature': .7}
        base = Path(tmp) / 'context_effects/full'
        row = json.loads((base / 'predictions.jsonl').read_text(encoding='utf-8'))
        assert row['status'] == 'complete' and len(row['prediction']) == 4
        settings = json.loads((base / 'config.json').read_text(encoding='utf-8'))
        assert settings['primary_effort'] == 'medium' and settings['fallback_effort'] == 'none'
        assert settings['primary_seconds'] == 60 and settings['fallback_seconds'] == 30
        assert settings['max_tokens'] is None
    print('Runner configuration, validated output and resume passed without network access')


def test_score_cli():
    records = replay_records('context_effects')
    with tempfile.TemporaryDirectory(prefix='mega-score-cli-') as tmp, contextlib.redirect_stdout(io.StringIO()):
        base = Path(tmp)
        fixture = base / 'offline-fixture.jsonl'
        fixture.write_text('\n'.join(json.dumps({'id': pid, 'status': 'complete', **record}) for pid, record in records.items()), encoding='utf-8')
        with patch.object(sys, 'argv', ['mega_scoring.py', '--study', 'context_effects', '--predictions', str(fixture), '--output', str(base / 'scores')]):
            mega_scoring.main()
        coverage = json.loads((base / 'scores/coverage.json').read_text(encoding='utf-8'))
        assert coverage['status'] == 'complete' and coverage['validated_predictions'] == 60
        assert coverage['not_attempted'] == coverage['published_total'] - 60
    print('Scoring entrypoint and coverage accounting passed with offline fixtures')


def test_scorers(include_creative=False):
    for file in sorted((mega.ROOT / 'data/mega/data').glob('*.parquet')):
        study = file.name.split('-00000')[0]
        if study == 'idea_generation' and not include_creative:
            continue  # Real Word2Vec resource is checked separately after download.
        human, replay = mega_scoring.frames_for(study, replay_records(study))
        fake = lambda *a, **k: {'status': 'complete', 'answer': '{"classification":"summary","confidence":99}'}
        with tempfile.TemporaryDirectory(prefix='mega-offline-') as tmp, contextlib.redirect_stdout(io.StringIO()), patch.object(mega_scoring, 'answer', fake):
            output = Path(tmp)
            mega_scoring.run_official(study, human, replay, output, judge=study == 'quantitative_intuition')
            assert (output / 'meta analysis.csv').exists(), study
        print(study, 'official scorer passed (offline replay)', flush=True)


def test_form_and_multiselect():
    q = {'QuestionID': 'QID_FORM', 'QuestionName': 'form', 'QuestionType': 'TE', 'Settings': {'Selector': 'FORM'}, 'Rows': ['first', 'second']}
    cells = list(mega_scoring.response_cells(q, {'Text': ['one', 'two']}, labels=True))
    assert [x[2] for x in cells] == ['one', 'two'] and [x[1] for x in cells] == ['form_1', 'form_2']
    q = {'QuestionID': 'QID_MULTI', 'QuestionName': 'multi', 'QuestionType': 'MC', 'Settings': {'Selector': 'MAVR'}, 'Options': ['first', 'second'], 'OptionsID': [3, 7]}
    assert list(mega_scoring.response_cells(q, {'SelectedByPosition': [2]})) == [('QID_MULTI_7', 'multi_7', 1)]
    print('FORM fields and source-ID multiselect regression checks passed')


if __name__ == '__main__':
    test_form_and_multiselect()
    test_prompt_and_validation()
    test_judge_adapter()
    test_runner()
    test_score_cli()
    test_scorers(include_creative='--with-word2vec' in sys.argv)
