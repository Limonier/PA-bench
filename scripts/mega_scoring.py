"""Invert official Qualtrics extraction and reuse author study-level scoring."""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from mega import ROOT, UPSTREAM, routed_questions, schema_for
from timed_nex import answer

os.environ.setdefault('GENSIM_DATA_DIR', str(ROOT / 'data/models/gensim'))

LABEL_STUDIES = {'context_effects', 'default_eric', 'idea_generation', 'preference_redistribution', 'quantitative_intuition'}


def reference(study):
    root = UPSTREAM / 'results' / study
    pattern = '*human*labels*.csv.gz' if study in LABEL_STUDIES else 'full_persona_without_reasoning*/consolidated_original_answers_values.csv.gz'
    files = sorted(root.glob(pattern)) or sorted(root.glob('*human*values*.csv.gz'))
    if not files:
        raise FileNotFoundError(f'No official human reference for {study}')
    frame = pd.read_csv(files[0], dtype=str)
    ids = pd.to_numeric(frame['TWIN_ID'], errors='coerce')
    return frame.loc[ids.notna()].assign(TWIN_ID=ids[ids.notna()].astype(int)).reset_index(drop=True)


def import_columns(study):
    root = UPSTREAM / 'results' / study
    files = sorted(root.glob('full_persona_without_reasoning*/consolidated_original_answers_values.csv.gz'))
    files += sorted(root.glob('*human*values*.csv.gz'))
    files += [UPSTREAM / f'.dat/{study}/raw_data/response.csv']
    for path in files:
        mapping = {}
        rows = pd.read_csv(path, nrows=2, dtype=str)
        for column in rows:
            for cell in rows[column]:
                try:
                    key = json.loads(cell).get('ImportId')
                except (ValueError, TypeError, AttributeError):
                    continue
                if key:
                    mapping.setdefault(key, []).append(column)
        if mapping:
            return mapping
    return {}


def response_cells(q, answers, *, labels=False):
    """Delegate field precedence and missing values to the author converter."""
    name = 'mega_author_json2csv'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, UPSTREAM / 'evaluation/json2csv.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    extractor = module.AnswerExtractor(module.ExtractionMode.TEXT if labels else module.ExtractionMode.NUMERIC)
    source = dict(q, Answers=answers)
    try:
        values = extractor._process_question(source, False)
    except (ValueError, TypeError, KeyError, IndexError):
        values = {}  # Author extract_from_file likewise skips a failed question.
    qid = q['QuestionID']
    prefix = qid.split('_QID')[0] + '_' if '_QID' in qid else ''
    tag = prefix + q.get('QuestionName', qid)
    for key, value in values.items():
        # Author extractor flattens a single slider to QID; Qualtrics CSV
        # retains its statement suffix. Adapt the column identity, not value.
        if key == qid and q.get('QuestionType') == 'Slider':
            ids = q.get('StatementsID', ['1'])
            if len(ids) == 1:
                key = qid + '_' + str(ids[0])
        yield key, tag + key[len(qid):], value


def target_column(importid, tag, mapping, columns):
    candidates = [c for c in mapping.get(importid, []) if c in columns]
    if tag in columns and (not candidates or tag in candidates):
        return tag
    if len(set(candidates)) == 1:
        return candidates[0]
    if importid in columns:
        return importid
    if tag.endswith('_TEXT') and tag[:-5] in columns:
        return tag[:-5]
    raise ValueError(f'Cannot uniquely map {importid}/{tag}: {candidates}')


def converted_row(document, prediction, survey, mapping, columns, labels):
    values = {}
    for key, q in routed_questions(document, schema_for(survey)).items():
        for importid, tag, value in response_cells(q, prediction.get(key, {}).get('Answers', {}), labels=labels):
            # Author recommendation_algorithms scorer distinguishes duplicated
            # Q16/Q20 export tags with pandas' .1 suffix. QID16 and QID20 are
            # the second occurrence in that study's exported CSV, not QID31/23.
            if re.fullmatch(r'[12]_QID(?:16|20)', q['QuestionID']) and tag + '.1' in columns:
                tag += '.1'
            if q.get('QuestionType') == 'MC' and q.get('Settings', {}).get('Selector') in ('MAVR', 'MAHR') and tag not in columns and not mapping.get(importid):
                # Valid published choice absent from the human CSV (e.g. nobody
                # selected Not applicable). Preserve it as an extra output column.
                values[tag] = value
                continue
            column = target_column(importid, tag, mapping, columns)
            if column in values and values[column] != value:
                raise ValueError(f'Duplicate mapped variable: {column}')
            values[column] = value
    return values


def frames_for(study, records):
    human = reference(study)
    mapping = import_columns(study)
    data = pd.read_parquet(ROOT / f'data/mega/data/{study}-00000-of-00001.parquet').set_index('PID')
    selected = {int(pid.removeprefix('pid_')) for pid in records}
    human = human[human.TWIN_ID.isin(selected)].copy()
    if set(human.TWIN_ID) != selected:
        raise ValueError('Prediction IDs absent from official analysis reference')
    twin = human.copy()
    qsf = json.loads((UPSTREAM / f'.dat/{study}/raw_data/survey.qsf').read_text(encoding='utf-8-sig'))
    tags = [e.get('Payload', {}).get('DataExportTag') for e in qsf['SurveyElements'] if e.get('Element') == 'SQ']
    # Preserve treatment/administrative metadata; erase ALL source response columns.
    erased = {c for key, cols in mapping.items() if 'QID' in key for c in cols}
    erased |= {c for c in twin if any(tag and (c == tag or c.startswith(tag + '_') or re.fullmatch(r'\d+_' + re.escape(tag) + r'(?:_.*)?', c)) for tag in tags)}
    twin.loc[:, list(erased & set(twin.columns))] = None
    for index, row in twin.iterrows():
        pid = 'pid_' + str(int(row.TWIN_ID))
        source = data.loc[pid]
        values = converted_row(json.loads(source.survey_json_with_human_response), records[pid]['prediction'], source.survey_text, mapping, twin.columns, study in LABEL_STUDIES)
        for column, value in values.items():
            twin.at[index, column] = value
    for frame in (human, twin):
        for column in frame:
            numeric = pd.to_numeric(frame[column], errors='coerce')
            if numeric.notna().sum() == frame[column].notna().sum():
                frame[column] = numeric
    return human, twin


class NexJudge:
    """OpenAI-shaped adapter for the author's fixed classification prompt."""
    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=self)
        self.output = kwargs.get('output')
        self.max_tokens = kwargs.get('max_tokens')

    def create(self, *, messages, **kwargs):
        prompt = '\n\n'.join(message['content'] for message in messages)
        options = {'max_tokens': self.max_tokens} if self.max_tokens is not None else {}
        result = answer(prompt, temperature=0, scoring=True)
        if self.output:
            with (self.output / 'judge_predictions.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps({'model': 'nex-n2.5-pro', 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), **result}, ensure_ascii=False) + '\n')
        if result['status'] != 'complete':
            raise RuntimeError('Nex judge did not produce a complete answer')
        text = result['answer'].strip()
        match = re.search(r'\{.*\}', text, re.S)
        parsed = json.loads(match[0]) if match else {}
        if parsed.get('classification') not in ('summary', 'synthesis'):
            raise ValueError('Invalid Nex judge classification')
        confidence = parsed.get('confidence')
        if type(confidence) not in (int, float) or not 0 <= confidence <= 100:
            raise ValueError('Invalid Nex judge confidence')
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(parsed)))])


def run_official(study, human, twin, output, *, judge=False):
    """Adapt dataframe transport only; keep author recoding/statistics unchanged."""
    if study == 'quantitative_intuition' and not judge:
        raise ValueError('Quantitative Intuition needs --judge for its open-answer outcome')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(UPSTREAM / 'mega_study_evaluation'))
    from common import data_loader
    original_loader = data_loader.load_standard_data
    data_loader.load_standard_data = lambda *a, **k: (human.copy(), twin.copy())
    script = UPSTREAM / f'mega_study_evaluation/{study}/mega_study_evaluation.py'
    spec = importlib.util.spec_from_file_location('official_' + study, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Never accidentally dispatch the author's optional judge to OpenAI.
    old_key = os.environ.pop('OPENAI_API_KEY', None)
    if study == 'quantitative_intuition':
        settings_file = output.parent / 'config.json'
        settings = json.loads(settings_file.read_text(encoding='utf-8')) if settings_file.exists() else {}
        module.OpenAI = lambda *a, **k: NexJudge(output=output, max_tokens=settings.get('max_tokens'))
        os.environ['OPENAI_API_KEY'] = 'nex-adapter-placeholder'
    argv = sys.argv
    original_cwd = Path.cwd()
    sys.argv = [str(script), '--human-data', 'in-memory-human', '--twin-data', 'in-memory-nex', '--output-dir', str(output)]
    try:
        # Some author scripts write relative filenames; contain those in this run.
        os.chdir(output)
        result = module.main()
        if study == 'quantitative_intuition':
            cache = list(output.glob('*open_classification_cache.csv'))
            if not cache or pd.read_csv(cache[0])['scenario_open_QI'].isna().any():
                raise RuntimeError('Incomplete judge outcomes; scoring is not complete')
        return result
    finally:
        os.chdir(original_cwd)
        sys.argv = argv
        data_loader.load_standard_data = original_loader
        os.environ.pop('OPENAI_API_KEY', None)
        if old_key is not None:
            os.environ['OPENAI_API_KEY'] = old_key


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--judge', action='store_true', help='Enable the locked Nex classifier required by quantitative_intuition')
    p.add_argument('--creativity-ratings', type=Path, help='Directory containing independently rated human and NEW Nex creativity responses')
    a = p.parse_args()
    records = {}
    attempted = set()
    abandoned = set()
    for line in a.predictions.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        attempted.add(row['id'])
        if row.get('status') == 'budget_exceeded':
            abandoned.add(row['id'])
        if row.get('status') == 'complete':
            records[row['id']] = row
    if not records:
        raise ValueError('No validated predictions to score')
    if a.study == 'quantitative_intuition' and not a.judge:
        p.error('quantitative_intuition requires --judge; partial scores are not reported as complete')
    human, twin = frames_for(a.study, records)
    a.output.mkdir(parents=True, exist_ok=True)
    if a.creativity_ratings:
        for name in ['idea generation human data creativity ratings.csv', 'idea generation twins data creativity ratings.csv']:
            ratings = a.creativity_ratings / name
            if not ratings.exists():
                raise FileNotFoundError(ratings)
            shutil.copyfile(ratings, a.output / name)
    total = len(pd.read_parquet(ROOT / f'data/mega/data/{a.study}-00000-of-00001.parquet', columns=['PID']))
    coverage = {'status': 'pending', 'model': 'nex-n2.5-pro', 'published_total': total, 'attempted': len(attempted), 'validated_predictions': len(records),
                'budget_abandoned': len(abandoned-set(records)), 'failed_or_invalid': len(attempted - set(records)), 'not_attempted': total - len(attempted),
                'score_scope': 'validated_predictions_after_official_filters',
                'missing_optional_outcomes': ['creativity_rating_byhuman'] if a.study == 'idea_generation' and not a.creativity_ratings else []}
    (a.output / 'coverage.json').write_text(json.dumps(coverage, indent=2), encoding='utf-8')
    human.to_csv(a.output / 'paired_human.csv', index=False)
    twin.to_csv(a.output / 'paired_nex.csv', index=False)
    try:
        run_official(a.study, human, twin, a.output, judge=a.judge)
    except Exception as exc:
        coverage.update(status='failed', error=str(exc))
        (a.output / 'coverage.json').write_text(json.dumps(coverage, indent=2), encoding='utf-8')
        raise
    coverage['status'] = 'complete_available_outcomes' if coverage['missing_optional_outcomes'] else 'complete'
    (a.output / 'coverage.json').write_text(json.dumps(coverage, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
