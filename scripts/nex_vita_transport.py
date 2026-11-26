"""Vita multi-role/tool transport, retaining official messages and fixed Nex policy."""
import os
from contextvars import ContextVar
from contextlib import contextmanager, nullcontext
from threading import RLock
from timed_nex import phase, failure_diagnostic, budget_trace, MODEL, PRIMARY_EFFORT, PREPARATION_EFFORT, FALLBACK_EFFORT, THINK_SECONDS, ANSWER_SECONDS, TASK_TOKEN_BUDGET, request_model

_remaining = ContextVar('vita_task_remaining', default=None)

class TaskBudget:
    """A task-local ledger shared by callbacks, including worker threads."""
    def __init__(self, limit=TASK_TOKEN_BUDGET, effort=PRIMARY_EFFORT, on_progress=None):
        self.limit = limit
        self.remaining = limit
        self.effort = effort
        self.on_progress = on_progress
        self.lock = RLock()
        self.failure = None
        self.usage_rows = []
        self.exhausting_request = None

@contextmanager
def task_budget(budget=None):
    budget = budget if budget is not None else TaskBudget()
    token = _remaining.set(budget)
    try:
        yield budget
    finally:
        _remaining.reset(token)

def token_cost(calls):
    """Sum exact server usage, separately marking unavailable fields."""
    rows = [c.get('usage') or c.get('phase',{}).get('usage') or {} for c in calls]
    fields = ('prompt_tokens','completion_tokens','reasoning_tokens','total_tokens')
    cost = {key:sum(u[key] for u in rows if isinstance(u.get(key),int)) for key in fields}
    cost.update(calls=len(rows), missing_usage_calls=sum(not isinstance(u.get('completion_tokens'),int) for u in rows),
                missing_fields={key:sum(not isinstance(u.get(key),int) for u in rows) for key in fields})
    return cost


class NexDiagnosticStop(RuntimeError):
    terminal_task_stop = True

    def __init__(self, phase_result):
        self.phase_result = phase_result
        self.reason = phase_result['status']
        self.tokens = phase_result.get('task_generated_tokens', phase_result.get('observed_reasoning_tokens', 0))
        self.seconds = phase_result['seconds']
        self.diagnostic = failure_diagnostic(phase_result,self.reason)
        self.budget_trace = phase_result.get('prior_budget_trace') or budget_trace(phase_result,self.reason)
        super().__init__(f'{self.reason}: observed_tokens={self.tokens}, seconds={self.seconds:.2f}')


def chat(messages, tools=None, tool_choice=None, temperature=0, max_tokens=16384, response_format=None):
    budget = _remaining.get()
    # Serialize requests belonging to the same task, not unrelated tasks.
    with budget.lock if budget is not None else nullcontext():
        if budget is not None and budget.failure is not None:
            raise budget.failure
        try:
            return _chat(messages, tools, tool_choice, temperature, max_tokens, budget, response_format)
        except NexDiagnosticStop as exc:
            if budget is not None:
                budget.failure = exc
            raise


def _chat(messages, tools, tool_choice, temperature, max_tokens, budget, response_format=None):
    diagnostic = os.getenv('VITA_DIAGNOSTIC_2048') == '1'
    remaining = budget.remaining if budget is not None else None
    if not diagnostic and remaining is not None and remaining <= 0:
        raise NexDiagnosticStop({'status':'budget_exceeded','task_generated_tokens':budget.limit,'seconds':0,
                                 'prior_budget_trace':budget.exhausting_request})
    options = dict(messages=messages, tools=tools, tool_choice=tool_choice, temperature=temperature, max_tokens=max_tokens if diagnostic else remaining)
    if response_format is not None:
        options['response_format'] = response_format
    # Only the task agent receives tools. User simulation and scoring are exempt.
    if diagnostic and tools:
        options['thinking_budget'] = 2048
    if budget is not None and budget.on_progress is not None:
        options['on_progress'] = budget.on_progress
    phases = [phase(None, budget.effort if budget is not None else PRIMARY_EFFORT, None, **options)]
    final = phases[-1]
    if budget is not None:
        budget.usage_rows.append({'usage':final.get('usage')})
    if not diagnostic and budget is not None:
        used = (final.get('usage') or {}).get('completion_tokens')
        if remaining is not None and used is not None:
            budget.remaining = remaining-used
            final['task_generated_tokens'] = budget.limit - budget.remaining
            if budget.remaining == 0 and final['status']=='complete':
                budget.exhausting_request = dict(budget_trace(final,'budget_exceeded'),source='previous_completed_request')
        if remaining is not None and ((final.get('finish_reason') == 'length' and (used is None or used>=remaining)) or (used is not None and used>remaining)):
            final['status'] = 'budget_exceeded'
        elif remaining is not None and final['status']=='complete' and used is None:
            final['status'] = 'usage_missing'
    if final['status'] != 'complete':
        final['diagnostic'] = failure_diagnostic(final)
        raise NexDiagnosticStop(final)
    usage = {key: sum((p.get('usage') or {}).get(key, 0) or 0 for p in phases) for key in ('prompt_tokens', 'completion_tokens', 'reasoning_tokens', 'total_tokens')}
    return {'model': request_model(budget.effort if budget is not None else PRIMARY_EFFORT), 'choices': [{'index': 0, 'finish_reason': final['finish_reason'],
            'message': {'role': 'assistant', 'content': final['answer'] or None, 'tool_calls': final.get('tool_calls')}}],
            'usage': usage, 'nex_phases': phases, 'usage_complete': all(p.get('usage') for p in phases)}
