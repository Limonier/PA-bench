"""One fresh track attempt after a non-budget failure; never select by score."""

def retryable(record, attempts=1):
    if attempts != 1 or record.get('status') in (None, 'complete', 'budget_exceeded', 'thinking_budget', 'manual_cancelled'):
        return False
    diagnostic = record.get('diagnostic') or {}
    if diagnostic.get('category') == 'model_token_limit':
        return False
    phases = (record.get('raw') or record).get('phases') or []
    return not any(p.get('finish_reason') == 'length' for p in phases)


def connection_failure(record):
    return (record.get('diagnostic') or {}).get('category') in {
        'network_before_submission', 'transport_interrupted', 'submission_timeout_unknown',
        'upstream_no_output_timeout', 'reasoning_stream_stalled', 'generation_stream_stalled',
        'stream_ended_without_done',
    }
