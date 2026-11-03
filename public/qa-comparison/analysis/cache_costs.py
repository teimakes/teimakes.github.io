"""Offline post-processing only. Never imports an API client or reads credentials.

Run: python analysis/cache_costs.py
Original live rows and approval fingerprints remain unchanged.
"""
from decimal import Decimal as D
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MILLION = D(1000000)


def priced(row, config):
    usage = row['raw_response']['usage']
    details = usage['input_tokens_details']
    total, cached = usage['input_tokens'], details['cached_tokens']
    written = details['cache_write_tokens']
    regular = total - cached - written
    assert min(regular, cached, written, usage['output_tokens']) >= 0
    assert total == row['input_tokens'] and cached == row['cached_input_tokens']
    rate = D(str(config['input_usd_per_million']))
    base = regular * rate + cached * D(str(config['cached_usd_per_million']))
    base += usage['output_tokens'] * D(str(config['output_usd_per_million']))
    split = details.get('cache_write_tokens_details')
    if split is not None:
        w30, w12 = split['ttl_30m_tokens'], split['ttl_12h_tokens']
        assert w30 + w12 == written and min(w30, w12) >= 0
        low = high = (base + rate * (w30 * D('1.25') + w12 * D('2.5'))) / MILLION
        assumption = 'usage_ttl_breakdown'
    else:
        # Requests omitted prompt_cache_options; documented default is 30m.
        # Preserve a 12h sensitivity value because responses omit the breakdown.
        assert 'prompt_cache_options' not in row['request']
        low = (base + written * rate * D('1.25')) / MILLION
        high = (base + written * rate * D('2.5')) / MILLION
        assumption = 'default_30m_assumed_no_ttl_breakdown'
    return {'id': row['id'], 'arm': row['arm'], 'repeat': row['repeat'],
            'ordinary_input_tokens': regular, 'cached_input_tokens': cached,
            'cache_write_tokens': written, 'input_tokens': total,
            'output_tokens': usage['output_tokens'], 'assumption': assumption,
            'cache_aware_estimate_usd': str(low),
            'write_12h_sensitivity_usd': str(high)}


def aggregate(rows):
    result = {'calls': len(rows)}
    for key in ('ordinary_input_tokens', 'cached_input_tokens', 'cache_write_tokens',
                'input_tokens', 'output_tokens'):
        result[key] = sum(r[key] for r in rows)
    for key in ('cache_aware_estimate_usd', 'write_12h_sensitivity_usd'):
        result[key] = str(sum((D(r[key]) for r in rows), D(0)))
    return result


def main():
    report = {'date': '2026-11-03', 'currency': 'USD', 'billing_invoice': False,
              'source': 'https://developers.openai.com/api/docs/guides/prompt-caching',
              'note': 'Default 30m write estimate, plus 12h sensitivity. Does not infer credit deductions or taxes.',
              'runs': {}}
    all_rows = []
    for name, count in [('smoke', 4), ('dev', 24)]:
        root = ROOT / 'outputs' / name
        raw = [json.loads(s) for s in (root / 'results.jsonl').read_text().splitlines()]
        assert len(raw) == count and all(r['status'] == 'ok' for r in raw)
        assert len({(r['id'], r['arm'], r['repeat']) for r in raw}) == count
        config = json.loads((ROOT / 'plans' / f'{name}.json').read_text())['config']
        rows = [priced(r, config) for r in raw]
        report['runs'][name] = {'total': aggregate(rows), 'arms': {
            arm: aggregate([r for r in rows if r['arm'] == arm]) for arm in ('full', 'rag')},
            'rows': rows}
        all_rows.extend(rows)
    report['total'] = aggregate(all_rows)
    ledger = json.loads((ROOT / 'budget-ledger.json').read_text())
    assert len(ledger['calls']) == len(all_rows)
    assert D(report['total']['write_12h_sensitivity_usd']) < D(str(ledger['reserved_usd'])) < D('0.25')
    report['reserved_usd'] = ledger['reserved_usd']
    report['approved_cap_usd'] = 0.25
    (ROOT / 'outputs' / 'cache_costs.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v['arms'] for k, v in report['runs'].items()}, indent=2))
    print('TOTAL', json.dumps(report['total']))


if __name__ == '__main__':
    main()
