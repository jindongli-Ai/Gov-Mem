"""Run frozen nano matrix, then export measured tables for Overleaf."""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines import run as matrix


def export(out):
    result = matrix.read(out / 'official_metrics.json')
    model = result['model']
    names = {'a_mem': 'A-Mem', 'mem0': 'Mem0', 'remem_i': 'ReMem-I', 'remem_s': 'ReMem-S'}
    systems = {}
    if model == 'gpt-5.4-nano':
        rag = matrix.read(ROOT / 'outputs/rag_naive_gpt54nano_responses_full2218_20260921/official_score.json')
        systems['Naive RAG'] = {'per_domain': rag['per_domain'], 'macro': rag['four_domain_average']}
    missing = []
    for method, name in names.items():
        if method not in result['methods']:
            continue
        systems[name] = result['methods'][method]
        for domain in matrix.DOMAINS:
            pred = matrix.rows(out / 'scored' / method / domain / 'predictions.jsonl')
            judged = matrix.rows(out / 'scored' / method / domain / 'official_eval/judge_scores.jsonl')
            expected = {c for j in matrix.read(out / 'protocol.json')['jobs'] if j['domain'] == domain for c in j['checkpoint_ids']}
            for rows in (pred, judged):
                assert len(rows) == len(expected) and {r['checkpoint_id'] for r in rows} == expected
            for row in judged:
                key = {'utility': 'utility_ok', 'privacy': 'privacy_leak', 'safety': 'deletion_leak'}[row['query_type']]
                if not row['judge'].get('parse_ok') or row['judge'].get(key) is None:
                    missing.append({'method': name, 'domain': domain, 'checkpoint_id': row['checkpoint_id'], 'field': key})
    keys = ('U', 'A', 'F', 'MGS')
    md = [f'# OpenLux / {model} full GateMem results', '',
          '91 episodes / 2218 queries per method. Percentages; U/MGS higher is better; A/F lower is better.', '',
          '| Method | U | A | F | MGS |', '|---|---:|---:|---:|---:|']
    tex = ['% Requires \\usepackage{booktabs}. All values are percentages.',
           f'% {model} via OpenLux; temperature 0; GPT-4o judge.',
           '\\begin{tabular}{lrrrr}', '\\toprule', 'Method & U $\\uparrow$ & A $\\downarrow$ & F $\\downarrow$ & MGS $\\uparrow$ \\\\', '\\midrule']
    for name, data in systems.items():
        values = [f'{100*data["macro"][k]:.2f}' for k in keys]
        md.append('| '+name+' | '+' | '.join(values)+' |')
        tex.append(name+' & '+' & '.join(values)+' \\\\')
    tex += ['\\bottomrule', '\\end{tabular}', '']
    for name, data in systems.items():
        md += ['', '## '+name, '', '| Domain | U | A | F | MGS |', '|---|---:|---:|---:|---:|']
        tex += ['% '+name, '\\begin{tabular}{lrrrr}', '\\toprule', 'Domain & U & A & F & MGS \\\\', '\\midrule']
        for domain in matrix.DOMAINS:
            values = [f'{100*data["per_domain"][domain][k]:.2f}' for k in keys]
            md.append('| '+domain.title()+' | '+' | '.join(values)+' |')
            tex.append(domain.title()+' & '+' & '.join(values)+' \\\\')
        tex += ['\\bottomrule', '\\end{tabular}', '']
    note = ('Missing applicable judge labels: '+str(len(missing))+'. '+
            ('Affected scores use available labels; see table_audit.json.' if missing else 'All applicable labels present.')+
            ' Macro MGS is the mean of domain MGS. Execution errors remain included; see official_metrics.json.')
    md += ['', note]
    tex.insert(0, '% '+note)
    dest = out / 'paper_tables'; dest.mkdir(exist_ok=True)
    (dest/'results.md').write_text('\n'.join(md)+'\n')
    (dest/'results.tex').write_text('\n'.join(tex)+'\n')
    matrix.dump(dest/'table_audit.json', {'missing_labels': missing, 'systems': systems})


if __name__ == '__main__':
    out = Path(sys.argv[sys.argv.index('--output')+1]).resolve()
    matrix.main()
    export(out)
