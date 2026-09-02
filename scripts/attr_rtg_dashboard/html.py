"""Render a self-contained, human-readable ATTR-RTG dashboard."""
from __future__ import annotations

from html import escape
from pathlib import Path


def _number(value: float, percent: bool = False) -> str:
    return f"{value * 100:.3f}%" if percent else f"{value:.4f}"


def _ci(row: dict, metric: str, percent: bool = False) -> str:
    return f"{_number(row[metric], percent)} [{_number(row[metric + '_ci_low'], percent)}, {_number(row[metric + '_ci_high'], percent)}]"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(item)}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(item))}</td>" for item in row) + "</tr>" for row in rows)
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _architecture_table(rows: list[dict]) -> str:
    values = [[row["regime"], row["model"], _ci(row, "nmse"), _ci(row, "nll")] for row in rows]
    return _table(["Regime", "Arquitetura", "NMSE (IC 95%)", "NLL (IC 95%)"], values)


def _governance_table(rows: list[dict]) -> str:
    values = [[row["regime"], row["model"], row["head"], _ci(row, "unsafe_rate", True),
               _ci(row, "safe_service", True), _ci(row, "coverage", True)] for row in rows]
    return _table(["Regime", "Arquitetura", "Cabeça", "Unsafe (IC 95%)", "Safe-service (IC 95%)", "Cobertura (IC 95%)"], values)


def _gate_cards(gates: list[dict]) -> str:
    return "".join(f"<li class='gate {'pass' if row['passed'] else 'fail'}'><b>{escape(row['gate'])}</b><span>{'PASSOU' if row['passed'] else 'NÃO PASSOU'}</span></li>" for row in gates)


def render_html(payload: dict, tables: dict, output: Path) -> None:
    gates = tables["gates"]
    passed = sum(row["passed"] for row in gates)
    total = len(gates)
    architecture = tables["architecture"]
    deltas = {(row["regime"], row["model"]): row for row in tables["g_vs_c"]}
    css = """
:root{--bg:#07111f;--card:#101d31;--ink:#e8f0fa;--muted:#9fb0c6;--cyan:#29c7ac;--amber:#ffb45b;--red:#ff7185}*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:40px 22px}
h1{font-size:clamp(2rem,5vw,4rem);margin:.2em 0}h2{margin-top:2.2em}h3{color:var(--cyan)}.eyebrow,.muted{color:var(--muted)}
.notice{border-left:4px solid var(--amber);padding:14px 18px;background:#172237}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}
.card,figure{background:var(--card);border:1px solid #263a55;border-radius:12px;padding:18px}.big{font-size:2rem;font-weight:750}.bad{color:var(--red)}
figure{margin:16px 0}figure img{width:100%;background:white;border-radius:7px}figcaption{color:var(--muted);margin-top:8px}.gates{list-style:none;padding:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:8px}
.gate{display:flex;justify-content:space-between;padding:9px 12px;border-radius:7px;background:#17263b}.gate span{font-size:.75rem}.gate.pass{border-left:4px solid var(--cyan)}.gate.fail{border-left:4px solid var(--red)}
.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;background:var(--card)}th,td{padding:9px 11px;border-bottom:1px solid #293c55;text-align:left;white-space:nowrap}th{color:var(--cyan)}a{color:#65d8ff}code{color:#c8dbef}footer{margin-top:48px;color:var(--muted)}
"""
    body = f"""<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>ATTR-RTG · evidência registrada</title><style>{css}</style></head><body><main>
<p class='eyebrow'>ATTR-RTG · resumo registrado · regimes ID / Shift / OOD</p><h1>ASM-X vs Transformer</h1>
<p class='notice'><b>Leitura correta:</b> este painel descreve apenas esta avaliação registrada. Ele não demonstra segurança, nem superioridade universal de ASM-X ou Transformer. “Unsafe” é uma métrica operacional do benchmark, não uma garantia de safety no mundo real.</p>
<p class='notice'><b>q95:</b> a banda usa o quantil empírico de 95% do resíduo absoluto na metade residual da calibração. q95 não é intervalo conformal, IC bootstrap, cobertura decisória ou garantia de safety; seus valores não estão neste resumo e não são inferidos pelo dashboard.</p>
<section class='grid'><article class='card'><div class='big bad'>{passed}/{total}</div><b>gates passaram</b><p class='muted'>Falhar um gate significa que seu critério registrado não foi satisfeito nesta avaliação.</p></article>
<article class='card'><div class='big'>5</div><b>seeds</b><p class='muted'>29, 43, 71, 89 e 107. Os pontos por seed mostram variação; os ICs 95% vêm do resumo.</p></article>
<article class='card'><div class='big'>3</div><b>regimes</b><p class='muted'>ID, mudança de distribuição (Shift) e fora de distribuição (OOD).</p></article></section>
<h2>O que os números dizem</h2><div class='grid'><article class='card'><h3>Qualidade preditiva</h3><p>Neste registro, Transformer tem NMSE e NLL menores que ASM-X nos três regimes. As diferenças ASM-X−Transformer são positivas em todos eles. Isso é resultado específico deste teste, não uma alegação geral.</p></article>
<article class='card'><h3>Governança G vs C</h3><p>C reduz levemente a taxa unsafe em troca de menor safe-service e cobertura. Os deltas usam direções explícitas: unsafe = C−G; safe-service e cobertura = G−C.</p></article>
<article class='card'><h3>Gates</h3><p>Somente <b>{escape(next(row['gate'] for row in gates if row['passed']))}</b> passou. Portanto, o conjunto registrado não sustenta uma conclusão ampla de sucesso.</p></article></div>
<h2>NMSE e NLL</h2><figure><img src='architecture_quality.svg' alt='NMSE e NLL por arquitetura e regime, com IC 95%'><figcaption>Menor é melhor. Barras de erro: IC 95% registrado.</figcaption></figure>{_architecture_table(architecture)}
<h2>Governança: unsafe, safe-service e cobertura</h2><figure><img src='governance.svg' alt='Métricas de governança para G e C'><figcaption>Resultados absolutos das cabeças G e C. Intervalos são IC 95% registrados.</figcaption></figure>{_governance_table(tables['governance'])}
<h2>Trade-off G vs C</h2><figure><img src='g_vs_c.svg' alt='Diferenças entre G e C'><figcaption>Valor zero significa ausência de diferença. A direção de cada delta está no título do painel.</figcaption></figure>
<p>Exemplo no OOD: em ASM-X, C−G unsafe = {_number(deltas[('OOD','ASM-X')]['delta_safety'], True)}, enquanto G−C safe-service = {_number(deltas[('OOD','ASM-X')]['delta_safe_service'], True)} e G−C cobertura = {_number(deltas[('OOD','ASM-X')]['coverage_difference'], True)}. A leitura é um trade-off, não “safety comprovada”.</p>
<h2>Seeds</h2><figure><img src='seed_differences.svg' alt='Diferenças de NMSE e NLL nas cinco seeds'><figcaption>Cada ponto é uma seed registrada. Δ positivo indica métrica maior de ASM-X; para NMSE/NLL, menor é melhor.</figcaption></figure>
<h2>Gates registrados</h2><ul class='gates'>{_gate_cards(gates)}</ul>
<h2>Dados reutilizáveis</h2><p><a href='architecture.csv'>architecture.csv</a> · <a href='governance.csv'>governance.csv</a> · <a href='g_vs_c.csv'>g_vs_c.csv</a> · <a href='seeds.csv'>seeds.csv</a> · <a href='gates.csv'>gates.csv</a> · <a href='dashboard_data.json'>dashboard_data.json</a></p>
<footer><p><b>Única fonte:</b> <code>registered_summary.json</code>. O dashboard não lê o run selado nem cria um novo RCM.</p><p>Fonte registrada SHA-256: <code>{escape(payload['source']['sha256'])}</code> · selo: <code>{escape(payload['metadata']['seal_sha256'])}</code> · completion receipt: <code>{escape(payload['source']['completion_receipt_sha256'])}</code>. Bootstrap: 1.000 réplicas, cluster seed-world-episode.</p></footer>
</main></body></html>"""
    (output / "index.html").write_text(body, encoding="utf-8")
