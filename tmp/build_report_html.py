import html
import json

with open("tmp/_report_rows.json", "r", encoding="utf-8") as f:
    rows = json.load(f)


def fmt_params(p):
    if not p:
        return "-"
    return f"{p / 1e9:g}B"


def fmt_ctx(c):
    if not c:
        return "-"
    return f"{c:,}"


body_rows = []
for r in rows:
    badge = (
        '<span class="badge pass">PASS</span>'
        if r["status"] == "PASS"
        else '<span class="badge fail">FAIL</span>'
    )
    caps = []
    for c in ("completion", "thinking", "tools", "vision"):
        if c in r["caps"].split(","):
            caps.append(f'<span class="cap {c}">{c}</span>')
    reply = html.escape(r["reply"]) or "&mdash;"
    body_rows.append(
        "<tr>"
        f'<td><b>{html.escape(r["name"])}</b>'
        f'<div class="sub">{html.escape(r["id"])}</div></td>'
        f'<td class="c" data-v="{0 if r["status"] == "PASS" else 1}">{badge}</td>'
        f'<td class="n" data-v="{r["latency"]}">{r["latency"]:g}</td>'
        f'<td>{"".join(caps)}</td>'
        f'<td>{html.escape(r["family"])}</td>'
        f'<td class="n" data-v="{r["params"]}">{fmt_params(r["params"])}</td>'
        f'<td class="n" data-v="{r["ctx"]}">{fmt_ctx(r["ctx"])}</td>'
        f'<td>{html.escape(r["quant"]) or "-"}</td>'
        f'<td class="reply">{reply}</td>'
        "</tr>"
    )

ok = sum(1 for r in rows if r["status"] == "PASS")
latencies = [r["latency"] for r in rows if r["status"] == "PASS"]
avg = sum(latencies) / len(latencies) if latencies else 0

page = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>NVIDIA 免费模型测试报表</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1a1f2e;--mut:#6b7280;--line:#e5e7eb;
  --green:#0a7d43;--greenbg:#e3f5ec;--red:#b4232a;--redbg:#fbe7e8;--accent:#0058cc}
*{box-sizing:border-box}
body{margin:0;font-family:'Segoe UI','Microsoft YaHei',system-ui,sans-serif;
  background:var(--bg);color:var(--ink)}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 48px}
h1{font-size:22px;margin:0 0 4px}
.sub2{color:var(--mut);font-size:13px;margin-bottom:22px}
.stats{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;
  padding:12px 20px;min-width:120px}
.stat b{display:block;font-size:22px}
.stat span{font-size:12px;color:var(--mut)}
.tablebox{background:var(--card);border:1px solid var(--line);
  border-radius:8px;overflow:auto}
table{border-collapse:collapse;width:100%;min-width:900px}
th{background:#f0f2f5;padding:10px 12px;text-align:left;font-size:13px;
  color:#374151;cursor:pointer;user-select:none;white-space:nowrap;
  border-bottom:2px solid var(--line)}
th:hover{background:#e6e9ee}
th .arr{font-size:10px;margin-left:4px;opacity:.35}
th.sorted{color:var(--accent)}
th.sorted .arr{opacity:1;color:var(--accent)}
td{padding:10px 12px;border-bottom:1px solid var(--line);font-size:13.5px;
  vertical-align:top}
tr:last-child td{border-bottom:none}
tr:hover td{background:#fafbfd}
.c{text-align:center}.n{text-align:right;font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:2px 10px;border-radius:10px;font-size:11.5px;
  font-weight:600}
.pass{background:var(--greenbg);color:var(--green)}
.fail{background:var(--redbg);color:var(--red)}
.cap{display:inline-block;padding:1px 8px;border-radius:8px;font-size:11px;
  margin:1px 3px 1px 0;background:#eef1f5;color:#4b5563}
.cap.vision{background:#e6f4fb;color:#0b6a99}
.cap.thinking{background:#fdf3dc;color:#8a6400}
.cap.tools{background:#ece8fb;color:#55419e}
.sub{color:var(--mut);font-size:11.5px;margin-top:2px}
.reply{max-width:230px;color:var(--mut);font-size:12px;word-break:break-all}
input#q{width:100%;padding:9px 12px;border:1px solid var(--line);
  border-radius:8px;font-size:14px;margin-bottom:14px;background:var(--card)}
</style>
</head>
<body>
<div class="wrap">
<h1>NVIDIA 免费模型可用性报表</h1>
<div class="sub2">白名单 __TOTAL__ 个（纯文本 + 图文），
推理实测 "Say OK" &middot; 点击表头排序 &middot; 支持搜索过滤</div>
<div class="stats">
<div class="stat"><b>__TOTAL__</b><span>模型总数</span></div>
<div class="stat"><b style="color:var(--green)">__OK__</b><span>可用 PASS</span></div>
<div class="stat"><b style="color:var(--red)">__FAIL__</b><span>失败 FAIL</span></div>
<div class="stat"><b>__AVG__s</b><span>PASS 平均延迟</span></div>
</div>
<input id="q" placeholder="搜索模型 / ID / 能力…">
<div class="tablebox">
<table id="t">
<thead><tr>
<th data-k="name">模型 <span class="arr">▲▼</span></th>
<th data-k="status" class="c">状态 <span class="arr">▲▼</span></th>
<th data-k="latency" data-n="1">延迟(s) <span class="arr">▲▼</span></th>
<th data-k="caps">能力 <span class="arr">▲▼</span></th>
<th data-k="family">Family <span class="arr">▲▼</span></th>
<th data-k="params" data-n="1">参数量 <span class="arr">▲▼</span></th>
<th data-k="ctx" data-n="1">上下文 <span class="arr">▲▼</span></th>
<th data-k="quant">精度 <span class="arr">▲▼</span></th>
<th>实测回复</th>
</tr></thead>
<tbody>__ROWS__</tbody>
</table>
</div>
</div>
<script>
const ths=[...document.querySelectorAll('th[data-k]')];
const state={};
ths.forEach(th=>{
  state[th.dataset.k]=null;
  th.addEventListener('click',()=>{
    const k=th.dataset.k, num=th.hasAttribute('data-n');
    const idx=[...th.parentNode.children].indexOf(th);
    Object.keys(state).forEach(x=>state[x]=null);
    // null -> asc -> desc -> asc ...
    state[k]=(state[k]===true)?false:true;
    ths.forEach(t=>{t.classList.remove('sorted');
      t.querySelector('.arr').textContent='▲▼';});
    th.classList.add('sorted');
    th.querySelector('.arr').textContent=state[k]?'▲':'▼';
    const tb=document.querySelector('#t tbody');
    [...tb.rows].sort((A,B)=>{
      const a=A.cells[idx], b=B.cells[idx];
      let va=a.dataset.v!==undefined?a.dataset.v:a.textContent.trim();
      let vb=b.dataset.v!==undefined?b.dataset.v:b.textContent.trim();
      let c;
      if(k==='status'){c=(+va||0)-(+vb||0);}
      else if(num){va=parseFloat(va)||0;vb=parseFloat(vb)||0;
        c=va<vb?-1:(va>vb?1:0);}
      else{c=va.localeCompare(vb,'zh');}
      return state[k]?c:-c;
    }).forEach(r=>tb.appendChild(r));
  });
});
document.getElementById('q').addEventListener('input',e=>{
  const q=e.target.value.toLowerCase();
  document.querySelectorAll('#t tbody tr').forEach(r=>{
    r.style.display=r.textContent.toLowerCase().includes(q)?'':'none';
  });
});
// 初始视图：状态升序(PASS 在前), 再点延迟列看最快模型
</script>
</body></html>
"""

page = (
    page.replace("__TOTAL__", str(len(rows)))
    .replace("__OK__", str(ok))
    .replace("__FAIL__", str(len(rows) - ok))
    .replace("__AVG__", f"{avg:.1f}")
    .replace("__ROWS__", "".join(body_rows))
)

with open("tmp/nvidia_whitelist_test_report.html", "w", encoding="utf-8") as f:
    f.write(page)

print(f"saved tmp/nvidia_whitelist_test_report.html ({len(page)} bytes)")
