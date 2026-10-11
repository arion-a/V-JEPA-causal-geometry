"""Inserts inline-SVG figures (theme tokens, no external libs) into reports/pilot_results.html at <!--FIG:name--> markers.
Every number below is copied from results/*.json (held-out: primary_heldout, dev: primary_dev / stage5 / ceiling / gate1_v3, stage 7: stage7). Intervals are drawn only where the result files contain them."""
import json, os, re, html
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
HO = json.load(open(f"{ROOT}/results/primary_heldout/primary.json")); DV = json.load(open(f"{ROOT}/results/primary_dev/primary.json")); S7 = json.load(open(f"{ROOT}/results/stage7/stage7.json"))
CE = json.load(open(f"{ROOT}/results/ceiling/ceiling.json")); G1 = json.load(open(f"{ROOT}/results/gate1_v3/gate1_v3.json"))
e = html.escape
def nice(lo, hi, n=6):
    import math
    step = (hi - lo) / n; mag = 10 ** math.floor(math.log10(step)); step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= step), default=mag)
    t = math.ceil(lo / step - 1e-9) * step; out = []
    while t <= hi + 1e-9: out.append(round(t, 6)); t += step
    return out
def barp(x0, x1, y, h, r=4):
    if abs(x1 - x0) < 1: return f"M{x0:.1f},{y:.1f}h1v{h}h-1z"
    r = min(r, abs(x1 - x0) / 2, h / 2); s = 1 if x1 >= x0 else -1
    return f"M{x0:.1f},{y:.1f}H{x1 - s * r:.1f}Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f}V{y + h - r:.1f}Q{x1:.1f},{y + h:.1f} {x1 - s * r:.1f},{y + h:.1f}H{x0:.1f}Z"
def colp(x, y0, y1, w, r=4):      # vertical bar from baseline y0 up to y1
    if abs(y1 - y0) < 1: return f"M{x:.1f},{y0:.1f}v-1h{w}v1z"
    r = min(r, abs(y0 - y1), w / 2)
    return f"M{x:.1f},{y0:.1f}V{y1 + r:.1f}Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f}H{x + w - r:.1f}Q{x + w:.1f},{y1:.1f} {x + w:.1f},{y1 + r:.1f}V{y0:.1f}Z"
HEAD = '<div class="fig"><svg viewBox="0 0 {W} {H}" role="img" aria-label="{aria}" style="min-width:{mw}px;width:100%;height:auto">'
def hbars(aria, rows, lo, hi, W=720, LW=170, band=None, xlabel="", note_lines=()):
    RH, BH = 34, 20; top = 12; H = top + RH * len(rows) + 46; X = lambda v: LW + (v - lo) / (hi - lo) * (W - LW - 24)
    o = [HEAD.format(W=W, H=H, aria=e(aria), mw=560)]
    for t in nice(lo, hi): o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{H - 38}" stroke="var(--line)" stroke-width="1"/><text x="{X(t):.1f}" y="{H - 22}" text-anchor="middle" fill="var(--muted)" font-size="12">{t:g}</text>')
    if band: o.append(f'<rect x="{X(band[0]):.1f}" y="{top}" width="{X(band[1]) - X(band[0]):.1f}" height="{H - 38 - top}" fill="var(--s1)" opacity=".10"/><text x="{X(band[1]) - 4:.1f}" y="{top + 11}" text-anchor="end" fill="var(--muted)" font-size="11.5">{e(band[2])}</text>')
    o.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{top}" y2="{H - 38}" stroke="var(--axis)" stroke-width="1.5"/>')
    for i, (lab, v, col, tip) in enumerate(rows):
        y = top + RH * i + (RH - BH) / 2
        o.append(f'<g class="mk"><title>{e(tip)}</title><text x="{LW - 10}" y="{y + 14:.1f}" text-anchor="end" fill="var(--fg)" font-size="13">{e(lab)}</text><path d="{barp(X(0), X(v), y, BH)}" fill="var({col})"/>'
                 f'<text x="{(X(v) + 6 if v >= 0.0 or abs(v) < 0.02 else X(v) - 6):.1f}" y="{y + 14:.1f}" text-anchor="{"start" if v >= 0 or abs(v) < 0.02 else "end"}" fill="var(--fg)" font-size="12.5" font-family="var(--mono)">{v:+.3f}</text></g>')
    o.append(f'<text x="{(LW + W - 24) / 2:.1f}" y="{H - 4}" text-anchor="middle" fill="var(--muted)" font-size="12">{e(xlabel)}</text></svg></div>')
    return "".join(o)
def forest(aria, rows, lo, hi, W=720, LW=190, xlabel=""):
    RH = 38; top = 26; H = top + RH * len(rows) + 50; X = lambda v: LW + (v - lo) / (hi - lo) * (W - LW - 24)
    o = [HEAD.format(W=W, H=H, aria=e(aria), mw=560)]
    for t in nice(lo, hi): o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top - 6}" y2="{H - 42}" stroke="var(--line)"/><text x="{X(t):.1f}" y="{H - 26}" text-anchor="middle" fill="var(--muted)" font-size="12">{t:g}</text>')
    o.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{top - 6}" y2="{H - 42}" stroke="var(--axis)" stroke-width="1.5"/>'
             f'<text x="{X(0) + 6:.1f}" y="14" fill="var(--muted)" font-size="12">V-JEPA higher →</text><text x="{X(0) - 6:.1f}" y="14" text-anchor="end" fill="var(--muted)" font-size="12">← trajectory higher</text>')
    for i, (lab, est, a, b, col, hollow, tip) in enumerate(rows):
        y = top + RH * i + RH / 2
        o.append(f'<g class="mk"><title>{e(tip)}</title><text x="{LW - 12}" y="{y + 4:.1f}" text-anchor="end" fill="var(--fg)" font-size="13">{e(lab)}</text>'
                 f'<line x1="{X(a):.1f}" x2="{X(b):.1f}" y1="{y}" y2="{y}" stroke="var({col})" stroke-width="2" stroke-linecap="round"/>'
                 f'<circle cx="{X(est):.1f}" cy="{y}" r="6" fill="{"var(--bg)" if hollow else f"var({col})"}" stroke="var({col})" stroke-width="2.5"/>'
                 f'<text x="{X(b) + 8:.1f}" y="{y + 4:.1f}" fill="var(--fg)" font-size="12" font-family="var(--mono)">{est:+.3f}  [{a:+.3f}, {b:+.3f}]</text></g>')
    o.append(f'<text x="{(LW + W - 24) / 2:.1f}" y="{H - 6}" text-anchor="middle" fill="var(--muted)" font-size="12">{e(xlabel)}</text></svg></div>')
    return "".join(o)
def legend(items): return '<div class="lg">' + "".join(f'<span><i style="background:var({c})"></i>{e(t)}</span>' for t, c in items) + "</div>"
def cols(aria, groups, series, ymax, W=720, thr=None, ylabel=""):
    H = 300; L, B, T = 52, 44, 16; ph = H - B - T; Y = lambda v: T + ph - v / ymax * ph; gw = (W - L - 16) / len(groups); bw = 24; gap = 2
    o = [HEAD.format(W=W, H=H, aria=e(aria), mw=560)]
    for t in nice(0, ymax): o.append(f'<line x1="{L}" x2="{W - 16}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="var(--line)"/><text x="{L - 8}" y="{Y(t) + 4:.1f}" text-anchor="end" fill="var(--muted)" font-size="12">{t:g}</text>')
    o.append(f'<line x1="{L}" x2="{W - 16}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="var(--axis)" stroke-width="1.5"/>')
    if thr: o.append(f'<line x1="{L}" x2="{W - 16}" y1="{Y(thr[0]):.1f}" y2="{Y(thr[0]):.1f}" stroke="var(--ceil)" stroke-width="1.5" stroke-dasharray="5 4"/>')
    for gi, (gl, vals) in enumerate(groups):
        n = len(series); tot = n * bw + (n - 1) * gap; x0 = L + gi * gw + (gw - tot) / 2
        for si, ((sl, col), v) in enumerate(zip(series, vals)):
            x = x0 + si * (bw + gap)
            o.append(f'<g class="mk"><title>{e(gl)} · {e(sl)}: {v:.3f}</title><path d="{colp(x, Y(0), Y(max(v, 0)), bw)}" fill="var({col})"/></g>')
            if v == max(vals): o.append(f'<text x="{x + bw / 2:.1f}" y="{Y(v) - 5:.1f}" text-anchor="middle" fill="var(--fg)" font-size="12" font-family="var(--mono)">{v:.2f}</text>')
        o.append(f'<text x="{L + gi * gw + gw / 2:.1f}" y="{H - 22}" text-anchor="middle" fill="var(--fg)" font-size="13">{e(gl)}</text>')
    o.append(f'<text x="14" y="{T + ph / 2:.1f}" transform="rotate(-90 14 {T + ph / 2:.1f})" text-anchor="middle" fill="var(--muted)" font-size="12">{e(ylabel)}</text></svg></div>')
    return "".join(o)
def interval_bars(aria, rows, lo, hi, W=720, LW=170):
    """rows: (label, value, ci or None, ceiling or None, color, tip); bar from 0, whisker for CI, tick for noise ceiling"""
    RH, BH = 36, 20; top = 12; H = top + RH * len(rows) + 46; X = lambda v: LW + (v - lo) / (hi - lo) * (W - LW - 120)
    o = [HEAD.format(W=W, H=H, aria=e(aria), mw=600)]
    for t in nice(lo, hi): o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{H - 38}" stroke="var(--line)"/><text x="{X(t):.1f}" y="{H - 22}" text-anchor="middle" fill="var(--muted)" font-size="12">{t:g}</text>')
    o.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{top}" y2="{H - 38}" stroke="var(--axis)" stroke-width="1.5"/>')
    for i, (lab, v, ci, ce, col, tip) in enumerate(rows):
        y = top + RH * i + (RH - BH) / 2; cy = y + BH / 2
        o.append(f'<g class="mk"><title>{e(tip)}</title><text x="{LW - 10}" y="{y + 14:.1f}" text-anchor="end" fill="var(--fg)" font-size="13">{e(lab)}</text><path d="{barp(X(0), X(v), y, BH)}" fill="var({col})" opacity=".9"/>')
        if ci: o.append(f'<line x1="{X(ci[0]):.1f}" x2="{X(ci[1]):.1f}" y1="{cy}" y2="{cy}" stroke="var(--fg)" stroke-width="2"/><line x1="{X(ci[0]):.1f}" x2="{X(ci[0]):.1f}" y1="{cy - 5}" y2="{cy + 5}" stroke="var(--fg)" stroke-width="2"/><line x1="{X(ci[1]):.1f}" x2="{X(ci[1]):.1f}" y1="{cy - 5}" y2="{cy + 5}" stroke="var(--fg)" stroke-width="2"/>')
        if ce: o.append(f'<line x1="{X(ce):.1f}" x2="{X(ce):.1f}" y1="{y - 4}" y2="{y + BH + 4}" stroke="var(--ceil)" stroke-width="3"/>')
        o.append(f'<text x="{W - 112}" y="{y + 14:.1f}" fill="var(--fg)" font-size="12.5" font-family="var(--mono)">{v:+.3f}</text></g>')
    o.append("</svg></div>")
    return "".join(o)
def F(name):
    H, D = HO["endpoints"], DV["endpoints"]; P = HO["per_rep"]
    if name == "scale":
        rows = [("V-JEPA, whole frame", P["pt_g"]["G"], "--s1", "V-JEPA whole-frame G = 0.293"), ("V-JEPA, object window", P["pt_w"]["G"], "--s3", "V-JEPA object-window G = 0.114"),
                ("Trajectory baseline", P["traj"]["G"], "--s2", "Tracked-trajectory G = 0.145"), ("Untrained, whole frame", P["rw_g"]["G"], "--neu", "Random-weight G = 0.020"),
                ("Untrained, object window", P["rw_w"]["G"], "--neu", "Random-weight G = 0.004"), ("Raw pixels", P["px"]["G"], "--neu", "Pixel G = -0.001")]
        return hbars("Held-out cross-scene gravity alignment G by representation on a 0 to 1 scale", rows, -0.1, 1.0, band=(0.656, 0.845, "noise ceiling on dev worlds"), xlabel="G = mean cosine across scene pairs (0 unrelated, 1 identical). Band: dev noise ceiling 0.66–0.85.") + legend([("V-JEPA whole frame", "--s1"), ("V-JEPA object window", "--s3"), ("Trajectory baseline", "--s2"), ("Controls (untrained, pixels)", "--neu")])
    if name == "forest":
        rows = [("Held-out · whole frame", H["global"]["diff_vs_trajectory"], *H["global"]["diff_vs_trajectory_interval"], "--s1", False, "Held-out whole frame"),
                ("Held-out · object window", H["window"]["diff_vs_trajectory"], *H["window"]["diff_vs_trajectory_interval"], "--s3", False, "Held-out object window"),
                ("Dev dry run · whole frame", D["global"]["diff_vs_trajectory"], *D["global"]["diff_vs_trajectory_interval"], "--s1", True, "Dev whole frame"),
                ("Dev dry run · object window", D["window"]["diff_vs_trajectory"], *D["window"]["diff_vs_trajectory_interval"], "--s3", True, "Dev object window")]
        return forest("Difference between V-JEPA and trajectory alignment with 97.5 percent intervals, held-out and dev", rows, -0.45, 0.9, xlabel="Δ = G(V-JEPA) − G(trajectory). Success needs the whole interval right of 0.") + legend([("Whole-frame pooling", "--s1"), ("Object-window pooling", "--s3"), ("Filled dot = held-out, hollow = dev", "--neu")])
    if name == "pairs":
        g = [(k, [P[r]["pairs"][k] for r in ("pt_g", "pt_w", "traj")]) for k in ("freefall~bounce", "freefall~incline", "bounce~incline")]
        return cols("Held-out gravity alignment for each pair of scenes", g, [("V-JEPA whole frame", "--s1"), ("V-JEPA object window", "--s3"), ("Trajectory baseline", "--s2")], 0.6, ylabel="cosine of gravity direction") + legend([("V-JEPA whole frame", "--s1"), ("V-JEPA object window", "--s3"), ("Trajectory baseline", "--s2")])
    if name == "ceiling":
        c = CE["pretrained|global|norm23"]["cross_scene"]
        rows = [(f"{k}", v["obs"], None, v["ceiling"], "--s1", f"{k}: observed {v['obs']:.3f}, ceiling {v['ceiling']:.3f}, corrected {v['corrected']:.2f}") for k, v in c.items()]
        return interval_bars("Observed dev alignment against the noise ceiling for each scene pair", rows, 0, 1.0, W=720) + '<div class="lg"><span><i style="background:var(--s1)"></i>Observed (V-JEPA whole frame, dev)</span><span><i style="background:var(--ceil)"></i>Noise ceiling</span></div>'
    if name == "stage7":
        R = S7["reps"]
        rows = [("V-JEPA, whole frame", R["pt_g"]["cos_world_mean"], R["pt_g"]["ci95"], R["pt_g"]["ceiling"], "--s1", "V-JEPA whole frame"), ("V-JEPA, object window", R["pt_w"]["cos_world_mean"], R["pt_w"]["ci95"], R["pt_w"]["ceiling"], "--s3", "V-JEPA object window"),
                ("Untrained, whole frame", R["rw_g"]["cos_world_mean"], R["rw_g"]["ci95"], R["rw_g"]["ceiling"], "--neu", "Untrained whole frame"), ("Raw pixels", R["px"]["cos_world_mean"], R["px"]["ci95"], R["px"]["ceiling"], "--neu", "Pixels"),
                ("Trajectory", R["traj"]["cos_world_mean"], R["traj"]["ci95"], R["traj"]["ceiling"], "--s2", "Trajectory")]
        return interval_bars("Stage 7 cosine between the gravity-change and initial-speed-change directions with intervals and noise ceilings", rows, -0.25, 1.05) + '<div class="lg"><span><i style="background:var(--fg);height:3px"></i>95% interval</span><span><i style="background:var(--ceil)"></i>noise ceiling tick</span></div>'
    if name == "gate1":
        cases = [("g freefall", ".871 .889"), ("g bounce", ".623 .654"), ("g incline", ".389 .665"), ("e bounce", ".608 .928"), ("μ incline", ".392 .368")]
        order = ["g@freefall", "g@bounce", "g@incline", "e@bounce", "mu@incline"]
        grp = [(cases[i][0], [G1["reliability"][f"{k}|global"]["norm23"]["mean"], G1["reliability"][f"{k}|window"]["norm23"]["mean"]]) for i, k in enumerate(order)]
        return cols("Split-half reliability at block 23 for each parameter and pooling", grp, [("Whole frame", "--s1"), ("Object window", "--s3")], 1.0, thr=(0.5, "0.5 gate level (rule needs it at 2 of 4 layers)"), ylabel="split-half reliability r") + legend([("Whole-frame pooling", "--s1"), ("Object-window pooling", "--s3")]) + '<div class="lg"><span><i style="background:var(--ceil);height:3px"></i>dashed line = 0.5 gate level (the rule needs it at 2 of 4 layers)</span></div>'
    raise KeyError(name)
p = f"{ROOT}/reports/pilot_results.html"; s = open(p).read()
s = re.sub(r"<!--FIG:(\w+)-->.*?<!--/FIG-->|<!--FIG:(\w+)-->", lambda m: f"<!--FIG:{m.group(1) or m.group(2)}-->{F(m.group(1) or m.group(2))}<!--/FIG-->", s, flags=re.S)
open(p, "w").write(s); print("figures:", re.findall(r"<!--FIG:(\w+)-->", s))
