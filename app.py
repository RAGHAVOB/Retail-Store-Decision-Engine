from itertools import combinations
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st

D = Path(__file__).parent / "data"
st.set_page_config(page_title="Next-store decision engine", layout="wide")

def sdf(*a, **k):   # works on old and new Streamlit versions
    try: return st.dataframe(*a, width="stretch", **k)
    except TypeError: return st.dataframe(*a, use_container_width=True, **k)

def spc(*a, **k):
    try: return st.plotly_chart(*a, width="stretch", **k)
    except TypeError: return st.plotly_chart(*a, use_container_width=True, **k)

CM = {"Open": "#2347d9", "Gated out": "#d92d20", "Not selected": "#98a2b3", "Existing": "#067647"}

def hav(a, b, c, d):
    p = np.pi / 180
    x = np.sin((c - a) * p / 2) ** 2 + np.cos(a * p) * np.cos(c * p) * np.sin((d - b) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(x))

def dist(A, B):
    return hav(A.lat.values[:, None], A.lon.values[:, None], B.lat.values[None], B.lon.values[None])

def analyse(S0, E, P, n):
    S = S0.copy()
    f = lambda T: (T.footfall_k * np.sqrt(T.income_lakh / 15) * (T.accessibility / 7) ** .3 * (T.population_k / 350) ** .2
                   / (1 + P.comp_pen * T.competitors))
    ratio = E.monthly_rev_lakh / f(E)
    k = ratio.median(); mape = (k * f(E) / E.monthly_rev_lakh - 1).abs().mean()
    sig = max(0.15, 2 * mape)                       # forecast error: calibration error, floored (only 6 stores)
    dec = lambda d: P.cmax * np.clip(1 - d / P.radius, 0, 1)
    dxs, dss = dist(S, E), dist(S, S)
    S["revenue"] = k * f(S) * P.dm
    S["cann"] = np.minimum(.5, dec(dxs).sum(1))
    S["nearest"] = E.store.values[dxs.argmin(1)]
    S["rent_cost"] = S.rent_psf * P.rs * P.sqft / 1e5
    t = np.arange(1, 85); df = (1 + P.dr) ** (-t / 12)
    ramp = np.where(t <= 12, .7, np.where(t <= 24, .9, 1.0))
    A = (S.revenue * P.gm * (ramp * df).sum()).values            # PV of gross profit before cannibalisation
    B0 = P.opex * df.sum() + P.fit                                  # PV of fixed costs and fit-out
    Br = (S.rent_cost * (df.sum() + P.dep * (1 - df[-1]))).values   # PV of rent and deposit (refunded in month 84)
    B = B0 + Br
    S["capex"] = P.fit + P.dep * S.rent_cost
    rng = np.random.default_rng(11); m = len(S)
    Z = np.exp(sig * (.5 * rng.standard_normal((3000, 1)) + np.sqrt(.75) * rng.standard_normal((3000, m))) - sig ** 2 / 2)
    G = 1 + .08 * rng.standard_normal((3000, 1)); R = np.exp(.10 * rng.standard_normal((3000, m)) - .005)   # margin and rent uncertainty
    one = A * Z * G * (1 - S.cann.values) - B0 - Br * R
    S["enpv"], S["p10"], S["p90"] = one.mean(0) / 100, np.percentile(one, 10, 0) / 100, np.percentile(one, 90, 0) / 100
    S["pv_gp"], S["pv_cann"] = A / 100, -A * S.cann.values / 100                # value bridge components, Rs Cr
    S["pv_cost"] = -((S.rent_cost + P.opex) * df.sum()).values / 100
    S["pv_inv"] = -(P.fit + P.dep * S.rent_cost * (1 - df[-1])).values / 100
    S["ppos"] = (one > 0).mean(0) * 100
    S["mos"] = (1 - B / (A * (1 - S.cann.values))) * 100          # revenue miss the site can absorb before NPV < 0
    ss = S.revenue * (1 - S.cann) * P.gm - S.rent_cost - P.opex
    S["payback"] = np.where(ss > 0, S.capex / ss.where(ss > 0, 1), np.nan)
    def pf(J):
        v = 0
        for j in J:
            c = min(.5, S.cann.values[j] + .5 * sum(dec(dss[j, l]) for l in J if l != j))
            v = v + A[j] * Z[:, j] * G[:, 0] * (1 - c) - B0 - Br[j] * R[:, j]
        return v / 100
    ok = [i for i in range(m) if S.ppos[i] >= P.gate]
    rows = []
    for J in combinations(ok, min(n, len(ok))):
        cx = S.capex.values[list(J)].sum() / 100
        if cx <= P.budget and J:
            v = pf(J); rows.append(dict(sites=" + ".join(S.site[list(J)]), idx=J, capex=cx, enpv=v.mean(), p10=np.percentile(v, 10), ppos=(v > 0).mean() * 100, rank_score=.5 * (v.mean() + np.percentile(v, 10))))
    port = pd.DataFrame(rows).sort_values("rank_score", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
    naive = tuple(S.footfall_k.sort_values(ascending=False).index[:n]); nv = pf(naive)
    best = port.idx[0] if len(port) else ()
    S["call"] = ["Open" if i in best else "Gated out" if S.ppos[i] < P.gate else "Not selected" for i in range(m)]
    return S, port, dict(naive=naive, nv=nv.mean(), nv10=np.percentile(nv, 10), k=k, mape=mape, sig=sig)

with st.sidebar:
    st.header("Decision constraints")
    n = st.slider("Stores to open", 1, 4, 3)
    bud = st.slider("Capex budget (Rs crore)", 2.0, 8.0, 5.0, .5)
    gate = st.slider("Min. probability a site is NPV-positive %", 0, 90, 60)
    st.header("Economics")
    P = NS(n=n, budget=bud, gate=gate, gm=st.slider("Gross margin %", 25, 50, 40) / 100,
           sqft=st.slider("Store size (sq ft)", 2000, 5000, 3000, 250),
           opex=st.slider("Other monthly costs (Rs lakh)", 4.0, 9.0, 5.5, .5),
           fit=st.slider("Fit-out (Rs lakh)", 60, 150, 90, 10), dep=st.slider("Deposit (months of rent)", 3, 12, 6),
           dr=st.slider("Discount rate %", 8, 18, 12) / 100, cmax=st.slider("Max cannibalisation %", 10, 60, 30) / 100,
           radius=st.slider("Cannibalisation radius (km)", 3, 12, 7), comp_pen=.10, rs=1.0, dm=1.0)
    up = st.file_uploader("Use your own site data (CSV, same columns)", type="csv")

S0 = pd.read_csv(up) if up else pd.read_csv(D / "candidate_sites.csv")
E = pd.read_csv(D / "existing_stores.csv")
S, port, nv = analyse(S0, E, P, n)
st.title("Where should the next store open?")
st.caption("Mumbai expansion decision engine. Ranks portfolios of stores on risk-adjusted 7-year NPV, not footfall. Synthetic data.")
t1, t2, t3, t4, tv, t5 = st.tabs(["Decision memo", "Site economics", "Portfolio options", "Stress tests", "Business value and limitation", "Data, method, next steps"])

with t1:
    if port.empty:
        st.error("No portfolio meets the budget and risk gate. Relax the gate or raise the budget.")
    else:
        b = port.iloc[0]; sel = S.loc[list(b.idx)]
        st.success(f"**Recommendation: open {', '.join(sel.site)}.**")
        c = st.columns(4)
        c[0].metric("Expected 7-yr NPV", f"Rs {b.enpv:.2f} Cr")
        c[1].metric("Downside (P10) NPV", f"Rs {b.p10:.2f} Cr")
        c[2].metric("Chance NPV > 0", f"{b.ppos:.0f}%")
        c[3].metric("Capex", f"Rs {b.capex:.2f} Cr")
        st.markdown(f"**Versus the footfall heuristic** (open the {n} highest-footfall sites: {', '.join(S.site[list(nv['naive'])])}): "
                    f"expected NPV Rs {nv['nv']:.2f} Cr and downside Rs {nv['nv10']:.2f} Cr. "
                    f"The recommended portfolio adds **Rs {b.enpv - nv['nv']:.2f} Cr** of expected value.")
        cmp = pd.DataFrame({"Plan": ["Top-3 footfall sites"] * 2 + ["Recommended portfolio"] * 2, "Measure": ["Expected NPV", "Downside (P10)"] * 2,
                            "Rs Cr": [nv["nv"], nv["nv10"], b.enpv, b.p10]})
        spc(px.bar(cmp, x="Plan", y="Rs Cr", color="Measure", barmode="group", text_auto=".1f", title="7-year NPV of the store plan (Rs Cr)",
                               color_discrete_sequence=["#2347d9", "#98a2b3"]))
        st.subheader("Why these sites, and the trade-offs")
        for r in sel.itertuples():
            st.markdown(f"- **{r.site}**: expected NPV Rs {r.enpv:.2f} Cr, {r.ppos:.0f}% chance of being NPV-positive, "
                        f"absorbs a {r.mos:.0f}% revenue miss before NPV turns negative; takes {r.cann:.0%} of sales from {r.nearest}; "
                        f"rent is {r.rent_cost / r.revenue:.0%} of revenue.")
        hf = S.loc[S.footfall_k.idxmax()]
        if hf.site in sel.site.values:
            st.markdown(f"- **Footfall check:** {hf.site} has the highest footfall ({hf.footfall_k:.0f}k a day) and is selected, but it has the thinnest cushion in the portfolio ({hf.mos:.0f}% revenue miss), so it is the first to swap out if demand disappoints.")
        else:
            st.markdown(f"- **Why not {hf.site}?** Highest footfall ({hf.footfall_k:.0f}k a day) but rent is {hf.rent_cost / hf.revenue:.0%} of revenue and {hf.cann:.0%} of its sales come from {hf.nearest}. Expected NPV Rs {hf.enpv:.2f} Cr.")
        w = sel.loc[sel.mos.idxmin()]
        st.markdown(f"- **What would change our mind:** if {w.site}'s revenue lands more than {w.mos:.0f}% below forecast, or its rent is renegotiated up, it should be swapped for the next-best site. "
                    "See the stress tests tab for portfolio-level breakpoints.")
        st.markdown("**Rollout:** open the two strongest sites first. Treat the third as a gated pilot, committing the next stores only if months 1-6 hit at least 85% of forecast revenue per sq ft and cannibalisation of the nearest existing store stays under the modelled share.")

with t2:
    show = S.assign(cann_pct=S.cann * 100, rent_pct=S.rent_cost / S.revenue * 100)[
        ["site", "call", "revenue", "cann_pct", "rent_pct", "enpv", "p10", "ppos", "mos", "payback", "capex"]].sort_values("enpv", ascending=False)
    sdf(show, hide_index=True, column_config={
        "revenue": st.column_config.NumberColumn("Revenue (Rs L/mo)", format="%.0f"), "cann_pct": st.column_config.NumberColumn("Cannibalised %", format="%.0f"),
        "rent_pct": st.column_config.NumberColumn("Rent % of revenue", format="%.0f"), "enpv": st.column_config.NumberColumn("Exp. NPV (Rs Cr)", format="%.2f"),
        "p10": st.column_config.NumberColumn("P10 NPV (Rs Cr)", format="%.2f"), "ppos": st.column_config.NumberColumn("P(NPV>0) %", format="%.0f"),
        "mos": st.column_config.NumberColumn("Revenue miss absorbed %", format="%.0f"), "payback": st.column_config.NumberColumn("Payback (months)", format="%.0f"),
        "capex": st.column_config.NumberColumn("Capex (Rs L)", format="%.0f")})
    fig = px.scatter(S, x="footfall_k", y="enpv", color="call", text="site", color_discrete_map=CM, error_y=S.p90 - S.enpv, error_y_minus=S.enpv - S.p10,
                     labels={"footfall_k": "Daily footfall (000)", "enpv": "Expected 7-yr NPV (Rs Cr), bars = P10 to P90"})
    fig.update_traces(textposition="top center"); fig.add_hline(y=0, line_dash="dot")
    spc(fig)
    st.caption("Footfall barely predicts value. Rent, competition and cannibalisation decide it.")
    r1, r2 = S.footfall_k.rank(ascending=False).astype(int), S.enpv.rank(ascending=False).astype(int)
    sg = go.Figure()
    for i in S.index:
        sg.add_trace(go.Scatter(x=[0, 1], y=[r1[i], r2[i]], mode="lines+markers+text", text=[S.site[i], S.site[i]], textposition=["middle left", "middle right"],
                                line=dict(color=CM[S.call[i]], width=3 if S.call[i] == "Open" else 1.5), showlegend=False, hoverinfo="text"))
    sg.update_layout(title="Rank by footfall (left) vs rank by risk-adjusted value (right)", height=480, xaxis=dict(visible=False, range=[-.6, 1.6]),
                     yaxis=dict(autorange="reversed", title="Rank (1 = best)", dtick=1))
    spc(sg)
    pick = st.selectbox("Value bridge for site", S.site.tolist(), index=int(S.footfall_k.idxmax()))
    r = S[S.site == pick].iloc[0]
    wf = go.Figure(go.Waterfall(x=["Gross profit (PV)", "Sales taken from existing stores", "Rent and running costs", "Fit-out and deposit", "Expected NPV"],
                                measure=["relative"] * 4 + ["total"], y=[r.pv_gp, r.pv_cann, r.pv_cost, r.pv_inv, r.pv_gp + r.pv_cann + r.pv_cost + r.pv_inv],
                                text=[f"{v:.1f}" for v in [r.pv_gp, r.pv_cann, r.pv_cost, r.pv_inv, r.pv_gp + r.pv_cann + r.pv_cost + r.pv_inv]],
                                increasing=dict(marker=dict(color="#067647")), decreasing=dict(marker=dict(color="#d92d20")), totals=dict(marker=dict(color="#2347d9"))))
    wf.update_layout(title=f"Where {pick}'s value goes (7-year, Rs Cr)", showlegend=False)
    spc(wf)

with t3:
    if not port.empty:
        sdf(port.head(8).drop(columns=["idx", "rank_score"]), hide_index=True, column_config={
            "sites": "Portfolio", "capex": st.column_config.NumberColumn("Capex (Rs Cr)", format="%.2f"), "enpv": st.column_config.NumberColumn("Exp. NPV (Rs Cr)", format="%.2f"),
            "p10": st.column_config.NumberColumn("P10 NPV (Rs Cr)", format="%.2f"), "ppos": st.column_config.NumberColumn("P(NPV>0) %", format="%.0f")})
        st.caption("Stores in the same portfolio also cannibalise each other, which is why the best portfolio is not simply the top sites ranked one by one.")
    M = pd.concat([S.assign(kind=S.call, label=S.site)[["label", "lat", "lon", "kind"]], E.assign(kind="Existing", label=E.store)[["label", "lat", "lon", "kind"]]])
    mp = px.scatter(M, x="lon", y="lat", color="kind", symbol="kind", text="label", color_discrete_map=CM, height=520,
                    symbol_map={"Open": "circle", "Not selected": "circle", "Gated out": "circle", "Existing": "diamond"},
                    labels={"lon": "Longitude", "lat": "Latitude", "kind": ""}, title="Candidate sites and existing stores (schematic map, no tiles needed)")
    mp.update_traces(marker=dict(size=14), textposition="top center")
    mp.update_yaxes(scaleanchor="x", scaleratio=1.06)
    spc(mp)

with t4:
    st.caption("Each scenario re-picks the best portfolio with the NPV gate switched off, so shifts show where the choice is fragile; the last column counts sites that still clear your gate.")
    sc = {"Base case": {}, "Rent +20%": dict(rs=1.2), "Gross margin -4 pts": dict(gm=P.gm - .04), "Demand -15%": dict(dm=.85),
          "Cannibalisation cap 50%": dict(cmax=.5), "Discount rate +4 pts": dict(dr=P.dr + .04)}
    rows = []
    for name, ov in sc.items():
        X, pp, _ = analyse(S0, E, NS(**{**vars(P), **ov, "gate": 0}), n)
        rows.append({"Scenario": name, "Best portfolio": pp.sites[0] if len(pp) else "none passes the gate", "Gate-passing sites": int((X.ppos >= P.gate).sum()),
                     "Exp. NPV (Rs Cr)": round(pp.enpv[0], 2) if len(pp) else None, "P(NPV>0) %": round(pp.ppos[0]) if len(pp) else None})
    R = pd.DataFrame(rows); sdf(R, hide_index=True)
    spc(px.bar(R, x="Scenario", y="Exp. NPV (Rs Cr)", text_auto=".1f", title="Expected NPV of the best portfolio under each scenario (Rs Cr)",
                           color_discrete_sequence=["#2347d9"]))
    sets = [set(s.split(" + ")) for s in R["Best portfolio"] if s != "none passes the gate"]
    core = set.intersection(*sets) if sets else set()
    st.markdown("**Robust core:** " + (", ".join(sorted(core)) or "none") + ". Sites outside the core are bets on a specific assumption.")

with tv:
    if port.empty:
        st.info("No portfolio meets the budget and risk gate, so there is no value case to show yet.")
    else:
        b = port.iloc[0]
        st.header("Business value")
        c = st.columns(3)
        c[0].metric("Extra expected value vs the footfall rule", f"Rs {b.enpv - nv['nv']:.2f} Cr")
        c[1].metric("Downside case (P10 NPV)", f"Rs {b.p10:.2f} Cr", delta=f"{b.p10 - nv['nv10']:.2f} Cr vs footfall rule")
        c[2].metric("Chance NPV is above zero", f"{b.ppos:.0f}%")
        l, r = st.columns(2)
        with l:
            st.success(f"""**What the business gets**
- Rs {b.enpv - nv['nv']:.2f} Cr more expected value than opening the highest-footfall sites
- One consistent, explainable method for every site
- Risk visible up front: downside case and chance of loss
- A gated rollout that limits the cost of being wrong""")
        with r:
            st.warning("""**One key limitation**

The data is synthetic and the revenue model is calibrated on only six existing stores. Trust the site ranking; treat the rupee values as indicative until they are recalibrated on the client's real data.

**Next:** recalibrate on real footfall, rent and store P&Ls, then pilot the first stores.""")

with t5:
    st.subheader("Synthetic data")
    sdf(S0, hide_index=True); sdf(E, hide_index=True)
    st.markdown(f"""
**Data note.** Figures are illustrative for Mumbai micro-markets, not sourced. Replace with footfall counts, broker rent data and the chain's store P&Ls.
**Revenue calibration.** Revenue index = footfall x (income / Rs 15 L)^0.5 x (accessibility / 7)^0.3 x (population / 350k)^0.2 / (1 + 10% per competitor); the elasticities are assumptions. Rs {nv['k']:.2f} L per index point is the median across the 6 existing stores (in-sample error {nv['mape']:.0%}). Forecast uncertainty is set to {nv['sig']:.0%}, floored because 6 stores is a thin base.
**Cannibalisation.** Up to {P.cmax:.0%} of a new store's sales come from each existing store, fading to zero at {P.radius} km, capped at 50%. New stores in one portfolio take half that share from each other.
**Valuation.** 84 months, 12% default discount rate, sales ramp at 70% (year 1), 90% (year 2), 100% after. Deposit refunded at month 84. Uncertainty: 3,000 simulations of revenue error (partly shared across sites), gross margin (±8%) and rent (±10%).
**Decision rule.** Drop sites below the NPV-positive gate, then pick the portfolio with the best risk-adjusted NPV (average of expected and P10 NPV) inside the capex budget and store count.
**Key limitation.** The model is only as good as the six-store calibration and the distance proxy for cannibalisation. Real trade-area data and loyalty-card switching are needed before committing capital.
**Next steps.** Pilot the weakest-margin site, measure revenue per sq ft and nearest-store sales change for 6 months, then recalibrate and release the next tranche.
""")