"""Acid-Base Titration Virtual Laboratory (Streamlit)"""
import os
from datetime import date
from io import BytesIO
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

st.set_page_config(page_title="Titration Virtual Lab", page_icon="⚗️", layout="wide")
st.markdown("<style>.main-title{text-align:center;font-size:40px;font-weight:bold;}</style>",
            unsafe_allow_html=True)
BASE = os.path.dirname(os.path.abspath(__file__))
LOGO = os.path.join(BASE, "srm_logo.jpeg")
DEVELOPER = "Aravinthan K and Edisun Fisher"
DEVELOPER_INFO = "Section Cyber B | SRM Institute of Science and Technology, Tiruchirappalli"
KW = 1e-14

# ---------------- data ----------------
ANALYTES = {  # name: (kind, K)   K=None means strong
    "Hydrochloric acid (HCl)": ("acid", None), "Nitric acid (HNO3)": ("acid", None),
    "Acetic acid (CH3COOH)": ("acid", 1.8e-5), "Formic acid (HCOOH)": ("acid", 1.8e-4),
    "Benzoic acid (C6H5COOH)": ("acid", 6.3e-5), "Hydrofluoric acid (HF)": ("acid", 6.8e-4),
    "Sodium hydroxide (NaOH)": ("base", None), "Ammonia (NH3)": ("base", 1.8e-5),
    "Methylamine (CH3NH2)": ("base", 4.4e-4), "Pyridine (C5H5N)": ("base", 1.7e-9),
}
INDICATORS = {  # name: (pH low, pH high, colour below, colour above)
    "Methyl orange": (3.1, 4.4, "#e8332a", "#f5d50a"),
    "Methyl red": (4.4, 6.2, "#e8332a", "#f5d50a"),
    "Bromothymol blue": (6.0, 7.6, "#f5d50a", "#2a5fe0"),
    "Phenolphthalein": (8.2, 10.0, "#f4f4f4", "#e0378f"),
}
QUIZ = [
    ("The point where moles of acid equal moles of base is the...", ["End point", "Equivalence point", "Boiling point"], "Equivalence point"),
    ("pH at equivalence for a strong acid with a strong base (25 °C) is...", ["7", "Less than 7", "More than 7"], "7"),
    ("Best indicator for a weak acid titrated with a strong base?", ["Methyl orange", "Phenolphthalein", "Litmus paper"], "Phenolphthalein"),
    ("At half-equivalence of a weak acid titration, pH equals...", ["pKa", "7", "pKw"], "pKa"),
    ("Which solution is normally placed in the burette?", ["The titrant of known concentration", "The unknown solution", "Distilled water"], "The titrant of known concentration"),
    ("pH at equivalence for a weak acid with a strong base is...", ["Exactly 7", "Below 7", "Above 7"], "Above 7"),
    ("Methyl orange turns what colour in basic solution?", ["Red", "Yellow", "Colourless"], "Yellow"),
    ("The steepest part of a titration curve is near the...", ["Start", "Equivalence point", "Very end"], "Equivalence point"),
    ("A buffer region is seen when titrating...", ["A strong acid with a strong base", "A weak acid with a strong base", "Water with water"], "A weak acid with a strong base"),
    ("For a 1:1 reaction, M1V1 = M2V2 is used to find...", ["Unknown concentration", "Temperature", "Molar mass of water"], "Unknown concentration"),
]


# ---------------- chemistry maths ----------------
def ph_at(kind, K, ca, va, ct, vt):
    """pH after adding vt mL of titrant (conc ct) to va mL of analyte (conc ca). Exact charge balance."""
    V = (va + vt) / 1000.0
    A, T = ca * va / 1000.0 / V, ct * vt / 1000.0 / V

    def f(ph):
        h = 10 ** (-ph)
        if kind == "acid":  # HA titrated with NaOH
            return T + h - A * (1.0 if K is None else K / (K + h)) - KW / h
        return A * (1.0 if K is None else h / (h + KW / K)) + h - KW / h - T  # B titrated with HCl

    lo, hi = -1.0, 15.0
    flo = f(lo)
    for _ in range(60):
        mid = (lo + hi) / 2
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return (lo + hi) / 2


@st.cache_data(show_spinner=False)
def curve(kind, K, ca, va, ct):
    veq = ca * va / ct
    vs = np.linspace(0, 2 * veq, 201)
    return vs, np.array([ph_at(kind, K, ca, va, ct, v) for v in vs])


def key_results(kind, K, ca, va, ct):
    veq = ca * va / ct
    p = lambda v: ph_at(kind, K, ca, va, ct, v)
    r = {"Equivalence volume (mL)": round(veq, 2), "Initial pH": round(p(0), 2),
         "pH at half-equivalence": round(p(veq / 2), 2), "pH at equivalence": round(p(veq), 2),
         "pH at 2 x equivalence": round(p(2 * veq), 2),
         "Steep region (pH)": f"{p(veq * 0.999):.2f} to {p(veq * 1.001):.2f}"}
    if K:
        r["pKa" if kind == "acid" else "pKb"] = round(-np.log10(K), 2)
    return r


def suitability(kind, K, ca, va, ct):
    veq = ca * va / ct
    a, b = sorted((ph_at(kind, K, ca, va, ct, veq * 0.999), ph_at(kind, K, ca, va, ct, veq * 1.001)))
    return [{"Indicator": n, "Range (pH)": f"{lo} - {hi}", "Suitable?": "Yes" if a <= (lo + hi) / 2 <= b else "No"} for n, (lo, hi, *_) in INDICATORS.items()]


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def flask_colour(ind, ph):
    lo, hi, c1, c2 = INDICATORS[ind]
    return mix(c1, c2, min(max((ph - lo) / (hi - lo), 0.0), 1.0))


def make_fig(vs, ph, veq, ind=None, mark=None, extra=None):
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(vs, ph, lw=2.2, color="#1f4e79", label="Selected")
    for label, (xs, ys) in (extra or {}).items():
        ax.plot(xs, ys, lw=1.5, ls="--", label=label)
    ax.axvline(veq, ls=":", color="gray")
    ax.text(veq, 0.4, f" Equivalence\n {veq:.1f} mL", fontsize=8, color="gray")
    if ind:
        lo, hi, _, c2 = INDICATORS[ind]
        ax.axhspan(lo, hi, color=c2, alpha=0.25, label=f"{ind} range")
    if mark:
        ax.plot(*mark, "o", color="red", ms=9, zorder=5)
    ax.set_xlabel("Volume of titrant added (mL)")
    ax.set_ylabel("pH")
    ax.set_ylim(0, 14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right" if not extra else "best")
    fig.tight_layout()
    return fig


def fig_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def build_pdf(analyte, kind, K, ca, va, ct, ind, student, reg):
    vs, ph = curve(kind, K, ca, va, ct)
    veq = ca * va / ct
    S = getSampleStyleSheet()
    N = S["Normal"]

    def grid(data, widths):
        t = Table([[Paragraph(escape(str(c)), N) for c in r] for r in data],
                  colWidths=[w * inch for w in widths], repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
        return t

    story = []
    if os.path.exists(LOGO):
        story.append(RLImage(LOGO, width=3.2 * inch, height=1.6 * inch))
    story += [Paragraph("Acid-Base Titration Report", S["Title"]),
              Paragraph(f"<b>Student:</b> {escape(student or 'Not provided')} &nbsp; "
                        f"<b>Reg. No.:</b> {escape(reg or 'Not provided')} &nbsp; "
                        f"<b>Date:</b> {date.today():%d-%m-%Y}", N), Spacer(1, 8)]
    titrant = "NaOH" if kind == "acid" else "HCl"
    sections = [
        ("1. Experiment setup", [["Item", "Value"], ["Analyte", analyte], ["Analyte concentration", f"{ca} M"],
                                 ["Analyte volume", f"{va} mL"], ["Titrant", f"{titrant}, {ct} M"], ["Indicator", ind]], [2.5, 3.7]),
        ("2. Key results", [["Result", "Value"]] + [[k, v] for k, v in key_results(kind, K, ca, va, ct).items()], [2.5, 3.7]),
    ]
    for head, data, w in sections:
        story += [Paragraph(head, S["Heading2"]), grid(data, w), Spacer(1, 6)]
    story += [Paragraph("3. Titration curve", S["Heading2"]),
              RLImage(BytesIO(fig_png(make_fig(vs, ph, veq, ind))), width=5.2 * inch, height=3.35 * inch)]
    sug = suitability(kind, K, ca, va, ct)
    story += [Paragraph("4. Indicator suitability", S["Heading2"]),
              grid([["Indicator", "Range (pH)", "Suitable?"]] + [[r["Indicator"], r["Range (pH)"], r["Suitable?"]] for r in sug], [2.5, 2, 1.7]),
              Paragraph("5. Titration data", S["Heading2"]),
              grid([["Volume (mL)", "pH"]] + [[f"{vs[i]:.2f}", f"{ph[i]:.2f}"] for i in range(0, 201, 10)], [2.5, 2])]
    buf = BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, topMargin=30, bottomMargin=30).build(story)
    return buf.getvalue()


# ---------------- UI ----------------
st.sidebar.title("⚗️ Titration Lab")
page = st.sidebar.radio("Navigation", ["🏠 Home", "📖 Theory", "🧪 Titration Simulator",
                                       "📊 Titration Data Report", "📈 Curve Analysis", "📝 Quiz"])
st.sidebar.divider()
student = st.sidebar.text_input("Student Name", key="student_name")
reg = st.sidebar.text_input("Registration Number", key="registration_number")
st.sidebar.divider()
st.sidebar.markdown("### 🔬 Set up the experiment")
analyte = st.sidebar.selectbox("Solution in the flask (analyte)", list(ANALYTES) + ["Custom weak acid", "Custom weak base"])
if analyte.startswith("Custom"):
    kind = "acid" if analyte.endswith("acid") else "base"
    try:
        K = float(st.sidebar.text_input("Ka" if kind == "acid" else "Kb", "1e-5"))
        assert K > 0
    except Exception:
        st.sidebar.error("Enter a positive number such as 1.8e-5. Using 1e-5.")
        K = 1e-5
else:
    kind, K = ANALYTES[analyte]
titrant = "NaOH" if kind == "acid" else "HCl"
ca = st.sidebar.number_input("Analyte concentration (M)", 0.001, 2.0, 0.1, 0.01, format="%.3f")
va = st.sidebar.number_input("Analyte volume (mL)", 5.0, 100.0, 25.0, 1.0)
ct = st.sidebar.number_input(f"{titrant} (titrant) concentration (M)", 0.001, 2.0, 0.1, 0.01, format="%.3f")
ind = st.sidebar.selectbox("Indicator", list(INDICATORS))
args = (kind, K, ca, va, ct)
veq = ca * va / ct
st.sidebar.download_button("📄 Download PDF report", build_pdf(analyte, *args, ind, student, reg),
                           file_name="titration_report.pdf", mime="application/pdf")

if page == "🏠 Home":
    if os.path.exists(LOGO):
        _, mid, _ = st.columns([1, 4, 1])
        mid.image(LOGO)
    st.markdown('<div class="main-title">⚗️ Acid–Base Titration Virtual Laboratory</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;font-size:22px;font-weight:700;color:#1f4e79;">'
                'SRM Institute of Science and Technology, Tiruchirappalli</div>'
                '<div style="text-align:center;font-size:17px;font-weight:600;">'
                'School of Sciences | Division of Chemistry</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("""
## Welcome
Run a titration on screen. Choose the acid or base, set the concentrations and volumes, add titrant drop by drop and watch the pH and the indicator colour change.

🧪 **Titration Simulator**: add titrant and see the flask colour and pH live

📊 **Titration Data Report**: key results, full data table and indicator suitability

📈 **Curve Analysis**: derivative curve, buffer region and comparison with other acids and bases

📄 **PDF report** with the titration curve and data, and a 📝 **Quiz**
""")
    st.info("All values are calculated from exact equilibrium equations at 25 °C, so you can change any input and get a new titration.")
    st.markdown(f"### 👨‍🎓 Developed by\n**{DEVELOPER}**  \n{DEVELOPER_INFO}")

elif page == "📖 Theory":
    st.title("📖 Theory")
    st.markdown("""
**Titration** finds the unknown concentration of a solution by reacting it with a solution of known concentration (the titrant).

**Equivalence point:** moles of acid = moles of base. For a 1:1 reaction, M₁V₁ = M₂V₂.

**End point:** the volume at which the indicator changes colour. A good indicator changes colour at the pH of the equivalence point.

**Strong acid + strong base:** pH = 7 at equivalence. **Weak acid + strong base:** pH > 7. **Weak base + strong acid:** pH < 7.

**Buffer region:** while a weak acid is being titrated, pH changes slowly around pH = pKa. At half-equivalence, pH = pKa.

**Henderson–Hasselbalch:** pH = pKa + log([A⁻]/[HA]).
""")

elif page == "🧪 Titration Simulator":
    st.title("🧪 Titration Simulator")
    st.caption(f"**{analyte}**, {ca} M, {va} mL in the flask. Titrant: **{titrant}**, {ct} M. Indicator: **{ind}**.")
    vt = st.slider("Volume of titrant added (mL)", 0.0, max(round(2 * veq, 1), 0.2), 0.0, 0.1)
    p = ph_at(kind, K, ca, va, ct, vt)
    stage = ("Before equivalence" if vt < veq - 0.05 else "At equivalence point" if abs(vt - veq) <= 0.05 else "After equivalence (excess titrant)")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f'<div style="width:150px;height:150px;border-radius:50%;background:{flask_colour(ind, p)};'
                    'border:4px solid #555;margin:10px auto;"></div>'
                    '<div style="text-align:center;font-weight:600;">Flask colour</div>', unsafe_allow_html=True)
        st.metric("pH", f"{p:.2f}")
        st.metric("Titrant added", f"{vt:.1f} mL")
        st.info(f"{stage}. Neutralised: {min(vt / veq * 100, 100):.0f}%")
    with c2:
        vs, ph = curve(*args)
        fig = make_fig(vs, ph, veq, ind, (vt, p))
        st.pyplot(fig)
        plt.close(fig)

elif page == "📊 Titration Data Report":
    st.title("📊 Titration Data Report")
    t1, t2, t3 = st.tabs(["⚗️ Results", "📋 Data table", "🎯 Indicator suitability"])
    with t1:
        res = key_results(*args)
        st.dataframe(pd.DataFrame({"Result": list(res), "Value": [str(v) for v in res.values()]}), hide_index=True)
        st.write(f"**Unknown concentration check:** {ca} M × {va} mL = {ct} M × **{veq:.2f} mL**")
    with t2:
        vs, ph = curve(*args)
        df = pd.DataFrame({"Volume of titrant (mL)": vs.round(2), "pH": ph.round(3)})
        st.dataframe(df, hide_index=True)
        st.download_button("⬇️ Data (CSV)", df.to_csv(index=False).encode("utf-8-sig"), file_name="titration_data.csv", mime="text/csv")
    with t3:
        st.dataframe(pd.DataFrame(suitability(*args)), hide_index=True)
        st.caption("An indicator is suitable if the middle of its colour range lies inside the steep jump around the equivalence point.")

elif page == "📈 Curve Analysis":
    st.title("📈 Curve Analysis")
    vs, ph = curve(*args)
    st.subheader("1. Titration curve and its derivative")
    c1, c2 = st.columns(2)
    with c1:
        fig = make_fig(vs, ph, veq, ind)
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        d = np.gradient(ph, vs)
        fig, ax = plt.subplots(figsize=(6.2, 4))
        ax.plot(vs, d, color="#c0392b", lw=2)
        ax.axvline(veq, ls=":", color="gray")
        ax.set_xlabel("Volume of titrant added (mL)")
        ax.set_ylabel("dpH/dV")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.caption("The peak of the derivative curve marks the equivalence point.")
    st.subheader("2. What the curve tells us")
    r = key_results(*args)
    st.write(f"- **Type:** {'strong' if K is None else 'weak'} {kind} titrated with a strong {'base' if kind == 'acid' else 'acid'}.")
    st.write(f"- **Equivalence point:** {veq:.2f} mL, pH = {r['pH at equivalence']}. " +
             ("It is neutral." if abs(r['pH at equivalence'] - 7) < 0.3 else "It is basic." if r['pH at equivalence'] > 7 else "It is acidic."))
    if K:
        pk = -np.log10(K)
        st.write(f"- **Buffer region:** pH {pk - 1:.2f} to {pk + 1:.2f} (around pK = {pk:.2f}); the pH at half-equivalence is {r['pH at half-equivalence']}.")
    else:
        st.write("- **No buffer region:** strong acids and bases give a flat start and a very sharp jump.")
    good = [x["Indicator"] for x in suitability(*args) if x["Suitable?"] == "Yes"]
    st.write("- **Suitable indicators:** " + (", ".join(good) if good else "none of the four listed ones."))
    st.subheader("3. Compare with other analytes")
    same = [n for n, (k, _) in ANALYTES.items() if k == kind and n != analyte]
    picks = st.multiselect("Overlay curves (same concentration and volume)", same, default=same[:2])
    extra = {n.split(" (")[0]: curve(kind, ANALYTES[n][1], ca, va, ct) for n in picks}
    fig = make_fig(vs, ph, veq, extra=extra)
    st.pyplot(fig)
    plt.close(fig)

elif page == "📝 Quiz":
    st.title("📝 Quiz")
    answers = [st.radio(f"Q{i}. {q}", o, index=None, key=f"quiz_{i}") for i, (q, o, _) in enumerate(QUIZ, 1)]
    if st.button("Submit Quiz"):
        if None in answers:
            st.warning("Please answer all the questions before submitting.")
        else:
            score = sum(a == k for a, (_, _, k) in zip(answers, QUIZ))
            pct = score / len(QUIZ) * 100
            st.success(f"{student or 'Student'}, your score: {score}/{len(QUIZ)} ({pct:.0f}%)")
            if pct >= 80:
                st.balloons()
            with st.expander("Review answers"):
                for i, (a, (q, _, k)) in enumerate(zip(answers, QUIZ), 1):
                    st.write(f"{'✅' if a == k else '❌'} Q{i}. {q} — correct answer: **{k}**")

st.divider()
st.caption("⚗️ Acid–Base Titration Virtual Laboratory | Python, Streamlit, NumPy and Matplotlib")
