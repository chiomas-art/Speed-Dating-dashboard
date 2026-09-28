import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="Speed Dating Dashboard", page_icon="💘", layout="wide")

RED, ORANGE, YELLOW, GREEN, BLUE, INDIGO, VIOLET = "#FF595E", "#FF924C", "#FFCA3A", "#8AC926", "#1982C4", "#6A4C93", "#C77DFF"
RAINBOW = [RED, ORANGE, YELLOW, GREEN, BLUE, INDIGO, VIOLET]
INK, PAPER = "#2B2140", "#FBF8FF"
GRAD = ", ".join(RAINBOW)

st.markdown(f"""
<style>
.stApp {{ background: {PAPER}; }}
.hero {{ background: linear-gradient(90deg, {GRAD}); padding: 24px 28px; border-radius: 18px; }}
.hero h1 {{ color: white !important; margin: 0; text-shadow: 0 2px 6px rgba(0,0,0,.35); }}
.hero p {{ color: white !important; margin: 6px 0 0 0; text-shadow: 0 1px 4px rgba(0,0,0,.35); }}
.kpi {{ padding: 16px 10px; border-radius: 14px; text-align: center; box-shadow: 0 3px 8px rgba(0,0,0,.12); }}
.kpi .l {{ font-size: 13px; }} .kpi .v {{ font-size: 28px; font-weight: 800; }}
.sec {{ padding-left: 12px; margin: 30px 0 10px 0; }}
.sec h2 {{ margin: 0; color: {INK}; }} .sec p {{ margin: 0; color: #6B5B8C; }}
.rbpanel {{ border: 5px solid transparent; border-radius: 20px; padding: 14px 22px; margin-top: 8px;
  background: linear-gradient({PAPER}, {PAPER}) padding-box, linear-gradient(90deg, {GRAD}) border-box; }}
h3 {{ color: {INK}; }}
</style>
""", unsafe_allow_html=True)

def kpi(label, value, color, text="white"):
    return f'<div class="kpi" style="background:{color};color:{text}"><div class="l">{label}</div><div class="v">{value}</div></div>'

def section(title, sub, color):
    st.markdown(f'<div class="sec" style="border-left:10px solid {color}"><h2>{title}</h2><p>{sub}</p></div>',
                unsafe_allow_html=True)

def paint(fig, *axes):
    fig.patch.set_facecolor(PAPER)
    for ax in axes:
        ax.set_facecolor("white")
        ax.tick_params(colors=INK, labelsize=8)
        ax.title.set_color(INK)
        for s in ax.spines.values():
            s.set_color(INK)

# ---------------- data ----------------
@st.cache_data(show_spinner="Fetching the Speed Dating dataset from OpenML…")
def fetch_from_openml():
    for kw in [dict(name="SpeedDating", version=1), dict(data_id=40536)]:
        try:
            return fetch_openml(as_frame=True, parser="auto", **kw).frame
        except Exception:
            continue
    return None

def get_raw():
    for p in ["speed_dating.csv", "data/speed_dating.csv", "SpeedDating.csv", "data/SpeedDating.csv",
              "Speed Dating Data.csv", "data/Speed Dating Data.csv"]:
        if Path(p).exists():
            return pd.read_csv(p, encoding="latin-1")
    raw = fetch_from_openml()
    if raw is not None:
        return raw
    up = st.sidebar.file_uploader("Upload the Speed Dating CSV", type="csv")
    if up is None:
        st.info("Couldn't fetch the dataset automatically. Upload the Speed Dating CSV in the sidebar to continue.")
        st.stop()
    return pd.read_csv(up, encoding="latin-1")

@st.cache_data
def prepare(raw):
    df = raw.copy()
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "category":
            s = df[c].astype(str).str.replace(r"^b'(.*)'$", r"\1", regex=True).str.strip()
            num = pd.to_numeric(s, errors="coerce")
            df[c] = num if num.notna().mean() > 0.8 else s.replace("nan", np.nan)
    return df

df = prepare(get_raw())

def find(*names):
    for n in names:
        if n in df.columns:
            return n
    return None

MATCH, GENDER, AGE, LIKE = find("match"), find("gender"), find("age"), find("like", "like_o")
if MATCH is None:
    st.error("Couldn't find a 'match' column in this file.")
    st.stop()
df[MATCH] = pd.to_numeric(df[MATCH], errors="coerce")
df = df[df[MATCH].isin([0, 1])].copy()

if GENDER:
    g = df[GENDER]
    df["Gender"] = g.map({0: "Female", 1: "Male"}) if pd.api.types.is_numeric_dtype(g) else g.astype(str).str.strip().str.title()
    df.loc[~df["Gender"].isin(["Male", "Female"]), "Gender"] = np.nan
else:
    df["Gender"] = np.nan

cands = [("Attractive", find("attractive_o", "attr_o")), ("Sincere", find("sinsere_o", "sinc_o", "sincere_o")),
         ("Intelligent", find("intelligence_o", "intel_o")), ("Fun", find("funny_o", "fun_o")),
         ("Ambitious", find("ambitous_o", "amb_o", "ambitious_o")),
         ("Shared Interests", find("shared_interests_o", "shar_o"))]
ATTRS = [(lab, c) for lab, c in cands if c and df[c].notna().any()]
FEATS = [c for _, c in ATTRS] + ([LIKE] if LIKE and df[LIKE].notna().any() else [])
LABELS = {c: lab for lab, c in ATTRS}
if LIKE:
    LABELS[LIKE] = "Overall liking"
if not FEATS:
    st.error("Couldn't find the rating columns in this file.")
    st.stop()

@st.cache_resource
def train(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    lr = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=2000, random_state=42))]).fit(Xtr, ytr)
    rf = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1))]).fit(Xtr, ytr)
    res = {n: (accuracy_score(yte, m.predict(Xte)), roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
           for n, m in [("Logistic Regression", lr), ("Random Forest", rf)]}
    return rf, res

rf, res = train(df[FEATS], df[MATCH])
base_rate = df[MATCH].mean()

# ---------------- header + filter bar ----------------
st.markdown("""<div class="hero"><h1>💘 Speed Dating Experiment Dashboard 🌈</h1>
<p>Who clicks with whom? Filter the floor, read the story, then test the chemistry yourself.</p></div>""",
            unsafe_allow_html=True)
st.write("")

f1, f2, f3 = st.columns([1, 2, 1])
genders = f1.multiselect("Gender", ["Female", "Male"], default=["Female", "Male"]) if GENDER else []
f = df.copy()
if genders:
    f = f[f["Gender"].isin(genders)]
if AGE and df[AGE].notna().any():
    lo, hi = int(df[AGE].min()), int(df[AGE].max())
    a_lo, a_hi = f2.slider("Age range", lo, hi, (lo, hi))
    f = f[f[AGE].between(a_lo, a_hi) | f[AGE].isna()]
f3.metric("Dates in view", f"{len(f):,}")
if len(f) < 30:
    st.warning("Too few dates match these filters. Widen them to see the charts.")
    st.stop()

# ---------------- KPI cards ----------------
cards = [("Total Dates", f"{len(f):,}", RED, "white"),
         ("Match Rate", f"{f[MATCH].mean()*100:.1f}%", ORANGE, "white")]
if ATTRS:
    cards.append(("Avg Attractiveness", f"{f[ATTRS[0][1]].mean():.1f} / 10", YELLOW, INK))
if LIKE:
    cards.append(("Avg Liking", f"{f[LIKE].mean():.1f} / 10", GREEN, INK))
for gname, color in [("Male", BLUE), ("Female", INDIGO)]:
    sub = f[f["Gender"] == gname]
    if len(sub):
        cards.append((f"{gname} Match Rate", f"{sub[MATCH].mean()*100:.1f}%", color, "white"))
for col, (l, v, c, t) in zip(st.columns(len(cards)), cards):
    col.markdown(kpi(l, v, c, t), unsafe_allow_html=True)

# ---------------- PAST ----------------
section("📜 Past — the dating floor", "What the data looks like", ORANGE)
a, b, c = st.columns(3)
with a:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    vc = f[MATCH].value_counts().reindex([0, 1]).fillna(0)
    ax.pie(vc, labels=["No match", "Match"], colors=[RED, GREEN], autopct="%1.1f%%",
           wedgeprops=dict(width=0.45, edgecolor="white"), textprops=dict(color=INK))
    ax.set_title("Did they match?"); fig.patch.set_facecolor(PAPER); st.pyplot(fig)
with b:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    gr = f.groupby("Gender")[MATCH].mean() * 100
    if len(gr):
        ax.bar(gr.index, gr.values, color=[BLUE if i == "Male" else VIOLET for i in gr.index], edgecolor=INK)
    ax.set_title("Match rate by gender (%)"); paint(fig, ax); st.pyplot(fig)
with c:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    series = f[AGE].dropna() if AGE else f[LIKE].dropna()
    _, _, patches = ax.hist(series, bins=14, edgecolor="white")
    cmap = plt.get_cmap("rainbow")
    for i, p in enumerate(patches):
        p.set_facecolor(cmap(i / max(len(patches) - 1, 1)))
    ax.set_title("Age of participants" if AGE else "Liking scores"); paint(fig, ax); st.pyplot(fig)

# ---------------- PRESENT ----------------
section("⚙️ Present — what drives a match", "Ratings, liking and the models behind them", GREEN)
d, e, g3 = st.columns(3)
with d:
    gap = pd.Series({LABELS[c]: f.loc[f[MATCH] == 1, c].mean() - f.loc[f[MATCH] == 0, c].mean() for c in FEATS}).dropna()
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.barh(gap.index, gap.values, color=[RAINBOW[i % 7] for i in range(len(gap))], edgecolor=INK)
    ax.set_title("Rating boost on matched dates"); paint(fig, ax); plt.tight_layout(); st.pyplot(fig)
with e:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    if LIKE:
        tmp = f[[LIKE, MATCH]].dropna()
        tmp["b"] = tmp[LIKE].round().clip(0, 10).astype(int)
        rate = tmp.groupby("b")[MATCH].mean() * 100
        ax.bar(rate.index, rate.values, color=[RAINBOW[i % 7] for i in range(len(rate))], edgecolor=INK)
        ax.set_xlabel("Liking score", color=INK)
    ax.set_title("Match rate by liking (%)"); paint(fig, ax); st.pyplot(fig)
with g3:
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=[LABELS[c] for c in FEATS]).sort_values()
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.barh(imp.index, imp.values, color=[RAINBOW[i % 7] for i in range(len(imp))], edgecolor=INK)
    ax.set_title("What the model watches"); paint(fig, ax); plt.tight_layout(); st.pyplot(fig)

m1, m2 = st.columns(2)
for col, (name, (acc, auc)), color in zip([m1, m2], res.items(), [BLUE, INDIGO]):
    col.markdown(kpi(name, f"Acc {acc*100:.1f}% | AUC {auc:.3f}", color), unsafe_allow_html=True)
st.caption("Only a minority of dates end in a match, so ROC-AUC is the fairer score than accuracy here.")

# ---------------- FUTURE ----------------
section("🔮 Future — the chemistry predictor", "Rate a date and see how likely it is to end in a match", VIOLET)
st.markdown('<div class="rbpanel">', unsafe_allow_html=True)
left, right = st.columns([3, 2])
vals = {}
with left:
    s1, s2 = st.columns(2)
    for i, c in enumerate(FEATS):
        default = float(np.clip(round(df[c].median() * 2) / 2, 0, 10)) if df[c].notna().any() else 5.0
        vals[c] = (s1 if i % 2 == 0 else s2).slider(LABELS[c], 0.0, 10.0, default, 0.5)
p = rf.predict_proba(pd.DataFrame([vals])[FEATS])[0][1]
ratio = p / base_rate if base_rate else 1
verdict = "🔥 Strong chemistry — well above the average date" if ratio >= 2 else \
          "🙂 Promising — better than the average date" if ratio >= 1 else "🌧️ Below-average odds for this date"
with right:
    st.markdown(f'<div style="background:linear-gradient(135deg,{GRAD});padding:28px;border-radius:18px;'
                f'text-align:center;color:white;text-shadow:0 2px 6px rgba(0,0,0,.4)">'
                f'<div>Match likelihood</div><div style="font-size:64px;font-weight:800">{p*100:.0f}%</div>'
                f'<div>{verdict}</div></div>', unsafe_allow_html=True)
    st.progress(min(int(p * 100), 100))
    st.caption(f"The average date in this dataset ends in a match {base_rate*100:.1f}% of the time.")
st.markdown('</div>', unsafe_allow_html=True)
st.caption("Note: these ratings are given after the date, so this shows what a match-worthy date looks like, "
           "not a prediction you could make before meeting someone.")