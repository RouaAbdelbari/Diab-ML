"""
app.py — Dashboard Streamlit BI-ML Diabète
==========================================
Utilise EXACTEMENT :
  - models/final_model.pkl        : pipeline complet
  - models/model_metadata.json    : threshold τ* réel
  - models/kmeans_artifacts.pkl   : K-Means validé

Bugs corrigés :
  - SQLite : migration automatique des colonnes manquantes
  - SHAP   : extraction correcte du vecteur 1D depuis shap_values RF/XGB
  - What-If: cast int/float sur les valeurs session_state
  - Historique: colonnes affichées selon ce qui est réellement en base
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, os, datetime, sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px

# ── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Système BI-ML — Prédiction du Diabète",
    layout="wide",
    initial_sidebar_state="expanded"
)
COLOR_HIGH     = "#E24B4A"
COLOR_MODERATE = "#EF9F27"
COLOR_LOW      = "#1D9E75"

FEATURE_NAMES_ORIG = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]
FEATURE_NAMES_FULL = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
    'glucose_bmi', 'bmi_category', 'glucose_category'
]

# ── SQLite avec migration automatique ─────────────────────────────────────────
DB_PATH = 'patients_history.db'

def init_db():
    """Crée la table et ajoute les colonnes manquantes si la DB est ancienne."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Création initiale
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name   TEXT,
            date           TEXT,
            glucose        REAL,
            bmi            REAL,
            age            REAL,
            blood_pressure REAL,
            risk_score     REAL
        )
    ''')
    conn.commit()
    # Migration : ajouter les colonnes si elles n'existent pas encore
    existing = {row[1] for row in c.execute("PRAGMA table_info(history)")}
    migrations = {
        'prediction':    'TEXT DEFAULT ""',
        'threshold_used':'REAL DEFAULT 0.5',
    }
    for col, typedef in migrations.items():
        if col not in existing:
            c.execute(f"ALTER TABLE history ADD COLUMN {col} {typedef}")
    conn.commit()
    conn.close()

init_db()


def save_history(name, glucose, bmi, age, bp, risk_score, prediction, threshold):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO history
            (patient_name, date, glucose, bmi, age, blood_pressure,
             risk_score, prediction, threshold_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (name, now, float(glucose), float(bmi), float(age),
          float(bp), float(risk_score), str(prediction), float(threshold)))
    conn.commit()
    conn.close()


def get_patient_history(name):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM history WHERE patient_name=? ORDER BY date ASC",
        conn, params=(name,)
    )
    conn.close()
    return df


def get_all_histories():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM history ORDER BY date DESC", conn)
    conn.close()
    return df


# ── PDF ───────────────────────────────────────────────────────────────────────
PDF_AVAILABLE = False
try:
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Rapport Medical - Evaluation du Risque de Diabete', 0, 1, 'C')
            self.ln(5)

    def generate_pdf(patient_name, data_dict, risk_pct, risk_level, recommendations):
        pdf = PDF()
        pdf.add_page()
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Patient: {patient_name}', 0, 1)
        pdf.cell(0, 10, f'Date: {datetime.date.today().strftime("%d/%m/%Y")}', 0, 1)
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, 'Donnees Cliniques:', 0, 1)
        pdf.set_font('Arial', '', 12)
        for k, v in data_dict.items():
            pdf.cell(0, 8, f'- {k}: {v}', 0, 1)
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f'Risque: {risk_pct:.1f}% ({risk_level})', 0, 1)
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, 'Recommandations:', 0, 1)
        pdf.set_font('Arial', '', 12)
        if not recommendations:
            pdf.cell(0, 8, '- Bons indicateurs. Maintenez vos habitudes.', 0, 1)
        for title, desc in recommendations:
            clean = f"- {title}: {desc}".encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 8, clean)
        pdf.ln(10)
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, 'Signature du Medecin: _______________________', 0, 1)
        os.makedirs('outputs', exist_ok=True)
        path = f"outputs/rapport_{patient_name.replace(' ', '_')}.pdf"
        pdf.output(path)
        return path

    PDF_AVAILABLE = True
except ImportError:
    pass


# ── Chargement des artefacts ──────────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    pipeline = None
    for path in ['models/final_model.pkl', 'models/best_model.pkl']:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                pipeline = pickle.load(f)
            break

    metadata = {}
    threshold = 0.5
    if os.path.exists('models/model_metadata.json'):
        with open('models/model_metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        threshold = float(metadata.get('threshold', 0.5))

    km_artifacts = None
    if os.path.exists('models/kmeans_artifacts.pkl'):
        with open('models/kmeans_artifacts.pkl', 'rb') as f:
            km_artifacts = pickle.load(f)

    return pipeline, threshold, metadata, km_artifacts


@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/diabetes.csv')
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            df[col] = df[col].replace(0, np.nan)
        return df
    except Exception as e:
        st.error(f"Erreur chargement données : {e}")
        return None


# ── Prédiction ────────────────────────────────────────────────────────────────
def predict_patient(pipeline, threshold, pregnancies, glucose, blood_pressure,
                    skin_thickness, insulin, bmi, dpf, age):
    """Prédit via le pipeline complet (preprocessing intégré)."""
    X = pd.DataFrame([[
        float(pregnancies), float(glucose), float(blood_pressure),
        float(skin_thickness), float(insulin), float(bmi),
        float(dpf), float(age)
    ]], columns=FEATURE_NAMES_ORIG)
    prob = float(pipeline.predict_proba(X)[0][1])
    pred = "Diabétique" if prob >= threshold else "Non Diabétique"
    return prob, pred


# ── SHAP local ────────────────────────────────────────────────────────────────
def compute_local_shap(pipeline, patient_values: list):
    """
    Calcule les valeurs SHAP locales pour un patient.
    Retourne (sv_1d, X_df) où sv_1d est un np.ndarray 1D de longueur 11.
    Gère correctement le format RF (list de 2 arrays) et XGB (array 2D ou 3D).
    """
    import shap as shap_lib

    prep      = pipeline.named_steps['imputer']
    feat_eng  = pipeline.named_steps['feature_eng']
    scaler    = pipeline.named_steps['scaler']
    model     = pipeline.named_steps['model']

    X_raw = pd.DataFrame([patient_values], columns=FEATURE_NAMES_ORIG)
    X_imp = prep.transform(X_raw)
    X_eng = feat_eng.transform(X_imp)
    X_sc  = scaler.transform(X_eng)
    X_df  = pd.DataFrame(X_sc, columns=FEATURE_NAMES_FULL)

    model_type = type(model).__name__
    if model_type in ('RandomForestClassifier', 'XGBClassifier',
                      'GradientBoostingClassifier', 'ExtraTreesClassifier'):
        explainer = shap_lib.TreeExplainer(model)
        sv_raw = explainer.shap_values(X_df)
        # RF: list [array(n,f), array(n,f)] → take class-1, row 0
        if isinstance(sv_raw, list):
            sv_1d = np.array(sv_raw[1][0], dtype=float)
        else:
            sv_arr = np.array(sv_raw)
            if sv_arr.ndim == 3:          # shape (n, f, classes)
                sv_1d = sv_arr[0, :, 1]
            else:                          # shape (n, f)
                sv_1d = sv_arr[0]
    else:
        explainer = shap_lib.LinearExplainer(model, X_df)
        sv_raw = explainer.shap_values(X_df)
        sv_1d = np.array(sv_raw[0] if isinstance(sv_raw, list) else sv_raw[0],
                         dtype=float)

    return sv_1d.flatten(), X_df


def get_recommendations(sv_1d: np.ndarray, data_dict: dict):
    """sv_1d is a plain 1D numpy array of length 11."""
    recs = []
    mapping = {
        'Glucose':                 ("Glucose élevé",
                                    "Réduire les sucres rapides, privilégier les aliments à faible index glycémique."),
        'BMI':                     ("IMC élevé",
                                    "Pratiquer 30 min de marche par jour et adopter une alimentation équilibrée."),
        'Age':                     ("Âge avancé",
                                    "Un dépistage annuel et un suivi médical régulier sont recommandés."),
        'BloodPressure':           ("Pression sanguine",
                                    "Surveiller la tension, réduire le sel, pratiquer une activité physique."),
        'DiabetesPedigreeFunction':("Antécédents familiaux",
                                    "Surveillance régulière conseillée en raison des antécédents génétiques."),
    }
    thresholds = {
        'Glucose': 100, 'BMI': 25, 'Age': 45,
        'BloodPressure': 80, 'DiabetesPedigreeFunction': 0.5,
    }
    for i, feat in enumerate(FEATURE_NAMES_FULL):
        sv_val = float(sv_1d[i])
        if sv_val > 0 and feat in mapping:
            if data_dict.get(feat, 0) > thresholds.get(feat, 0):
                recs.append(mapping[feat])
    return recs


def plot_shap_bar(sv_1d: np.ndarray, title: str = "Explication SHAP"):
    """Barplot SHAP horizontal, sorted by |value|."""
    sv = np.array(sv_1d, dtype=float).flatten()
    indices = np.argsort(np.abs(sv))
    colors = [COLOR_HIGH if sv[i] > 0 else COLOR_LOW for i in indices]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([FEATURE_NAMES_FULL[i] for i in indices],
            sv[indices], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Impact sur le risque (SHAP)")
    plt.tight_layout()
    return fig


# ── Chargement global ─────────────────────────────────────────────────────────
pipeline, threshold, metadata, km_artifacts = load_model_artifacts()
data = load_data()

if pipeline is None:
    st.error("⚠️ Modèle non trouvé. Exécutez d'abord : `python run_all.py`")
    st.stop()

model_display_name = metadata.get('model_name', 'Modèle ML') if metadata else 'Modèle ML'

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='text-align:center;color:#1D9E75;font-size:3em;'>BI-ML</h1>",
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>Prédiction du Diabète</h3>",
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"**Modèle actif :** {model_display_name}")
    st.markdown(f"**Seuil τ :** `{threshold:.4f}`")
    st.markdown("---")
    role = st.radio("Connexion", ["Espace Patient", "Espace Médecin"])
    st.markdown("---")
    st.caption("Projet AICCSA 2026 · τ* chargé depuis model_metadata.json")


# ═════════════════════════════════════════════════════════════════════════════
# ESPACE PATIENT
# ═════════════════════════════════════════════════════════════════════════════
if role == "Espace Patient":
    st.title("👤 Espace Patient — Votre Profil Santé")
    patient_name = st.text_input("Nom d'utilisateur", "Patient_Demo")

    tab_actuel, tab_whatif, tab_hist = st.tabs([
        "📊 Profil Actuel & Recommandations",
        "🎮 Simulateur What-If",
        "📈 Mon Historique"
    ])

    # Session state defaults
    _defaults = {'preg': 3, 'gluc': 120, 'bp': 72, 'skin': 29,
                 'ins': 94, 'bmi': 28.0, 'dpf': 0.470, 'age': 35, 'act': 2}
    for k, v in _defaults.items():
        if f'pat_{k}' not in st.session_state:
            st.session_state[f'pat_{k}'] = v

    # ── Tab 1 : Profil actuel ──────────────────────────────────────────────
    with tab_actuel:
        st.subheader("Saisissez vos données actuelles")
        with st.form("patient_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age  = st.number_input("Âge",            18, 100, int(st.session_state.pat_age))
                bmi  = st.number_input("IMC (kg/m²)",    10.0, 60.0, float(st.session_state.pat_bmi), 0.1)
                preg = st.number_input("Grossesses",     0,   20,  int(st.session_state.pat_preg))
            with c2:
                gluc = st.number_input("Glucose (mg/dL)",50, 300, int(st.session_state.pat_gluc))
                bp   = st.number_input("Pression (mmHg)",40, 150, int(st.session_state.pat_bp))
                act  = st.selectbox("Activité (1=Faible, 5=Forte)", [1,2,3,4,5],
                                    index=int(st.session_state.pat_act) - 1)
            with c3:
                ins  = st.number_input("Insuline (µU/mL)",0, 500, int(st.session_state.pat_ins))
                skin = st.number_input("Épaisseur peau (mm)", 0, 100, int(st.session_state.pat_skin))
                dpf  = st.number_input("Antécédents familiaux", 0.0, 2.5,
                                        float(st.session_state.pat_dpf), 0.01)
            submit = st.form_submit_button("Analyser mon risque", use_container_width=True)

        if submit:
            # Persist to session state
            st.session_state.pat_age  = age
            st.session_state.pat_bmi  = bmi
            st.session_state.pat_preg = preg
            st.session_state.pat_gluc = gluc
            st.session_state.pat_bp   = bp
            st.session_state.pat_act  = act
            st.session_state.pat_ins  = ins
            st.session_state.pat_skin = skin
            st.session_state.pat_dpf  = dpf

            prob, prediction = predict_patient(
                pipeline, threshold, preg, gluc, bp, skin, ins, bmi, dpf, age
            )
            prob_pct = prob * 100
            st.session_state.current_risk_pct = prob_pct

            # BI risk categories (separate from τ* classification threshold)
            if prob_pct < 30:
                risk_level, risk_color = "Faible",  COLOR_LOW
            elif prob_pct <= 60:
                risk_level, risk_color = "Modéré",  COLOR_MODERATE
            else:
                risk_level, risk_color = "Élevé",   COLOR_HIGH

            save_history(patient_name, gluc, bmi, age, bp,
                         prob_pct, prediction, threshold)

            st.markdown("---")
            st.markdown(
                f"### Risque : <span style='color:{risk_color}'>{prob_pct:.1f}% ({risk_level})</span> — "
                f"**Décision (τ={threshold:.2f}) : {prediction}**",
                unsafe_allow_html=True
            )
            st.markdown(
                f"""<div style="width:100%;background-color:#e0e0e0;border-radius:10px;">
                    <div style="width:{min(prob_pct,100):.1f}%;background-color:{risk_color};
                    height:30px;border-radius:10px;"></div></div><br>""",
                unsafe_allow_html=True
            )

            # SHAP local
            recs = []
            try:
                sv_1d, _ = compute_local_shap(
                    pipeline, [preg, gluc, bp, skin, ins, bmi, dpf, age]
                )
                data_dict = {
                    'Glucose': gluc, 'BMI': bmi, 'Age': age,
                    'BloodPressure': bp, 'DiabetesPedigreeFunction': dpf,
                }
                recs = get_recommendations(sv_1d, data_dict)
                st.session_state.current_recs = recs

                st.markdown("### 💊 Recommandations Personnalisées")
                if not recs:
                    st.success("Vos indicateurs sont bons. Maintenez une vie saine.")
                else:
                    for r_title, r_desc in recs:
                        st.info(f"**{r_title}** : {r_desc}", icon="💡")

                fig = plot_shap_bar(sv_1d, "Explication SHAP locale")
                st.pyplot(fig)
                plt.close()

            except Exception as e:
                st.warning(f"SHAP local non disponible : {e}")
                st.markdown("### 💊 Recommandations")
                st.info("Analysez vos données pour obtenir des recommandations personnalisées.")

            if PDF_AVAILABLE:
                try:
                    pdf_path = generate_pdf(
                        patient_name,
                        {'Glucose': gluc, 'IMC': bmi,
                         'Age': age, 'Pression': bp},
                        prob_pct, risk_level,
                        st.session_state.get('current_recs', [])
                    )
                    with open(pdf_path, 'rb') as fh:
                        st.download_button(
                            "🧾 Télécharger le rapport PDF", fh,
                            file_name=f"rapport_{patient_name}.pdf",
                            mime='application/pdf', type="primary"
                        )
                except Exception as e:
                    st.warning(f"PDF non généré : {e}")

    # ── Tab 2 : What-If ────────────────────────────────────────────────────
    with tab_whatif:
        st.subheader("🎮 Simulateur What-If")
        st.caption("Modifiez les curseurs pour voir l'impact en temps réel.")
        col_w1, col_w2 = st.columns([2, 1])

        with col_w1:
            sim_gluc = st.slider("Glucose simulé",        50,   300,
                                  int(st.session_state.pat_gluc))
            sim_bmi  = st.slider("IMC simulé",            10.0, 60.0,
                                  float(st.session_state.pat_bmi))
            sim_bp   = st.slider("Pression simulée",      40,   150,
                                  int(st.session_state.pat_bp))
            sim_act  = st.slider("Activité simulée",      1, 5,
                                  int(st.session_state.pat_act))

        # Adjustment: more activity lowers effective glucose and BMI slightly
        eff_gluc = float(sim_gluc) - (int(sim_act) - int(st.session_state.pat_act)) * 2.0
        eff_bmi  = float(sim_bmi)  - (int(sim_act) - int(st.session_state.pat_act)) * 0.5

        with col_w2:
            try:
                sim_prob, sim_pred = predict_patient(
                    pipeline, threshold,
                    int(st.session_state.pat_preg),
                    eff_gluc, int(sim_bp),
                    int(st.session_state.pat_skin),
                    int(st.session_state.pat_ins),
                    eff_bmi,
                    float(st.session_state.pat_dpf),
                    int(st.session_state.pat_age)
                )
                sim_pct   = sim_prob * 100
                sim_color = (COLOR_LOW if sim_pct < 30
                             else COLOR_MODERATE if sim_pct <= 60
                             else COLOR_HIGH)
                curr_risk = st.session_state.get('current_risk_pct', None)

                st.markdown("#### Résultat simulation")
                st.markdown(f"<h1 style='color:{sim_color}'>{sim_pct:.1f}%</h1>",
                            unsafe_allow_html=True)
                st.markdown(f"**Décision :** {sim_pred}")

                if curr_risk is not None:
                    diff = sim_pct - curr_risk
                    if diff < -0.5:
                        st.success(f"📉 Baisse : {diff:.1f}%")
                    elif diff > 0.5:
                        st.error(f"📈 Hausse : +{diff:.1f}%")
                    else:
                        st.info("Risque stable.")
                else:
                    st.info("Analysez d'abord votre profil actuel (onglet 1).")
            except Exception as e:
                st.warning(f"Erreur simulateur : {e}")

    # ── Tab 3 : Historique ─────────────────────────────────────────────────
    with tab_hist:
        st.subheader("📈 Évolution du risque dans le temps")
        df_hist = get_patient_history(patient_name)

        if len(df_hist) > 0:
            df_hist['date'] = pd.to_datetime(df_hist['date'])

            if len(df_hist) > 1:
                last = float(df_hist.iloc[-1]['risk_score'])
                prev = float(df_hist.iloc[-2]['risk_score'])
                if last < prev:
                    st.success(f"Tendance : Amélioration ({prev - last:.1f}%)")
                elif last > prev:
                    st.warning(f"Tendance : Aggravation (+{last - prev:.1f}%)")
                else:
                    st.info("Tendance : Stable")

            fig_line = px.line(df_hist, x='date', y='risk_score', markers=True,
                               title="Historique du score de risque (%)",
                               labels={'risk_score': 'Risque (%)', 'date': 'Date'})
            st.plotly_chart(fig_line, use_container_width=True)

            # Display only columns that actually exist in this DB
            base_cols = ['date', 'glucose', 'bmi', 'risk_score']
            extra_cols = [c for c in ['prediction', 'threshold_used']
                          if c in df_hist.columns]
            st.dataframe(df_hist[base_cols + extra_cols], use_container_width=True)
        else:
            st.info("Aucun historique. Saisissez vos données pour créer le premier enregistrement.")


# ═════════════════════════════════════════════════════════════════════════════
# ESPACE MÉDECIN
# ═════════════════════════════════════════════════════════════════════════════
elif role == "Espace Médecin":
    st.title("🧑‍⚕️ Espace Médecin — Dashboard & Gestion")

    med_tab1, med_tab2, med_tab3 = st.tabs([
        "📊 Vue Population (BI)",
        "📋 Gestion des Patients",
        "🔍 Outil Diagnostic Rapide"
    ])

    # ── Tab 1 : Vue Population ─────────────────────────────────────────────
    with med_tab1:
        if data is not None:
            k1, k2, k3, k4, k5 = st.columns(5)
            diabetic = int(data['Outcome'].sum())
            k1.metric("Total patients",    len(data))
            k2.metric("Cas diabétiques",   f"{diabetic} ({diabetic/len(data):.1%})")
            k3.metric("Âge moyen",         f"{data['Age'].mean():.1f}")
            k4.metric("Glucose moyen",
                      f"{data['Glucose'].fillna(data['Glucose'].mean()).mean():.1f}")
            k5.metric("IMC moyen",
                      f"{data['BMI'].fillna(data['BMI'].mean()).mean():.1f}")
            st.markdown("---")

            cc1, cc2 = st.columns(2)

            with cc1:
                df_plot = data.copy()
                df_plot['Statut'] = df_plot['Outcome'].map(
                    {0: 'Non Diabétique', 1: 'Diabétique'})
                fig_h = px.histogram(
                    df_plot, x='Glucose', color='Statut',
                    color_discrete_map={'Non Diabétique': 'blue', 'Diabétique': 'red'},
                    title="Distribution du glucose par statut", barmode='overlay'
                )
                st.plotly_chart(fig_h, use_container_width=True)

            with cc2:
                if km_artifacts is not None:
                    try:
                        km_model       = km_artifacts['kmeans']
                        km_imp         = km_artifacts['imputer']
                        km_scl         = km_artifacts['scaler']
                        cluster_labels = km_artifacts['cluster_labels']

                        df_km = data.drop(columns=['Outcome'], errors='ignore').copy()
                        for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
                            df_km[col] = df_km[col].replace(0, np.nan)

                        raw_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                                    'Insulin','BMI','DiabetesPedigreeFunction','Age']
                        X_km = pd.DataFrame(
                            km_imp.transform(df_km[raw_cols]), columns=raw_cols
                        )
                        X_km['glucose_bmi']      = X_km['Glucose'] * X_km['BMI']
                        X_km['bmi_category']     = np.where(X_km['BMI']<25, 0,
                                                    np.where(X_km['BMI']<30, 1, 2))
                        X_km['glucose_category'] = np.where(X_km['Glucose']<100, 0,
                                                    np.where(X_km['Glucose']<126, 1, 2))
                        feat_order = raw_cols + ['glucose_bmi','bmi_category','glucose_category']
                        X_km_sc = km_scl.transform(X_km[feat_order])

                        labels_pred  = km_model.predict(X_km_sc)
                        label_names  = [cluster_labels.get(int(c), f'Cluster {c}')
                                        for c in labels_pred]
                        cnt_df = (pd.Series(label_names).value_counts()
                                    .reset_index().rename(columns={'index':'Cluster',0:'Count'}))
                        cnt_df.columns = ['Cluster', 'Count']
                        fig_pie = px.pie(cnt_df, values='Count', names='Cluster',
                                         title="Segments patients (K-Means validé)")
                        st.plotly_chart(fig_pie, use_container_width=True)
                    except Exception as e:
                        st.warning(f"K-Means : {e}")
                else:
                    st.warning("K-Means non chargé. Exécutez `python run_all.py`.")

            st.markdown("---")
            st.subheader("Prévalence des facteurs de risque")
            df_f = data.copy()
            for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
                df_f[col] = df_f[col].fillna(df_f[col].median())
            risk_factors = {
                "Glucose ≥ 126":              (df_f['Glucose'] >= 126).mean() * 100,
                "Obésité IMC ≥ 30":           (df_f['BMI'] >= 30).mean() * 100,
                "Âge ≥ 45":                   (df_f['Age'] >= 45).mean() * 100,
                "Antécédents familiaux > 0.5":(df_f['DiabetesPedigreeFunction'] > 0.5).mean() * 100,
                "Insuline < 50":              (df_f['Insulin'] < 50).mean() * 100,
            }
            df_risk = (pd.DataFrame(list(risk_factors.items()), columns=['Facteur','%'])
                         .sort_values('%'))
            st.plotly_chart(
                px.bar(df_risk, x='%', y='Facteur', orientation='h',
                       title="Prévalence des facteurs de risque (%)"),
                use_container_width=True
            )

            st.markdown("---")
            tab_ml, tab_shap_c = st.tabs(["Figures ML", "SHAP & Clustering"])

            def render_images(images):
                cols = st.columns(2)
                for i, (path, title) in enumerate(images):
                    with cols[i % 2]:
                        if os.path.exists(path):
                            st.image(path, caption=title, use_container_width=True)
                        else:
                            st.caption(f"⚠️ Non trouvée : {path}")

            safe_name = (model_display_name
                         .replace(' ','_').replace('(','').replace(')',''))
            with tab_ml:
                render_images([
                    ('results/figures/roc_curves_comparison.png',  'Comparaison ROC'),
                    (f'results/figures/cm_{safe_name}.png',         f'Confusion Matrix — {model_display_name}'),
                    (f'results/figures/threshold_curve_{safe_name}.png', 'Courbe seuil OOF'),
                    ('results/figures/pr_curves_comparison.png',   'Courbes PR'),
                ])
            with tab_shap_c:
                render_images([
                    ('outputs/shap_summary_beeswarm.png',           'SHAP Summary'),
                    ('outputs/shap_bar_global.png',                  'SHAP Feature Importance'),
                    ('outputs/clustering_kmeans.png',                'K-Means PCA'),
                    ('results/figures/clustering_profiles.png',     'Profils clusters'),
                ])

            st.markdown("---")
            st.subheader(f"Patients à risque élevé (prob ≥ {threshold:.2f})")
            try:
                df_inf = data.copy()
                for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
                    df_inf[col] = df_inf[col].replace(0, np.nan)
                probs_all = pipeline.predict_proba(df_inf[FEATURE_NAMES_ORIG])[:, 1]
                df_inf['Risque (%)'] = np.round(probs_all * 100, 1)
                df_inf['Décision']   = np.where(probs_all >= threshold,
                                                'Diabétique', 'Non diabétique')
                hr = df_inf[probs_all >= threshold].copy()
                if len(hr) > 0:
                    st.dataframe(
                        hr[['Glucose','BMI','Age','DiabetesPedigreeFunction',
                            'Risque (%)','Décision']]
                          .sort_values('Risque (%)', ascending=False).head(15),
                        use_container_width=True
                    )
                else:
                    st.info("Aucun patient au-dessus du seuil.")
            except Exception as e:
                st.warning(f"Tableau non généré : {e}")

    # ── Tab 2 : Gestion patients ───────────────────────────────────────────
    with med_tab2:
        st.subheader("Base de données patients")
        all_hist = get_all_histories()
        if len(all_hist) > 0:
            st.dataframe(all_hist, use_container_width=True)
            pat_sel  = st.selectbox("Sélectionner un patient",
                                     all_hist['patient_name'].unique())
            pat_data = all_hist[all_hist['patient_name'] == pat_sel]
            st.plotly_chart(
                px.line(pat_data, x='date', y='risk_score', markers=True,
                        title=f"Évolution de {pat_sel}"),
                use_container_width=True
            )
        else:
            st.info("Base de patients vide.")

    # ── Tab 3 : Diagnostic rapide ──────────────────────────────────────────
    with med_tab3:
        st.subheader("Simulateur Rapide de Risque")
        st.caption(f"Seuil τ* = {threshold:.4f} (lu depuis model_metadata.json)")

        with st.form("doc_form"):
            cc1, cc2 = st.columns(2)
            with cc1:
                d_gluc = st.slider("Glucose",            50,   300, 120)
                d_bmi  = st.slider("IMC",                15.0, 55.0, 25.0)
                d_age  = st.slider("Âge",                18,   90,  45)
            with cc2:
                d_bp   = st.slider("Pression artérielle",40,   150, 80)
                d_preg = st.slider("Grossesses",         0,    15,  0)
                d_dpf  = st.slider("Antécédents familiaux", 0.0, 2.5, 0.3)
            doc_sub = st.form_submit_button("Calculer le risque")

        if doc_sub:
            prob, pred = predict_patient(
                pipeline, threshold,
                d_preg, d_gluc, d_bp, 20, 80, d_bmi, d_dpf, d_age
            )
            col_r1, col_r2 = st.columns(2)
            col_r1.metric("Risque estimé",      f"{prob*100:.1f}%")
            col_r2.metric(f"Décision (τ={threshold:.2f})", pred)

            try:
                sv_1d, _ = compute_local_shap(
                    pipeline, [d_preg, d_gluc, d_bp, 20, 80, d_bmi, d_dpf, d_age]
                )
                fig = plot_shap_bar(sv_1d, "Explication SHAP")
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f"SHAP non disponible : {e}")
