import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sqlite3
import datetime
from fpdf import FPDF
import os

# --- CONFIGURATION GÉNÉRALE ---
st.set_page_config(
    page_title="Système BI-ML — Prédiction du Diabète",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Couleurs principales
COLOR_HIGH = "#E24B4A"    # Rouge
COLOR_MODERATE = "#EF9F27" # Orange
COLOR_LOW = "#1D9E75"     # Vert

# --- INITIALISATION BASE DE DONNÉES ---
def init_db():
    conn = sqlite3.connect('patients_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            date TEXT,
            glucose REAL,
            bmi REAL,
            age REAL,
            blood_pressure REAL,
            risk_score REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- FONCTIONS UTILITAIRES ---
def save_history(name, glucose, bmi, age, bp, risk):
    conn = sqlite3.connect('patients_history.db')
    c = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO history (patient_name, date, glucose, bmi, age, blood_pressure, risk_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (name, now, glucose, bmi, age, bp, risk))
    conn.commit()
    conn.close()

def get_patient_history(name):
    conn = sqlite3.connect('patients_history.db')
    df = pd.read_sql_query(f"SELECT * FROM history WHERE patient_name='{name}' ORDER BY date ASC", conn)
    conn.close()
    return df

def get_all_histories():
    conn = sqlite3.connect('patients_history.db')
    df = pd.read_sql_query("SELECT * FROM history ORDER BY date DESC", conn)
    conn.close()
    return df

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
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Donnees Cliniques Actuelles:', 0, 1)
    pdf.set_font('Arial', '', 12)
    for k, v in data_dict.items():
        if k in ['Glucose', 'IMC', 'Age', 'Pression Arterielle']:
            pdf.cell(0, 8, f'- {k}: {v}', 0, 1)
            
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'Evaluation du Risque: {risk_pct:.1f}% ({risk_level})', 0, 1)
    
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Recommandations Personnalisees:', 0, 1)
    pdf.set_font('Arial', '', 12)
    if not recommendations:
         pdf.cell(0, 8, '- Aucune recommandation particuliere. Maintenez vos bonnes habitudes.', 0, 1)
    for title, desc in recommendations:
        clean_title = title.encode('latin-1', 'replace').decode('latin-1')
        clean_desc = desc.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 8, f"- {clean_title}: {clean_desc}")
        
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 10, 'Signature du Medecin: _______________________', 0, 1)
    
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    filepath = f"outputs/rapport_{patient_name.replace(' ', '_')}.pdf"
    pdf.output(filepath)
    return filepath

def get_recommendations(shap_values, feature_names, data_dict):
    recs = []
    for idx, sv in enumerate(shap_values):
        if sv > 0: # Augmente le risque
            feature = feature_names[idx]
            if feature == 'Glucose' and data_dict.get('Glucose', 0) > 100:
                recs.append(("Glucose eleve", "Reduire les sucres rapides, privilegier les aliments a faible index glycemique et consulter un nutritionniste."))
            elif feature == 'BMI' and data_dict.get('BMI', 0) > 25:
                recs.append(("IMC eleve", "Pratiquer 30 minutes de marche par jour et adopter une alimentation equilibree pour reduire le poids."))
            elif feature == 'Age' and data_dict.get('Age', 0) >= 45:
                recs.append(("Age avance", "Un depistage annuel et un suivi medical regulier sont recommandes."))
            elif feature == 'BloodPressure' and data_dict.get('BloodPressure', 0) > 80:
                recs.append(("Pression sanguine", "Surveiller votre tension, reduire l'apport en sel et pratiquer une activite physique reguliere."))
            elif feature == 'DiabetesPedigreeFunction' and data_dict.get('DiabetesPedigreeFunction', 0) > 0.5:
                recs.append(("Antecedents familiaux", "Surveillance reguliere conseillee en raison de vos antecedents genetiques. Informez votre famille."))
    
    if data_dict.get('Activity', 0) < 3:
        recs.append(("Sedentarite", "Augmentez votre activite physique (ex: 10 000 pas/jour) pour ameliorer la sensibilite a l'insuline."))
        
    return recs

# --- CHARGEMENT DES DONNÉES ET MODÈLES ---
@st.cache_resource
def load_models():
    try:
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles : {e}")
        return None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/diabetes.csv')
        cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)
        df[cols_to_replace] = df[cols_to_replace].fillna(df[cols_to_replace].median())
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None

def predict_risk(model, scaler, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    glucose_bmi = glucose * bmi
    bmi_category = 0 if bmi < 25 else (1 if bmi < 30 else 2)
    glucose_category = 0 if glucose < 100 else (1 if glucose < 126 else 2)
    
    patient_data = np.array([[
        pregnancies, glucose, blood_pressure, skin_thickness, 
        insulin, bmi, dpf, age, glucose_bmi, bmi_category, glucose_category
    ]])
    patient_scaled = scaler.transform(patient_data)
    prob = model.predict_proba(patient_scaled)[0][1]
    return prob, patient_scaled

model, scaler = load_models()
data = load_data()

FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 
    'glucose_bmi', 'bmi_category', 'glucose_category'
]

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #1D9E75; font-size: 3em;'>BI-ML</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Système BI-ML — Prédiction du Diabète</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    role = st.radio("Connexion", ["Espace Patient", "Espace Médecin"])
    
    st.markdown("---")
    st.markdown("<br>" * 10, unsafe_allow_html=True)
    st.caption("Projet BI-ML · 2025–2026 · Système interactif de santé")

# --- ESPACE PATIENT ---
if role == "Espace Patient":
    st.title("👤 Espace Patient - Votre Profil Santé")
    
    patient_name = st.text_input("Nom d'utilisateur (pour l'historique)", "Patient_Demo")
    
    tab_actuel, tab_whatif, tab_hist = st.tabs(["📊 Profil Actuel & Recommandations", "🎮 Simulateur What-If", "📈 Mon Historique"])
    
    if 'pat_preg' not in st.session_state: st.session_state.pat_preg = 3
    if 'pat_gluc' not in st.session_state: st.session_state.pat_gluc = 120
    if 'pat_bp' not in st.session_state: st.session_state.pat_bp = 72
    if 'pat_skin' not in st.session_state: st.session_state.pat_skin = 29
    if 'pat_ins' not in st.session_state: st.session_state.pat_ins = 94
    if 'pat_bmi' not in st.session_state: st.session_state.pat_bmi = 28.0
    if 'pat_dpf' not in st.session_state: st.session_state.pat_dpf = 0.470
    if 'pat_age' not in st.session_state: st.session_state.pat_age = 35
    if 'pat_act' not in st.session_state: st.session_state.pat_act = 2
    
    with tab_actuel:
        st.subheader("Saisissez vos données actuelles")
        with st.form("patient_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.pat_age = st.number_input("Âge", min_value=18, max_value=100, value=st.session_state.pat_age)
                st.session_state.pat_bmi = st.number_input("IMC (kg/m²)", min_value=10.0, max_value=60.0, value=st.session_state.pat_bmi, step=0.1)
                st.session_state.pat_preg = st.number_input("Grossesses", min_value=0, max_value=20, value=st.session_state.pat_preg)
            with c2:
                st.session_state.pat_gluc = st.number_input("Glucose (mg/dL)", min_value=50, max_value=300, value=st.session_state.pat_gluc)
                st.session_state.pat_bp = st.number_input("Pression artérielle (mmHg)", min_value=40, max_value=150, value=st.session_state.pat_bp)
                st.session_state.pat_act = st.selectbox("Activité physique (1=Faible, 5=Forte)", [1, 2, 3, 4, 5], index=st.session_state.pat_act-1)
            with c3:
                st.session_state.pat_ins = st.number_input("Insuline (µU/mL)", min_value=0, max_value=500, value=st.session_state.pat_ins)
                st.session_state.pat_skin = st.number_input("Épaisseur peau (mm)", min_value=0, max_value=100, value=st.session_state.pat_skin)
                st.session_state.pat_dpf = st.number_input("Antécédents familiaux (0-2.5)", min_value=0.0, max_value=2.5, value=st.session_state.pat_dpf, step=0.01)
                
            submit_btn = st.form_submit_button("Analyser mon risque", use_container_width=True)
            
        if submit_btn and model is not None:
            prob, p_scaled = predict_risk(
                model, scaler, st.session_state.pat_preg, st.session_state.pat_gluc,
                st.session_state.pat_bp, st.session_state.pat_skin, st.session_state.pat_ins,
                st.session_state.pat_bmi, st.session_state.pat_dpf, st.session_state.pat_age
            )
            prob_pct = prob * 100
            st.session_state.current_risk_pct = prob_pct
            
            save_history(patient_name, st.session_state.pat_gluc, st.session_state.pat_bmi, st.session_state.pat_age, st.session_state.pat_bp, prob_pct)
            
            risk_color = COLOR_LOW if prob_pct < 30 else (COLOR_MODERATE if prob_pct <= 60 else COLOR_HIGH)
            risk_level = "Faible" if prob_pct < 30 else ("Modéré" if prob_pct <= 60 else "Élevé")
            
            st.markdown("---")
            st.markdown(f"### Votre Risque Actuel : <span style='color:{risk_color}'>{prob_pct:.1f}% ({risk_level})</span>", unsafe_allow_html=True)
            
            st.markdown(
                f"""
                <div style="width: 100%; background-color: #e0e0e0; border-radius: 10px;">
                    <div style="width: {prob_pct}%; background-color: {risk_color}; height: 30px; border-radius: 10px; transition: width 0.5s;"></div>
                </div>
                <br>
                """, unsafe_allow_html=True
            )
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(p_scaled)
            sv = shap_values[1][0] if isinstance(shap_values, list) else (shap_values[0, :, 1] if len(shap_values.shape)==3 else shap_values[0])
            
            data_dict = {
                'Glucose': st.session_state.pat_gluc, 'BMI': st.session_state.pat_bmi,
                'Age': st.session_state.pat_age, 'BloodPressure': st.session_state.pat_bp,
                'DiabetesPedigreeFunction': st.session_state.pat_dpf, 'Activity': st.session_state.pat_act,
                'Insulin': st.session_state.pat_ins
            }
            
            recs = get_recommendations(sv, FEATURE_NAMES, data_dict)
            st.session_state.current_recs = recs
            
            st.markdown("### 💊 Recommandations Personnalisées")
            if not recs:
                st.success("Bravo ! Vos indicateurs sont bons. Maintenez une vie saine.")
            else:
                for r_title, r_desc in recs:
                    st.info(f"**{r_title}** : {r_desc}", icon="💡")
                    
            pdf_path = generate_pdf(
                patient_name, 
                {'Glucose': st.session_state.pat_gluc, 'IMC': st.session_state.pat_bmi, 'Age': st.session_state.pat_age, 'Pression Arterielle': st.session_state.pat_bp},
                prob_pct, risk_level, recs
            )
            
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="🧾 Télécharger mon rapport médical (PDF)",
                    data=f,
                    file_name=f"rapport_medical_{patient_name.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    type="primary"
                )

    with tab_whatif:
        st.subheader("🎮 Simulateur interactif : Quel est l'impact de vos actions ?")
        st.write("Modifiez les curseurs ci-dessous pour voir comment votre risque évolue en temps réel.")
        
        col_w1, col_w2 = st.columns([2, 1])
        with col_w1:
            sim_gluc = st.slider("Glucose simulé", 50, 300, int(st.session_state.pat_gluc))
            sim_bmi = st.slider("IMC simulé", 10.0, 60.0, float(st.session_state.pat_bmi))
            sim_bp = st.slider("Pression artérielle simulée", 40, 150, int(st.session_state.pat_bp))
            sim_act = st.slider("Activité physique simulée", 1, 5, int(st.session_state.pat_act))
            
            eff_gluc = sim_gluc - (sim_act - st.session_state.pat_act) * 2
            eff_bmi = sim_bmi - (sim_act - st.session_state.pat_act) * 0.5
            
        with col_w2:
            if model:
                sim_prob, _ = predict_risk(
                    model, scaler, st.session_state.pat_preg, eff_gluc, sim_bp,
                    st.session_state.pat_skin, st.session_state.pat_ins, eff_bmi,
                    st.session_state.pat_dpf, st.session_state.pat_age
                )
                sim_prob_pct = sim_prob * 100
                sim_color = COLOR_LOW if sim_prob_pct < 30 else (COLOR_MODERATE if sim_prob_pct <= 60 else COLOR_HIGH)
                
                curr_risk = st.session_state.get('current_risk_pct', None)
                
                st.markdown("#### Résultat de la simulation")
                st.markdown(f"<h1 style='color:{sim_color}'>{sim_prob_pct:.1f}%</h1>", unsafe_allow_html=True)
                
                if curr_risk is not None:
                    diff = sim_prob_pct - curr_risk
                    if diff < -0.5:
                        st.success(f"📉 Baisse du risque : {diff:.1f}%")
                        st.markdown(f"**Si vous adoptez ces valeurs, votre risque passe de {curr_risk:.1f}% à {sim_prob_pct:.1f}%.**")
                    elif diff > 0.5:
                        st.error(f"📈 Hausse du risque : +{diff:.1f}%")
                        st.markdown(f"**Attention, ces changements augmenteraient votre risque de {curr_risk:.1f}% à {sim_prob_pct:.1f}%.**")
                    else:
                        st.info("Risque stable.")
                        
    with tab_hist:
        st.subheader("📈 Évolution de mon risque dans le temps")
        df_hist = get_patient_history(patient_name)
        
        if len(df_hist) > 0:
            df_hist['date'] = pd.to_datetime(df_hist['date'])
            fig_line = px.line(df_hist, x='date', y='risk_score', markers=True, 
                               title="Historique du score de risque (%)", 
                               labels={'risk_score': 'Risque (%)', 'date': 'Date'})
            
            if len(df_hist) > 1:
                last = df_hist.iloc[-1]['risk_score']
                prev = df_hist.iloc[-2]['risk_score']
                if last < prev:
                    st.success(f"Tendance : Amélioration (Le risque a baissé de {prev - last:.1f}%)")
                elif last > prev:
                    st.warning(f"Tendance : Aggravation (Le risque a augmenté de {last - prev:.1f}%)")
                else:
                    st.info("Tendance : Stable")
                    
            st.plotly_chart(fig_line, use_container_width=True)
            st.dataframe(df_hist[['date', 'glucose', 'bmi', 'risk_score']].style.format({'risk_score': "{:.1f}%"}))
        else:
            st.info("Aucun historique trouvé. Saisissez vos données pour créer le premier enregistrement.")

# --- ESPACE MÉDECIN ---
elif role == "Espace Médecin":
    st.title("🧑‍⚕️ Espace Médecin - Dashboard & Gestion")
    
    med_tab1, med_tab2, med_tab3 = st.tabs(["📊 Vue Population (BI)", "📋 Gestion des Patients", "🔍 Outil d'Analyse Rapide"])
    
    with med_tab1:
        if data is not None:
            k_col1, k_col2, k_col3, k_col4, k_col5 = st.columns(5)
            total_patients = len(data)
            diabetic_cases = data['Outcome'].sum() if 'Outcome' in data.columns else 268
            diabetic_pct = (diabetic_cases / total_patients) * 100
            
            k_col1.metric("Total patients", total_patients)
            k_col2.metric("Cas diabétiques", f"{diabetic_cases} ({diabetic_pct:.1f}%)")
            k_col3.metric("Âge moyen", f"{data['Age'].mean():.1f}")
            k_col4.metric("Glucose moyen", f"{data['Glucose'].mean():.1f}")
            k_col5.metric("IMC moyen", f"{data['BMI'].mean():.1f}")
            
            st.markdown("---")
            c_col1, c_col2 = st.columns(2)
            
            with c_col1:
                if 'Outcome' in data.columns:
                    df_plot = data.copy()
                    df_plot['Statut'] = df_plot['Outcome'].map({0: 'Non Diabétique (0)', 1: 'Diabétique (1)'})
                    fig_hist = px.histogram(
                        df_plot, x="Glucose", color="Statut", 
                        color_discrete_map={'Non Diabétique (0)': 'blue', 'Diabétique (1)': 'red'},
                        title="Distribution du glucose par statut", barmode="overlay"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
            with c_col2:
                df_cluster = data.drop(columns=['Outcome', 'Statut'], errors='ignore').copy()
                df_cluster = df_cluster.fillna(df_cluster.median())
                scaler_km = StandardScaler()
                df_scaled = scaler_km.fit_transform(df_cluster)
                kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
                clusters = kmeans.fit_predict(df_scaled)
                
                cluster_labels = {0: "Cluster A - Faible risque", 1: "Cluster B - Modéré", 2: "Cluster C - Élevé", 3: "Cluster D - Critique"}
                colors_cluster = {"Cluster A - Faible risque": "green", "Cluster B - Modéré": "orange", "Cluster C - Élevé": "red", "Cluster D - Critique": "purple"}
                cluster_counts = pd.Series(clusters).map(cluster_labels).value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Count']
                fig_pie = px.pie(cluster_counts, values='Count', names='Cluster', color='Cluster', color_discrete_map=colors_cluster, title="Segments de patients (K-Means)")
                st.plotly_chart(fig_pie, use_container_width=True)
                
    with med_tab2:
        st.subheader("Base de données des historiques patients")
        all_hist = get_all_histories()
        if len(all_hist) > 0:
            st.dataframe(all_hist.style.format({'risk_score': "{:.1f}%"}), use_container_width=True)
            
            st.subheader("Analyse comparative d'un patient")
            pat_sel = st.selectbox("Sélectionner un patient", all_hist['patient_name'].unique())
            pat_data = all_hist[all_hist['patient_name'] == pat_sel]
            
            fig_p = px.line(pat_data, x='date', y='risk_score', markers=True, title=f"Évolution de {pat_sel}")
            st.plotly_chart(fig_p)
            
            latest = pat_data.iloc[-1]
            pdf_path = generate_pdf(
                pat_sel, 
                {'Glucose': latest['glucose'], 'IMC': latest['bmi'], 'Age': latest['age'], 'Pression Arterielle': latest['blood_pressure']},
                latest['risk_score'], 
                "Elevé" if latest['risk_score'] > 60 else ("Modéré" if latest['risk_score'] > 30 else "Faible"),
                []
            )
            with open(pdf_path, "rb") as f:
                st.download_button("Télécharger le dernier rapport PDF", data=f, file_name=f"rapport_{pat_sel}.pdf", mime="application/pdf")
        else:
            st.info("La base de patients est vide.")
            
    with med_tab3:
        st.subheader("Simulateur Rapide de Risque (Outil Diagnostic)")
        with st.form("doc_form"):
            cc1, cc2 = st.columns(2)
            with cc1:
                d_gluc = st.slider("Glucose", 50, 300, 120)
                d_bmi = st.slider("IMC", 15.0, 55.0, 25.0)
                d_age = st.slider("Âge", 18, 90, 45)
            with cc2:
                d_bp = st.slider("Pression artérielle", 40, 150, 80)
                d_preg = st.slider("Grossesses", 0, 15, 0)
                d_dpf = st.slider("Antécédents", 0.0, 2.5, 0.3)
            
            doc_sub = st.form_submit_button("Calculer le risque")
            
        if doc_sub and model:
            prob, p_scaled = predict_risk(model, scaler, d_preg, d_gluc, d_bp, 20, 80, d_bmi, d_dpf, d_age)
            st.metric("Risque estimé", f"{prob*100:.1f}%")
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(p_scaled)
            sv = shap_values[1][0] if isinstance(shap_values, list) else (shap_values[0, :, 1] if len(shap_values.shape)==3 else shap_values[0])
            
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['red' if x > 0 else 'green' for x in sv]
            ax.barh(FEATURE_NAMES, sv, color=colors)
            ax.set_title("Explication de la prédiction (SHAP)")
            st.pyplot(fig)
