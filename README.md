# KULDIO ESG Compliance Platform — README

**Team:** Cyber Walruses

**Project:** AI-Powered Carbon Compliance & ESG Reporting Automation (Hackathon Prototype)

**Short description**
A Streamlit prototype that demonstrates an end-to-end approach for automated ESG data integration, basic emissions forecasting, EU Taxonomy alignment checks and AI-powered recommendations. The app ingests heterogeneous CSV files (utility bills, ERP extracts, meter dumps), normalizes dates/units, derives CO₂ metrics where missing, and produces simple CSRD-style reports and dashboards.

---

## Problem statement (summary)

Nordic companies face increasing regulatory pressure (CSRD, EU Taxonomy) to produce reliable ESG disclosures. Many firms rely on manual spreadsheets and fragmented systems (ERP, utility bills, meters), causing non-compliance risks. This prototype automates data integration, computes derived emissions metrics, checks against simple EU Taxonomy rules, and offers AI-backed recommendations to help close compliance gaps.

---

## Our Solution

KULDIO provides a single, unified platform to automate this entire process. Our dashboard allows companies to ingest, analyze, and act on their environmental data in one place.

### Key Features

* **🤖 Smart Data Integration:** Our core feature. Upload multiple, inconsistent CSV files from any source (ERP, utility bills, etc.). The platform automatically:
    * **Maps Columns:** Intelligently maps column names (e.g., 'energy\_mwh', 'consumption', 'power' all become `energy_consumption_mwh`).
    * **Normalizes Dates:** Understands various date formats (e.g., `Jan-2023`, `2023-01`, `15/01/2023`).
    * **Converts Units:** Automatically normalizes units (e.g., converts `kWh` to `MWh` and `kg` to `tons`).
    * **Calculates Metrics:** Derives new metrics like CO2 emissions from energy data if not provided directly.

* **📊 Interactive Compliance Dashboard:** A clear UI to visualize key metrics, including:
    * **Green Credit Score:** A single, holistic score of your company's ESG health.
    * **EU Taxonomy Alignment:** A percentage-based gauge showing compliance with sector-specific EU thresholds.
    * **Emissions Forecasting:** A predictive model that forecasts future emissions based on historical trends.

* **💡 AI-Powered Recommendations:** The platform uses a local Large Language Model (LLM) to analyze your company's specific data (sector, emission trends, compliance gaps) and provide actionable, qualitative recommendations for improvement.

* **📋 One-Click Reporting:** Instantly generate and download:
    * A simulated **CSRD Compliance Report**.
    * A clean, unified **ESG Data Export (CSV)**.

---

## Files

* `app.py` (or the single-file Streamlit script): main application code (the code you provided)
* `esg_data_template.csv`: downloadable CSV template created inside the app (sample rows)

> Note: In this prototype everything is bundled in one Streamlit file for ease of demonstration.

---

## Dependencies

Create a virtual environment and install the Python packages below.

**Suggested `requirements.txt`**

```
streamlit>=1.20.0
pandas>=1.5.0
numpy>=1.22.0
plotly>=5.0.0
scikit-learn>=1.0.0
joblib>=1.2.0
chardet>=5.0.0
requests>=2.28.0
```

You can create `requirements.txt` using the list above or run `pip freeze` after installing.

---

## Setup & Run (step-by-step)

1. **Clone / copy the project**

   * Place the `app.py` file (your Streamlit script) in a project folder.

2. **Create a virtual environment**

   * Linux / macOS:

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   * Windows (PowerShell):

     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   * Windows (cmd):

     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   If you don't have `requirements.txt`, install packages directly:

   ```bash
   pip install streamlit pandas numpy plotly scikit-learn joblib chardet requests
   ```

4. **CRITICAL: Expose LM Studio / local LLM (for AI recommendations)**

   * The app attempts to call a local LM Studio-like HTTP endpoint at `http://localhost:1234/v1/chat/completions`.
   * If you have a local LM (LM Studio, Llama server, etc.), start it and expose an API compatible with the `messages` chat-completion format.
   * If a local LM is NOT available, the app falls back to built-in rule-based suggestions automatically.

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

6. **Open in browser**

   * After Streamlit starts, open `http://localhost:8501` (or the URL given in your terminal).

