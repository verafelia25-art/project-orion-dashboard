# Project ORION Management Dashboard

A Streamlit management dashboard that accepts the original Project ORION tenant workbook and produces management-relevant commercial, financial, operational, risk and data-quality insights.

## Input

Upload:

`Project_ORION_Tenant_Dataset_5000_Rows (1).xlsx`

Required sheet:

`Tenant Dataset`

Required columns:

`tenant_id, region, commercial_status, planned_activation, actual_activation, monthly_arpu, bandwidth_usage, installation_cost, sla_percentage, invoice_amount, payment_days, churn_probability`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

This repository is ready for Streamlit Community Cloud. Use `app.py` as the main file.

## Dashboard sections

- Executive
- Commercial
- Financial
- Operations
- Data Quality
- Risk / Monte Carlo

## Reproducibility

The dashboard:
- profiles the source data before cleaning,
- preserves missing values instead of inventing replacements,
- deduplicates deterministically,
- derives median ARPU and churn from cleaned tenant data,
- uses a fixed Monte Carlo seed (`20260909`),
- runs 10,000 simulations.

## Important modelling caveat

The case states Sales OPEX of Rp12.3B over 5 years while also stating a flat Sales bandwidth cost of Rp320M/month. Rp320M × 60 months = Rp19.2B, so the Sales OPEX cannot be reconciled from the supplied information.

This dashboard uses a conservative reviewer assumption that Rp12.3B is other/non-network OPEX and separately applies the actual network-cost formula.
