import numpy as np
import pandas as pd
from sklearn import *
import xgboost as xgb
import os

df_dim_date=pd.read_csv(r"C:\Users\SarthakMohapatra\Downloads\Gold Table\dim_date.csv")
df_prm=pd.read_csv(r"C:\Users\SarthakMohapatra\Downloads\Gold Table\fact_provider_metrics.csv")
df_provider=pd.read_csv(r"C:\Users\SarthakMohapatra\Downloads\Gold Table\dim_provider.csv")
df_beneficiary=pd.read_csv(r"C:\Users\SarthakMohapatra\Downloads\Gold Table\dim_beneficiary.csv")
df_outpatient=pd.read_csv(r"C:\Users\SarthakMohapatra\Downloads\Gold Table\fact_outpatient.csv")
df_inpatient=pd.read_csv(r"C:\Users\SarthakMohapatra\Downloads\Gold Table\fact_inpatient.csv")
df_pas=pd.read_csv(r"C:\Users\SarthakMohapatra\Downloads\Gold Table\fact_patient_summary.csv")
# print(df_pas.head())

def replace_na(df):
    return df.replace("N/A",np.nan)
replace_na(df_dim_date)
replace_na(df_prm)
replace_na(df_provider)
replace_na(df_beneficiary)
replace_na(df_outpatient)
replace_na(df_inpatient)
replace_na(df_pas)
# print(df_inpatient.head(20))
df_union=pd.merge(df_pas,df_beneficiary,'left','BeneID')
# print(df_union.head())
diagnosis_cols = [f'ClmDiagnosisCode_{i}' for i in range(1, 11)]
procedure_cols = [f'ClmProcedureCode_{i}' for i in range(1, 7)]
# print(diagnosis_cols)
# print(procedure_cols)

df_inpatient_agg=df_inpatient.groupby("BeneID").agg(
    inpatient_claim_cnt=('ClaimID','count'),
    avg_length_of_stay=('length_of_stay','mean'),
    avg_inpatient_reimbursement=('insurance_amt_reimbursed','mean'),
    unq_inpatient_physicians=('AttendingPhysician',pd.Series.nunique)
).reset_index()

all_diagnosis_codes = df_inpatient.melt(id_vars='BeneID', value_vars=diagnosis_cols, value_name='DiagnosisCode')
unique_diagnosis_counts = all_diagnosis_codes.groupby('BeneID')['DiagnosisCode'].nunique().reset_index()
unique_diagnosis_counts.rename(columns={'DiagnosisCode': 'Unique_inpatient_DiagnosisCodes'}, inplace=True)
all_procedure_codes = df_inpatient.melt(id_vars='BeneID', value_vars=procedure_cols, value_name='ProcedureCode')
unique_procedure_counts = all_procedure_codes.groupby('BeneID')['ProcedureCode'].nunique().reset_index()
unique_procedure_counts.rename(columns={'ProcedureCode': 'Unique_inpatient_ProcedureCodes'}, inplace=True)
inpatient_agg = pd.merge(df_inpatient_agg, unique_diagnosis_counts, on='BeneID', how='left')
inpatient_agg = pd.merge(inpatient_agg, unique_procedure_counts, on='BeneID', how='left')

df_outpatient_agg=df_outpatient.groupby("BeneID").agg(
    outpatient_claim_cnt=("ClaimID","count"),
    avg_outpatient_reimbursement=("insurance_amt_reimbursed","mean"),
    unq_outpatient_physicans=("AttendingPhysician",pd.Series.nunique)

).reset_index()


all_diagnosis_codes = df_outpatient.melt(id_vars='BeneID', value_vars=diagnosis_cols, value_name='DiagnosisCode')
unique_diagnosis_counts = all_diagnosis_codes.groupby('BeneID')['DiagnosisCode'].nunique().reset_index()
unique_diagnosis_counts.rename(columns={'DiagnosisCode': 'Unique_Outpatient_DiagnosisCodes'}, inplace=True)
all_procedure_codes = df_outpatient.melt(id_vars='BeneID', value_vars=procedure_cols, value_name='ProcedureCode')
unique_procedure_counts = all_procedure_codes.groupby('BeneID')['ProcedureCode'].nunique().reset_index()
unique_procedure_counts.rename(columns={'ProcedureCode': 'Unique_Outpatient_ProcedureCodes'}, inplace=True)
outpatient_agg = pd.merge(df_outpatient_agg, unique_diagnosis_counts, on='BeneID', how='left')
# outpatient_agg = pd.merge(outpatient_agg, unique_procedure_counts, on='BeneID', how='left')
# print("done")
all_claims=pd.concat([df_inpatient[['BeneID','Provider']],df_outpatient[['BeneID','Provider']]],ignore_index=True)
claim_with_provider=pd.merge(all_claims,df_prm,'left','Provider')
provider_features_per_patient = claim_with_provider.groupby('BeneID').agg(
    Average_Provider_TotalClaims=('total_claims', 'mean'),
    Average_Provider_TotalReimbursement=('total_reimbursed', 'mean')
).reset_index()
final_dataset = pd.merge(df_union, provider_features_per_patient, on='BeneID', how='left')
final_dataset = pd.merge(final_dataset, inpatient_agg, on="BeneID", how="left")
final_dataset = pd.merge(final_dataset, outpatient_agg, on="BeneID", how="left")
print("Done with creation")
final_dataset.to_csv('final_patient_data.csv', index=False)
print("done saving")