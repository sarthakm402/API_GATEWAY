import numpy as np
import pandas as pd
from sklearn import *
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler,RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
import optuna
import joblib
# Load dataset
final_dataset=pd.read_csv(r"C:\Users\SarthakMohapatra\Desktop\sarthak_dev_code\Veersa Training\final_patient_data.csv")
converted_dataset=pd.read_csv(r"Veersa Training/final_patient_data_converted.csv")

y=converted_dataset["total_reimbursed"]
x=converted_dataset.drop(columns=['total_reimbursed'])


# Features
numeric_features = ['inpatient_claims','outpatient_claims','total_ip_reimbursed',
 'total_op_reimbursed','total_claims','Part_A_COV_months','Part_B_COV_months',
 'IPAnnualReimbursementAmt','IPAnnualDeductibleAmt','OPAnnualReimbursementAmt',
 'OPAnnualDeductibleAmt','Average_Provider_TotalClaims',
 'Average_Provider_TotalReimbursement','inpatient_claim_cnt','avg_length_of_stay',
 'avg_inpatient_reimbursement','unq_inpatient_physicians','outpatient_claim_cnt',
 'avg_outpatient_reimbursement','unq_outpatient_physicans']

categorical_features = ['Gender','Race','State','County']
flag_features = [c for c in final_dataset.columns if c.startswith('ChronicCond_')] + ['RenalDiseaseIndicator']

#  creating  leakage-free dataset
leakage_features = [
    'total_ip_reimbursed','total_op_reimbursed',
    'IPAnnualReimbursementAmt','OPAnnualReimbursementAmt',
    'Average_Provider_TotalReimbursement',
    'avg_inpatient_reimbursement','avg_outpatient_reimbursement'
]

x_leakfree = x.drop(columns=leakage_features, errors='ignore')
numeric_features_lf = [c for c in numeric_features if c not in leakage_features]
# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", RobustScaler(), numeric_features_lf),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("flag", "passthrough", flag_features),
    ]
)
# Optuna Tuning
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
        "n_jobs": -1
    }

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(**params))
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, x_leakfree, y, cv=kf, scoring="neg_mean_absolute_error")
    return -scores.mean()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("Best Optuna Params:", study.best_params)
# Retrain best model
best_params = study.best_params

xtrain_lf, xtest_lf, ytrain_lf, ytest_lf = train_test_split(x_leakfree, y, test_size=0.2, random_state=42)

best_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(**best_params))
])

best_model.fit(xtrain_lf, ytrain_lf)
joblib.dump(best_model,"Veersa_model.pkl")
best_model=joblib.load("Veersa_model.pkl")
y_pred = best_model.predict(xtest_lf)
y_train_pred = best_model.predict(xtrain_lf)

mae = mean_absolute_error(ytest_lf, y_pred)
r2 = r2_score(ytest_lf, y_pred)
rmse = np.sqrt(mean_squared_error(ytest_lf, y_pred))

train_mae = mean_absolute_error(ytrain_lf, y_train_pred)
train_r2 = r2_score(ytrain_lf, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(ytrain_lf, y_train_pred))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(best_model, x_leakfree, y, cv=kf, scoring="r2").mean()
cv_mae = -cross_val_score(best_model, x_leakfree, y, cv=kf, scoring="neg_mean_absolute_error").mean()
cv_rmse = np.mean(np.sqrt(-cross_val_score(best_model, x_leakfree, y, cv=kf, scoring="neg_mean_squared_error")))

# Adjusted R²
def adjusted_r2(r2, n, k):
    return 1 - (1-r2) * (n-1)/(n-k-1)

n, k = xtrain_lf.shape
adj_r2 = adjusted_r2(r2, n, k)
adj_train_r2 = adjusted_r2(train_r2, n, k)

print("\nFinal Tuned Leakage-Free Model Results:")
print(" MAE:", f'{mae:,.2f}')
print(" R²:", r2)
print(" Adjusted R²:", f'{adj_r2:,.2f}')
print(" RMSE:", f'{rmse:,.2f}')
print(" Train MAE:", f'{train_mae:,.2f}')
print(" Train R²:", f'{train_r2:,.2f}')
print(" Adjusted Train R²:", f'{adj_train_r2:,.2f}')
print(" Train RMSE:", f'{train_rmse:,.2f}')
print(" CV MAE:", f'{cv_mae:,.2f}')
print(" CV R²:", f'{cv_r2:,.2f}')
print(" CV RMSE:", f'{cv_rmse:,.2f}')

# # residual plots
# residuals = ytest_lf - y_pred
# residuals_log = ytest_lf - y_pred_log

# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# sns.scatterplot(x=ytest_lf, y=residuals)
# plt.axhline(0, color="red", linestyle="--")
# plt.title("Residuals (Leakage-Free Normal Model)")
# plt.xlabel("Actual")
# plt.ylabel("Residuals")

# plt.subplot(1,2,2)
# sns.scatterplot(x=ytest_lf, y=residuals_log)
# plt.axhline(0, color="red", linestyle="--")
# plt.title("Residuals (Leakage-Free Log Model)")
# plt.xlabel("Actual")
# plt.ylabel("Residuals")

# plt.tight_layout()
# plt.show()




























# import numpy as np
# import pandas as pd
# from sklearn import *
# import xgboost as xgb
# from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
# from xgboost import XGBRegressor
# import matplotlib.pyplot as plt
# import seaborn as sns

# final_dataset=pd.read_csv(r"C:\Users\SarthakMohapatra\Desktop\sarthak_dev_code\misc\final_patient_data.csv")
# converted_dataset=pd.read_csv(r"C:\Users\SarthakMohapatra\Desktop\sarthak_dev_code\final_patient_data_converted.csv")

# # print(dataset.head())
# # cat_cols=dataset.select_dtypes(include=["object","category"]).columns.to_list()
# # num_cols=dataset.select_dtypes(include=["float64","int64"]).columns.to_list()
# # print(cat_cols)
# # print(num_cols)
# # date_cols = ['DOB','DOD','created_at_x','created_at_y']
# # numeric_cols = ['inpatient_claims','outpatient_claims','total_ip_reimbursed','total_op_reimbursed','total_claims','total_reimbursed','Part_A_COV_months','Part_B_COV_months','IPAnnualReimbursementAmt','IPAnnualDeductibleAmt','OPAnnualReimbursementAmt','OPAnnualDeductibleAmt','Average_Provider_TotalClaims','Average_Provider_TotalReimbursement','inpatient_claim_cnt','avg_length_of_stay','avg_inpatient_reimbursement','unq_inpatient_physicians','outpatient_claim_cnt','avg_outpatient_reimbursement','unq_outpatient_physicans']
# # categorical_cols = ['BeneID','Gender','Race','State','County','Unique_inpatient_DiagnosisCodes','Unique_inpatient_ProcedureCodes','Unique_Outpatient_DiagnosisCodes','gender_map','record_status_x','record_status_y']
# # flag_cols = [c for c in final_dataset.columns if c.startswith('ChronicCond_')] + ['RenalDiseaseIndicator']
# # diag_list_cols = ['Unique_inpatient_DiagnosisCodes','Unique_Outpatient_DiagnosisCodes','Unique_inpatient_ProcedureCodes']

# # for c in date_cols:
# #     if c in final_dataset.columns:
# #         final_dataset[c] = pd.to_datetime(final_dataset[c], errors='coerce')
# # for c in numeric_cols:
# #     if c in final_dataset.columns:
# #         final_dataset[c] = pd.to_numeric(final_dataset[c], errors='coerce')
# # for c in categorical_cols:
# #     if c in final_dataset.columns:
# #         final_dataset[c] = final_dataset[c].astype('string').replace('nan', pd.NA).astype('category')
# # for c in flag_cols:
# #     if c in final_dataset.columns:
# #         if pd.api.types.is_numeric_dtype(final_dataset[c]):
# #             final_dataset[c] = final_dataset[c].fillna(0).astype(int)
# #         else:
# #             s = final_dataset[c].astype(str).str.strip().str.upper().replace({'Y':'1','N':'0','YES':'1','NO':'0','TRUE':'1','FALSE':'0','NONE':pd.NA})
# #             final_dataset[c] = pd.to_numeric(s, errors='coerce').fillna(0).astype(int)
# # for c in diag_list_cols:
# #     if c in final_dataset.columns:
# #         s = final_dataset[c].astype('string').fillna('')
# #         final_dataset[c + '_count'] = s.apply(lambda t: 0 if t.strip()=='' else (len(t.split('|')) if '|' in t else (len(t.split(',')) if ',' in t else 1)))

# # final_dataset.to_csv('final_patient_data_converted.csv', index=False)
# y=converted_dataset["total_reimbursed"]
# x=converted_dataset.drop(columns=['total_reimbursed'])
# xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)

# numeric_features = ['inpatient_claims','outpatient_claims','total_ip_reimbursed',
#  'total_op_reimbursed','total_claims','Part_A_COV_months','Part_B_COV_months',
#  'IPAnnualReimbursementAmt','IPAnnualDeductibleAmt','OPAnnualReimbursementAmt',
#  'OPAnnualDeductibleAmt','Average_Provider_TotalClaims',
#  'Average_Provider_TotalReimbursement','inpatient_claim_cnt','avg_length_of_stay',
#  'avg_inpatient_reimbursement','unq_inpatient_physicians','outpatient_claim_cnt',
#  'avg_outpatient_reimbursement','unq_outpatient_physicans']

# categorical_features = ['Gender','Race','State','County']
# flag_features = [c for c in final_dataset.columns if c.startswith('ChronicCond_')] + ['RenalDiseaseIndicator']

# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(), numeric_features),
#         ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
#         ("flag", "passthrough", flag_features),
#     ]
# )

# xgb_model = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("model", XGBRegressor(
#         n_estimators=500,
#         learning_rate=0.05,
#         max_depth=6,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         reg_alpha=0.5,
#         reg_lambda=1.0,
#         random_state=42,
#         n_jobs=-1
#     ))
# ])

# xgb_model.fit(xtrain, ytrain)
# y_pred = xgb_model.predict(xtest)
# y_train_pred = xgb_model.predict(xtrain)

# mae = mean_absolute_error(ytest, y_pred)
# r2 = r2_score(ytest, y_pred)
# rmse=np.sqrt(mean_squared_error(ytest,y_pred))

# train_mae = mean_absolute_error(ytrain, y_train_pred)
# train_r2 = r2_score(ytrain, y_train_pred)
# train_rmse = np.sqrt(mean_squared_error(ytrain, y_train_pred))

# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# cv_r2 = cross_val_score(xgb_model, x, y, cv=kf, scoring="r2").mean()
# cv_mae = -cross_val_score(xgb_model, x, y, cv=kf, scoring="neg_mean_absolute_error").mean()
# cv_rmse = np.mean(np.sqrt(-cross_val_score(xgb_model, x, y, cv=kf, scoring="neg_mean_squared_error")))

# y_log = np.log1p(y)
# xtrain_log, xtest_log, ytrain_log, ytest_log = train_test_split(x, y_log, test_size=0.2, random_state=42)

# xgb_model_log = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("model", XGBRegressor(
#         n_estimators=500,
#         learning_rate=0.05,
#         max_depth=6,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         reg_alpha=0.5,
#         reg_lambda=1.0,
#         random_state=42,
#         n_jobs=-1
#     ))
# ])

# xgb_model_log.fit(xtrain_log, ytrain_log)
# y_pred_log = np.expm1(xgb_model_log.predict(xtest_log))
# y_train_pred_log = np.expm1(xgb_model_log.predict(xtrain_log))

# mae_log = mean_absolute_error(ytest, y_pred_log)
# r2_log = r2_score(ytest, y_pred_log)
# rmse_log = np.sqrt(mean_squared_error(ytest, y_pred_log))

# train_mae_log = mean_absolute_error(ytrain, y_train_pred_log)
# train_r2_log = r2_score(ytrain, y_train_pred_log)
# train_rmse_log = np.sqrt(mean_squared_error(ytrain, y_train_pred_log))

# y_pred_log_cv = cross_val_predict(xgb_model_log, x, y_log, cv=kf)
# y_pred_log_cv_orig = np.expm1(y_pred_log_cv)
# y_log_orig = np.expm1(y_log)

# cv_mae_log = mean_absolute_error(y_log_orig, y_pred_log_cv_orig)
# cv_r2_log = r2_score(y_log_orig, y_pred_log_cv_orig)
# cv_rmse_log = np.sqrt(mean_squared_error(y_log_orig, y_pred_log_cv_orig))

# def adjusted_r2(r2, n, k):
#     return 1 - (1-r2) * (n-1)/(n-k-1)

# n, k = xtrain.shape
# adj_r2 = adjusted_r2(r2, n, k)
# adj_train_r2 = adjusted_r2(train_r2, n, k)
# adj_r2_log = adjusted_r2(r2_log, n, k)
# adj_train_r2_log = adjusted_r2(train_r2_log, n, k)

# print(" MAE:", mae)
# print(" R²:", r2)
# print(" Adjusted R²:", adj_r2)
# print(" RMSE:",rmse)
# print(" Train MAE:", train_mae)
# print(" Train R²:", train_r2)
# print(" Adjusted Train R²:", adj_train_r2)
# print(" Train RMSE:", train_rmse)
# print(" CV MAE:", cv_mae)
# print(" CV R²:", cv_r2)
# print(" CV RMSE:", cv_rmse)

# print(" Log- MAE:", mae_log)
# print(" Log- R²:", r2_log)
# print(" Log- Adjusted R²:", adj_r2_log)
# print(" Log- RMSE:", rmse_log)
# print(" Log-Train MAE:", train_mae_log)
# print(" Log-Train R²:", train_r2_log)
# print(" Log- Adjusted Train R²:", adj_train_r2_log)
# print(" Log-Train RMSE:", train_rmse_log)
# print(" Log-CV MAE:", cv_mae_log)
# print(" Log-CV R²:", cv_r2_log)
# print(" Log-CV RMSE:", cv_rmse_log)

# # # residual plots
# # residuals = ytest - y_pred
# # residuals_log = ytest - y_pred_log

# # plt.figure(figsize=(12,5))
# # plt.subplot(1,2,1)
# # sns.scatterplot(x=ytest, y=residuals)
# # plt.axhline(0, color="red", linestyle="--")
# # plt.title("Residuals (Normal Model)")
# # plt.xlabel("Actual")
# # plt.ylabel("Residuals")

# # plt.subplot(1,2,2)
# # sns.scatterplot(x=ytest, y=residuals_log)
# # plt.axhline(0, color="red", linestyle="--")
# # plt.title("Residuals (Log Model)")
# # plt.xlabel("Actual")
# # plt.ylabel("Residuals")

# # plt.tight_layout()
# # plt.show()
