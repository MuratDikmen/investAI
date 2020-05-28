import pickle
import numpy as numpy
import uvicorn
import pandas as pd
from fastapi import FastAPI
from catboost import Pool, CatBoostClassifier
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "https://p2plending.netlify.com/",
    "https://p2plending.netlify.com/loans/*",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = pickle.load(open('model.pickle', 'rb'))
shap_values = pickle.load(open('shap_values.pickle', 'rb'))


class Data(BaseModel):
    loan_id: float
    loan_amnt: float
    mths_since_recent_inq: float
    term: str
    inq_last_12m: float
    inq_last_6mths: float
    revol_util: float
    bc_open_to_buy: float
    bc_util: float
    num_rev_tl_bal_gt_0: float
    revol_bal: float
    delinq_2yrs: float
    num_op_rev_tl: float
    fico_range_low: float
    sec_app_fico_range_low: float
    mo_sin_rcnt_rev_tl_op: float
    percent_bc_gt_75: float
    num_rev_accts: float
    mths_since_last_delinq: float


@app.post("/predict")
def predict(data: Data):
    arr = []
    for key, value in data:
        if key != = loan_id:
        arr.append(value)
        # arr.append(value)
    # ja = [[35000.0, 4.0, "60 months", 3.0, 2.0, 65.1, 19167.0, 74.7, 3.0,
    #        56633.0, 0.0, 10.0, 705.0, 540.0, 4.0, 42.9, 15.0, 226.0, 23.4]]
    ja = [arr]
    df = pd.DataFrame(ja, columns=['loan_amnt',
                                   'mths_since_recent_inq',
                                   'term',
                                   "inq_last_12m",
                                   "inq_last_6mths",
                                   "revol_util",
                                   "bc_open_to_buy",
                                   "bc_util",
                                   "num_rev_tl_bal_gt_0",
                                   "revol_bal",
                                   "delinq_2yrs",
                                   "num_op_rev_tl",
                                   "fico_range_low",
                                   "sec_app_fico_range_low",
                                   "mo_sin_rcnt_rev_tl_op",
                                   "percent_bc_gt_75",
                                   "num_rev_accts",
                                   "mths_since_last_delinq"])
    prediction = model.predict_proba(df)[0][1]
    shap_vals = shap_values[loan_id]
    data = {"prediction": prediction, "shap_vals": shap_vals}
    return data
    # return prediction
