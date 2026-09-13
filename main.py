from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("xgb_pipeline.pkl")

class ChurnInput(BaseModel):
    Gender: str
    Age: int
    Married: str
    Number_of_Dependents: int
    Number_of_Referrals: int
    Tenure_in_Months: int
    Offer: str
    Phone_Service: str
    Avg_Monthly_GB_Download: int
    Multiple_Lines: str
    Internet_Service: str
    Internet_Type: str
    Avg_Monthly_Long_Distance_Charges: float
    Online_Security: str
    Online_Backup: str
    Device_Protection_Plan: str
    Premium_Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str
    Streaming_Music: str
    Unlimited_Data: str
    Contract: str
    Paperless_Billing: str
    Payment_Method: str
    Monthly_Charge: float
    Total_Refunds: float
    Total_Extra_Data_Charges: float

@app.post("/predict")
def predict_churn(data: ChurnInput):
    d = data.dict()
    
    tenure = max(1, d['Tenure_in_Months'])
    monthly = d['Monthly_Charge']
    calc_total = monthly * tenure
    total_ld = d['Avg_Monthly_Long_Distance_Charges'] * tenure

    full_data = {
        'Gender': d['Gender'],
        'Age': d['Age'],
        'Married': d['Married'],
        'Number of Dependents': d['Number_of_Dependents'],
        'Number of Referrals': d['Number_of_Referrals'],
        'Tenure in Months': tenure,
        'Offer': d['Offer'],
        'Phone Service': d['Phone_Service'],
        'Avg Monthly GB Download': d['Avg_Monthly_GB_Download'],
        'Multiple Lines': d['Multiple_Lines'],
        'Internet Service': d['Internet_Service'],
        'Internet Type': d['Internet_Type'],
        'Avg Monthly Long Distance Charges': d['Avg_Monthly_Long_Distance_Charges'],
        'Online Security': d['Online_Security'],
        'Online Backup': d['Online_Backup'],
        'Device Protection Plan': d['Device_Protection_Plan'],
        'Premium Tech Support': d['Premium_Tech_Support'],
        'Streaming TV': d['Streaming_TV'],
        'Streaming Movies': d['Streaming_Movies'],
        'Streaming Music': d['Streaming_Music'],
        'Unlimited Data': d['Unlimited_Data'],
        'Contract': d['Contract'],
        'Paperless Billing': d['Paperless_Billing'],
        'Payment Method': d['Payment_Method'],
        'Monthly Charge': monthly,
        'Total Charges': calc_total,
        'Total Refunds': d['Total_Refunds'],
        'Total Extra Data Charges': d['Total_Extra_Data_Charges'],
        'Total Long Distance Charges': total_ld,
        'Total Revenue': calc_total + total_ld - d['Total_Refunds'] + d['Total_Extra_Data_Charges']
    }
    
    input_df = pd.DataFrame([full_data])
    
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    
    churn_result = "Yes" if str(prediction) in ["1", "Yes"] else "No"
    churn_prob = float(probabilities[1] if len(probabilities) > 1 else probabilities[0])
    
    return {
        "churn": churn_result,
        "churn_probability": f"{churn_prob:.2%}"
    }