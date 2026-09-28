import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types


GRANICA = 90

model = joblib.load("models/final_model.pkl")
threshold = joblib.load("models/threshold.pkl")
gemini_client = genai.Client()
df = pd.read_csv("data/processed/final.csv")

app = FastAPI(
    title="OULAD API",
    description="API za predikciju rizičnih studenata na osnovu OULAD podataka.",
    version="1.0"
)


class StudentRequest(BaseModel):
    id_student: int
    code_module: str
    code_presentation: str


@app.get("/")
def home():
    return {
        "message": "API za predikciju rizičnih studenata je pokrenut.",
        "granica": GRANICA
    }


@app.get("/features")
def get_features():
    expected_features = model.named_steps["preprocessor"].feature_names_in_.tolist()
    return {"expected_features": expected_features}


def generisi_llm_preporuku(student_row, probability, threshold):
    prompt = f"""
Na osnovu rezultata prediktivnog modela napiši kratko objašnjenje i preporuku podrške za studenta.

Rezultat modela:
- Student je označen kao rizičan.
- Verovatnoća rizika: {round(float(probability), 4)}
- Prag odlučivanja: {float(threshold)}

Podaci o aktivnosti i uspehu studenta:
- Prosečna TMA ocena: {student_row["avg_score_TMA"].iloc[0]}
- Broj predatih TMA zadataka: {student_row["count_submitted_TMA"].iloc[0]}
- Ukupan broj TMA zadataka: {student_row["total_assessments_TMA"].iloc[0]}
- Stopa predaje TMA zadataka: {student_row["submission_rate_TMA"].iloc[0]}
- Prosečan broj dana pre roka za TMA: {student_row["avg_days_before_deadline_TMA"].iloc[0]}
- Prosečna CMA ocena: {student_row["avg_score_CMA"].iloc[0]}
- Broj predatih CMA zadataka: {student_row["count_submitted_CMA"].iloc[0]}
- Ukupan broj CMA zadataka: {student_row["total_assessments_CMA"].iloc[0]}
- Stopa predaje CMA zadataka: {student_row["submission_rate_CMA"].iloc[0]}
- Ukupan broj klikova na VLE platformi: {student_row["total_clicks"].iloc[0]}
- Poslednji dan aktivnosti: {student_row["last_active_day"].iloc[0]}
- Broj različitih aktivnosti: {student_row["unique_activities"].iloc[0]}

Uputstvo:
Napiši odgovor na srpskom jeziku u 4 do 6 rečenica.
Objasni moguće razloge rizika i predloži 2 do 3 mere podrške.
Ne koristi demografske ili osetljive atribute.
Ako je ukupan broj CMA ili TMA zadataka 0, nemoj to tumačiti kao problem studenta.
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text



@app.post("/predict")
def predict(student: StudentRequest):
    student_row = df[
        (df["id_student"] == student.id_student) &
        (df["code_module"] == student.code_module) &
        (df["code_presentation"] == student.code_presentation)
    ]

    if student_row.empty:
        raise HTTPException(
            status_code=404,
            detail="Student sa unetim podacima nije pronađen u pripremljenom skupu podataka."
        )

    expected_features = model.named_steps["preprocessor"].feature_names_in_.tolist()
    input_data = student_row.iloc[[0]][expected_features]
    probability = model.predict_proba(input_data)[:, 1][0]
    prediction = int(probability >= threshold)
    if prediction == 1:
        label = "Rizičan student"
    else:
        label = "Uspešan student"

    llm_explanation = None

    if prediction == 1:
        llm_explanation = generisi_llm_preporuku(
            student_row,
            probability,
            threshold
        )

    return {
        "id_student": student.id_student,
        "code_module": student.code_module,
        "code_presentation": student.code_presentation,
        "granica": GRANICA,
        "prediction": prediction,
        "label": label,
        "risk_probability": round(float(probability), 4),
        "threshold": float(threshold),
        "llm_explanation": llm_explanation
    }

