import requests
import streamlit as st


st.set_page_config(
    page_title="Predikcija rizika studenata",
    page_icon="🎓",
    layout="centered"
)

st.title("Predikcija rizičnih studenata")

st.write(
    "Aplikacija koristi FastAPI servis i prethodno trenirani model "
    "za procenu rizika od neuspešnog završetka online kursa."
)

st.info(
    "Predikcija se vrši na osnovu već određene vremenske granice posmatranja "
    "i prethodno pripremljenog skupa podataka."
)

st.subheader("Unos podataka o studentu")

id_student = st.number_input(
    "ID studenta",
    min_value=0,
    step=1
)

code_module = st.selectbox(
    "Kurs",
    ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
)

code_presentation = st.selectbox(
    "Prezentacija kursa",
    ["2013B", "2013J", "2014B", "2014J"]
)

if st.button("Predvidi rizik"):
    data = {
        "id_student": int(id_student),
        "code_module": code_module,
        "code_presentation": code_presentation
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data
        )

        if response.status_code == 200:
            result = response.json()

            st.subheader("Rezultat predikcije")
            st.write("ID studenta:", result["id_student"])
            st.write("Kurs:", result["code_module"])
            st.write("Prezentacija kursa:", result["code_presentation"])
            st.write("Granica posmatranja:", result["granica"])
            st.write("Verovatnoća rizika:", result["risk_probability"])
            st.write("Korišćeni prag:", result["threshold"])

            if result["prediction"] == 1:
                st.error(result["label"])
            else:
                st.success(result["label"])

            if result["llm_explanation"] is not None:
                st.subheader("LLM objašnjenje i preporuka podrške")
                st.write(result["llm_explanation"])

        else:
            st.error("Student nije pronađen ili je došlo do greške.")
            st.write(response.json()["detail"])

    except requests.exceptions.ConnectionError:
        st.error("FastAPI servis nije pokrenut.")

