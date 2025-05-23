import streamlit as st
from keras.models import load_model
import numpy as np
from preprocessing.preprocessing import preprocess_text

st.title("Human vs AI Text Generator Classification")

model = load_model('model/model.keras')

st.write("Input text to classify:")
input_text = st.text_area("Text Input", height=200)

if st.button("Classify!"):
    text = preprocess_text(input_text)
    result = model.predict(text)
    res_class = (result > 0.5).astype("int32")
    if res_class[0] == 0:
        st.success("The text is generated by a human.")
    else:
        st.success("The text is generated by an AI.")
    st.write("Model Prediction Probability:", result[0][0])
    print(result)