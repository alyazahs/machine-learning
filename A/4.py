import streamlit as st

yes_checked = st.checkbox("yes")

if st.button("Click"):
    st.write("Button clicked!")

st.write("Pick your gender")
gender_radio = st.radio("Pick your gender", options=["Male", "Female"], key="radio_gender")

st.write("Pick your gender")
gender_dropdown = st.selectbox("Pick your gender", options=["Male", "Female"], key="dropdown_gender")

planet = st.selectbox("Choose a planet", options=["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"])

st.write("Pick a mark")
mark = st.select_slider(
    "Pick a mark:",
    options=["Bad", "Good", "Excellent"],
    value="Good"  
)

number = st.slider("Pick a number", min_value=0, max_value=50, value=9, step=1)