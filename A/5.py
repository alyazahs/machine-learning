import streamlit as st

number = st.number_input("Pick a number", min_value=0, max_value=100, value=1)

email = st.text_input("Email address")

travel_date = st.date_input("Travelling date")

school_time = st.time_input("School time", value=None)

description = st.text_area("Description")

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "png", "jpeg"], help="Limit 200MB per file")

color = st.color_picker("Choose your favourite color", value="#800080")