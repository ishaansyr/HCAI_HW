import streamlit as st

# Define your pages
hw1 = st.Page("hw_1.py", title="HW 1")
hw2 = st.Page("hw_2.py", title="HW 2")

# Build the navigation
pg = st.navigation([hw2, hw1])

# Run the selected page
pg.run()