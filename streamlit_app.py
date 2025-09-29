import streamlit as st

# Define your pages
hw1 = st.Page("homeworks/hw_1.py", title = "HW 1")
hw2 = st.Page("homeworks/hw_2.py", title = "HW 2")
hw3 = st.Page("homeworks/hw_3.py", title = "HW 3")
hw4 = st.Page("homeworks/hw_4.py", title = "HW 4: iSchool RAG Chatbot")
hw5 = st.Page("homeworks/hw_5.py", title = "HW 5: Intelligent iSchool RAG Chatbot")

# Build the navigation
pg = st.navigation([hw5, hw4, hw3, hw2, hw1])

# Run the selected page
pg.run()