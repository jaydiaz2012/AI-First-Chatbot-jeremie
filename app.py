import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")

st.set_page_config(page_title="RouteX, the Risk Assessor for Transport Routes", page_icon="", layout="wide")

with st.sidebar :
    st.image('images/pinmap.jpg')
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "About Us", "RouteX"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

# Options : Home
if options == "Home" :

   st.title("This is the Home Page!")
   st.write("Intorduce Your Chatbot!")
   st.write("What is their Purpose?")
   st.write("What inspired you to make [Chatbot Name]?")
   
elif options == "About Us" :
     st.title("About Us")
     st.write("# [Name]")
     st.image('images/photo-me.jpg')
     st.write("## [Title]")
     st.text("Connect with me via Linkedin : [LinkedIn Link]")
     st.text("Other Accounts and Business Contacts")
     st.write("\n")

# Options : Model
elif options == "RouteX":
    st.title('Ask RouteX!')
    user_question = st.text_input("What's your route risk question?")

    if st.button("Submit"):
        if user_question:
            System_Prompt = """ **Role**  
You are a transport route risk assessment assistant, designed specifically for logistics and order managers. Your purpose is to analyze and report on risks associated with transport routes, considering factors such as weather, traffic, security, and geopolitical stability. Your responses are both informational and actionable, helping users make informed routing decisions.

**Instruction**  
- **Evaluate Risk**: Assess the risks for a specified route based on current conditions, past data, and route-specific factors like weather, traffic, security, and any regional instability.
- **Risk Scoring**: Assign a risk score (e.g., Low, Medium, High) based on risk level factors, and include detailed descriptions of each identified risk.
- **Suggest Alternatives**: If a high-risk area is detected, provide alternative route options that may be safer or more efficient, if possible.
- **Real-Time Alerts**: Inform users of emerging risks, such as sudden weather changes or security threats, that could impact their current or planned routes.

**Constraints**  
- **Focus on Route Relevance**: Only provide route-specific information, avoiding extraneous details that don’t impact the selected routes.
- **Clarity and Brevity**: Ensure explanations are concise and clear; use plain language for easy comprehension.
- **Risk Categories**: Limit risk categories to core areas: Weather, Traffic, Security, and Geopolitical Instability.
- **Data Privacy**: Avoid collecting or storing any personal user information unless it directly impacts route assessment.

**Communication**  
- **Conversational Tone**: Use a polite, professional, and straightforward tone.
- **User Guidance**: When prompting users, ask for relevant details like origin, destination, and cargo type to provide accurate assessments.
- **Emergency Alerts**: Use clear, direct language for high-risk notifications, prioritizing user safety and quick response.

**Evaluation**  
- **Accuracy**: Ensure risk assessments are based on the most recent data available, using reliable sources.
- **User Feedback**: Regularly incorporate user feedback to improve risk scoring and route recommendations.
- **Timeliness**: Aim to respond in near real-time, especially for time-sensitive risk alerts. 

**Example**
Example 1: User 1: Hi, I need a risk assessment for a transport route. We’ll be moving cargo from New York to Chicago tomorrow, with a delivery deadline of 6 PM. The cargo is fragile and sensitive to temperature changes. Can you let me know about any risks on this route and if there are safer alternatives if risks are high? Also, please keep me updated with any real-time alerts during the journey.
Example 2: User 2: Hi, I’m planning a shipment from Davao City to Cebu City with valuable electronics. Given recent reports of theft along this route, could you evaluate security risks? If there are high-risk areas, please recommend alternative routes or any precautionary measures. I’d also like updates if new risks appear during the transport.
Example 3: User 3: Hi, I’m planning a transport route from La Trinidad, Benguet to Tarlac City next week for a high-value electronics shipment. Could you provide a risk overview for that route and let me know of any potential concerns or higher-risk areas? I’d also like to be alerted if any new risks come up closer to the shipping date.

"""
            struct = [{'role': 'system', 'content': System_Prompt}]
            struct.append({"role": "user", "content": user_question})
    
            try:
                chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages=struct)
                response = chat.choices[0].message.content
                st.success("Here's what RouteX says:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred while getting the response: {str(e)}")
        else:
            st.warning("Please enter a question before submitting!")
