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

st.set_page_config(page_title="RouteX, the Risk Assessor for Transport Routes", page_icon="📍", layout="wide")

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

   st.title("Ask RouteX!")
   st.write("RouteX evaluate potential risks based on factors like weather, road conditions, traffic, political stability, and security concerns on various routes when making deliveries. After assessing risks, RouteX could recommend alternative, safer routes if high-risk areas are detected!")
   st.write("In logistics, you don't control the weather, certain conditions, and overall circumstances that will hinder your deliveries to the point of affecting damage to the deliveries to unsuccessful deliveries. RouteX eliminates those problems by telling you the risks.")
   st.write("\n")
   
elif options == "About Us" :
     st.title("About Us")
     st.write("# Jeremie Diaz, AI Bot Developer")
     st.image('images/photo-me1.jpg', use_container_width=True)
     st.write("## Let's connect!")
     st.text("Connect with me via Linkedin : https://www.linkedin.com/in/jandiaz/")
     st.text("Conncet with me via Kaggle: https://www.kaggle.com/jeremiediaz1/")
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
- **Data Limitations**: Only focus your response to what is provided by the user. 
- **Focus on Route Relevance**: Only provide route-specific information, avoiding extraneous details that don’t impact the selected routes.
- **Clarity and Brevity**: Ensure explanations are concise and clear; use plain language for easy comprehension.
- **Risk Categories**: Limit risk categories to core areas: Weather Hazards, Traffic, Security Risks, and Political Instability.
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
Example 2: User 2: Hi, I’m planning a shipment from Caguas to Los Angeles City with valuable electronics. Given recent reports of theft along this route, could you evaluate security risks? If there are high-risk areas, please recommend alternative routes or any precautionary measures. I’d also like updates if new risks appear during the transport.
Example 3: User 3: Hi, I’m planning a transport route from West New York to Sousa next week for a high-value electronics shipment. Could you provide a risk overview for that route and let me know of any potential concerns or higher-risk areas? I’d also like to be alerted if any new risks come up closer to the shipping date.

"""
#            struct = [{'role': 'system', 'content': System_Prompt}]
#            struct.append({"role": "user", "content": user_question})
#    
#            try:
#                chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages=struct)
#                response = chat.choices[0].message.content
#                st.success("Here's what RouteX says:")
#                st.write(response)
#            except Exception as e:
#                st.error(f"An error occurred while getting the response: {str(e)}")
#        else:
#            st.warning("Please enter a question before submitting!")


if openai.api_key and (openai.api_key.startswith('sk-') and len(openai.api_key) == 164):
    st.title("RouteX: Risk Assessor Assistant")
    
    origin = st.text_input("Enter the source location of delivery (city):")
    destination = st.text_input("Enter the destination location (city):")
    item_type = st.selectbox("Select the risk:", ["weather hazards", "security risks", "traffic", "political instability"])
    delivery_date = st.selectbox("Select days of delivery:", ["1 day", "3 days", "7 days", "other"])
    
    if st.button("Get Risk Assessment"):
        # Load the dataset and create embeddings only when the button is pressed
        dataframed = pd.read_csv('https://raw.githubusercontent.com/jaydiaz2012/AI-First-Chatbot-jeremie/refs/heads/main/delivery_logistics_data.csv')
        dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        documents = dataframed['combined'].tolist()

        # Generate embeddings for the documents
        embeddings = [get_embedding(doc, engine="text-embedding-3-small") for doc in documents]
        embedding_dim = len(embeddings[0])
        embeddings_np = np.array(embeddings).astype('float32')

        # Create a FAISS index for efficient similarity search
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings_np)

        user_message = f"Hello, I need a quick risk assessment for a {delivery_date} delivery from {origin} to {destination}. The cargo is important, so {item_type} could be a problem. Could you assess potential risks and suggest alternative routes if necessary?"
        # Generate embedding for the user message
        query_embedding = get_embedding(user_message, engine='text-embedding-3-small')
        query_embedding_np = np.array([query_embedding]).astype('float32')

        # Search for similar documents
        _, indices = index.search(query_embedding_np, 2)
        retrieved_docs = [documents[i] for i in indices[0]]
        context = ' '.join(retrieved_docs)

        # Prepare structured prompt
        structured_prompt = f"Context:\n{context}\n\nQuery:\n{user_message}\n\nResponse:"

        # Call OpenAI API
        chat = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": System_Prompt}] + [{"role": "user", "content": structured_prompt}],
            temperature=0.5,
            max_tokens=1500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        # Get response
        response = chat.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display the response
        st.write(response)

    # Follow-up question input
    if st.session_state.messages:
        follow_up_question = st.text_input("Ask a follow-up question:")
        if st.button("Submit Follow-Up"):
            # Append the user's follow-up question to the messages
            st.session_state.messages.append({"role": "user", "content": follow_up_question})

            # Prepare the structured prompt for the follow-up question
            follow_up_context = ' '.join([msg['content'] for msg in st.session_state.messages if msg['role'] == 'assistant'])
            follow_up_prompt = f"Context:\n{follow_up_context}\n\nQuery:\n{follow_up_question}\n\nResponse:"

            # Call OpenAI API for the follow-up question
            follow_up_chat = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": System_Prompt}] + [{"role": "user", "content": follow_up_prompt}],
                temperature=0.5,
                max_tokens=1500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            # Get response for the follow-up question
            follow_up_response = follow_up_chat.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": follow_up_response})

            # Display the follow-up response
            st.write(follow_up_response)
else:
    st.warning("Please enter your OpenAI API key to use the chatbot.")
