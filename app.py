import os
import openai
import numpy as np
import pandas as pd
import json
import re
import requests 
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
from utils.routex import routex
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
warnings.filterwarnings("ignore")

def extract_geolocation(response):
    """
    Extract geolocation details from a RouteGuru response and convert them into a dictionary.

    Parameters:
    response (str): The RouteGuru response containing geolocation details.

    Returns:
    dict: A dictionary with 'origin', 'destination', and 'waypoints' as keys.
    """
    # Regex patterns to capture geolocation data
    origin_pattern = r"Origin:\s*([\d.]+),([\d.]+)"
    destination_pattern = r"Destination:\s*([\d.]+),([\d.]+)"
    waypoints_pattern = r"Waypoints:\s*((?:[\d.]+,[\d.]+\s*\|\s*)*[\d.]+,[\d.]+)"

    # Extract origin, destination, and waypoints
    origin_match = re.search(origin_pattern, response)
    destination_match = re.search(destination_pattern, response)
    waypoints_match = re.search(waypoints_pattern, response)

    # Convert matches to dictionary format
    geolocation_data = {
        "origin": f"{origin_match.group(1)},{origin_match.group(2)}" if origin_match else None,
        "destination": f"{destination_match.group(1)},{destination_match.group(2)}" if destination_match else None,
        "waypoints": waypoints_match.group(1).replace(" ", "").split("|") if waypoints_match else []
    }

    return geolocation_data

class GoogleMAP:

    def __init__(self, key):
        self.api_key =  key

    def getEmbededMapsSource(self, origin, destination, waypoints):
        return f"https://www.google.com/maps/embed/v1/directions?key={self.api_key}&origin={origin}&destination={destination}&waypoints={waypoints}&avoid=tolls|highways"
    
    def getMapCoordinates(self, address):
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': address,
            'key': self.api_key
        }

        response = requests.get(url, params=params)
        data = response.json()
        if data['status'] == 'OK':
            location = {
                "longitude": data['results'][0]['geometry']['location']['lng'],
                "latitude": data['results'][0]['geometry']['location']['lat']
            }
            return location
        else:
            return f"Error: {data['status']}"

st.set_page_config(page_title="RouteX, the Risk Assessor for Transport Routes", page_icon="📍", layout="wide")
# Sidebar for navigation and API key input
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
rg = RouteGuru(api_key=api_key)

with st.sidebar :
    st.image('images/pinmap.jpg')
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    rx = routex(openai.api_key = openai.api_key)
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
    st.header('RouteX: Your Delivery Route Expert!')
    
    deliveries = []

        if "click_count" not in st.session_state:
            st.session_state.click_count = 1

        def add_delivery_input():
            st.session_state.click_count += 1

        if st.button("➕ Add Delivery Details"):
            add_delivery_input()

        for i in range(st.session_state.click_count):
            delivery_address = st.text_input(f"Enter Delivery Address #{i + 1}:")
            delivery_time_window = st.text_input(f"Enter Delivery Time Window #{i + 1}:")
            deliveries.append(f"{delivery_address} ({delivery_time_window})")
            
        origin_location = st.text_input("Enter the origin location:")
    
    struct = []
        if st.button("Recommend Route!"):
            structured_prompt = rg.get_structured_prompt(deliveries=deliveries, origin=origin_location)
            resp = rg.route_recommendation(structured_prompt)
            st.session_state.messages.append({"role": "assistant", "content": resp["response"]})
            struct = resp["struct"]

            geo_locations = extract_geolocation(resp["response"])
            gm = GoogleMAP(key=config('GOOGLE_MAPS_API_KEY'))
            gm_source = gm.getEmbededMapsSource(
                origin=geo_locations["origin"],
                destination=geo_locations["destination"],
                waypoints="|".join(geo_locations["waypoints"])
            )
            # Google Maps iframe embed example
            iframe_html = f"""
            <iframe src={gm_source} width="800" height="600"></iframe>
            """

            # Display the iframe in the Streamlit app
            components.html(iframe_html, height=600)
            
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

         if st.session_state.messages:
             if prompt := st.chat_input("Are there any more questions?"):
                 # Add user message to chat history
                 st.session_state.messages.append({"role": "user", "content": prompt})
                 # Display user message in chat message container
                 with st.chat_message("user"):
                     st.markdown(prompt)
                with st.chat_message("assistant"):
                    response = st.write_stream(rg.route_guru_chat(struct, st.session_state.messages))
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
