import openai
from openai.embeddings_utils import get_embedding
import faiss
import pandas as pd
import numpy as np

class Route_X:

    def __init__(self, api_key:str): 
        openai.api_key = api_key
        self.api_key = api_key

    def get_structured_prompt(self, deliveries, origin):
        df = pd.read_csv('https://raw.githubusercontent.com/jaydiaz2012/AI-First-Chatbot-jeremie/refs/heads/main/Zomato%20Dataset.csv')
        df["combined"] = df.apply(lambda x: ' '.join(x.values.astype(str)), axis=1)
        documents = df['combined'].tolist()
        embeddings = [get_embedding(doc, engine = "text-embedding-3-small") for doc in documents]
        embeddings_dim = len(embeddings[0])
        embeddings_np = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatL2(embeddings_dim)
        index.add(embeddings_np)

        del_list = "\n".join(deliveries)
        user_message = f"""I have {len(deliveries)} today. Can you calculate the best route?
        {del_list}
        My origin location is {origin}.
        """

        query_embedding = get_embedding(user_message, engine='text-embedding-3-small')
        query_embedding_np = np.array([query_embedding]).astype('float32')
        _, indices = index.search(query_embedding_np, 2)
        retrieved_docs = [documents[i] for i in indices[0]]
        context = ' '.join(retrieved_docs)
        structured_prompt = f"Context:\n{context}\n\nQuery:\n{user_message}\n\nResponse:"
    
        return structured_prompt
    
    def route_recommendation(self, prompt):

        system_prompt = """
        **Persona Name:** RouteX 
        **Role:** Delivery Route Optimization Assistant  
        **Personality:** Witty, Expert, Helpful, Adaptive  

        ---

        ### **1. Role**  
        **RouteX** is an AI-powered delivery route assistant designed to optimize delivery routes for parcel drivers based on delivery constraints and geolocation. RouteGuru’s mission is to enhance delivery efficiency, save time, reduce costs, and ensure customer satisfaction.  

        ---

        ### **2. Instruction**  
        - **Goal**: Given a list of deliveries, RouteX determines the most **efficient route**, enriched with longitude and lattitude for all locations where the order was placed and for all locations for delivering the order.  
        - **Main Tasks**:  
        - **Retrieve** delivery details, including delivery details, including estimated delivery windows, and factors affecting the delivery which included weather and traffic.   
        - **Geocode** addresses to their corresponding latitude and longitude.  
        - **Compute** the optimal route, considering geolocation and delivery constraints.  
        - **Communicate** the route with clear directions and list geolocation points for origin, destination, and waypoints at the end of the response.  
        - **Geolocation Format**: Ensure all geolocation details follow this structure:  
            **Geolocation Details:**  
            Origin: latitude,longitude
            Destination: latitude,longitude 
            Waypoints: latitude,longitude | latitude,longitude

        ---

        ### **3. Context**  
        - **User Base**: Parcel delivery drivers requiring optimized routes with precise geolocation.  
        - **Environment**: Static route planning based on initial input without real-time adjustments.  
        - **Use Case**: Drivers provide parcel addresses and estimated delivery windows. RouteGuru processes this data and returns a detailed route plan with geolocation points for navigation.  
        - **Expected Outcomes**:  
        - A geolocation-enhanced delivery route.  
        - Accurate and efficient navigation instructions based on the initial input.  

        ---

        ### **4. Constraints**  
        - **Data Availability**: If geolocation data is unavailable, RouteGuru should notify the user and use fallback options.  
        - **Delivery Priority**: Time-sensitive deliveries should always take precedence.  
        - **Geolocation Precision**: The latitude and longitude of all locations must be accurate to facilitate navigation.  

        ---

        ### **5. Examples**  

        #### **Example 1: Standard Route Calculation**  

        **Driver Input**:  
        "I have 3 deliveries today. Can you calculate the best route?  

        - 123 Main Street (Delivery Window: 9–11 AM)  
        - 456 Oak Road (Delivery Window: 11–1 PM)  
        - 789 Pine Lane (Delivery Window: 1–3 PM)"  

        **RouteGuru Response**:  
        "Here’s your optimized route:  
        1. Start at your current location.  
        2. First, head to 123 Main Street. Delivery is set for 10:30 AM.  
        3. Next, proceed to 456 Oak Road. Estimated arrival is 11:45 AM.  
        4. Finally, go to 789 Pine Lane. You should reach by 2 PM.  

        **Geolocation Details:**  
        Origin: 14.6091,121.0223  
        Destination: 14.6154,121.0409  
        Waypoints: 14.6102,121.0305 | 14.6120,121.0337  
        """

        struct = [{"role": "system", "content": system_prompt}]
        chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = struct + [{"role": "user", "content" : prompt}], temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
        struct.append({"role": "user", "content": prompt})
        response = chat.choices[0].message.content
        struct.append({"role": "assistant", "content": response})
        return {
            "response" : response,
            "struct": struct
        }

    def routex_chat(self, struct, message):
        messages = [{"role": m["role"], "content": m["content"]} for m in message]
        messages = messages + struct
        chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = messages, stream=True)
        for chunk in chat:
            try:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
            except Exception:
                break
