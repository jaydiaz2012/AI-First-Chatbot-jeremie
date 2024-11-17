import requests

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
    