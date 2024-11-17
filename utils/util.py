import re

def extract_geolocation(response):
    """
    Extract geolocation details from a RouteX response and convert them into a dictionary.

    Parameters:
    response (str): The RouteX response containing geolocation details.

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
