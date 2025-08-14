from flask import Flask, render_template_string, jsonify, request
import folium
from folium import plugins
from datetime import datetime, timedelta
import pandas as pd
import os
import json
from tropycal import realtime
import numpy as np
import math
import threading
import time
import openai  # pip install openai
import re
import pickle
from dotenv import load_dotenv  # pip install python-dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder='static', static_url_path='/static')

# In-memory storage for locations and AI predictions
locations_data = []
ai_predictions_cache = {}
last_ai_update = None

# Configure OpenAI from environment variable (secure)
openai.api_key = os.getenv('OPENAI_API_KEY')

# Cost tracking
cost_tracker = {
    'total_cost': 0.0,
    'requests_today': 0,
    'last_reset': datetime.now()
}

def get_ai_hurricane_prediction(hurricane_info):
    """
    Get AI prediction for hurricane path - COST OPTIMIZED VERSION
    Uses GPT-3.5-turbo for low cost (~$0.001 per request)
    """
    global cost_tracker
    
    # Check cache first (6-hour cache to save money)
    storm_id = hurricane_info['id']
    if storm_id in ai_predictions_cache:
        cached_time = ai_predictions_cache[storm_id].get('timestamp')
        if cached_time and (datetime.now() - cached_time).seconds < 21600:  # 6 hours
            print(f"  Using cached AI prediction for {hurricane_info['name']}")
            return ai_predictions_cache[storm_id]['prediction']
    
    # If no API key, use free climatological model
    if not openai.api_key:
        print("  No OpenAI API key found, using free climatological model")
        return get_free_climatological_prediction(hurricane_info)
    
    try:
        # Prepare concise prompt to minimize tokens (saves money)
        lat = hurricane_info['lat']
        lon = hurricane_info['lon']
        wind = hurricane_info.get('wind_speed', 75)
        movement = hurricane_info.get('movement_dir', 'W')
        speed = hurricane_info.get('movement_speed', 15)
        
        prompt = f"""Storm {hurricane_info['name']} at {lat}N,{lon}W, {wind}mph winds, moving {movement} at {speed}mph.
Provide 5-day forecast. Format each line exactly as: Hours,Lat,Lon,Wind,ErrorRadius
Example: 12,25.5,-80.3,95,35
12h:
24h:
36h:
48h:
60h:
72h:
84h:
96h:
108h:
120h:"""

        # Use GPT-3.5-turbo for cost efficiency
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Only $0.001 per request
            messages=[
                {"role": "system", "content": "You are a meteorologist. Give only numbers, no explanation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150  # Limit response size
        )
        
        # Track costs
        cost = 0.001  # Approximate cost for GPT-3.5-turbo
        cost_tracker['total_cost'] += cost
        cost_tracker['requests_today'] += 1
        print(f"  AI prediction cost: ${cost:.4f} (Total today: ${cost_tracker['total_cost']:.2f})")
        
        # Parse response
        ai_text = response.choices[0].message.content
        prediction = parse_ai_forecast(ai_text, hurricane_info)
        
        # Cache the prediction
        ai_predictions_cache[storm_id] = {
            'prediction': prediction,
            'timestamp': datetime.now()
        }
        
        return prediction
        
    except Exception as e:
        print(f"  AI prediction failed: {e}, using free backup")
        return get_free_climatological_prediction(hurricane_info)

def parse_ai_forecast(ai_text, hurricane_info):
    """Parse AI response into forecast points"""
    forecast_points = []
    
    try:
        lines = ai_text.strip().split('\n')
        for line in lines:
            # Try to extract numbers from each line
            numbers = re.findall(r'-?\d+\.?\d*', line)
            if len(numbers) >= 4:
                hours = float(numbers[0])
                lat = float(numbers[1])
                lon = float(numbers[2])
                wind = float(numbers[3])
                error = float(numbers[4]) if len(numbers) > 4 else (20 + hours * 1.2)
                
                forecast_points.append({
                    'hours': hours,
                    'lat': lat,
                    'lon': lon,
                    'wind_speed': wind,
                    'cone_radius_nm': error,
                    'time': (datetime.now() + timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M')
                })
    except:
        pass
    
    # If parsing fails, return climatological forecast
    if not forecast_points:
        return get_free_climatological_prediction(hurricane_info)
    
    return {
        'forecast_points': forecast_points,
        'confidence': 'high' if len(forecast_points) >= 8 else 'medium',
        'source': 'AI-Enhanced (GPT-3.5)'
    }

def get_free_climatological_prediction(hurricane_info):
    """
    FREE fallback prediction using Atlantic hurricane climatology
    No API costs!
    """
    lat = hurricane_info['lat']
    lon = hurricane_info['lon']
    wind = hurricane_info.get('wind_speed', 75)
    
    forecast_points = []
    
    for hours in [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]:
        days = hours / 24
        
        # Atlantic hurricane climatology patterns
        if lat < 20:  # Trade winds belt
            new_lat = lat + days * 0.8
            new_lon = lon - days * 3.5  # Westward
        elif lat < 25:  # Subtropical ridge
            new_lat = lat + days * 1.2
            new_lon = lon - days * 2.5
        elif lat < 30:  # Recurvature zone
            new_lat = lat + days * 2.0
            new_lon = lon - days * 1.0 + (days - 2) * 0.5  # Start curving
        else:  # Extratropical
            new_lat = lat + days * 3.0
            new_lon = lon + days * 2.5  # Northeast
        
        # Wind decay over ocean (slower) vs land (faster)
        if new_lon > -95 and new_lat < 45:  # Over water
            new_wind = max(wind - days * 8, 35)
        else:  # Approaching/over land
            new_wind = max(wind - days * 15, 25)
        
        # Standard NHC error cone growth
        error_cone = {
            12: 35, 24: 50, 36: 65, 48: 80, 60: 95,
            72: 110, 84: 125, 96: 140, 108: 155, 120: 170
        }.get(hours, 170)
        
        forecast_points.append({
            'hours': hours,
            'lat': round(new_lat, 1),
            'lon': round(new_lon, 1),
            'wind_speed': new_wind,
            'cone_radius_nm': error_cone,
            'time': (datetime.now() + timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M')
        })
    
    return {
        'forecast_points': forecast_points,
        'confidence': 'medium',
        'source': 'Climatological Model (Free)'
    }

def create_ai_uncertainty_cone(hurricane, ai_prediction):
    """Create uncertainty cone from AI predictions"""
    if not ai_prediction or not ai_prediction.get('forecast_points'):
        return None
    
    forecast_points = ai_prediction['forecast_points']
    
    # Build cone polygon
    cone_left_points = [[hurricane['lat'], hurricane['lon']]]
    cone_right_points = [[hurricane['lat'], hurricane['lon']]]
    center_line = [[hurricane['lat'], hurricane['lon']]]
    
    for i, point in enumerate(forecast_points):
        center_line.append([point['lat'], point['lon']])
        
        # Calculate perpendicular to track
        if i > 0:
            prev = forecast_points[i-1]
            dlat = point['lat'] - prev['lat']
            dlon = point['lon'] - prev['lon']
            length = math.sqrt(dlat**2 + dlon**2)
            
            if length > 0:
                perp_lat = -dlon / length
                perp_lon = dlat / length
            else:
                perp_lat, perp_lon = 0, 1
        else:
            # First point
            if len(forecast_points) > 1:
                next_pt = forecast_points[1]
                dlat = next_pt['lat'] - point['lat']
                dlon = next_pt['lon'] - point['lon']
                length = math.sqrt(dlat**2 + dlon**2)
                if length > 0:
                    perp_lat = -dlon / length
                    perp_lon = dlat / length
                else:
                    perp_lat, perp_lon = 0, 1
            else:
                perp_lat, perp_lon = 0, 1
        
        # Convert error radius to degrees
        error_deg_lat = point['cone_radius_nm'] / 60.0
        error_deg_lon = point['cone_radius_nm'] / (60.0 * math.cos(math.radians(point['lat'])))
        
        # Calculate cone boundaries
        left_lat = point['lat'] + perp_lat * error_deg_lat
        left_lon = point['lon'] + perp_lon * error_deg_lon
        right_lat = point['lat'] - perp_lat * error_deg_lat
        right_lon = point['lon'] - perp_lon * error_deg_lon
        
        cone_left_points.append([left_lat, left_lon])
        cone_right_points.append([right_lat, right_lon])
    
    # Create polygon
    cone_polygon = cone_left_points + list(reversed(cone_right_points))
    
    return {
        'polygon': cone_polygon,
        'center_line': center_line,
        'forecast_points': forecast_points,
        'confidence': ai_prediction.get('confidence', 'medium'),
        'source': ai_prediction.get('source', 'AI')
    }

def get_hurricane_data_tropycal():
    """Enhanced version that includes AI predictions and full historical data"""
    hurricane_data = []
    
    try:
        print("=" * 50)
        print("Fetching real-time tropical cyclone data...")
        
        realtime_obj = realtime.Realtime()
        
        atlantic_ids = realtime_obj.list_active_storms(basin='north_atlantic')
        pacific_ids = realtime_obj.list_active_storms(basin='east_pacific')
        
        all_storm_ids = atlantic_ids + pacific_ids
        print(f"Active storms: {all_storm_ids}")
        
        for storm_id in all_storm_ids:
            try:
                storm = getattr(realtime_obj, storm_id)
                
                storm_info = {
                    'id': storm_id,
                    'name': storm.name,
                    'lat': None,
                    'lon': None,
                    'past_track': [],
                    'forecast_track': [],
                    'ai_forecast': None,
                    'wind_speed': None,
                    'movement_speed': None,
                    'movement_dir': None,
                    'category': None,
                    'pressure': None
                }
                
                # Get COMPLETE track data - ALL historical points
                if hasattr(storm, 'dict') and storm.dict:
                    lats = storm.dict.get('lat', [])
                    lons = storm.dict.get('lon', [])
                    times = storm.dict.get('time', [])
                    winds = storm.dict.get('vmax', [])
                    pressures = storm.dict.get('mslp', [])
                    types = storm.dict.get('type', [])
                    
                    if lats and lons:
                        storm_info['lat'] = lats[-1]
                        storm_info['lon'] = lons[-1]
                        
                        if winds:
                            storm_info['wind_speed'] = winds[-1]
                        
                        if pressures:
                            storm_info['pressure'] = pressures[-1]
                        
                        # Determine category
                        if storm_info['wind_speed']:
                            wind = storm_info['wind_speed']
                            if wind >= 157:
                                storm_info['category'] = "Category 5"
                            elif wind >= 130:
                                storm_info['category'] = "Category 4"
                            elif wind >= 111:
                                storm_info['category'] = "Category 3"
                            elif wind >= 96:
                                storm_info['category'] = "Category 2"
                            elif wind >= 74:
                                storm_info['category'] = "Category 1"
                            elif wind >= 39:
                                storm_info['category'] = "Tropical Storm"
                            else:
                                storm_info['category'] = "Tropical Depression"
                    
                    # Get ALL past track points (complete history)
                    for i in range(len(lats)):
                        track_point = {
                            'lat': lats[i],
                            'lon': lons[i],
                            'time': times[i].strftime('%Y-%m-%d %H:%M UTC') if i < len(times) else '',
                            'winds': winds[i] if i < len(winds) else None,
                            'pressure': pressures[i] if i < len(pressures) else None,
                            'type': types[i] if i < len(types) else '',
                            'category': None
                        }
                        
                        # Add category for each point
                        if track_point['winds']:
                            w = track_point['winds']
                            if w >= 157:
                                track_point['category'] = "Cat 5"
                            elif w >= 130:
                                track_point['category'] = "Cat 4"
                            elif w >= 111:
                                track_point['category'] = "Cat 3"
                            elif w >= 96:
                                track_point['category'] = "Cat 2"
                            elif w >= 74:
                                track_point['category'] = "Cat 1"
                            elif w >= 39:
                                track_point['category'] = "TS"
                            else:
                                track_point['category'] = "TD"
                        
                        storm_info['past_track'].append(track_point)
                    
                    print(f"  Loaded {len(storm_info['past_track'])} historical points for {storm.name}")
                
                # Calculate movement
                if len(storm_info['past_track']) >= 2:
                    lat1, lon1 = storm_info['past_track'][-2]['lat'], storm_info['past_track'][-2]['lon']
                    lat2, lon2 = storm_info['past_track'][-1]['lat'], storm_info['past_track'][-1]['lon']
                    
                    # Calculate bearing
                    dlon = np.radians(lon2 - lon1)
                    lat1_rad = np.radians(lat1)
                    lat2_rad = np.radians(lat2)
                    
                    x = np.cos(lat2_rad) * np.sin(dlon)
                    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
                    bearing = np.degrees(np.arctan2(x, y))
                    
                    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
                    index = int((bearing + 11.25) / 22.5) % 16
                    storm_info['movement_dir'] = directions[index]
                    
                    # Calculate speed
                    R = 3959
                    dlat = np.radians(lat2 - lat1)
                    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
                    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                    distance = R * c
                    storm_info['movement_speed'] = int(distance / 6)
                
                # Get AI prediction for this storm
                print(f"  Getting AI prediction for {storm_info['name']}...")
                storm_info['ai_forecast'] = get_ai_hurricane_prediction(storm_info)
                
                hurricane_data.append(storm_info)
                print(f"  Added: {storm_info['name']} ({storm_info['category']}) with {len(storm_info['past_track'])} data points")
                
            except Exception as e:
                print(f"  Error processing storm {storm_id}: {e}")
                continue
        
        print(f"\nTotal storms with AI predictions: {len(hurricane_data)}")
        print(f"Total API cost this session: ${cost_tracker['total_cost']:.4f}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error fetching hurricane data: {e}")
    
    return hurricane_data

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in miles using Haversine formula"""
    R = 3959  # Earth radius in miles
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def is_mobile_request():
    """Detect if request is from mobile device"""
    user_agent = request.headers.get('User-Agent', '').lower()
    mobile_agents = ['android', 'iphone', 'ipad', 'mobile', 'tablet']
    return any(agent in user_agent for agent in mobile_agents)

def create_hurricane_map():
    """Create Folium map with complete historical data and enhanced property info"""
    
    # Check if mobile
    is_mobile = is_mobile_request()
    
    # Create base map
    m = folium.Map(
        location=[25, -60],
        zoom_start=4,
        tiles=None,
        prefer_canvas=True,
        png_enabled=True,
        zoom_control=True,
        control_scale=True
    )
    
    # Add tile layers
    folium.TileLayer(
        tiles='CartoDB positron',
        name='Light',
        attr='CartoDB',
        overlay=False,
        control=True,
        show=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='CartoDB dark_matter',
        name='Dark',
        attr='CartoDB',
        overlay=False,
        control=True,
        show=False
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='ESRI',
        name='Satellite',
        overlay=False,
        control=True,
        show=False
    ).add_to(m)
    
    # Get hurricane data with AI predictions
    hurricanes = get_hurricane_data_tropycal()
    
    # Create feature groups
    property_group = folium.FeatureGroup(name='Properties', show=True)
    threatened_property_group = folium.FeatureGroup(name='Properties Under Threat (< 300 miles)', show=True)
    current_position_group = folium.FeatureGroup(name='Current Positions', show=True)
    threat_zone_group = folium.FeatureGroup(name='300 Mile Threat Zones', show=True)
    past_track_group = folium.FeatureGroup(name='Complete Historical Track', show=True)
    ai_forecast_group = folium.FeatureGroup(name='AI Forecast', show=True)
    ai_cone_group = folium.FeatureGroup(name='AI Uncertainty Cone', show=True)
    
    # Add properties with distance calculations to hurricanes
    print(f"Adding {len(locations_data)} properties to map...")
    print(f"Mobile device detected: {is_mobile} - Using larger touch-friendly markers")
    threatened_properties = []
    
    for location in locations_data:
        if location.get('lat') and location.get('lon'):
            # Calculate distance to each hurricane
            min_distance = float('inf')
            closest_hurricane = None
            hurricane_distances = []
            
            for hurricane in hurricanes:
                if hurricane.get('lat') and hurricane.get('lon'):
                    distance = calculate_distance(
                        location['lat'], location['lon'],
                        hurricane['lat'], hurricane['lon']
                    )
                    hurricane_distances.append({
                        'name': hurricane['name'],
                        'distance': distance,
                        'category': hurricane.get('category', 'Unknown')
                    })
                    if distance < min_distance:
                        min_distance = distance
                        closest_hurricane = hurricane['name']
            
            # Sort hurricanes by distance
            hurricane_distances.sort(key=lambda x: x['distance'])
            
            # Build comprehensive popup with iOS-style design
            popup_min_width = "150px" if is_mobile else "250px"
            popup_max_width = "175px" if is_mobile else "350px"
            
            popup_content = f"""
            <div class='property-popup' style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; min-width: {popup_min_width}; max-width: {popup_max_width};'>
                <h3 style='color: #000000; margin: 5px 0; border-bottom: 1px solid #E5E5EA; padding-bottom: 8px; font-size: 17px; font-weight: 600;'>
                    {location['location']}
                </h3>
            """
            
            # Add hurricane threat information if any storms exist
            if hurricane_distances:
                bg_color = "#FFF2F2" if min_distance < 300 else "#F2FFF4"
                text_color = "#FF3B30" if min_distance < 300 else "#34C759"
                popup_content += f"""
                <div style='background-color: {bg_color}; padding: 8px; margin-bottom: 8px; border-radius: 8px;'>
                    <b style='color: {text_color}; font-size: 14px;'>
                        {'⚠️ HURRICANE THREAT' if min_distance < 300 else '✓ No Immediate Threat'}
                    </b><br>
                    <span style='font-size: 13px; color: #3C3C43;'>
                        <b>Closest Storm:</b> {closest_hurricane} - {min_distance:.0f} miles
                    </span>
                </div>
                """
                
                # Show all hurricane distances
                popup_content += """
                <div style='margin: 8px 0; padding: 8px; background-color: #F2F2F7; border-radius: 8px;'>
                    <b style='font-size: 13px; color: #000000;'>Distance to Active Storms:</b><br>
                """
                for h in hurricane_distances[:3]:  # Show top 3 closest
                    color = '#FF3B30' if h['distance'] < 300 else '#3C3C43'
                    popup_content += f"""
                    <span style='color: {color}; font-size: 12px;'>
                        • {h['name']} ({h['category']}): <b>{h['distance']:.0f} miles</b>
                    </span><br>
                """
                popup_content += "</div>"
            
            popup_content += "<table style='font-size: 13px; width: 100%; border-collapse: collapse;'>"
            
            # Always show TIV if available
            if location.get('tiv') and location['tiv'] != 'N/A':
                popup_content += f"""
                    <tr style='border-bottom: 1px solid #E5E5EA;'>
                        <td style='padding: 8px 0; font-weight: 600; color: #8E8E93;'>Total Insured Value:</td>
                        <td style='padding: 8px 0; color: #0E7490; font-weight: 600;'>{location['tiv']}</td>
                    </tr>
                """
            
            # Always show Hurricane Deductible if available
            if location.get('hurricane_deductible') and location['hurricane_deductible'] != 'N/A':
                popup_content += f"""
                    <tr style='border-bottom: 1px solid #E5E5EA;'>
                        <td style='padding: 8px 0; font-weight: 600; color: #8E8E93;'>NWS Deductible:</td>
                        <td style='padding: 8px 0; color: #FF9500; font-weight: 600;'>{location['hurricane_deductible']}</td>
                    </tr>
                """
            
            # Show other deductibles if they exist
            if location.get('deductible') and location['deductible'] != 'N/A':
                popup_content += f"""
                    <tr style='border-bottom: 1px solid #E5E5EA;'>
                        <td style='padding: 8px 0; color: #8E8E93;'><b>Standard Deductible:</b></td>
                        <td style='padding: 8px 0; color: #3C3C43;'>{location['deductible']}</td>
                    </tr>
                """
            
            # Show category/risk level if available
            if location.get('category') and location['category'] != 'N/A':
                popup_content += f"""
                    <tr style='border-bottom: 1px solid #E5E5EA;'>
                        <td style='padding: 8px 0; color: #8E8E93;'><b>Risk Category:</b></td>
                        <td style='padding: 8px 0; color: #3C3C43;'>{location['category']}</td>
                    </tr>
                """
            

            
            # Show claims contact if available
            if location.get('claims_contact') and location['claims_contact'] != 'N/A':
                popup_content += f"""
                    <tr style='border-bottom: 1px solid #E5E5EA;'>
                        <td style='padding: 8px 0; color: #8E8E93;'><b>Claims Contact:</b></td>
                        <td style='padding: 8px 0; color: #3C3C43;'>{location['claims_contact']}</td>
                    </tr>
                """
            
            # Always show the phone number
            popup_content += f"""
                <tr style='border-bottom: 1px solid #E5E5EA;'>
                    <td style='padding: 8px 0; color: #8E8E93;'><b>Phone:</b></td>
                    <td style='padding: 8px 0; color: #007AFF;'><a href='tel:8136826198' style='color: #007AFF; text-decoration: none;'>(813) 682-6198</a></td>
                </tr>
            """
            
            # Always show the email
            popup_content += f"""
                <tr style='border-bottom: 1px solid #E5E5EA;'>
                    <td style='padding: 8px 0; color: #8E8E93;'><b>Email:</b></td>
                    <td style='padding: 8px 0; color: #007AFF;'><a href='mailto:leann.saliga@franklinst.com' style='color: #007AFF; text-decoration: none;'>leann.saliga@franklinst.com</a></td>
                </tr>
            """
            
            # Create appropriate icon based on threat status
            if hurricanes and min_distance < 300:
                # Enhanced tooltip for threatened properties
                tooltip_text = f"""⚠️ {location['location']}
                THREAT: {closest_hurricane} - {min_distance:.0f} miles
                TIV: {location.get('tiv', 'N/A')}
                Hurricane Ded: {location.get('hurricane_deductible', 'N/A')}"""
                
                # Red icon for threatened properties - bigger for mobile clicking
                if is_mobile:
                    # Larger red circle for mobile threatened properties
                    folium.CircleMarker(
                        location=[location['lat'], location['lon']],
                        radius=12,  # Bigger radius for easier clicking
                        popup=folium.Popup(popup_content, max_width=175),  # Half size popup
                        tooltip=tooltip_text,
                        color='white',
                        weight=2,
                        fill=True,
                        fillColor='#FF3B30',
                        fillOpacity=0.9
                    ).add_to(threatened_property_group)
                else:
                    # Regular icon for desktop
                    folium.Marker(
                        location=[location['lat'], location['lon']],
                        popup=folium.Popup(popup_content, max_width=350),
                        tooltip=tooltip_text,
                        icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
                    ).add_to(threatened_property_group)
                threatened_properties.append(location['location'])
            else:
                # Normal tooltip
                tooltip_text = f"<b>{location['location']}</b>"
                
                # Bigger circles for easier mobile clicking
                marker_radius = 11 if is_mobile else 6
                popup_width = 175 if is_mobile else 350  # Half size for mobile
                folium.CircleMarker(
                    location=[location['lat'], location['lon']],
                    radius=marker_radius,
                    popup=folium.Popup(popup_content, max_width=popup_width),
                    tooltip=tooltip_text,
                    color='white',
                    weight=2,
                    fill=True,
                    fillColor='#0E7490',
                    fillOpacity=0.9
                ).add_to(property_group)
    
    # Print summary of threatened properties
    if threatened_properties:
        print(f"\n⚠️  WARNING: {len(threatened_properties)} properties within 300 miles of hurricane")
        for prop in threatened_properties:
            print(f"    - {prop}")
    else:
        print(f"✓ All properties are safe (>300 miles from any hurricane)")
    
    # Color scheme for storms
    def get_storm_color(wind_speed):
        if wind_speed is None:
            return 'gray'
        elif wind_speed >= 157:
            return 'purple'
        elif wind_speed >= 130:
            return 'darkred'
        elif wind_speed >= 111:
            return 'red'
        elif wind_speed >= 96:
            return 'orange'
        elif wind_speed >= 74:
            return '#DAA520'
        elif wind_speed >= 39:
            return 'green'
        else:
            return 'blue'
    
    # Add hurricanes with complete historical data
    for hurricane in hurricanes:
        if hurricane.get('lat') and hurricane.get('lon'):
            
            # Current position with iOS-styled popup
            hurricane_html = f"""
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; min-width: 300px;">
                <h3 style="color: #000000; margin-bottom: 10px; font-size: 17px; font-weight: 600;">{hurricane['name']}</h3>
                <table style="font-size: 13px; width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #E5E5EA;">
                        <td style="padding: 6px 0; color: #8E8E93;"><b>Category:</b></td>
                        <td style="padding: 6px 0; color: #000000;">{hurricane.get('category', 'Unknown')}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E5E5EA;">
                        <td style="padding: 6px 0; color: #8E8E93;"><b>Position:</b></td>
                        <td style="padding: 6px 0; color: #000000;">{hurricane['lat']:.1f}°N, {abs(hurricane['lon']):.1f}°W</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E5E5EA;">
                        <td style="padding: 6px 0; color: #8E8E93;"><b>Max Winds:</b></td>
                        <td style="padding: 6px 0; color: #000000;">{hurricane.get('wind_speed', 'Unknown')} mph</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E5E5EA;">
                        <td style="padding: 6px 0; color: #8E8E93;"><b>Pressure:</b></td>
                        <td style="padding: 6px 0; color: #000000;">{hurricane.get('pressure', 'Unknown')} mb</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E5E5EA;">
                        <td style="padding: 6px 0; color: #8E8E93;"><b>Movement:</b></td>
                        <td style="padding: 6px 0; color: #000000;">{hurricane.get('movement_dir', 'Unknown')} at {hurricane.get('movement_speed', 'Unknown')} mph</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E5E5EA;">
                        <td style="padding: 6px 0; color: #8E8E93;"><b>AI Forecast:</b></td>
                        <td style="padding: 6px 0; color: #8E8E93;">{hurricane.get('ai_forecast', {}).get('source', 'Processing...')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: #8E8E93;"><b>Track Points:</b></td>
                        <td style="padding: 6px 0; color: #000000;">{len(hurricane.get('past_track', []))} historical positions</td>
                    </tr>
                </table>
            </div>
            """
            
            # Hurricane icon sizing - moderate size for mobile
            icon_size = "14px" if is_mobile else "16px"
            font_size = "12px" if is_mobile else "13px"
            
            hurricane_icon = folium.features.DivIcon(
                html=f"""
                <div style="text-align: center;">
                    <div style="font-size: 12px; width: {icon_size}; height: {icon_size}; 
                         background-color: #FF3B30; border-radius: 50%; 
                         border: 3px solid white; margin: 0 auto; box-shadow: 0 2px 8px rgba(0,0,0,0.2);"></div>
                    <div style="font-size: {font_size}; font-weight: 600; color: #000000; 
                         margin-top: 3px; text-shadow: 1px 1px 3px rgba(255,255,255,0.8), -1px -1px 3px rgba(255,255,255,0.8);
                         font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;">
                        {hurricane['name']}
                    </div>
                </div>
                """
            )
            
            folium.Marker(
                location=[hurricane['lat'], hurricane['lon']],
                popup=folium.Popup(hurricane_html, max_width=175 if is_mobile else 350),
                icon=hurricane_icon
            ).add_to(current_position_group)
            
            # Add 300-mile threat zone circle with iOS styling
            folium.Circle(
                location=[hurricane['lat'], hurricane['lon']],
                radius=300 * 1609.34,  # Convert 300 miles to meters
                color='#FF3B30',  # iOS red
                fill=True,
                fillColor='#FF3B30',
                fillOpacity=0.08,  # Very subtle fill
                weight=1.5,  # Consistent weight
                opacity=0.3,
                dash_array='8,4',
                tooltip=f"{hurricane['name']} - 300 Mile Threat Zone"
            ).add_to(threat_zone_group)
            
            # Add COMPLETE past track with ALL historical points
            if hurricane.get('past_track'):
                past_coords = [[pt['lat'], pt['lon']] for pt in hurricane['past_track']]
                
                if len(past_coords) > 1:
                    # Main track line with neutral styling
                    folium.PolyLine(
                        locations=past_coords,
                        color='#8E8E93',  # iOS gray for neutral past track
                        weight=2 if is_mobile else 2.5,  # Maintain visibility
                        opacity=0.7,
                        tooltip=f"{hurricane['name']} Complete Track ({len(past_coords)} points)"
                    ).add_to(past_track_group)
                    
                    # Add markers for ALL historical positions (every 4th point for visibility)
                    point_interval = 6 if is_mobile else 4  # Show fewer points on mobile
                    marker_radius = 4 if is_mobile else 3  # Larger minimum size for visibility
                    
                    for i, point in enumerate(hurricane['past_track']):
                        # Show fewer points on mobile
                        if i % point_interval == 0 or i == 0 or i == len(hurricane['past_track']) - 1:
                            # Use neutral gray for historical points
                            point_color = '#8E8E93'  # iOS gray
                            
                            # Detailed popup for each historical point with iOS styling
                            point_popup = f"""
                            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 12px;">
                                <b style="color: #000000; font-size: 14px;">{hurricane['name']} - Historical Position</b><br>
                                <span style="color: #8E8E93;"><b>Time:</b></span> {point.get('time', 'Unknown')}<br>
                                <span style="color: #8E8E93;"><b>Position:</b></span> {point['lat']:.1f}°N, {abs(point['lon']):.1f}°W<br>
                                <span style="color: #8E8E93;"><b>Winds:</b></span> {point.get('winds', 'Unknown')} mph<br>
                                <span style="color: #8E8E93;"><b>Pressure:</b></span> {point.get('pressure', 'Unknown')} mb<br>
                                <span style="color: #8E8E93;"><b>Category:</b></span> {point.get('category', 'Unknown')}<br>
                                <span style="color: #8E8E93;"><b>Type:</b></span> {point.get('type', 'Unknown')}
                            </div>
                            """
                            
                            folium.CircleMarker(
                                location=[point['lat'], point['lon']],
                                radius=marker_radius,
                                popup=folium.Popup(point_popup, max_width=175 if is_mobile else 250),
                                tooltip=f"{point.get('time', '')} - {point.get('category', 'Unknown')}",
                                color=point_color,
                                fill=True,
                                fillColor=point_color,
                                fillOpacity=0.6,
                                weight=1
                            ).add_to(past_track_group)
            
            # AI Forecast and Cone
            if hurricane.get('ai_forecast'):
                cone_data = create_ai_uncertainty_cone(hurricane, hurricane['ai_forecast'])
                
                if cone_data:
                    # iOS-style neutral confidence colors
                    confidence_colors = {
                        'high': '#C7C7CC',    # iOS light gray
                        'medium': '#AEAEB2',  # iOS medium gray
                        'low': '#8E8E93'      # iOS gray
                    }
                    cone_color = confidence_colors.get(cone_data['confidence'], '#AEAEB2')
                    
                    # Add center line (forecast track) with neutral gray styling
                    folium.PolyLine(
                        locations=cone_data['center_line'],
                        color='#8E8E93',  # iOS gray - neutral
                        weight=2,  # Consistent weight for visibility
                        opacity=0.6,
                        dash_array='10,5',
                        tooltip=f"AI Forecast - {cone_data['source']}"
                    ).add_to(ai_forecast_group)
                    
                    # Add uncertainty cone with subtle styling
                    folium.Polygon(
                        locations=cone_data['polygon'],
                        color=cone_color,
                        weight=1.5,  # Consistent weight
                        fill=True,
                        fillColor=cone_color,
                        fillOpacity=0.15,  # More subtle
                        tooltip=f"AI Cone - {cone_data['confidence']} confidence"
                    ).add_to(ai_cone_group)
                    
                    # Add forecast position markers with iOS styling
                    forecast_interval = 3 if is_mobile else 2  # Show fewer points on mobile
                    for idx, point in enumerate(cone_data['forecast_points']):
                        if idx % forecast_interval == 0:  # Show fewer points on mobile
                            # Convert nautical miles to regular miles for display
                            error_miles = point['cone_radius_nm'] * 1.15078
                            popup_html = f"""<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;">
                                     <b style="color: #000000; font-size: 14px;">AI Forecast +{point['hours']}h</b><br>
                                     <span style="color: #8E8E93;">Position:</span> {point['lat']:.1f}°N, {abs(point['lon']):.1f}°W<br>
                                     <span style="color: #8E8E93;">Winds:</span> {point['wind_speed']:.0f} mph<br>
                                     <span style="color: #8E8E93;">Time:</span> {point['time']}<br>
                                     <span style="color: #8E8E93;">Error:</span> ±{error_miles:.0f} miles
                                     </div>"""
                            folium.CircleMarker(
                                location=[point['lat'], point['lon']],
                                radius=5 if is_mobile else 4,  # Larger for better zoom visibility
                                popup=folium.Popup(popup_html, max_width=175 if is_mobile else 300),
                                color='#8E8E93',  # iOS gray - neutral forecast color
                                fill=True,
                                fillColor='#FFFFFF',  # White fill
                                fillOpacity=0.9,
                                weight=1.5  # Consistent weight
                            ).add_to(ai_forecast_group)
    
    # Add all groups to map
    property_group.add_to(m)
    threatened_property_group.add_to(m)
    current_position_group.add_to(m)
    threat_zone_group.add_to(m)
    past_track_group.add_to(m)
    ai_forecast_group.add_to(m)
    ai_cone_group.add_to(m)
    
    # Layer control
    folium.LayerControl(collapsed=False, position='topright').add_to(m)
    
    # Tools - only add measure control on desktop
    if not is_mobile:
        plugins.MeasureControl(primary_length_unit='miles').add_to(m)
    plugins.Fullscreen().add_to(m)
    
    # Title with iOS-style design and logo beside title
    threat_count = len(threatened_properties) if 'threatened_properties' in locals() else 0
    threat_color = '#FF3B30' if threat_count > 0 else '#34C759'  # iOS red and green
    threat_text = f'⚠️ {threat_count} Properties Under Threat' if threat_count > 0 else '✓ All Properties Safe'
    
    # Smaller title for mobile
    title_padding = "8px 16px" if is_mobile else "12px 24px"
    title_font_size = "15px" if is_mobile else "17px"
    subtitle_font_size = "11px" if is_mobile else "13px"
    logo_height = "30px" if is_mobile else "40px"
    
    title_html = f'''
    <div id="title" style="position: fixed; 
                top: 10px; 
                left: 50%; 
                transform: translateX(-50%);
                width: auto; 
                max-width: 90%;
                height: auto; 
                background-color: rgba(255,255,255,0.98);
                border-radius: 16px;
                padding: {title_padding};
                z-index: 9999;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08), 0 1px 3px rgba(0,0,0,0.05);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);">
        <div style="display: flex; align-items: center; justify-content: center; gap: 12px;">
            <div style="text-align: center;">
                <h3 style="color: #000000; margin: 0; font-size: {title_font_size}; font-weight: 600; letter-spacing: -0.4px;">
                    Fitch Irick Hurricane Risk Tracker
                </h3>
                <p style="margin: 4px 0 0 0; font-size: {subtitle_font_size}; color: #8E8E93; font-weight: 400;">
                    Click on Locations for More Information
                </p>
            </div>
            <img src="/static/logo.png" alt="Fitch Irick Logo" style="height: {logo_height};">
        </div>

    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

def load_properties_from_excel():
    """Load properties from Excel file with ALL columns"""
    LOCATIONS_FILE = "locations.xlsx"
    
    print("\n" + "="*60)
    print("LOADING EXCEL FILE WITH COMPLETE DATA")
    print("="*60)
    
    if not os.path.exists(LOCATIONS_FILE):
        print(f"ERROR: {LOCATIONS_FILE} NOT FOUND!")
        return []
    
    try:
        df = pd.read_excel(LOCATIONS_FILE)
        print(f"SUCCESS: Loaded {LOCATIONS_FILE}")
        print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns found: {list(df.columns)}")
        
        locations = []
        
        # Find key columns
        name_col = None
        lat_col = None
        lon_col = None
        tiv_col = None
        hurricane_ded_col = None
        standard_ded_col = None
        pm_col = None
        claims_col = None
        category_col = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Required columns
            if not name_col and any(x in col_lower for x in ['name', 'location', 'property', 'address', 'site', 'building']):
                name_col = col
                print(f"Found name column: '{col}'")
            if not lat_col and any(x in col_lower for x in ['lat', 'latitude']):
                lat_col = col
                print(f"Found latitude column: '{col}'")
            if not lon_col and any(x in col_lower for x in ['lon', 'long', 'longitude']):
                lon_col = col
                print(f"Found longitude column: '{col}'")
            
            # Financial columns
            if not tiv_col and any(x in col_lower for x in ['tiv', 'total insured', 'insured value', 'value']):
                tiv_col = col
                print(f"Found TIV column: '{col}'")
            if not hurricane_ded_col and any(x in col_lower for x in ['hurricane', 'wind']) and 'deductible' in col_lower:
                hurricane_ded_col = col
                print(f"Found hurricane deductible column: '{col}'")
            if not standard_ded_col and 'deductible' in col_lower and not any(x in col_lower for x in ['hurricane', 'wind']):
                standard_ded_col = col
                print(f"Found standard deductible column: '{col}'")
            
            # Contact columns
            if not pm_col and any(x in col_lower for x in ['property manager', 'manager', 'pm']):
                pm_col = col
                print(f"Found property manager column: '{col}'")
            if not claims_col and any(x in col_lower for x in ['claims', 'contact', 'phone']):
                claims_col = col
                print(f"Found claims contact column: '{col}'")
            
            # Risk category
            if not category_col and any(x in col_lower for x in ['category', 'risk', 'tier', 'level']):
                category_col = col
                print(f"Found category column: '{col}'")
        
        # Fallback for required columns
        if not (name_col and lat_col and lon_col):
            if len(df.columns) >= 3:
                name_col = df.columns[0] if not name_col else name_col
                lat_col = df.columns[1] if not lat_col else lat_col
                lon_col = df.columns[2] if not lon_col else lon_col
                print(f"Using fallback columns: {name_col}, {lat_col}, {lon_col}")
        
        # Process all rows
        print(f"\nProcessing {len(df)} rows...")
        
        for index, row in df.iterrows():
            try:
                location_name = row[name_col] if name_col else f"Property {index+1}"
                lat = float(row[lat_col]) if lat_col and pd.notna(row[lat_col]) else None
                lon = float(row[lon_col]) if lon_col and pd.notna(row[lon_col]) else None
                
                if not lat or not lon:
                    continue
                
                location = {
                    'id': len(locations) + 1,
                    'location': str(location_name),
                    'lat': lat,
                    'lon': lon,
                    'tiv': 'N/A',
                    'hurricane_deductible': 'N/A',
                    'deductible': 'N/A',
                    'property_manager': 'N/A',
                    'claims_contact': 'N/A',
                    'category': 'N/A'
                }
                
                # Add TIV with formatting
                if tiv_col and pd.notna(row.get(tiv_col)):
                    try:
                        tiv_val = float(row[tiv_col])
                        location['tiv'] = f"${tiv_val:,.0f}"
                    except:
                        location['tiv'] = str(row[tiv_col])
                
                # Add hurricane deductible
                if hurricane_ded_col and pd.notna(row.get(hurricane_ded_col)):
                    ded_val = row[hurricane_ded_col]
                    # Format percentage or dollar amount
                    try:
                        if isinstance(ded_val, (int, float)):
                            if ded_val < 1:  # Assume percentage if less than 1
                                location['hurricane_deductible'] = f"{ded_val*100:.1f}%"
                            elif ded_val < 100:  # Assume percentage if less than 100
                                location['hurricane_deductible'] = f"{ded_val:.1f}%"
                            else:  # Dollar amount
                                location['hurricane_deductible'] = f"${ded_val:,.0f}"
                        else:
                            location['hurricane_deductible'] = str(ded_val)
                    except:
                        location['hurricane_deductible'] = str(ded_val)
                
                # Add standard deductible
                if standard_ded_col and pd.notna(row.get(standard_ded_col)):
                    location['deductible'] = str(row[standard_ded_col])
                
                # Add other fields
                if pm_col and pd.notna(row.get(pm_col)):
                    location['property_manager'] = str(row[pm_col])
                
                if claims_col and pd.notna(row.get(claims_col)):
                    location['claims_contact'] = str(row[claims_col])
                
                if category_col and pd.notna(row.get(category_col)):
                    location['category'] = str(row[category_col])
                
                locations.append(location)
                print(f"  [{len(locations)}] {location['location']}: TIV={location['tiv']}, Hurricane Ded={location['hurricane_deductible']}")
                
            except Exception as e:
                print(f"  ERROR on row {index}: {e}")
                continue
        
        print(f"\nSUCCESS: Loaded {len(locations)} properties with complete data")
        print("="*60)
        return locations
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []

# Auto-update function
def auto_update_loop():
    """Background thread that updates every 6 hours"""
    while True:
        try:
            print(f"\n{'='*60}")
            print(f"Auto-update at {datetime.now()}")
            print(f"Current cost: ${cost_tracker['total_cost']:.2f}")
            
            # Reset daily cost tracker
            if (datetime.now() - cost_tracker['last_reset']).days >= 1:
                cost_tracker['total_cost'] = 0.0
                cost_tracker['requests_today'] = 0
                cost_tracker['last_reset'] = datetime.now()
            
            # Update hurricane data with AI predictions
            hurricanes = get_hurricane_data_tropycal()
            
            # Save cache to disk
            with open('ai_cache.pkl', 'wb') as f:
                pickle.dump(ai_predictions_cache, f)
            
            print(f"Update complete. Next update in 6 hours.")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"Error in auto-update: {e}")
        
        # Wait 6 hours
        time.sleep(21600)  # 6 hours in seconds

@app.route('/')
def index():
    """Render the Folium map"""
    hurricane_map = create_hurricane_map()
    map_html = hurricane_map._repr_html_()
    
    hurricanes = get_hurricane_data_tropycal()
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fitch Irick Hurricane Risk Tracker</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="screen-orientation" content="landscape">
        <style>
            html, body {
                height: 100%;
                width: 100%;
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
                overflow: hidden;
            }
            #map {
                position: absolute;
                top: 0;
                bottom: 0;
                right: 0;
                left: 0;
                height: 100vh;
                width: 100vw;
            }
            
            /* Mobile-specific marker sizing WITHOUT zoom scaling */
            @media screen and (max-width: 768px) {
                /* Don't scale markers - let them maintain size for visibility */
                .leaflet-clickable {
                    cursor: pointer;
                }
                
                /* Ensure minimum touch target size */
                .leaflet-marker-icon {
                    min-width: 24px !important;
                    min-height: 24px !important;
                }
                
                /* Make click areas bigger for circle markers */
                .leaflet-interactive {
                    pointer-events: visiblePainted;
                }
            }
            
            /* Landscape orientation lock for mobile */
            @media screen and (max-width: 768px) and (orientation: portrait) {
                #rotate-message {
                    display: flex;
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background-color: rgba(0, 0, 0, 0.95);
                    z-index: 99999;
                    align-items: center;
                    justify-content: center;
                    flex-direction: column;
                }
                #map {
                    display: none;
                }
            }
            
            @media screen and (max-width: 768px) and (orientation: landscape) {
                #rotate-message {
                    display: none;
                }
                #map {
                    display: block;
                }
            }
            
            @media screen and (min-width: 769px) {
                #rotate-message {
                    display: none;
                }
            }
            
            #rotate-message {
                display: none;
                color: white;
                text-align: center;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            }
            
            #rotate-icon {
                font-size: 72px;
                margin-bottom: 20px;
                animation: rotate 2s infinite ease-in-out;
            }
            
            @keyframes rotate {
                0% { transform: rotate(0deg); }
                50% { transform: rotate(90deg); }
                100% { transform: rotate(90deg); }
            }
            
            #rotate-text {
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 10px;
            }
            
            #rotate-subtext {
                font-size: 14px;
                color: #999;
            }
            
            /* Mobile popup sizing - makes popups much smaller on mobile devices */
            @media screen and (max-width: 768px) {
                .leaflet-popup-content {
                    width: 160px !important;
                    max-width: 175px !important;
                    font-size: 11px !important;
                }
                .property-popup {
                    min-width: 150px !important;
                    max-width: 175px !important;
                }
                .property-popup h3 {
                    font-size: 13px !important;
                    margin: 2px 0 !important;
                    padding-bottom: 4px !important;
                }
                .property-popup table {
                    font-size: 10px !important;
                }
                .property-popup td {
                    padding: 3px 0 !important;
                }
                .property-popup div {
                    font-size: 10px !important;
                    padding: 4px !important;
                }
                .property-popup b {
                    font-size: 11px !important;
                }
            }
        </style>
    </head>
    <body>
        <div id="map">{{ map_html|safe }}</div>
        
        <!-- Rotation message for mobile portrait mode -->
        <div id="rotate-message">
            <div id="rotate-icon">📱↻</div>
            <div id="rotate-text">Please Rotate Your Device</div>
            <div id="rotate-subtext">This hurricane tracker works best in landscape mode</div>
        </div>
        
        <script>
            // Auto-refresh every 30 minutes for UI updates
            setTimeout(function() {
                location.reload();
            }, 1800000);
            
            // Force landscape on mobile devices
            if (screen && screen.orientation && screen.orientation.lock) {
                screen.orientation.lock('landscape').catch(function(error) {
                    console.log('Orientation lock failed:', error);
                });
            }
            
            // Handle orientation change
            window.addEventListener('orientationchange', function() {
                if (window.innerWidth < 768) {
                    if (window.orientation === 0 || window.orientation === 180) {
                        // Portrait mode - show rotation message
                        document.getElementById('rotate-message').style.display = 'flex';
                        document.getElementById('map').style.display = 'none';
                    } else {
                        // Landscape mode - show map
                        document.getElementById('rotate-message').style.display = 'none';
                        document.getElementById('map').style.display = 'block';
                    }
                }
            });
            
            // Check initial orientation on mobile
            if (window.innerWidth < 768) {
                if (window.orientation === 0 || window.orientation === 180) {
                    document.getElementById('rotate-message').style.display = 'flex';
                    document.getElementById('map').style.display = 'none';
                }
            }
            
            // Add touch optimization for mobile
            if ('ontouchstart' in window) {
                document.addEventListener('gesturestart', function (e) {
                    e.preventDefault();
                });
            }
        </script>
    </body>
    </html>
    """
    
    return render_template_string(
        html_template, 
        map_html=map_html,
        hurricanes=hurricanes,
        locations=locations_data
    )

@app.route('/api/hurricanes')
def api_hurricanes():
    """API endpoint for hurricane data with AI predictions"""
    hurricanes = get_hurricane_data_tropycal()
    return jsonify(hurricanes)

@app.route('/api/locations')
def api_locations():
    """API endpoint for location data"""
    return jsonify(locations_data)

@app.route('/api/status')
def api_status():
    """API endpoint for system status"""
    return jsonify({
        'last_update': datetime.now().isoformat(),
        'cost_today': cost_tracker['total_cost'],
        'requests_today': cost_tracker['requests_today'],
        'cached_storms': len(ai_predictions_cache),
        'api_mode': 'OpenAI' if openai.api_key else 'Free Climatological'
    })

# Initialize app
print("Fitch Irick Hurricane Risk Tracker - AI Enhanced")
print("=" * 50)
print("Loading property portfolio with complete data...")
locations_data = load_properties_from_excel()
print(f"Loaded {len(locations_data)} properties")

# Load cache if exists
try:
    with open('ai_cache.pkl', 'rb') as f:
        ai_predictions_cache = pickle.load(f)
        print(f"Loaded {len(ai_predictions_cache)} cached AI predictions")
except:
    print("No cache found, starting fresh")

# Check API configuration
if openai.api_key:
    print("✓ OpenAI API configured - Using GPT-3.5-turbo ($0.001/request)")
    print("  Estimated cost: $2-5/year for typical usage")
else:
    print("✗ No OpenAI API key found - Using FREE climatological model")
    print("  To enable AI: Set OPENAI_API_KEY in .env file")

print("\nStarting auto-update thread (every 6 hours)...")
update_thread = threading.Thread(target=auto_update_loop, daemon=True)
update_thread.start()

print("\nFitch Irick Hurricane Risk Tracker Features:")
print("- Mobile-optimized with larger touch-friendly markers (11-12px)")
print("- Compact mobile popups (175px max width)")
print("- Complete historical hurricane track data")
print("- Enhanced property popups with TIV and deductibles")
print("- AI-powered hurricane path prediction")
print("- Automatic updates every 6 hours")
print("- iOS-themed interface design")
print("\nEndpoints:")
print("- / - Main map interface")
print("- /api/hurricanes - Hurricane JSON with AI predictions")
print("- /api/locations - Property JSON data")
print("- /api/status - System status and costs")
print("=" * 50)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
