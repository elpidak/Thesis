import glob
import os
import sys
import time
import numpy as np
from datetime import datetime

try:
     sys.path.append(glob.glob('.\carla\carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

def generate_waypoints(world, starting_waypoint, total_distance):
    waypoints = [starting_waypoint]
    current_waypoint = starting_waypoint
    distance_accumulated = 0

    while distance_accumulated < total_distance:
       
        if current_waypoint.road_id == 391 and current_waypoint.lane_id == -1:
            
            next_waypoints = current_waypoint.next_until_lane_end(2.0)
        else:
            next_waypoints = current_waypoint.next(2.0)

        if next_waypoints:
            if distance_accumulated <  650:
                next_waypoint = next_waypoints[-1]  
            else :
                next_waypoint = current_waypoint.next(1.0)[0]
            waypoints.append(next_waypoint)
            distance_accumulated += next_waypoint.transform.location.distance(current_waypoint.transform.location)
            current_waypoint = next_waypoint
        else:
            
            print("No further waypoints found; accumulated distance is: ", distance_accumulated)
            break

    return waypoints


def draw_waypoints_old(world, route_waypoints):
    # Draw the route waypoints
  
   
        
    for i in range(len(lap_waypoints) - 1):
        world.debug.draw_line(
            lap_waypoints[i].transform.location + carla.Location(z=0.25),
            lap_waypoints[i+1].transform.location + carla.Location(z=0.25),
            thickness=1.0, color=carla.Color(0, 255, 0), life_time=120.0,
            persistent_lines=True)

def draw_waypoints(world, route_waypoints):
    # Draw the route waypoints
    for i in range(len(route_waypoints) - 1):
        world.debug.draw_line(
            route_waypoints[i].transform.location + carla.Location(z=0.25),
            route_waypoints[i+1].transform.location + carla.Location(z=0.25),
            thickness=1.0, color=carla.Color(255, 0, 0), life_time=120.0,
            persistent_lines=True)
    
    # Mark the starting waypoint with a blue mark and the word 'Start'
    start_location = route_waypoints[0].transform.location + carla.Location(z=0.25)
    world.debug.draw_point(
        start_location, size=0.1, color=carla.Color(0, 0, 255), life_time=120.0,
        persistent_lines=True)
    world.debug.draw_string(
        start_location, 'Start', draw_shadow=False,
        color=carla.Color(0, 0, 255), life_time=120.0, persistent_lines=True)

    # Mark the ending waypoint with a green circle and the word 'End'
    end_location = route_waypoints[-1].transform.location + carla.Location(z=0.25)
    world.debug.draw_point(
        end_location, size=0.1, color=carla.Color(0, 255, 0), life_time=120.0,
        persistent_lines=True)
    world.debug.draw_string(
        end_location, 'End', draw_shadow=False,
        color=carla.Color(0, 255, 0), life_time=120.0, persistent_lines=True)
    
    circle_radius = 0.5  # This is a small radius, you can adjust as needed
    number_of_points = 30  # This is the number of points to form the circle
    for i in range(number_of_points):
        angle = i * 2 * np.pi / number_of_points
        dx = circle_radius * np.cos(angle)
        dy = circle_radius * np.sin(angle)
        world.debug.draw_point(
            start_location + carla.Location(x=dx, y=dy),
            size=0.05, color=carla.Color(0, 0, 255), life_time=120.0,
            persistent_lines=True)
        world.debug.draw_point(
            end_location + carla.Location(x=dx, y=dy),
            size=0.05, color=carla.Color(0, 255, 0), life_time=120.0,
            persistent_lines=True)

    
# Connect to the CARLA server
client = carla.Client("localhost", 2000)
client.set_timeout(20.0)
world = client.get_world()

# Choose the map
world = client.load_world('Town02')

weather = carla.WeatherParameters(
    cloudiness=0.0,
    precipitation=0.0,
    sun_altitude_angle=70.0  # Adjust this for the sun's angle
)
world.set_weather(weather)

# Starting location from your training code

starting_transform = world.get_map().get_spawn_points()[1] # Town02 starting point
starting_waypoint = world.get_map().get_waypoint(starting_transform.location)

# Distance for the lap from your training code
total_distance = 780  # Total distance for a lap in Town02

# Generate waypoints for the perfect lap
lap_waypoints = generate_waypoints(world, starting_waypoint, total_distance)


# Draw the route
draw_waypoints(world, lap_waypoints)
# # Draw the route
# draw_waypoints(world, lap_waypoints)

# Wait for a moment to let the server draw the waypoints
time.sleep(2)

# Capture the screenshot
spectator = world.get_spectator()
spectator.set_transform(carla.Transform(starting_waypoint.transform.location + carla.Location(z=50),
                                        carla.Rotation(pitch=-90)))


time.sleep(10)

