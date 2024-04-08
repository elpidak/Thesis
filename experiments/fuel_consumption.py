import glob
import os
import sys
import time
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import pygame
import random

try:
    sys.path.append(glob.glob('.\carla\carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def init_pygame(window_width, window_height):
    pygame.init()
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("CARLA Random Drive Visualization")
    return window

def draw_image(surface, image):
    if image is None:
        return
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface.blit(pygame.surfarray.make_surface(array.swapaxes(0, 1)), (0, 0))

def compute_fuel_consumption(throttle):
    return 0.0086 * (abs(throttle * 100)**2) + 0.083 * abs(throttle * 100) + 0.2001

client = carla.Client("localhost", 2000)
client.set_timeout(20.0)

window_width = 800
window_height = 600
window = init_pygame(window_width, window_height)
clock = pygame.time.Clock()
client.load_world('Town01')
world = client.get_world()
blueprint_library = world.get_blueprint_library()
weather = carla.WeatherParameters(
    cloudiness=0.0,
    precipitation=0.0,
    sun_altitude_angle=70.0  # 
)
world.set_weather(weather)
vehicle_bp = blueprint_library.filter('model3')[0]
vehicle_bp.set_attribute('color', '255,0,0')
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(True)

camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(window_width))
camera_bp.set_attribute('image_size_y', str(window_height))
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

current_image = None

def process_image(image):
    global current_image
    current_image = image

camera.listen(lambda image: process_image(image))

start_time = time.time()
duration_seconds = 3600 
total_fuel_consumption = 0.0
timesteps = 0
writer = SummaryWriter(f"runs/fuel_consumption_second/Town1")
try:
    while True:
        if time.time() - start_time > duration_seconds:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        throttle = vehicle.get_control().throttle
        fuel_consumption = compute_fuel_consumption(throttle)
        total_fuel_consumption += fuel_consumption
        writer.add_scalar('Fuel Consumption/timestep', fuel_consumption, timesteps)
        timesteps += 1

        draw_image(window, current_image)
        pygame.display.flip()
        clock.tick(30)  
       

except KeyboardInterrupt:
    print("\nSimulation interrupted by user.")

finally:
    mean_fuel_consumption = total_fuel_consumption / timesteps if timesteps else 0
    writer.add_scalar('Mean Fuel Consumption', mean_fuel_consumption, 1)
    print(f"Mean Fuel Consumption: {mean_fuel_consumption} per timestep over {timesteps} timesteps")
    camera.destroy()
    vehicle.destroy()
    pygame.quit()

   