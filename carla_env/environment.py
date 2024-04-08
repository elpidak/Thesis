import time
import random
import numpy as np
import pygame
from carla_env.client import carla
from carla_env.sensors import SemanticSegmentationCamera, RgbCamera, CollisionSensor


pedestrians = 10
car = 'c3'

class CarlaEnvironment():

    def __init__(self, client, world, town, checkpoint_frequency=100, continuous_action=True) -> None:

        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.action_space = self.get_discrete_action_space()
        self.continous_action_space = continuous_action
        self.visual_display = True
        self.vehicle = None
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0
        self.episode_start=True
        self.checkpoint_frequency = checkpoint_frequency
        self.route_waypoints = None
        self.town = town
        self.sensors = list()
        self.actors = list()
        self.walker_list = list()
        self.create_pedestrians()
        self.camera_obj = None
        self.env_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None

    def reset(self):

        try:
            if len(self.actors) != 0 or len(self.sensors) != 0:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensors])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actors])
                self.sensors.clear()
                self.actors.clear()
            self.remove_sensors()

            vehicle = self.get_vehicle(car)
            if self.town == "Town02":
                transform = self.map.get_spawn_points()[1] 
                self.total_distance = 780
            else:
                transform = random.choice(self.map.get_spawn_points())
                self.total_distance = 250

            self.vehicle = self.world.try_spawn_actor(vehicle, transform)
            self.actors.append(self.vehicle)
            self.camera_obj = SemanticSegmentationCamera(self.vehicle)
            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0001)
            self.image_obs = self.camera_obj.front_camera.pop(-1)
            self.sensors.append(self.camera_obj.sensor)
            if self.visual_display:
                self.env_camera_obj = RgbCamera(self.vehicle)
                self.sensors.append(self.env_camera_obj.sensor)
            self.collision_obj = CollisionSensor(self.vehicle)
            self.collisions = self.collision_obj.collision_data
            self.sensors.append(self.collision_obj.sensor)

            self.timesteps = 0      
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.previous_location = self.vehicle.get_location()
            self.distance_traveled = 0.0
            self.center_lane_deviation = 0.0
            self.target_speed = 22 
            self.max_speed = 40.0
            self.min_speed = 15.0
            self.max_distance_from_center = 2
            self.throttle = float(0.0)
            self.previous_steer = float(0.0)
            self.velocity = float(0.0)
            self.distance_from_center = float(0.0)
            self.angle = float(0.0)
            self.center_lane_deviation = 0.0
            self.distance_covered = 0.0
            self.fuel_consumption_average =0
            self.fuel_consumption_sum= 0
            #fuel_consumption = 0.0086x^2+0.083x+0.2001 where x is %
            self.min_fuel_consumption=0.2001 #throttle = 0, 0%
            self.max_fuel_consumption = 94.5001 #throttle = 1, 100%      
            self.max_steering_diff = 0.1  
            self.jerk_penalty_weight = 1.0 
            self.previous_steer = 0.0
            self.MAX_STEERING_DIFF = self.max_steering_diff
            self.MAX_STEERING = 1
            self.MIN_STEERING = - 1
            

            if self.episode_start:
                self.current_waypoint_index = 0
                self.route_waypoints = list()
                self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
                current_waypoint = self.waypoint
                self.route_waypoints.append(current_waypoint)
                for x in range(self.total_distance):
                    if self.town == "Town02":
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[0]
                    else:
                        next_waypoint = current_waypoint.next(1.0)[0]
                    self.route_waypoints.append(next_waypoint)
                    current_waypoint = next_waypoint
            else:
                waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
                transform = waypoint.transform
                self.vehicle.set_transform(transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index

            self.navigation_obs = np.array([self.throttle, self.velocity, self.previous_steer, self.distance_from_center, self.angle])
            time.sleep(0.5)
            self.collisions.clear()

            self.episode_start_time = time.time()
            return [self.image_obs, self.navigation_obs]

        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensors])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actors])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensors.clear()
            self.actors.clear()
            self.remove_sensors()
            if self.visual_display:
                pygame.quit()

    def step(self, action_idx):
        try:

            self.timesteps+=1
            self.episode_start = False
            velocity = self.vehicle.get_velocity()
            self.velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            
            if self.continous_action_space:
                steer = float(action_idx[0])
                steer = max(min(steer, 1.0), -1.0)
                steer_change = abs(steer - self.previous_steer)
                max_diff = (self.MAX_STEERING_DIFF - 1e-5) * (self.MAX_STEERING - self.MIN_STEERING)
                if steer_change > self.MAX_STEERING_DIFF:
                    jerk_penalty = self.jerk_penalty_weight * (steer_change - self.MAX_STEERING_DIFF)**2
                    jerk_penalty_factor = 1 - self.jerk_penalty_weight * (steer_change - self.MAX_STEERING_DIFF)**2
                  
                else:
                    jerk_penalty = 0
                    jerk_penalty_factor = 1

                jerk_penalty_factor = max(jerk_penalty_factor, 0)
                self.previous_steer = steer    
                throttle = float((action_idx[1] + 1.0)/2)
                throttle = max(min(throttle, 1.0), 0.0)

                steer_change = steer - self.previous_steer
                diff = np.clip(steer_change, -max_diff, max_diff)
                controlled_steer = self.previous_steer + diff

                self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 +controlled_steer*0.1,throttle=self.throttle*0.9 + throttle*0.1))
                self.throttle = throttle
                
            else:
                steer = self.action_space[action_idx]
                if self.velocity < 20.0:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, throttle=1.0))
                else:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1))
                self.previous_steer = steer
                self.throttle = 1.0
            
            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)

            self.collisions = self.collision_obj.collision_data            
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.location = self.vehicle.get_location()
            waypoint_index = self.current_waypoint_index

            for _ in range(len(self.route_waypoints)):
                next_waypoint_index = waypoint_index + 1
                wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
                dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],self.vector(self.location - wp.transform.location)[:2])
                if dot > 0.0:
                    waypoint_index += 1
                else:
                    break

            self.current_waypoint_index = waypoint_index
            self.current_waypoint = self.route_waypoints[ self.current_waypoint_index    % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
            self.distance_from_center = self.distance_from_the_center(self.vector(self.current_waypoint.transform.location),self.vector(self.next_waypoint.transform.location),self.vector(self.location))
            self.center_lane_deviation += self.distance_from_center

            fwd = self.vector(self.vehicle.get_velocity())
            wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
            self.angle  = self.angle_difference(fwd, wp_fwd)
            if not self.episode_start:
                if self.checkpoint_frequency is not None:
                    self.checkpoint_waypoint_index = (self.current_waypoint_index // self.checkpoint_frequency) * self.checkpoint_frequency

 
            epsisode_finished = False
            reward = 0
            if len(self.collisions) != 0:
                epsisode_finished = True
                reward = -10
                
            elif self.velocity > self.max_speed:
                reward = -10
                epsisode_finished = True
               
            elif self.episode_start_time + 10 < time.time() and self.velocity < 1.0:
                reward = -10
                epsisode_finished = True

            elif self.distance_from_center > self.max_distance_from_center:
                epsisode_finished = True
                reward = -10

               
            angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)
            centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)

            if not epsisode_finished:
                
                fuel_consumption =  0.0086* (abs(self.throttle*100)**2)+0.083* abs(self.throttle*100) +0.2001  
                normalized_fuel_consumption = (fuel_consumption - self.min_fuel_consumption) / (self.max_fuel_consumption - self.min_fuel_consumption) 
                self.fuel_consumption_sum +=fuel_consumption   
                
                if self.continous_action_space:
                    if self.velocity < self.min_speed:
                        reward = (self.velocity / self.min_speed) * centering_factor * angle_factor * (1 - normalized_fuel_consumption)
                    elif self.velocity > self.target_speed:               
                        reward = (1.0 - (self.velocity-self.target_speed) / (self.max_speed-self.target_speed)) * centering_factor * angle_factor *  (1 - normalized_fuel_consumption)
                    else:  
                        reward = 1.0* centering_factor * angle_factor * (1 - normalized_fuel_consumption)
                else:
                    reward =  1.0* centering_factor * angle_factor * (1 - normalized_fuel_consumption)
                reward = reward*jerk_penalty_factor

            if self.timesteps >= 7500:
                epsisode_finished = True
            elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
                epsisode_finished = True
                self.episode_start = True
                if self.checkpoint_frequency is not None:
                    if self.checkpoint_frequency < self.total_distance//2:
                        self.checkpoint_frequency += 2
                    else:
                        self.checkpoint_frequency = None
                        self.checkpoint_waypoint_index = 0

            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0001)

            self.image_obs = self.camera_obj.front_camera.pop(-1)
            normalized_velocity = self.velocity/self.target_speed
            normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
            normalized_angle = abs(self.angle / np.deg2rad(20))
            self.navigation_obs = np.array([self.throttle, self.velocity, normalized_velocity, normalized_distance_from_center, normalized_angle])
            
            if epsisode_finished:
                self.center_lane_deviation = self.center_lane_deviation / self.timesteps
                self.distance_covered = abs(self.current_waypoint_index - self.checkpoint_waypoint_index)
                self.fuel_consumption_average = self.fuel_consumption_sum / self.timesteps     
                for sensor in self.sensors:
                    sensor.destroy()
                
                self.remove_sensors() 
                for actor in self.actors:
                    actor.destroy()
            
            return [self.image_obs, self.navigation_obs], reward, epsisode_finished, [self.distance_covered, self.center_lane_deviation, self.fuel_consumption_average]
       
        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensors])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actors])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensors.clear()
            self.actors.clear()
            self.remove_sensors()
            if self.visual_display:
                pygame.quit()


    def create_pedestrians(self):
        try:
            walker_spawn_points = []
            for i in range(pedestrians):
                spawn_point_ = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point_.location = loc
                    walker_spawn_points.append(spawn_point_)
            for spawn_point_ in walker_spawn_points:
                walker_bp = random.choice(
                    self.blueprint_library.filter('walker.pedestrian.*'))
                walker_controller_bp = self.blueprint_library.find(
                    'controller.ai.walker')
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute(
                        'speed', (walker_bp.get_attribute('speed').recommended_values[1]))
                else:
                    walker_bp.set_attribute('speed', 0.0)
                walker = self.world.try_spawn_actor(walker_bp, spawn_point_)
                if walker is not None:
                    walker_controller = self.world.spawn_actor(
                        walker_controller_bp, carla.Transform(), walker)
                    self.walker_list.append(walker_controller.id)
                    self.walker_list.append(walker.id)
            all_actors = self.world.get_actors(self.walker_list)
            for i in range(0, len(self.walker_list), 2):
                all_actors[i].start()
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())
        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])

    def angle_difference(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi: angle -= 2 * np.pi
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle

    def distance_from_the_center(self, A, B, p):
        num  = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom

    def vector(self, v):
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])

    def get_discrete_action_space(self):
        action_space = \
            np.array([-0.50, -0.30,-0.10, 0.0, 0.10,0.30, 0.50])
        return action_space
  
    def get_vehicle(self, vehicle_name):
        blueprint = self.blueprint_library.filter(vehicle_name)[0]
        if blueprint.has_attribute('color'):
            car_color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', car_color)
        return blueprint

    def remove_sensors(self):
        self.camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.env_camera_obj = None
        self.front_camera = None
        self.collisions = None
        self.wrong_maneuver = None

