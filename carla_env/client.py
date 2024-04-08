import os
import sys
import glob

try:
    sys.path.append(glob.glob('.\carla\carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print('Couldn\'t import Carla egg properly')

import carla

host = "localhost"
port = 2000
timeout = 20.0

class ClientConnection:
    def __init__(self, town):
        self.client = None
        self.town = town

    def setup(self):
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(timeout)
            self.world = self.client.load_world(self.town)
            self.world.set_weather(carla.WeatherParameters.CloudyNoon)
            return self.client, self.world
        except Exception as e:
            print('An error occured: {}'.format(e))
