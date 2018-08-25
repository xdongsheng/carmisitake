#!/usr/bin/env python3
from __future__ import print_function
# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic CARLA client example."""


import numpy as np
import argparse
import logging
import random
import time
from xdsDQN import DeepQNetwork

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from carla.image_converter import depth_to_local_point_cloud, to_rgb_array
import carla.transform as transform

actions=[-1.0,-0.5,0,0.5,1.0,0.5,1.0]
k = 1
def run_carla_client(args):
    # Here we will run 3 episodes with 300 frames each.
    number_of_episodes = 500000
    frames_per_episode = 300000000000000
    global k




    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')


        #########################
        for episode in range(350, number_of_episodes):
            # Start a new episode.
            print(episode)
            if args.settings_filepath is None:

                # Create a CarlaSettings object. This object is a wrapper around
                # the CarlaSettings.ini file. Here we set the configuration we
                # want for the new episode.
                settings = CarlaSettings()
                settings.set(
                    SynchronousMode=True,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=0,
                    NumberOfPedestrians=0,
                    # WeatherId=random.choice([1, 3, 7, 8, 14]),
                    WeatherId=1,
                    QualityLevel=args.quality_level)
                settings.randomize_seeds()

                # Now we want to add a couple of cameras to the player vehicle.
                # We will collect the images produced by these cameras every
                # frame.

                # The default camera captures RGB images of the scene.
                camera0 = Camera('CameraRGB')
                # Set image resolution in pixels.
                camera0.set_image_size(210, 160)
                # Set its position relative to the car in meters.
                camera0.set_position(0.30, 0, 1.30)
                settings.add_sensor(camera0)

                # Let's add another camera producing ground-truth depth.
            #     camera1 = Camera('CameraDepth', PostProcessing='Depth')
            #     camera1.set_image_size(210, 160)
            #     camera1.set_position(0.30, 0, 1.30)
            #     settings.add_sensor(camera1)
            #
            #     if args.lidar:
            #         lidar = Lidar('Lidar32')
            #         lidar.set_position(0, 0, 2.50)
            #         lidar.set_rotation(0, 0, 0)
            #         lidar.set(
            #             Channels=32,
            #             Range=50,
            #             PointsPerSecond=100000,
            #             RotationFrequency=10,
            #             UpperFovLimit=10,
            #             LowerFovLimit=-30)
            #         settings.add_sensor(lidar)
            #
            # else:
            #
            #     # Alternatively, we can load these settings from a file.
            #     with open(args.settings_filepath, 'r') as fp:
            #         settings = fp.read()

            # Now we load these settings into the server. The server replies
            # with a scene description containing the available start spots for
            # the player. Here we can provide a CarlaSettings object or a
            # CarlaSettings.ini file as string.
            scene = client.load_settings(settings)

            # Choose one player start at random.
            #从哪里开始
           # number_of_player_starts = len(scene.player_start_spots)
        # player_start = random.randint(0, max(0, number_of_player_starts - 1))

            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.
            print('Starting new episode at %r...' % scene.map_name)
            #client.start_episode(player_start)
            client.start_episode(0)
            #--------------------------------------------------------------------

            # Iterate every frame in the episode.
            # Read the data produced by the server this frame.
            # 这个应该是获取到的数据
            measurements, sensor_data = client.read_data()

            # Print some of the measurements.
            image_RGB = to_rgb_array(sensor_data['CameraRGB'])
            image_RGB_real=image_RGB.flatten()

            print(image_RGB_real)
            print(image_RGB_real.shape[0])
            ###############################################################################################
            if k==1:
                # player_measurements = measurements.player_measurements
                # loc = np.array([player_measurements.transform.location.x, player_measurements.transform.location.y])
                RL = DeepQNetwork(n_actions=7,
                                  n_features=image_RGB_real.shape[0],
                                  learning_rate=0.0001, e_greedy=0.9,
                                  replace_target_iter=1000, memory_size=2000,
                                  e_greedy_increment=0.000001, )

                total_steps = 0

            else:

                print("rl运行完毕")
            k=2
            while True:

                measurements, sensor_data = client.read_data()
                player_measurements = measurements.player_measurements
                other_lane = 100 * player_measurements.intersection_otherlane
                offroad = 100 * player_measurements.intersection_offroad
                # loc1=np.array([player_measurements.transform.location.x, player_measurements.transform.location.y])
                # print(offroad)
                # print(other_lane)
                # print(loc1)
                image_RGB = to_rgb_array(sensor_data['CameraRGB'])
                image_RGB_real = image_RGB.flatten()
                print_measurements(measurements)
                observation = image_RGB_real
                ep_r = 0
                done = False
                # while True:
                #     # env.render()

                action = RL.choose_action(observation)

                if not args.autopilot:
                    brake1=0.0
                    steer1=0.0
                    if (action > 4):
                        brake1 = actions[action]
                    else:
                        steer1 = actions[action]

                    client.send_control(
                        # steer=random.uniform(-1.0, 1.0),

                        steer=steer1,
                        throttle=0.6,
                        brake=brake1,
                        hand_brake=False,
                        reverse=False)


                else:

                    # Together with the measurements, the server has sent the
                    # control that the in-game autopilot would do this frame. We
                    # can enable autopilot by sending back this control to the
                    # server. We can modify it if wanted, here for instance we
                    # will add some noise to the steer.

                    control = measurements.player_measurements.autopilot_control
                    # control.steer += random.uniform(-0.1, 0.1)
                    client.send_control(control)

                # loc2=np.array([player_measurements.transform.location.x, player_measurements.transform.location.y])
                image_RGB = to_rgb_array(sensor_data['CameraRGB'])
                image_RGB_rea2 = image_RGB.flatten()
                observation_ = image_RGB_rea2
                reward = -other_lane - offroad+np.sqrt(np.square(player_measurements.transform.location.x-271.0)+np.square(player_measurements.transform.location.y-129.5))
                print(offroad)
                col=player_measurements.collision_other
                speed=player_measurements.forward_speed * 3.6
                print(col)
                if offroad > 10 or other_lane>10 or col > 0 :
                    print(111111111111111111111)
                    done = True



                RL.store_transition(observation, action, reward, observation_)

                ep_r += reward
                if total_steps > 100:
                    RL.learn(do_train=1)

                if done:
                    print('episode: ',
                              'ep_r: ', round(ep_r, 2),
                              ' epsilon: ', round(RL.epsilon, 2))

                    # client.start_episode(0)
                    break

                observation = observation_
                total_steps += 1

              #  RL.plot_cost()



                # Save the images to disk if requested.
                if args.save_images_to_disk:
                    for name, measurement in sensor_data.items():
                        filename = args.out_filename_format.format(episode, name, frame)
                        measurement.save_to_disk(filename)
            continue
                # We can access the encoded data of a given image as numpy
                # array using its "data" property. For instance, to get the
                # depth value (normalized) at pixel X, Y
                #
                #     depth_array = sensor_data['CameraDepth'].data
                #     value_at_pixel = depth_array[Y, X]
                #

                # Now we have to send the instructions to control the vehicle.
                # If we are in synchronous mode the server will pause the
                # simulation until we send this control.




def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.0f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed * 3.6, # m/s -> km/h
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print_over_same_line(message)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    while True:
        try:

            run_carla_client(args)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
