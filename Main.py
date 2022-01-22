import CoDrone
from CoDrone import Direction
print("Creating drone object")
drone = CoDrone.CoDrone()
print("Getting ready to pair")
drone.pair(drone.Nearest)
print("Paired!")

height = drone.get_height()
print(height)
drone.takeoff()

print("taking off")

# FORWARD AND LANDING
h = drone.get_height()
print('Before pitch')
print(h)

throttle_p = -25
while h > 50:
    drone.set_pitch(12)
    drone.move(1)

    h = drone.get_height()
    if h <= 50:
        break

    # drone.set_pitch(0)
    # drone.move(1)
    #
    # h = drone.get_height()
    # if h <= 50:
    #     break

    #print(drone.get_height())
    throttle_p = throttle_p + 3
    if throttle_p > -15:
        drone.land()
        break
    else:
        drone.set_throttle(throttle_p + 3)
        drone.move(1)
    #print(drone.get_height())

    h = drone.get_height()
    print('After throttle')
    print(h)



# SQUARE
# drone.go(Direction.FORWARD, 1, 60)    # moves the drone forward for 2 seconds at 30% power
# drone.go(Direction.LEFT, 1, 60)            # moves the drone left for 2 seconds at 30 power
# drone.go(Direction.BACKWARD, 1, 60)    # moves the drone backward for 2 seconds a 30% power
# drone.go(Direction.RIGHT, 1, 60)            # moves the drone right for 2 seconds at 30% power

# TESTING PITCH, ROLL, YAW
# height = drone.get_height()
# print(height)
# angles = drone.get_gyro_angles()
# print(angles.ROLL, angles.PITCH, angles.YAW)
# drone.hover(2)
# height = drone.get_height()
# print(height)
# # drone.go(Direction.DOWN, 2, 50)
# drone.set_roll(-30)
# angles = drone.get_gyro_angles()
# print(angles.ROLL, angles.PITCH, angles.YAW)
# drone.move(1)
# angles = drone.get_gyro_angles()
# print(angles.ROLL, angles.PITCH, angles.YAW)
#
# #drone.set_roll(0)
# angles = drone.get_gyro_angles()
# print(angles.ROLL, angles.PITCH, angles.YAW)
# #drone.move(1)
# angles = drone.get_gyro_angles()
# print(angles.ROLL, angles.PITCH, angles.YAW)
#
# drone.set_roll(30)
# angles = drone.get_gyro_angles()
# print(angles.ROLL, angles.PITCH, angles.YAW)
# drone.move(1)
# angles = drone.get_gyro_angles()
# print(angles.ROLL, angles.PITCH, angles.YAW)
# height = drone.get_height()
# print(height)
# # drone.hover(2)
# height = drone.get_height()
# print(height)

# TEST HEIGHT MOVEMENT
# for i in range(3):
#     height = drone.get_height()
#     print(height)
#     drone.go(Direction.UP, 2, 20)
#     height = drone.get_height()
#     if height > 100:
#         drone.go(Direction.DOWN, 0, 50)
#     if height < 50:
#         drone.go(Direction.UP, 2, 50)
#     # sleep(0.1)
drone.land()
# print("Hovering")
# # drone.set_pitch(45)
# angles = drone.get_gyro_angles()
# print(angles.ROLL, angles.PITCH, angles.YAW)
# drone.hover(10)
# angles = drone.get_gyro_angles()
# print(angles.ROLL, angles.PITCH, angles.YAW)# Set positive pitch to 30% power
# # drone.move(3)            # forward for 2 seconds
# drone.land()
print("landing")
drone.close()

