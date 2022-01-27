import CoDrone
from CoDrone import Direction

print("Creating drone object")
drone = CoDrone.CoDrone()
print("Getting ready to pair")
drone.pair(drone.Nearest)
print("Paired!")


def straight_land(start_coords, land_coords = [0,0,0]):
    height_threshold = 80 # height before drone.land()
    # x:left-right, y:height, z:depth
    xs,ys,zs = start_coords
    xl,yl,zl = land_coords

    xs, ys, zs = xs*1000,ys*1000,zs*1000
    xl, yl, zl = xl*1000,yl*1000,zl*1000

    h = drone.get_height()  # relative to the ground
    print('Before pitch')
    print(h)

    pitch_p = zs/(20*3)  # 20 is the scale factor for pitch, assume while loop repeat 3 times
    # for now assume that camera is on the ground, drone is starting out higher than camera
    throttle_p = -50 #-ys/9  # 7 is the scale factor for throttle
    while h > height_threshold:
        drone.set_pitch(pitch_p)
        drone.move(1)

        h = drone.get_height()
        if h <= height_threshold:
            break

        throttle_p = throttle_p + 3
        if throttle_p > -15:
            drone.land()
            break
        else:
            drone.set_throttle(throttle_p)
            drone.move(1)
        # print(drone.get_height())

        h = drone.get_height()
        print('After throttle')
        print(h)

    drone.land()
    print("landing")


height = drone.get_height()
print('before takeoff', height)
drone.takeoff()
print("taking off")

height = drone.get_height()
print('after takeoff, before land', height)

height = drone.get_height()  # in millimeter
xs,ys,zs = 0,height/1000,1  # in meter
start_coords = [xs,ys,zs]

straight_land(start_coords)
height = drone.get_height()
print('after land', height)

drone.close()