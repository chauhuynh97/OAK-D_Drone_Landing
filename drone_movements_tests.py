import CoDrone
from CoDrone import Direction

drone = CoDrone.CoDrone()
drone.pair(drone.Nearest)
drone.set_arm_led(CoDrone.Mode.OFF)

# drone.pair() if paired to CoDrone before
drone.takeoff()            # takeoff for 2 seconds

#drone.go_to_height(500)
drone.set_pitch(15)
while True:
    #drone.go(Direction.FORWARD, 0, 50)
    drone.move()
    if input() == 'q':
        break

print('Outside of loop')
drone.go_to_height(200)
drone.land()            # lands the CoDrone
drone.close()           # disconnects CoDrone

