##timestamp
float64 timestamp 			#ros time
int32 	count               #1,2,3,4,5,...

float64 brake_cmd 			#brake pedal opening [0-1, 100%]
float64 throttle_cmd 		#throttle pedal openning [0-1, 100%]

# Drive wheel angular speed commands, rad/s.
# These two fields are the direct outputs [W_l, W_r] of Learning_preview_controller.
# The bywire driver converts them to motor rpm before encoding to CAN.
float64 left_drive_wheel_speed_cmd
float64 right_drive_wheel_speed_cmd
int16 gear_cmd            # NONE:0, PARK:1, REVERSE:2, NEUTRAL:3, DRIVE:4, LOW:5
int16 turn_signal_cmd     # NONE:0, LEFT:1, RIGHT:2

float64 acceleration_cmd    #acceleration cmd, m/s/s

int16 bywire_control_enable #self-driving enabled, 0-off, 1-on
int16 emerg_brake           #0:normal brake, 1:emergency brake
int16 front_light           #0-off, 1-on

int16 engine_enable         #0-off 1-on
int16 park_enable           #0-off, 1-on
int16 vehicle_mode          #0-pure EV, 1-Hybrid
float64 yaw_speed_cmd       #rad/s


##
float64 ey_out              #tracking error, lateral offset
float64 ephi_out            #tracking error, heading angle
