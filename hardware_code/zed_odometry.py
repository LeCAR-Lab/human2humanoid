import pyzed.sl as sl
import math
import numpy as np
import sys
import math
import cv2
import time
import torch

#for ros image publish
import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Header, Float64MultiArray
from lpf import ActionFilterButter, ActionFilterButterTorch

def compute_xyz_vel_in_camera_frame(vel_xyz_world, world_rotation):
    vel_x_in_world = vel_xyz_world[0]
    vel_y_in_world = vel_xyz_world[1]
    rotation_z_in_world = world_rotation[2]

    vel_x_in_camera = vel_x_in_world * np.cos(rotation_z_in_world) + vel_y_in_world * np.sin(rotation_z_in_world)
    vel_y_in_camera = -vel_x_in_world * np.sin(rotation_z_in_world) + vel_y_in_world * np.cos(rotation_z_in_world)

    compute_xy_vel_in_camera_frame = np.array([vel_x_in_camera, vel_y_in_camera, vel_xyz_world[2]])
    return compute_xy_vel_in_camera_frame

class NNLinvelInfo:
    def __init__(self) -> None:
        self.get_info = False
        self.nn_linvel = np.zeros((1, 3), dtype=np.float64)
    
    def nn_linvel_callback(self, msg):
        
        self.get_info = True
        self.nn_linvel = np.array(msg.data, dtype=np.float64).reshape(1, 3)

    def check(self):
        if not self.get_info:
            rospy.logwarn_throttle(1, "No nn_linvel info received")



def main():
    USE_LPF_FOR_ZED = False
    PLOT_NN_LINVEL = True
    lpf_filter = ActionFilterButter(lowcut=np.zeros(1*3),
                                    highcut=np.ones(1*3) * 4.0, 
                                    sampling_rate=60.0, num_joints=1*3)
    
    #ros init
    rospy.init_node('zed_odometry', anonymous=True)
    pub_position_xyz_linvel_xyz_rotation_z = rospy.Publisher('zed_odometry', Float64MultiArray, queue_size=1)
    nnlinvelinfo = NNLinvelInfo()
    rospy.Subscriber("nn_linvel", Float64MultiArray, nnlinvelinfo.nn_linvel_callback, queue_size=1)
    nnlinvelinfo_gru = NNLinvelInfo()
    rospy.Subscriber("nn_linvel_gru", Float64MultiArray, nnlinvelinfo_gru.nn_linvel_callback, queue_size=1)
    bridge = CvBridge()



    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init_params.depth_maximum_distance = 6       # Set the maximum depth perception distance to 40m
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Use ULTRA depth mode
    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.enable_fill_mode = True

    i = 0
    depth = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))


    tracking_params = sl.PositionalTrackingParameters() #set parameters for Positional Tracking
    tracking_params.enable_imu_fusion = True 
    tracking_params.enable_area_memory = False
    tracking_params.enable_pose_smoothing = True
    status = zed.enable_positional_tracking(tracking_params) #enable Positional Tracking
    if status != sl.ERROR_CODE.SUCCESS:
        print("Enable Positional Tracking : "+repr(status)+". Exit program.")
        zed.close()
        exit()

    camera_pose = sl.Pose()
    camera_info = zed.get_camera_information()

    py_translation = sl.Translation()
    pose_data = sl.Transform()
    last_translation = np.array([0, 0, 0])
    last_timestamp = 0

    
    depth_sensing_time_total = 0
    velo_estimate_time_total = 0
    model_inference_time_total = 0
    zed_time_total = 0
    ros_pub_time_total = 0
    time_start = time.time()
    rate = rospy.Rate(100)
    last_vel_xyz_camera = np.array([0, 0, 0])
    lin_vel_to_plot_raw = []
    liv_vel_to_plot_smooth = []
    liv_vel_to_plot_lpf = []
    nn_linvel_to_plot = []
    nn_linvel_to_plot_gru = []
    while True:
        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed_time_start = time.time()
            i += 1    
            velo_estimate_time_start = time.time()
            tracking_state = zed.get_position(camera_pose,sl.REFERENCE_FRAME.WORLD) #Get the position of the camera in a fixed reference frame (the World Frame)
            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                #Get rotation and translation and displays it
                rotation = camera_pose.get_rotation_vector()
                translation = camera_pose.get_translation(py_translation)
                timestamp = camera_pose.timestamp.get_nanoseconds()
                
                time_now = time.time()
                FPS = i / (time_now - time_start)
                if i == 0:  
                    last_translation = np.array([translation.get()[0], translation.get()[1], translation.get()[2]])
                    last_timestamp = timestamp
                else:
                    # print(timestamp)
                    dt = (timestamp - last_timestamp) / 1e9
                    # dt = 1/60.
                    vel_xyz_world = np.array([translation.get()[0] - last_translation[0], translation.get()[1] - last_translation[1], translation.get()[2] - last_translation[2]]) / dt


                    vel_xyz_camera = compute_xyz_vel_in_camera_frame(vel_xyz_world, rotation)
                    if USE_LPF_FOR_ZED:
                        vel_xyz_world_lpf = lpf_filter.filter(vel_xyz_camera)
                        liv_vel_to_plot_lpf.append(vel_xyz_world_lpf)

                    lin_vel_to_plot_raw.append(vel_xyz_camera)

                    smooth_vel_xyz_camera_ = 0.5 * last_vel_xyz_camera + 0.5 * vel_xyz_camera
                    smooth_vel_xyz_camera = smooth_vel_xyz_camera_
                    liv_vel_to_plot_smooth.append(smooth_vel_xyz_camera_)
                    if PLOT_NN_LINVEL:
                        nn_linvel_to_plot.append(nnlinvelinfo.nn_linvel[0].copy())
                        nn_linvel_to_plot_gru.append(nnlinvelinfo_gru.nn_linvel[0].copy())
                    last_vel_xyz_camera = smooth_vel_xyz_camera_
                    last_translation = np.array([translation.get()[0], translation.get()[1], translation.get()[2]])
                    last_timestamp = timestamp
                    # print("vel_xy_camera: ", vel_xy_camera)
            velo_estimate_time_end = time.time()
            velo_estimate_time_total += (velo_estimate_time_end - velo_estimate_time_start)



            zed_time_end = time.time()
            zed_time_total += zed_time_end - zed_time_start
            

           
            
            ros_pub_time_start = time.time()

            
            position_xyz_linvel_xyz_rotation_z_msg = Float64MultiArray()
            
            position_xyz_linvel_xyz_rotation_z_data = list(np.array([translation.get()[0], translation.get()[1], translation.get()[2]+0.95, 
                                                                     smooth_vel_xyz_camera[0], smooth_vel_xyz_camera[1], smooth_vel_xyz_camera[2],
                                                                     rotation[2]]).reshape(-1))
                                                                     # rotation[2]-3.14/2]).reshape(-1))
            # position_xy_rotation_z_data = list(np.array([1, 0, 0]).reshape(-1))
            position_xyz_linvel_xyz_rotation_z_msg.data = position_xyz_linvel_xyz_rotation_z_data
            pub_position_xyz_linvel_xyz_rotation_z.publish(position_xyz_linvel_xyz_rotation_z_msg)

            

            if i % 100 == 0:
                print(i)
                time_sofar = time.time() - time_start
                print("Time so far: ", time_sofar)
                print("ZED FPS: ", i/zed_time_total)
                print("Velo Estimation FPS: ", i/velo_estimate_time_total)
                print("ROS Pub FPS: ", i/ros_pub_time_total)
                print("FPS: ", i/time_sofar)  
            ros_pub_time_end = time.time()
            ros_pub_time_total += ros_pub_time_end - ros_pub_time_start                                                                 
            
            
            
            rate.sleep()




    zed.close()

if __name__ == "__main__":

    main()