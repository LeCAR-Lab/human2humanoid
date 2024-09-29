#include "damiao_mvp/damiao.h"
#include <cmath>
#include <thread> // 包含 <thread> 标头以使用 std::this_thread::sleep_for
#include <csignal>

// ros msg
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <std_msgs/Float64MultiArray.h>


float left_wrist_q = 0;
float right_wrist_q = 0;

bool g_shutdown_requested = false;

// Signal handler for Ctrl+C
void signalHandler(int signum)
{
  if (signum == SIGINT)
  {
    ROS_INFO("Ctrl+C received. Shutting down gracefully...");
    g_shutdown_requested = true;
  }
}


void h1_damiao_callback(const std_msgs::Float64MultiArray::ConstPtr& msg)
{
  // ROS_INFO("I heard: [%s]", msg->data.c_str());
  std::cout << "I heard: "  << msg->data[0] << msg->data[1] << std::endl;
  // std::cout << "I heard: " << msg->data[0] << " " << msg->data[1] << " " << msg->data[2] << " " << msg->data[3] << " " << msg->data[4] << std::endl;
  left_wrist_q = msg->data[0];
  right_wrist_q = msg->data[1];
}

int cnt = 0;
int main(int argc , char** argv)
{ 
  ros::init(argc, argv, "damiao_command");
  // Set up the signal handler
  signal(SIGINT, signalHandler);
  ros::NodeHandle n;
  ros::Subscriber h1_damiao_subscriber = n.subscribe("damiao_command", 1, h1_damiao_callback);
  
  auto serial = std::make_shared<SerialPort>("/dev/ttyACM0", B921600);
  // auto serial = std::make_shared<SerialPort>("/dev/ttyACM0", B1000000);
  auto dm = std::make_shared<damiao::Motor>(serial);

  dm->addMotor(0x01, 0x00); // default id
  dm->addMotor(0x02, 0x00); // default id

  sleep(0.5);
  dm->enable(0x01);
  sleep(0.5);
  dm->enable(0x02);

  


  while (ros::ok() && !g_shutdown_requested)
  {
    ros::spinOnce();
    float q = sin(std::chrono::system_clock::now().time_since_epoch().count() / 1e9);

    // dm->control(0x01, 0, 0, 0, 0, 0);
    // dm->control(0x02, 0, 0, 0, 0, 0);
    
    // std::cout << q << std::endl;
    // dm->control(0x01, 7.5, 0.4, left_wrist_q, 0, 0);
    // // dm->control(0x02, 5, 0.3, q*12, 0, 0);

    // std::this_thread::sleep_for(std::chrono::microseconds(500));
    // dm->control(0x01, 7.5, 0.4, left_wrist_q, 0, 0);

    dm->control(0x01, 5, 0.4, -1.5*right_wrist_q, 0, 0);
    dm->control(0x02, 5, 0.4, 1.5*left_wrist_q, 0, 0);
    std::cout << "sending command to left wrist: " << left_wrist_q << std::endl;
    std::cout << "sending command to right wrist: " << right_wrist_q << std::endl;
    
    auto & m1 = dm->motors[0x01];
    std::cout << "m1: " << m1->state.q << " " << m1->state.dq << " " << m1->state.tau << std::endl;

    std::this_thread::sleep_for(std::chrono::microseconds(500));
    auto & m2 = dm->motors[0x02];
    std::cout << "m2: " << m2->state.q << " " << m2->state.dq << " " << m2->state.tau << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    cnt++;
    std::cout << "cnt: " << cnt << std::endl;
  }

  return 0;
}