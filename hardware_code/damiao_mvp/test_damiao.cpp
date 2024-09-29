#include "damiao.h"
#include <cmath>
#include <thread> // 包含 <thread> 标头以使用 std::this_thread::sleep_for

int main(int argc , char** argv)
{
  auto serial = std::make_shared<SerialPort>("/dev/ttyACM0", B921600);
  // auto serial = std::make_shared<SerialPort>("/dev/ttyACM0", B1000000);
  auto dm = std::make_shared<damiao::Motor>(serial);

  dm->addMotor(0x01, 0x00); // default id
  dm->addMotor(0x02, 0x00); // default id
  // dm->addMotor(0x02, 0x00);

  // dm->clear_error(0x01);
  // std::this_thread::sleep_for(std::chrono::microseconds(500));
  // dm->clear_error(0x02);
  // std::this_thread::sleep_for(std::chrono::microseconds(500));

  // dm->reset(0x01);
  // dm->reset(0x02);
  
  // sleep(0.5);
  


  // std::string confirmation;
  // std::cout << "请确认大拇指指天后按 y 继续：";
  // std::cin >> confirmation;

  // if (confirmation != "y") {
  //     std::cout << "确认失败，程序退出" << std::endl;
  //     return 1; // 或者采取其他适当的措施
  // }

  // dm->zero_position(0x01);
  // dm->zero_position(0x02);

  // sleep(0.5);
  //dm->enable(0x01);
  // dm->reset(0x02);
  // dm->enable(0x01);
  sleep(0.5);
  dm->enable(0x01);
  sleep(0.5);
  dm->enable(0x02);


  while (true)
  {
    float q = sin(std::chrono::system_clock::now().time_since_epoch().count() / 1e9);

    // dm->control(0x01, 0, 0, 0, 0, 0);
    // dm->control(0x02, 0, 0, 0, 0, 0);
    std::cout << q << std::endl;
    dm->control(0x01, 7.5, 0.4, -q*1.5, 0, 0);
    // dm->control(0x02, 5, 0.3, q*12, 0, 0);
    // 添加延迟
    std::this_thread::sleep_for(std::chrono::microseconds(500));
    dm->control(0x02, 7.5, 0.4, q*1.5, 0, 0);

    auto & m1 = dm->motors[0x01];
    std::cout << "m1: " << m1->state.q << " " << m1->state.dq << " " << m1->state.tau << std::endl;
    // 添加延迟
    std::this_thread::sleep_for(std::chrono::microseconds(500));
    auto & m2 = dm->motors[0x02];
    std::cout << "m2: " << m2->state.q << " " << m2->state.dq << " " << m2->state.tau << std::endl;
    sleep(0.10);
  }

  return 0;
}