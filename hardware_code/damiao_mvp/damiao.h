#ifndef DAMIAO_H
#define DAMIAO_H

#include "SerialPort.h"
#include <vector>
#include <unordered_map>

namespace damiao
{

#pragma pack(1)

typedef struct
{
  uint8_t freamHeader;
  uint8_t CMD;// 命令 0x00: 心跳 
              //     0x01: receive fail 0x11: receive success 
              //     0x02: send fail 0x12: send success
              //     0x03: set baudrate fail 0x13: set baudrate success
              //     0xEE: communication error 此时格式段为错误码
              //           8: 超压 9: 欠压 A: 过流 B: MOS过温 C: 电机线圈过温 D: 通讯丢失 E: 过载
  uint8_t canDataLen: 6; // 数据长度
  uint8_t canIde: 1; // 0: 标准帧 1: 扩展帧
  uint8_t canRtr: 1; // 0: 数据帧 1: 远程帧
  uint32_t CANID; // 电机反馈的ID
  uint8_t canData[8];
  uint8_t freamEnd; // 帧尾
} CAN_Recv_Fream;

typedef struct 
{
  uint8_t freamHeader[2] = {0x55, 0xAA}; // 帧头
  uint8_t freamLen = 0x1e; // 帧长
  uint8_t CMD = 0x01; // 命令 1：转发CAN数据帧 2：PC与设备握手，设备反馈OK 3: 非反馈CAN转发，不反馈发送状态
  uint32_t sendTimes = 1; // 发送次数
  uint32_t timeInterval = 10; // 时间间隔
  uint8_t IDType = 0; // ID类型 0：标准帧 1：扩展帧
  uint32_t CANID; // CAN ID 使用电机ID作为CAN ID
  uint8_t frameType = 0; // 帧类型 0： 数据帧 1：远程帧
  uint8_t len = 0x08; // len
  uint8_t idAcc;
  uint8_t dataAcc;
  uint8_t data[8];
  uint8_t crc; // 未解析，任意值

  void modify(const id_t id, const uint8_t* send_data)
  {
    CANID = id;
    std::copy(send_data, send_data+8, data);
  }
} CAN_Send_Fream;

#pragma pack()


typedef struct 
{
  float Q_MIN = -12.5;
  float Q_MAX = 12.5;
  float DQ_MAX = 30;
  float TAU_MAX = 10;

  struct {
    float kp;
    float kd;
    float q;
    float dq;
    float tau;
  } cmd;

  struct {
    float q;
    float dq;
    float tau;
  } state;

} MotorParam;

/**
 * @brief 达妙科技 DM-J4310-2EC 电机控制
 * 
 * 使用USB转CAN进行通信，linux做虚拟串口
 */
class Motor
{
public:
  Motor(SerialPort::SharedPtr serial = nullptr)
  : serial_(serial)
  {
    if (serial_ == nullptr) {
      serial_ = std::make_shared<SerialPort>("/dev/ttyACM0", B921600);
    }
  }

  void enable(id_t id) { control_cmd(id, 0xFC); }
  void reset(id_t id) { control_cmd(id, 0xFD); }
  void clear_error(id_t id) { control_cmd(id, 0xFB); }
  void zero_position(id_t id) { control_cmd(id, 0xFE); }

  void control(id_t id, float kp, float kd, float q, float dq, float tau)
  {
    // 位置、速度和扭矩采用线性映射的关系将浮点型数据转换成有符号的定点数据
    static auto float_to_uint = [](float x, float xmin, float xmax, uint8_t bits) -> uint16_t {
      float span = xmax - xmin;
      float data_norm = (x - xmin) / span;
      uint16_t data_uint = data_norm * ((1 << bits) - 1);
      return data_uint;
    };

    if(motors.find(id) == motors.end())
    {
      throw std::runtime_error("Motor id not found");
    }

    auto& m = motors[id];

    m->cmd = {kp, kd, q, dq, tau}; // 保存控制命令

    uint16_t kp_uint = float_to_uint(kp, 0, 500, 12);
    uint16_t kd_uint = float_to_uint(kd, 0, 5, 12);
    uint16_t q_uint = float_to_uint(q, m->Q_MIN, m->Q_MAX, 16);
    uint16_t dq_uint = float_to_uint(dq, -m->DQ_MAX, m->DQ_MAX, 12);
    uint16_t tau_uint = float_to_uint(tau, -m->TAU_MAX, m->TAU_MAX, 12);

    std::array<uint8_t, 8> data_buf;
    data_buf[0] = (q_uint >> 8) & 0xff;
    data_buf[1] = q_uint & 0xff;
    data_buf[2] = dq_uint >> 4;
    data_buf[3] = ((dq_uint & 0xf) << 4) | ((kp_uint >> 8) & 0xf);
    data_buf[4] = kp_uint & 0xff;
    data_buf[5] = kd_uint >> 4;
    data_buf[6] = ((kd_uint & 0xf) << 4) | ((tau_uint >> 8) & 0xf);
    data_buf[7] = tau_uint & 0xff;

    send_data.modify(id, data_buf.data());
    serial_->send((uint8_t*)&send_data, sizeof(CAN_Send_Fream));
    this->recv();
  }

  void recv()
  {
    serial_->recv((uint8_t*)&recv_data, 0xAA, sizeof(CAN_Recv_Fream)); 

    if(recv_data.CMD == 0x11 && recv_data.freamEnd == 0x55) // receive success
    { 
      static auto uint_to_float = [](uint16_t x, float xmin, float xmax, uint8_t bits) -> float {
        float span = xmax - xmin;
        float data_norm = float(x) / ((1 << bits) - 1);
        float data = data_norm * span + xmin;
        return data;
      };

      auto & data = recv_data.canData;

      uint16_t q_uint = (uint16_t(data[1]) << 8) | data[2];
      uint16_t dq_uint = (uint16_t(data[3]) << 4) | (data[4] >> 4);
      uint16_t tau_uint = (uint16_t(data[4] & 0xf) << 8) | data[5];

      if(motors.find(recv_data.CANID) == motors.end())
      {
        std::cout << "Unknown motor id: " << std::hex << recv_data.CANID << std::endl;
        return;
      }

      auto & m = motors[recv_data.CANID];
      m->state.q = uint_to_float(q_uint, m->Q_MIN, m->Q_MAX, 16);
      m->state.dq = uint_to_float(dq_uint, -m->DQ_MAX, m->DQ_MAX, 12);
      m->state.tau = uint_to_float(tau_uint, -m->TAU_MAX, m->TAU_MAX, 12);
      
      return;
    } 
    else if (recv_data.CMD == 0x01) // receive fail
    {
      /* code */
    } 
    else if (recv_data.CMD == 0x02) // send fail
    {
      /* code */
    } 
    else if (recv_data.CMD == 0x03) // send success
    {
      /* code */
    }
    else if (recv_data.CMD == 0xEE) // communication error
    {
      /* code */
    }
  }

  std::unordered_map<id_t, std::shared_ptr<MotorParam>> motors;

  /**
   * @brief 添加电机
   * 
   * 实现不同的MOTOR_ID和MASTER_ID都指向同一个MotorParam
   * 确保MOTOR_ID和MASTER_ID都未使用
   */
  void addMotor(id_t MOTOR_ID, id_t MASTER_ID)
  {
    motors.insert({MOTOR_ID, std::make_shared<MotorParam>()});
    motors[MASTER_ID] = motors[MOTOR_ID];
  }

private:
  void control_cmd(id_t id , uint8_t cmd)
  {
    std::array<uint8_t, 8> data_buf = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, cmd};
    send_data.modify(id, data_buf.data());
    serial_->send((uint8_t*)&send_data, sizeof(CAN_Send_Fream));
    usleep(1000);
    recv();
  }

  SerialPort::SharedPtr serial_;
  CAN_Send_Fream send_data;
  CAN_Recv_Fream recv_data;
};

}; // namespace damiao

#endif // DAMIAO_H