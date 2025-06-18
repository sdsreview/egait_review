import robot_interface as sdk


class RobotInterface(): 

    def __init__(self):
        hIGHLEVEL = 0xee
        LOWLEVEL  = 0xff
        self.udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10",8007)
        self.safe = sdk.Safety(sdk.LeggedType.Go1)

        self.cmd = sdk.LowCmd()
        self.state = sdk.LowState()

        
    
        self.udp.InitCmdData(self.cmd)


    def receive_observation(self):
        self.udp.Recv()
        self.udp.GetRecv(self.state)
        return self.state


    def send_command(self,motorcmd):
        self.cmd.levelFlag = 0xff
        for motor_id in range(12):
            self.cmd.motorCmd[motor_id].mode = 0x0A
            self.cmd.motorCmd[motor_id].q = motorcmd[motor_id * 5]
            self.cmd.motorCmd[motor_id].Kp = motorcmd[motor_id * 5 + 1]
            self.cmd.motorCmd[motor_id].dq = motorcmd[motor_id * 5 + 2]
            self.cmd.motorCmd[motor_id].Kd = motorcmd[motor_id * 5 + 3]
            self.cmd.motorCmd[motor_id].tau = motorcmd[motor_id * 5 + 4]
        self.safe.PositionLimit(self.cmd)
        self.udp.SetSend(self.cmd)
        self.udp.Send()



    