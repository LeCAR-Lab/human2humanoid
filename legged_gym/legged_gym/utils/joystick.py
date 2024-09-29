

def key_response_fn(mode='vel'):
    if mode == 'vel':
        return key_response_vel
    elif mode == 'joint_monkey':
        return key_response_joint_monkey


def key_response_vel(key, command_state, env, ):
    if key.char == 'w':
        if command_state['vel_forward'] < 4.5:
            command_state['vel_forward'] += 0.5
        env.command_ranges["lin_vel_x"] = [command_state['vel_forward'], command_state['vel_forward']]
        env._resample_commands([0])
        print(f'{key.char} pressed, forward speed: ', command_state['vel_forward'])

    elif key.char == 's':
        if command_state['vel_forward'] > -1.0:
            command_state['vel_forward']-= 0.5
        env.command_ranges["lin_vel_x"] = [command_state['vel_forward'], command_state['vel_forward']]
        env._resample_commands([0])
        print(f'{key.char} pressed, forward speed: ', command_state['vel_forward'])
    elif key.char == 'd':
        if command_state['vel_side'] > -0.75:
            command_state['vel_side'] -= 0.15
        env.command_ranges["lin_vel_y"] = [command_state['vel_side'], command_state['vel_side']]
        env._resample_commands([0])
        print(f'{key.char} pressed, side speed: ', command_state['vel_side'])
    elif key.char == 'a':
        print(f'{key.char} pressed')
        if command_state['vel_side'] < 0.75:
            command_state['vel_side'] += 0.15
        env.command_ranges["lin_vel_y"] = [command_state['vel_side'], command_state['vel_side']]
        env._resample_commands([0])
        print(f'{key.char} pressed, side speed: ', command_state['vel_side'])
    elif key.char == 'q':
        if command_state['orientation'] < 2.0:
            command_state['orientation'] += 0.4
        env.command_ranges["ang_vel_yaw"] = [command_state['orientation'], command_state['orientation']]
        env._resample_commands([0])
        print(f'{key.char} pressed, face orientation: ', command_state['orientation'])
    elif key.char == 'e':
        if command_state['orientation'] > -2.0:
            command_state['orientation'] -= 0.4
        env.command_ranges["ang_vel_yaw"] = [command_state['orientation'], command_state['orientation']]
        env._resample_commands([0])
        print(f'{key.char} pressed, face orientation: ', command_state['orientation'])




def key_response_joint_monkey(key, command_state, env, ): 
    if key.char == '[':
        if command_state['torso'] < 2.5:
            command_state['torso'] += 0.5
        env.command_ranges["torso_angle"] = [command_state['torso'], command_state['torso']]
        env._resample_commands([0])
        print(f'{key.char} pressed, torso orientation: ', command_state['torso'])
    elif key.char == ']':
        if command_state['torso'] > -2.5:
            command_state['torso'] -= 0.5
        env.command_ranges["torso_angle"] = [command_state['torso'], command_state['torso']]
        env._resample_commands([0])
        print(f'{key.char} pressed, torso orientation: ', command_state['torso'])

    elif key.char == ',':
        if command_state['left_elbow'] < 2.61:
            command_state['left_elbow'] += 0.25
        env.command_ranges["left_elbow_angle"] = [command_state['left_elbow'], command_state['left_elbow']]
        env._resample_commands([0])
        print(f'{key.char} pressed, left elbow orientation: ', command_state['left_elbow'])
    elif key.char == '<':
        if command_state['left_elbow'] > -1.25:
            command_state['left_elbow'] -= 0.25
        env.command_ranges["left_elbow_angle"] = [command_state['left_elbow'], command_state['left_elbow']]
        env._resample_commands([0])
        print(f'{key.char} pressed, left elbow orientation: ', command_state['left_elbow'])
    elif key.char == '.':
        if command_state['right_elbow'] < 2.61:
            command_state['right_elbow'] += 0.25
        env.command_ranges["right_elbow_angle"] = [command_state['right_elbow'], command_state['right_elbow']]
        env._resample_commands([0])
        print(f'{key.char} pressed, torso orientation: ', command_state['right_elbow'])
    elif key.char == '>':
        if command_state['right_elbow'] > -1.25:
            command_state['right_elbow'] -= 0.25
        env.command_ranges["right_elbow_angle"] = [command_state['right_elbow'], command_state['right_elbow']]
        env._resample_commands([0])
        print(f'{key.char} pressed, torso orientation: ', command_state['right_elbow'])

    # 'n' for left shoulder yaw increase, 'N' for decrease, range [-4.45, 1.3]
    elif key.char == 'n':
        if command_state['left_shoulder_yaw'] < 1.3:
            command_state['left_shoulder_yaw'] += 0.25
        env.command_ranges["left_shoulder_yaw_angle"] = [command_state['left_shoulder_yaw'], command_state['left_shoulder_yaw']]
        env._resample_commands([0])
        print(f'{key.char} pressed, left_shoulder_yaw: ', command_state['left_shoulder_yaw'])
    elif key.char == 'N':
        if command_state['left_shoulder_yaw'] > -4.45:
            command_state['left_shoulder_yaw'] -= 0.25
        env.command_ranges["left_shoulder_yaw_angle"] = [command_state['left_shoulder_yaw'], command_state['left_shoulder_yaw']]
        env._resample_commands([0])
        print(f'{key.char} pressed, left_shoulder_yaw: ', command_state['left_shoulder_yaw'])

    # 'm' for right shoulder yaw increase, 'M' for decrease, range [-1.3, 4.45]
    elif key.char == 'm':
        if command_state['right_shoulder_yaw'] < 1.3:
            command_state['right_shoulder_yaw'] += 0.25
        env.command_ranges["right_shoulder_yaw_angle"] = [command_state['right_shoulder_yaw'], command_state['right_shoulder_yaw']]
        env._resample_commands([0])
        print(f'{key.char} pressed, right_shoulder_yaw: ', command_state['right_shoulder_yaw'])
    elif key.char == 'M':
        if command_state['right_shoulder_yaw'] > -4.45:
            command_state['right_shoulder_yaw'] -= 0.25
        env.command_ranges["right_shoulder_yaw_angle"] = [command_state['right_shoulder_yaw'], command_state['right_shoulder_yaw']]
        env._resample_commands([0])
        print(f'{key.char} pressed, right_shoulder_yaw: ', command_state['right_shoulder_yaw'])

    # 'h' for left shoulder pitch increase, 'H' for decrease, range [-2.87, 2.87]
    elif key.char == 'h':
        if command_state['left_shoulder_pitch'] < 2.87:
            command_state['left_shoulder_pitch'] += 0.25
        env.command_ranges["left_shoulder_pitch_angle"] = [command_state['left_shoulder_pitch'], command_state['left_shoulder_pitch']]
        env._resample_commands([0])
        print(f'{key.char} pressed, left_shoulder_pitch: ', command_state['left_shoulder_pitch'])
    elif key.char == 'H':
        if command_state['left_shoulder_pitch'] > -2.87:
            command_state['left_shoulder_pitch'] -= 0.25
        env.command_ranges["left_shoulder_pitch_angle"] = [command_state['left_shoulder_pitch'], command_state['left_shoulder_pitch']]
        env._resample_commands([0])
        print(f'{key.char} pressed, left_shoulder_pitch: ', command_state['left_shoulder_pitch'])

    # 'j' for right shoulder pitch increase, 'J' for decrease, range [-2.87, 2.87]
    elif key.char == 'j':
        if command_state['right_shoulder_pitch'] < 2.87:
            command_state['right_shoulder_pitch'] += 0.25
        env.command_ranges["right_shoulder_pitch_angle"] = [command_state['right_shoulder_pitch'], command_state['right_shoulder_pitch']]
        env._resample_commands([0])
        print(f'{key.char} pressed, right_shoulder_pitch: ', command_state['right_shoulder_pitch'])
    elif key.char == 'J':
        if command_state['right_shoulder_pitch'] > -2.87:
            command_state['right_shoulder_pitch'] -= 0.25
        env.command_ranges["right_shoulder_pitch_angle"] = [command_state['right_shoulder_pitch'], command_state['right_shoulder_pitch']]
        env._resample_commands([0])
        print(f'{key.char} pressed, right_shoulder_pitch: ', command_state['right_shoulder_pitch'])

    # 'y' for left shoulder roll increase, 'Y' for decrease, range [-3.11, 0.34]
    elif key.char == 'y':
        if command_state['left_shoulder_roll'] < 0.34:
            command_state['left_shoulder_roll'] += 0.25
        env.command_ranges["left_shoulder_roll_angle"] = [command_state['left_shoulder_roll'], command_state['left_shoulder_roll']]
        env._resample_commands([0])
        print(f'{key.char} pressed, left_shoulder_roll: ', command_state['left_shoulder_roll'])
    elif key.char == 'Y':
        if command_state['left_shoulder_roll'] > -3.11:
            command_state['left_shoulder_roll'] -= 0.25
        env.command_ranges["left_shoulder_roll_angle"] = [command_state['left_shoulder_roll'], command_state['left_shoulder_roll']]
        env._resample_commands([0])
        print(f'{key.char} pressed, left_shoulder_roll: ', command_state['left_shoulder_roll'])

    # 'u' for right shoulder roll increase, 'U' for decrease, range [-3.11, 0.34]
    elif key.char == 'u':
        if command_state['right_shoulder_roll'] < 0.34:
            command_state['right_shoulder_roll'] += 0.25
        env.command_ranges["right_shoulder_roll_angle"] = [command_state['right_shoulder_roll'], command_state['right_shoulder_roll']]
        env._resample_commands([0])
        print(f'{key.char} pressed, right_shoulder_roll: ', command_state['right_shoulder_roll'])
    elif key.char == 'U':
        if command_state['right_shoulder_roll'] > -3.11:
            command_state['right_shoulder_roll'] -= 0.25
        env.command_ranges["right_shoulder_roll_angle"] = [command_state['right_shoulder_roll'], command_state['right_shoulder_roll']]
        env._resample_commands([0])
        print(f'{key.char} pressed, right_shoulder_roll: ', command_state['right_shoulder_roll'])