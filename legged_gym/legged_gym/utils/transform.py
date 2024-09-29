from pytorch3d.transforms import quaternion_apply, axis_angle_to_matrix, matrix_to_quaternion, quaternion_multiply


def apply_rotation_to_quat_z(original_quat, rotation_angle):

    rotation_matrix = axis_angle_to_matrix(rotation_angle)
    # import ipdb; ipdb.set_trace()
    # rotation_matrix[:, :2, :2] = torch.stack([
    #     torch.cos(rotation_angle), -torch.sin(rotation_angle),
    #     torch.sin(rotation_angle), torch.cos(rotation_angle)
    # ], dim=-2)
    rotation_quat = matrix_to_quaternion(rotation_matrix)
    rotated_quaternions = quaternion_multiply(original_quat, rotation_quat)

    # rotated_quaternions = matrix_to_quaternion(rotated_quaternions_matrix)
    return rotated_quaternions



