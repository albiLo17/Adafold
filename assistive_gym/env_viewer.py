import gym, argparse
import numpy as np
import time
import os
from moviepy.editor import ImageSequenceClip
import cv2
from assistive_gym.envs.panda_cloth_env import ClothObjectPandaEnv

def save_numpy_as_gif(array, filename, fps=20, scale=1.0):
    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip


def run(env_name):
    env = ClothObjectPandaEnv()

    rgb_arrays = []
    env.render(width=720, height=720)
    env.setup_camera(camera_eye=[-1, 1, 1.5], camera_target=[-0.1, 0, 0.2])
    observation = env.reset(
        spring_elastic_stiffness=10,
        spring_damping_stiffness=0.1,
        spring_bending_stiffness=0,
    )
    
    t = time.time()
    for t_idx in range(120):
        if t_idx < 40:
            action = np.zeros_like(env.action_space.sample())
        if t_idx >= 40 and t_idx <= 20 + 40:
            # 7 action dim per robot
            # For each robot, first 3 is delta position, second 4 is delat orientation.
            # this stage is pulling the cloht outwards to stretch it
            action = np.array([0.2, 0, 0, 0, 0, 0, 0, -0.2, 0, 0, 0, 0, 0, 0])
        else:
            # this stage is pulling the cloth downwards towards the object
            action = np.array([0, 0, -0.2, 0, 0, 0, 0, 0, 0, -0.2, 0, 0, 0, 0])
            
        observation, _, _, _ = env.step(action)
        rgb, depth = env.get_camera_image_depth(shadow=True)
        rgb = rgb.astype(np.uint8)
        rgb_arrays.append(rgb)
        zfar = 1000.
        znear = 0.01
        depth = (zfar + znear - (2. * depth - 1.) * (zfar - znear))
        depth = (2. * znear * zfar) / depth
        print('Runtime: %.2f s, FPS: %.2f' % (time.time() - t, t_idx / (time.time() - t)))
        # cv2.imshow("rgb", rgb)
        # cv2.waitKey()

    
    save_numpy_as_gif(np.asarray(rgb_arrays), './tmp-3.gif')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default='ClothObjectPandaEnv-v1',
                        help='Environment to test (default: ScratchItchJaco-v1)')
    args = parser.parse_args()

    run(args.env)
