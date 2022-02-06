import glob
import json
import os
import sys

import cv2
import pygame as pg
from PIL import Image, ImageDraw, ImageFont

import numpy as np
from numpy import int32, uint8, uint

main_dir = os.path.split(os.path.abspath(__file__))[0]

WINDOW_LENGTH = 7
TRAJ_HEIGHT = 4
IMG_SCALING = 3


def start_ui(get_frame_fn, name):
    "displays a surface, waits for user to continue"
    horz_scroll_index = 0
    array_img = get_frame_fn(horz_scroll_index)
    display_frame(array_img, name)
    while True:
        e = pg.event.wait()
        if e.type == pg.KEYDOWN and e.key == pg.K_RIGHT:
            # Move window forward one frame
            horz_scroll_index += 1
            array_img = get_frame_fn(horz_scroll_index)
            display_frame(array_img, name)
        elif e.type == pg.KEYDOWN and e.key == pg.K_LEFT and horz_scroll_index > 0:
            horz_scroll_index -= 1
            array_img = get_frame_fn(horz_scroll_index)
            display_frame(array_img, name)
        elif e.type == pg.QUIT:
            pg.quit()
            raise SystemExit()


def display_frame(array_img, name):
    screen = pg.display.set_mode(array_img.shape[:2], 0, 32)
    pg.surfarray.blit_array(screen, array_img)
    pg.display.flip()
    pg.display.set_caption(name)


def main():
    pg.init()
    if len(sys.argv) == 1:
        runs = sorted(glob.glob('../images/viz_traj/*'))
        latest = runs[-1]
        print('Grabbing latest run from ', latest)
        run_id = latest.split('_')[-5]
    else:
        run_id = sys.argv[1]

    frame_paths = sorted(glob.glob(f'../images/viz_traj/*{run_id}*/*.png'))
    traj_paths = sorted(glob.glob(f'../images/viz_traj/*{run_id}*/*.json'))
    cluster_paths = sorted(glob.glob(f'../images/viz_clusters/*{run_id}*'))

    cluster_path = get_cluster(cluster_paths, run_id)
    cluster_img_paths = sorted(glob.glob(f'{cluster_path}/*.png'))

    print("Press right to advance frames.")

    def _get_frame(horz_scroll_index):
        ret = get_frame(cluster_img_paths, frame_paths, traj_paths, horz_scroll_index)
        return ret
    start_ui(_get_frame, "learnmax predicted trajectories per step")

    # # rgbarray
    # imagename = os.path.join(main_dir, "arraydemo.bmp")
    # imgsurface = pg.image.load(imagename)
    # rgbarray = surfarray.array3d(imgsurface)
    #
    # # scaleup
    # # the element type is required for N.zeros in numpy else
    # # an #array of floats is returned.
    # shape = rgbarray.shape
    # scaleup = np.zeros((shape[0] * 2, shape[1] * 2, shape[2]), int32)
    # scaleup[::2, ::2, :] = rgbarray
    # scaleup[1::2, ::2, :] = rgbarray
    # scaleup[:, 1::2] = scaleup[:, ::2]
    # show(scaleup, "scaleup")

    # alldone
    pg.quit()


def get_frame(cluster_img_paths, frame_paths, traj_paths, horz_scroll_index):
    start_frame = WINDOW_LENGTH * horz_scroll_index
    frame_paths = frame_paths[start_frame:start_frame + WINDOW_LENGTH]
    frame_imgs = []
    # death example '../images/viz_traj/2022_01_26_14_52_05_913927_r_SOUKMA_e_3_b_5000/000000000.png'
    #  Write top row actual images then write column with latent predicted trajectory for that state
    for _i, frame_path in enumerate(frame_paths):
        frame_i = start_frame + _i
        frame_img = get_np_surface(frame_path, IMG_SCALING)
        traj_path = traj_paths[frame_i]
        traj_index = int(traj_path.split('.')[-2][-9:])
        assert traj_index == frame_i
        with open(traj_path) as tfp:
            traj = json.load(tfp)
        traj_imgs = []
        for z_q_ind in traj['z_q_ind'][:TRAJ_HEIGHT]:
            traj_img_path = cluster_img_paths[z_q_ind]
            _traj_i = int(traj_img_path.split('/')[-1].split('.')[0])
            assert _traj_i == z_q_ind
            traj_img = get_np_surface(traj_img_path, IMG_SCALING)
            txt_img = get_np_txt(traj_img)
            traj_imgs.append(traj_img)
            # traj_imgs.append(np.concatenate((traj_img, txt_img), axis=1))

        traj_img = np.concatenate(traj_imgs, axis=1)

        frame_img = np.concatenate((frame_img, traj_img), axis=1)
        frame_imgs.append(frame_img)

        # img_resized.show()

        # TODO: Display entropy in a bar below image normalized or something
        # TODO: Display action
        # TODO: Also write top row z_q index so we can compare with an overlay / residual
    win_frame_img = np.concatenate(frame_imgs, axis=0)
    return win_frame_img


def get_np_txt(traj_img):
    image = Image.new('RGBA', (traj_img.shape[0] * 4, traj_img.shape[0]), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    # Search for your system's own truetype font if this doesn't work, sorry!
    font = ImageFont.truetype(font='/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf', size=120)
    draw.text((10, 0), "yoyoyo", (0, 0, 0), font=font)
    img_resized = image.resize((traj_img.shape[0], traj_img.shape[0] // 4), Image.ANTIALIAS)
    np_txt = np.array(img_resized)[:, :, :3].transpose(1, 0, 2)
    return np_txt


def get_np_surface(frame_path, img_scaling):
    img = get_image(frame_path, img_scaling)
    surface = pg.surfarray.make_surface(img)
    np_surface = pg.surfarray.array3d(surface)
    return np_surface


def get_image(frame_path, img_scaling):
    img = cv2.imread(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(1, 0, 2)
    img = cv2.resize(img, dsize=(round(img.shape[0] * img_scaling),
                                 round(img.shape[1] * img_scaling)), interpolation=cv2.INTER_CUBIC)
    return img


def get_cluster(cluster_paths, run_id):
    count_with_run_id = 0
    # Check that we have the right cluster
    for cluster_path in cluster_paths[::-1]:
        if run_id in cluster_path:
            if count_with_run_id > 0:
                raise ValueError('Should only be one matching cluster until we train DVQ and GPT together')
            count_with_run_id += 1
    assert count_with_run_id == 1
    cluster_path = cluster_paths[-1]
    assert run_id in cluster_path  # We just write clusters at beginning of training right now
    return cluster_path


if __name__ == "__main__":
    main()
