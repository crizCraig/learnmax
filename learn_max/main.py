import gym
import atari_py
import fire
from gym import wrappers
from loguru import logger as log
from matplotlib import pyplot as plt
import pytorch_lightning as pl

from dvq.vqvae import VQVAE


def main(env_id='MontezumaRevenge-v0'):
    # MontezumaRevenge-v0 has 'repeat_action_probability': 0.25
    # MontezumaRevenge-v4 is nondeterministic => Whether this environment is non-deterministic even after seeding
    log.info(atari_py.list_games())
    env = gym.make(env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    # agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    frame_encoder = VQVAE(n_hid=64, num_embeddings=1024, embedding_dim=64, loss_flavor='l2', input_channels=3,
                          enc_dec_flavor='deepmind', vq_flavor='vqvae')

    # TODO: Set this up as a lightning module
    # TODO: Instantiate transformer - look into using detach to separate training of two. See if you can actually join them???

    # transformer =

    for i in range(episode_count):
        ob = env.reset()
        while True:
            # action = agent.act(ob, reward, done)
            action = 0
            ob, reward, done, _ = env.step(action)
            plt.imshow(ob, interpolation='nearest')
            plt.show()
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()


if __name__ == '__main__':
    fire.Fire(main)

