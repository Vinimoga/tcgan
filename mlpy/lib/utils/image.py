import os
import imageio
import glob


def create_gif(input_path, anim_file='vis.gif'):
    """
        reference: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb
    """
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(input_path, 'image*.png'))
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(os.path.join(input_path, filename))
            writer.append_data(image)

