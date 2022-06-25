import os
import tensorflow as tf


def freeze_graph(input_checkpoint, output_pb, output_node_names):
    """ freeze a model and save to .pb file
    reference: https://gist.github.com/moodoki/e37a85fb0258b045c005ca3db9cbc7f6
    
    :param input_checkpoint: the directory of checkpoint
    :param output_pb: the directory of .pb file
    :param output_node_names: a str split by comma.
    :return: 
    """

    # Load meta graph to Tensorflow default graph.
    # Devices should be cleared to allow Tensorflow to control placement of graph when loading on different machines.
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    # fix batch norm nodes
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
    # save model to .pb file
    with tf.Session(graph=graph) as sess:
        # restore weights
        saver.restore(sess, input_checkpoint)
        # export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.replace(" ", "").split(",")
        )
        # serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_pb, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("{} ops in the final graph.".format(len(output_graph_def.node)))


def load_pb(pb_dir, name=''):
    """ load .pb file 
    
    :param pb_dir: the directory of .pb file 
    :param name: set the name of graph
    :return: 
        The tensorflow graph
    """
    with tf.gfile.GFile(pb_dir, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=name)
    print("graph operations: ")
    for op in graph.get_operations():
        print(op.name)
    return graph


def save_ckpt(saver, sess, dir_checkpoint, step, model_name=None):
    """ save current session to checkpoint file.
    
    :param saver: 
    :param sess: 
    :param dir_checkpoint: 
    :param step: 
    :param model_name: 
    :return: 
    """
    if (model_name is None) or model_name == '':
        dir_save = dir_checkpoint
    else:
        dir_save = os.path.join(dir_checkpoint, model_name)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    prefix = os.path.join(dir_save, "cache.ckpt")
    saver.save(sess, prefix, global_step=step)


def load_ckpt(saver, sess, dir_checkpoint, model_name=None):
    """ load checkpoint file to current session
    
    :param saver: 
    :param sess: 
    :param dir_checkpoint: 
    :param model_name: 
    :return: 
        The operating result(True or False), 
        The counter.
    """
    import re
    print("[Info]  Reading checkpoints ...")
    if model_name is None:
        dir_load = dir_checkpoint
    else:
        dir_load = os.path.join(dir_checkpoint, model_name)
    ckpt = tf.train.get_checkpoint_state(dir_load)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(dir_load, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [Info] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [Error] Failed to find a checkpoint")
        return False, 0


def latest_ckpt(checkpoint_dir, latest_filename=None):
    """Finds the filename of latest saved checkpoint file.
        Refer: tf.train.latest_checkpoint(),
        Motivation: address the inconsistent root path between 'checkpoint_dir' and 'ckpt.model_checkpoint_path'.
      Args:
        checkpoint_dir: Directory where the variables were saved.
        latest_filename: Optional name for the protocol buffer file that
          contains the list of most recent checkpoint filenames.
          See the corresponding argument to `Saver.save()`.

      Returns:
        The full path to the latest checkpoint or `None` if no checkpoint was found.
    """
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir, latest_filename)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path = os.path.join(checkpoint_dir, os.path.split(ckpt.model_checkpoint_path)[-1])
        return ckpt_path
    return None