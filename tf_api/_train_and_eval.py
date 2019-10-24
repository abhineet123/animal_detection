import argparse, sys, os, subprocess
from pprint import pprint
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description="DeepLab_ResNet Network")
    parser.add_argument("--train_script_path", type=str,
                        default='~/models/research/object_detection/legacy/train.py',
                        help="path to training script")

    parser.add_argument("--eval_script_path", type=str,
                        default='~/models/research/object_detection/legacy/eval.py',
                        help="path to evaluation script")

    parser.add_argument("--checkpoint_dir", type=str,
                        default='',
                        help="path to checkpoint dir")

    parser.add_argument("--pipeline_config_path", type=str,
                        help="path to config path")

    parser.add_argument("--steps_between_eval", type=int,
                        help="training steps between successive evaluations")

    parser.add_argument("--total_steps", type=int,
                        help="total training steps")

    args, unknown_args = parser.parse_known_args()

    train_script_path = args.train_script_path
    eval_script_path = args.eval_script_path
    checkpoint_dir = args.checkpoint_dir
    pipeline_config_path = args.pipeline_config_path
    steps_between_eval = args.steps_between_eval
    total_steps = args.total_steps

    pprint(sys.argv)
    pprint(unknown_args)

    train_command = ''
    eval_command = ''

    # pre_args = []
    # for _arg in sys.argv:
    #     if _arg == 'python3':
    #         break
    #     pre_args.append(_arg)
    #     train_command = '{} {}'.format(train_command, _arg) if train_command else _arg
    #     eval_command = '{} {}'.format(eval_command, _arg) if eval_command else _arg

    train_command = 'python3 {} --pipeline_config_path={} --train_dir={}'.format(
        train_script_path, pipeline_config_path, checkpoint_dir)
    eval_command = 'python2 {} --pipeline_config_path={} --checkpoint_dir={}'.format(
        eval_script_path, pipeline_config_path, checkpoint_dir)

    for _arg in unknown_args:
        if _arg.startswith('--train.'):
            train_command = '{} {}'.format(train_command, _arg.replace('--train.', '--'))
        elif _arg.startswith('--eval.'):
            eval_command = '{} {}'.format(eval_command, _arg.replace('--eval.', '--'))

    # print('train_command: {}'.format(train_command))
    # print('eval_command: {}'.format(eval_command))

    n_steps = 0
    ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt_path:
        print('Continuing from {}'.format(ckpt_path))
        start_epoch = int(ckpt_path.split('-')[-1]) + 1
        n_steps = start_epoch

    while n_steps < total_steps:
        n_steps += steps_between_eval

        _train_command = '{} --n_steps={}'.format(train_command, n_steps)
        print('Running training:\n{}\n'.format(_train_command))
        # os.system(_train_command)
        subprocess.check_call(_train_command, shell=True)

        _eval_command = '{}'.format(eval_command)
        print('Running evaluation:\n{}\n'.format(_eval_command))
        # os.system(_eval_command)
        subprocess.check_call(_eval_command, shell=True)



if __name__ == '__main__':
    main()
