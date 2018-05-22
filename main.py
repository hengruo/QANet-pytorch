from config import config
from preproc import preproc
from absl import app
from absl import logging

def main(_):
    if config.mode == "train":
        train(config)
    elif config.mode == "data":
        preproc(config)
    elif config.mode == "debug":
        config.num_steps = 2
        config.val_num_batches = 1
        config.checkpoint = 1
        config.period = 1
        train(config)
    elif config.mode == "test":
        test(config)
    elif config.mode == "demo":
        demo(config)
    else:
        print("Unknown mode")
        exit(0)

if __name__ == '__main__':
    app.run(main)