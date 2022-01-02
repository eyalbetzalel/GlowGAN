from datasets import get_GMMSD

path_train = "/home/dsi/eyalbetzalel/pytorch-generative-v6/train_imagegpt.h5"
path_test = "/home/dsi/eyalbetzalel/pytorch-generative-v6/test_imagegpt.h5"

get_GMMSD(path_train, path_test)