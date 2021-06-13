import cav as cav
import model as model
import tcav as tcav
import utils as utils
import utils_plot as utils_plot  # utils_plot requires matplotlib
import os
import torch
import pickle
import activation_generator as act_gen
import tensorflow as tf

# source_dir: where images of concepts, target class and random images (negative samples when learning CAVs) live. Each should be a sub-folder within this directory.
# Note that random image directories can be in any name. In this example, we are using random500_0, random500_1,.. for an arbitrary reason. You need roughly 50-200 images per concept and target class (10-20 pictures also tend to work, but 200 is pretty safe).

# cav_dir: directory to store CAVs (None if you don't want to store)

# target, concept: names of the target class (that you want to investigate) and concepts (strings) - these are folder names in source_dir

# bottlenecks: list of bottleneck names (intermediate layers in your model) that you want to use for TCAV. These names are defined in the model wrapper below.


working_dir = "./tcav_class_test"
activation_dir = working_dir + "/activations/"
cav_dir = working_dir + "/cavs/"
dataset="CUB"

if dataset == "CUB":
    source_dir = "/home/computer/WBH/bmvc/CUB_200_2011/concepts/"
    concepts = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
    # concepts = ["1", "2"]
    LABEL_PATH = './cub_200_2011_labels.txt'
    target_exclude = ['001.Black_footed_Albatross']
    targets = tf.io.gfile.GFile(LABEL_PATH).read().splitlines()
    mymodel = model.CUBResNet50Wrapper(LABEL_PATH, "./82.12_best_model.tar")
else:
    source_dir = "./image_net_subsets"
    concepts = ["dotted", "striped"]
    LABEL_PATH = './imagenet_comp_graph_label_strings.txt'
    target_exclude = ['001.Black_footed_Albatross']
    targets = ['zebra']
    # mymodel = model.InceptionV3Wrapper(LABEL_PATH)
    mymodel = model.ResNet50Wrapper(LABEL_PATH)

# bottlenecks = ['Mixed_5d', 'Conv2d_2a_3x3']
# bottlenecks = ['Conv2d_2a_3x3']
bottlenecks = ["layer3"]
# bottlenecks = ['layer4']

utils.make_dir_if_not_exists(working_dir)
utils.make_dir_if_not_exists(activation_dir)
utils.make_dir_if_not_exists(cav_dir)

# this is a regularizer penalty parameter for linear classifier to get CAVs.
alphas = [0.01]

# random_counterpart = 'random500_1'
print(targets)

act_generator = act_gen.ImageActivationGenerator(
    mymodel, source_dir, activation_dir, max_examples=40
)

# ---------------------------------------------------------------------------------------------------------------
# num_random_exp: number of experiments to confirm meaningful concept direction. TCAV will search for this many folders named random500_0, random500_1, etc. You can alternatively set the random_concepts keyword to be a list of folders of random concepts. Run at least 10-20 for meaningful tests.

# random_counterpart: as well as the above, you can optionally supply a single folder with random images as the "positive set" for statistical testing. Reduces computation time at the cost of less reliable random TCAV scores.

tf.compat.v1.logging.set_verbosity(20)
num_random_exp = 100  # folders (random500_0, random500_1)

for target in targets:
    if target in target_exclude:
        continue

    mytcav = tcav.TCAV(target,
                       concepts,
                       bottlenecks,
                       act_generator,
                       alphas,
                       cav_dir=cav_dir,
                       num_random_exp=num_random_exp)
                       
    print('Loading mytcav')
    results = mytcav.run()
    for result in results:
        print(result["cav_accuracies"])
    utils_plot.plot_results(results, num_random_exp=num_random_exp, save_name=target+"_results.jpg")
    
    with open(target+'_saved_results.pkl','wb') as f:
        pickle.dump(results,f)
