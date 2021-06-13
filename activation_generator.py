from abc import ABCMeta
from abc import abstractmethod
from multiprocessing import dummy as multiprocessing
import os.path
import numpy as np
import PIL.Image
import tensorflow as tf
import torch
from torchvision import transforms


class ActivationGeneratorInterface(object):
    """Interface for an activation generator for a model"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def process_and_load_activations(self, bottleneck_names, concepts):
        pass

    @abstractmethod
    def get_model():
        pass


class ActivationGeneratorBase(ActivationGeneratorInterface):
    """Basic abstract activation generator for a model"""

    def __init__(self, model, acts_dir, max_examples=500, sample_num=5):
        self.model = model
        self.acts_dir = acts_dir
        self.max_examples = max_examples
        self.sample_num = sample_num
        if self.sample_num < 1:
            tf.compat.v1.logging.info("sample_num must be >= 1")

    def get_model(self):
        return self.model

    @abstractmethod
    def get_examples_for_concept(self, concept):
        pass

    def get_activations_for_concept(self, concept, bottleneck):
        examples = self.get_examples_for_concept(concept)
        return self.get_activations_for_examples(examples, bottleneck)

    def get_activations_for_examples(self, examples, bottleneck):
        acts = self.model.run_examples(examples, bottleneck)
        # print("acts.shape: ",acts.shape)
        # print("reshaped: ",self.model.reshape_activations(acts).squeeze().shape)
        return self.model.reshape_activations(acts).squeeze()

    def process_and_load_activations(
        self, bottleneck_names, concepts, targets=[]
    ):
        acts = {}
        if self.acts_dir and not tf.io.gfile.exists(self.acts_dir):
            tf.io.gfile.makedirs(self.acts_dir)

        for concept in concepts:
            if concept not in acts:
                acts[concept] = {}
            for bottleneck_name in bottleneck_names:
                acts_path = (
                    os.path.join(
                        self.acts_dir,
                        "acts_{}_{}".format(concept, bottleneck_name),
                    )
                    if self.acts_dir
                    else None
                )
                if acts_path and tf.io.gfile.exists(acts_path):
                    with tf.io.gfile.GFile(acts_path, "rb") as f:
                        acts[concept][bottleneck_name] = np.load(f).squeeze()
                        tf.compat.v1.logging.info(
                            "Loaded {} shape {}".format(
                                acts_path, acts[concept][bottleneck_name].shape
                            )
                        )
                else:
                    acts[concept][
                        bottleneck_name
                    ] = self.get_activations_for_concept(
                        concept, bottleneck_name
                    )
                    if self.sample_num > 1 and concept not in targets:
                        for _ in range(self.sample_num - 1):
                            acts[concept][bottleneck_name] = np.concatenate(
                                (
                                    acts[concept][bottleneck_name],
                                    self.get_activations_for_concept(
                                        concept, bottleneck_name
                                    ),
                                ),
                                axis=0,
                            )
                    if acts_path:
                        tf.compat.v1.logging.info(
                            "{} does not exist, Making one...".format(acts_path)
                        )
                        with tf.io.gfile.GFile(acts_path, "w") as f:
                            np.save(
                                f,
                                acts[concept][bottleneck_name],
                                allow_pickle=False,
                            )
        return acts


class ImageActivationGenerator(ActivationGeneratorBase):
    """Activation generator for a basic image model"""

    def __init__(
        self, model, source_dir, acts_dir, max_examples=10, sample_num=5
    ):
        self.source_dir = source_dir
        super(ImageActivationGenerator, self).__init__(
            model, acts_dir, max_examples
        )

    def get_examples_for_concept(self, concept):
        concept_dir = os.path.join(self.source_dir, concept)
        img_paths = [
            os.path.join(concept_dir, d)
            for d in tf.io.gfile.listdir(concept_dir)
        ]
        imgs = self.load_images_from_files(
            img_paths, self.max_examples, shape=self.model.get_image_shape()[:2]
        )
        return imgs

    def load_image_from_file(self, filename, shape):
        """Given a filename, try to open the file. If failed, return None.
        Args:
          filename: location of the image file
          shape: the shape of the image file to be scaled
        Returns:
          the image if succeeds, None if fails.
        Rasies:
          exception if the image was not the right shape.
        """
        image_size = shape[0]
        transform = transforms.Compose(
            [
                transforms.Resize(int(image_size / 0.875)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        if not tf.io.gfile.exists(filename):
            tf.compat.v1.logging.error("Cannot find file: {}".format(filename))
            return None
        # try:
        # ensure image has no transparency channel
        # img = np.array(
        #     PIL.Image.open(tf.io.gfile.GFile(filename, "rb"))
        #     .convert("RGB")
        #     .resize(shape, PIL.Image.LANCZOS)
        # )
        image = PIL.Image.open(tf.io.gfile.GFile(filename, "rb")).convert("RGB")
        img = transform(image)

        if not (len(img.shape) == 3 and img.shape[0] == 3):
            tf.compat.v1.logging.error("wrong shape: {}, shape is : {}".format(filename, img.shape))
            return None
        else:
            return img

        # except Exception as e:
        #     tf.compat.v1.logging.info(e)
        #     return None
        return img

    def load_images_from_files(
        self,
        filenames,
        max_imgs=500,
        do_shuffle=True,
        run_parallel=True,
        shape=(299, 299),
        num_workers=10,
    ):
        """Return image arrays from filenames.
        Args:
          filenames: locations of image files.
          max_imgs: maximum number of images from filenames.
          do_shuffle: before getting max_imgs files, shuffle the names or not
          run_parallel: get images in parallel or not
          shape: desired shape of the image
          num_workers: number of workers in parallelization.
        Returns:
          image arrays
        """
        image_size = shape[0]
        imgs = []
        # First shuffle a copy of the filenames.
        filenames = filenames[:]
        if do_shuffle:
            np.random.shuffle(filenames)

        imgs = torch.empty((0, 3, image_size, image_size))
        if run_parallel:
            pool = multiprocessing.Pool(num_workers)
            img_pool = pool.map(
                lambda filename: self.load_image_from_file(filename, shape),
                filenames[:max_imgs],
            )
            # print(img_pool)
            for img in img_pool:
                if img is not None:
                    # imgs.append(img)
                    img = img.view(1, 3, shape[0], shape[1])
                    imgs = torch.cat([imgs, img], dim=0)
            if imgs.shape[0] <= 1:
                raise ValueError(
                    "You must have more than 1 image in each class to run TCAV."
                )
        else:
            for filename in filenames:
                img = self.load_image_from_file(filename, shape)
                if img is not None:
                    # imgs.append(img)
                    img = img.view(1, 3, shape[0], shape[1])
                    imgs = torch.cat([imgs, img], dim=0)
                if imgs.shape[0] <= 1:
                    raise ValueError(
                        "You must have more than 1 image in each class to run TCAV."
                    )
                elif imgs.shape[1] >= max_imgs:
                    break
                # if len(imgs) <= 1:
                #     raise ValueError(
                #         "You must have more than 1 image in each class to run TCAV."
                #     )
                # elif len(imgs) >= max_imgs:
                #     break

        return imgs
        # return np.array(imgs)
