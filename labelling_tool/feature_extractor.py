import numpy as np
import cv2
import torch
from torch import nn
from tqdm import tqdm
from torchvision import models, transforms


def resizeAR(src_img, width, height, return_factors=False, add_border=True):
    src_height, src_width, n_channels = src_img.shape
    src_aspect_ratio = float(src_width) / float(src_height)

    if width == 0 or height == 0:
        raise SystemError('Neither width nor height can be zero')

    aspect_ratio = float(width) / float(height)

    if add_border:
        if src_aspect_ratio == aspect_ratio:
            dst_width = src_width
            dst_height = src_height
            start_row = start_col = 0
        elif src_aspect_ratio > aspect_ratio:
            dst_width = src_width
            dst_height = int(src_width / aspect_ratio)
            start_row = int((dst_height - src_height) / 2.0)
            start_col = 0
        else:
            dst_height = src_height
            dst_width = int(src_height * aspect_ratio)
            start_col = int((dst_width - src_width) / 2.0)
            start_row = 0

        dst_img = np.zeros((dst_height, dst_width, n_channels), dtype=np.uint8)
        dst_img[start_row:start_row + src_height, start_col:start_col + src_width, :] = src_img
        dst_img = cv2.resize(dst_img, (width, height))
        if return_factors:
            resize_factor = float(height) / float(dst_height)
            return dst_img, resize_factor, start_row, start_col
        else:
            return dst_img
    else:
        if src_aspect_ratio < aspect_ratio:
            dst_width = width
            dst_height = int(dst_width / src_aspect_ratio)
        else:
            dst_height = height
            dst_width = int(dst_height * src_aspect_ratio)
        dst_img = cv2.resize(src_img, (dst_width, dst_height))
        start_row = start_col = 0
        if return_factors:
            resize_factor = float(src_height) / float(dst_height)
            return dst_img, resize_factor, start_row, start_col
        else:
            return dst_img


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]

    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


def run(all_img_paths, n_images, crop, save_path, load=0):
    if load:
        print('Loading features from {}'.format(save_path))

        loaded = np.load(save_path)

        features = loaded['features']
        feature_colors = loaded['feature_colors']

        return features, feature_colors

    model = models.vgg16(pretrained=True)
    # model = models.resnet101(pretrained=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    new_model = FeatureExtractor(model)

    transform_list = [
        transforms.ToPILImage(),
    ]

    if crop:
        transform_list.append(transforms.CenterCrop(512))
        transform_list.append(transforms.Resize(448))

    transform_list.append(transforms.ToTensor())

    transform = transforms.Compose(transform_list)

    class_colors = {
        'diff': [1, 0, 0],  # red
        'ipsc': [0, 1, 0]  # green
    }

    feature_colors = []
    features = []

    print('extracting features....')

    for img_id in tqdm(range(n_images)):
        img_path, class_label = all_img_paths[img_id]

        img_cv = cv2.imread(img_path)

        if not crop:
            img_cv = resizeAR(img_cv, 448, 448)

        img = transform(img_cv)

        img = img.reshape(1, 3, 448, 448)
        img = img.to(device)

        # img_disp = img.cpu().detach().numpy().squeeze().transpose([1, 2, 0])

        # cv2.imshow('img_cv', img_cv)
        # cv2.imshow('img_disp', img_disp)
        # cv2.waitKey(0)

        feature = new_model(img)

        feature_colors.append(class_colors[class_label])

        feature = feature.cpu().detach().numpy().reshape(-1)

        features.append(feature)

    features = np.asarray(features)
    feature_colors = np.asarray(feature_colors)

    if not load and save_path:
        print('Saving features to {}'.format(save_path))

        np.savez_compressed(save_path, features=features, feature_colors=feature_colors)

    return features, feature_colors


def main():
    pass


if __name__ == '__main__':
    main()
