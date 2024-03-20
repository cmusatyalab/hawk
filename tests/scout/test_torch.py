# Copyright (c) 2024 Carnegie Mellon University
# SPDX-License-Identifier: BSD-3-Clause

# Some of the following code is derived from the torchvision documentation.
# torchvision is licensed as BSD-3-Clause, so I assume it's documentation
# examples are probably similar.

import pytest
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models import resnet18  # , ResNet18_Weights


@pytest.mark.scout
@pytest.mark.benchmark(group="torchvision_transform")
def test_torchvision_transforms(benchmark, reference_image):
    # weights = ResNet18_Weights.DEFAULT
    # preprocess = weights.transforms()
    preprocess = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transformed_img = benchmark(preprocess, reference_image)
    assert isinstance(transformed_img, torch.Tensor)


@pytest.mark.scout
@pytest.mark.benchmark(group="resnet18_inference")
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_resnet18_model_eval(benchmark, reference_image, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    benchmark.extra_info.update(
        {
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
        }
    )
    # Initialize model
    # weights = ResNet18_Weights.DEFAULT
    # model = resnet18(weights=weights)
    model = resnet18(pretrained=True)
    model.to(torch.device(device))
    model.eval()
    # if device == "cuda":
    #    model = model.cuda()

    # initialize transforms
    # preprocess = weights.transforms()
    preprocess = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # apply inference
    batch = preprocess(reference_image).unsqueeze(0)
    if device == "cuda":
        batch = batch.cuda()

    # get predicted category
    evaluation = benchmark(model, batch)
    prediction = evaluation.squeeze(0).softmax(0)

    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    # category_name = weights.meta["categories"][class_id]
    # print(f"{category_name}: {100 * score:.1f}%")
    # assert category_name == ""

    # assume we will detect the same class and score with a 10% error margin?
    assert class_id == 652
    assert abs(score - 0.4946037828922272) < 0.1
