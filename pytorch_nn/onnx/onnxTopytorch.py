import onnx 
import onnxruntime as ort
import onnx2pytorch

import torch 
from torchvision import datasets, transforms
import numpy as np


def main():
    # dataset = datasets.MNIST(root="./data/", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    
    # idx = torch.zeros(len(dataset.targets)).bool()
    # idx = idx | (dataset.targets == 0)
    # dataset.data = dataset.data[idx, :, :]
    # dataset.targets = dataset.targets[idx]
    
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # model_path = "./mnist/mnist_relu_5_100.onnx"
    # session = ort.InferenceSession(model_path)
    
    # count_true = 0
    # for id, (image, label) in enumerate(dataloader):
    #     input_data = image.numpy()
    #     input_name = session.get_inputs()[0].name
    #     output_name = session.get_outputs()[0].name
        
    #     output = session.run([output_name], {input_name: input_data})
    #     logits = output[0]
    #     precdicated_class = np.argmax(logits, axis=1)
        
    #     # print(f"Predicated Class: {precdicated_class}, True Label: {label.numpy()}")
    #     if precdicated_class == label.numpy():
    #         count_true += 1
    # print(f"count true = {count_true}")

    print("Transformation starting ...")
    model_path = ["./mnist/ffnnRELU__PGDK_w_0.1_6_500.onnx", 
                  "./mnist/ffnnRELU__PGDK_w_0.3_6_500.onnx",
                  "./mnist/ffnnRELU__Point_6_500.onnx",
                  "./mnist/mnist_relu_3_50.onnx",
                  "./mnist/mnist_relu_3_100.onnx",
                  "./mnist/mnist_relu_4_1024.onnx",
                  "./mnist/mnist_relu_5_100.onnx",
                  "./mnist/mnist_relu_6_100.onnx",
                  "./mnist/mnist_relu_6_200.onnx",
                  "./mnist/mnist_relu_9_100.onnx",
                  "./mnist/mnist_relu_9_200.onnx",
                  "./cifar10/cifar_relu_4_100.onnx",
                  "./cifar10/cifar_relu_6_100.onnx",
                  "./cifar10/cifar_relu_7_1024.onnx",
                  "./cifar10/cifar_relu_9_200.onnx",
                  "./cifar10/ffnnRELU__PGDK_w_0.0078_6_500.onnx",
                  "./cifar10/ffnnRELU__PGDK_w_0.0313_6_500.onnx",
                  "./cifar10/ffnnRELU__Point_6_500.onnx"]
    
    for m in model_path:
        pytorch_model = onnx2pytorch.ConvertModel(onnx.load(m), experimental=True)
        model_layers = []
        for p in pytorch_model.named_parameters():
            if ".weight" in p[0]:
                model_layers.append({"type": "Linear",
                                    "parameters": [p[1].shape[1], p[1].shape[0]]})
            elif ".bias" in p[0]:
                model_layers.append({"type": "ReLU"})
        state = {
            "state_dict": pytorch_model.state_dict(),
            "model_layers": model_layers
        }
        torch.save(state, f"{m.split('.onnx')[0]}.pth")
    print("Transformation finished.")

    return


if __name__ == "__main__":
    main()