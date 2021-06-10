# AttentionNet on CIFAR10

AttentionNet is an new image classification architecture designed by Liao Qu.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+
- vit_pytorch 0.19.4

## About

Below is the designed Attention-ResNet Block and the structure of AttentionNet.

![image-attention-resnet-block](https://github.com/QuLiao1117/AttentionNet/blob/master/img/Attention-ResNet-Block.png)

![image-attentionnet](https://github.com/QuLiao1117/AttentionNet/blob/master/img/AttentionNet.png)

## Training

```
# Start training with: 
python main.py --lr=0.003

# You can manually resume the training with: 
python main.py --resume --lr=0.003
```

## Visualization

After the begin of training a model, the project will automatically generate an events file in `/runs`. By using `tensorboard --logdir=./` you can visualize the training process and compare    different models.