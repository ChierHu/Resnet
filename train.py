import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from model import resnet34
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# 使用tensorboard对训练过程中的损失进行可视化
# from tensorboardX import SummaryWriter
# # 将log保存在output中
# writer = SummaryWriter('./output/log')
#
# train_tags = ['Train-Acc','Train-Loss']
# test_tags = ['Val-Acc', 'Val-Loss']

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


    train_path = '../train'
    assert os.path.exists(train_path), "{} path does not exist.".format(train_path)

    val_path = '../test'
    assert os.path.exists(val_path), "{} path does not exist.".format(val_path)

    train_dataset = datasets.ImageFolder(root=train_path,
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=val_path,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./pretrained_weights/resnet34-333f7ec4.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 7)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 40
    best_acc = 0.0
    save_path = './model_weight/ResNet-34-0422.pth'
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    # for epoch in range(epochs):
    #     # train
    #     net.train()
    #     training_loss = 0.0
    #     train_bar = tqdm(train_loader)
    #     for step, data in enumerate(train_bar):
    #         images, labels = data
    #         optimizer.zero_grad()
    #         logits = net(images.to(device))
    #         loss = loss_function(logits, labels.to(device))
    #         loss.backward()
    #         optimizer.step()
    #         # print statistics
    #         training_loss += loss.item()
    #         train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
    #                                                                  epochs,
    #                                                                  loss)
    #     # validate
    #     net.eval()
    #     val_acc = 0.0  # accumulate accurate number / epoch
    #     with torch.no_grad():
    #         val_bar = tqdm(validate_loader)
    #         for val_data in val_bar:
    #             val_images, val_labels = val_data
    #             outputs = net(val_images.to(device))
    #             predict_y = torch.max(outputs, dim=1)[1]
    #             val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
    #
    #             val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
    #                                                        epochs)
    #
    #
    #     val_accurate = val_acc / val_num
    #
    #     print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
    #           (epoch + 1, training_loss / train_steps, val_accurate))
    #
    #     import matplotlib.pyplot as plt
    #
    #     train_losses = []
    #     val_accuracies = []



    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # train
        net.train()
        training_loss = 0.0
        train_acc = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            predict_train = torch.max(logits, dim=1)[1]
            train_acc += torch.eq(predict_train, labels.to(device)).sum().item()
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            training_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        train_accuracy = train_acc / train_num
        train_loss = training_loss / train_steps
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # validate
        net.eval()
        val_acc = 0.0  # accumulate accurate number / epoch
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_loss += loss_function(outputs, val_labels.to(device)).item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accuracy = val_acc / val_num
        val_loss = val_loss / val_steps
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, train_loss, val_accuracy))

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

    # plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='train accuracy')
    plt.plot(val_accuracies, label='val accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.show()




if __name__ == '__main__':
    main()
