import os
import json

import torch
from PIL import Image
from torchvision import transforms
from model import resnet34,resnet50
from tqdm import tqdm

def main(image_path,model,json_path,output_file):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    file_write = open(output_file,'w')

    filelist = os.listdir(image_path)
    for file in tqdm(filelist):
        img_path = os.path.join(image_path,file)
        assert os.path.exists(image_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path).convert('RGB')
        # plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        # read class_indict
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            score = predict[predict_cla].numpy()
        file_write.write(str(file) + '\t' + str(score) + '\t' + str(class_indict[str(predict_cla)]) + '\n')
    file_write.close()

            # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
            #                                              predict[predict_cla].numpy())
        # plt.title(print_res)
        # for i in range(len(predict)):
        #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
        #                                               predict[i].numpy()))
        # plt.show()


if __name__ == '__main__':

    # prediction
    image_path = '../test_all'
    weights_path = './model_weight/Resnet-34-0422.pth'
    json_path = './class_indices.json'
    output_file = './output/ResNet_test_data_res_0422.txt'

    # create model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet34(num_classes=7).to(device)

    # model = resnet18(num_classes=3).to(device)
    # model = Resnet18(num_classes=3, improved=False).to(device)
    # load model weights
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    main(image_path,model,json_path,output_file)
