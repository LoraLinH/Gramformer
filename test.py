import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg_graph import vgg19_trans
import argparse
import math
from tqdm import tqdm
from glob import glob
from thop import profile
args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default=r'F:\Dataset\Counting\UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='model/Gramformer_UCF.pth',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)

    device = torch.device('cuda')
    model = vgg19_trans(0.3, 18)
    model.to(device)
    model.eval()
    log_list = []
    epoch_minus = []
    model.load_state_dict(torch.load(args.save_dir, device))
    for inputs, count, name in tqdm(dataloader):
        inputs = inputs.to(device)
        b, c, h, w = inputs.shape
        h, w = int(h), int(w)
        assert b == 1, 'the batch size should equal to 1 in validation mode'
        input_list = []
        c_size = 1024
        if h > c_size or w > c_size:
            h_stride = int(math.ceil(1.0 * h / c_size))
            w_stride = int(math.ceil(1.0 * w / c_size))
            h_step = h // h_stride
            w_step = w // w_stride
            for i in range(h_stride):
                for j in range(w_stride):
                    h_start = i * h_step
                    if i != h_stride - 1:
                        h_end = (i + 1) * h_step
                    else:
                        h_end = h
                    w_start = j * w_step
                    if j != w_stride - 1:
                        w_end = (j + 1) * w_step
                    else:
                        w_end = w
                    input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
            with torch.set_grad_enabled(False):
                pre_count = 0.0
                for idx, input in enumerate(input_list):
                    output = model(input)[0]
                    pre_count += torch.sum(output)
            res = count[0].item() - pre_count.item()
            epoch_minus.append(res)
        else:
            with torch.set_grad_enabled(False):
                outputs = model(inputs)[0]
                res = count[0].item() - torch.sum(outputs).item()
                epoch_minus.append(res)

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'mae {}, mse {}\n'.format(mae, mse)
    print(log_str)
    log_list.append(log_str)