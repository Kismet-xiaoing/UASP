import time
import numpy as np
import torch
import os
import argparse
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn import preprocessing


def set_seed(seed=1028):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
set_seed()

parser = argparse.ArgumentParser(description='Calculate CI')

parser.add_argument(
    '--arch',
    type=str,
    default='resnet_110',
    choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50'),
    help='architecture to calculate feature maps')

parser.add_argument(
    '--repeat',
    type=int,
    default=5,
    help='repeat times')

parser.add_argument(
    '--num_layers',
    type=int,
    default=109,
    help='conv layers in the model')

parser.add_argument(
    '--feature_map_dir',
    type=str,
    default='./100conv_feature_map',
    help='feature maps dir')

args = parser.parse_args()


def Spearmancorr(x,y):
    def _rank_corr_(a,b):
        n = torch.tensor(a.shape[0])
        upper = 6 * torch.sum((b - a).pow(2), dim=1)
        down = n * (n.pow(2) - 1.0)
        return (1.0 - (upper / down)).mean(dim=-1)
    x = x.sort(dim=1)[1]
    y = y.sort(dim=1)[1]
    corr = _rank_corr_(x.float(), y.float())
    return corr

def Personcorr(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-7)
    return cost

def Coscorr(x, y):
    cos = torch.nn.functional.cosine_similarity(torch.flatten(x), torch.flatten(y), dim=0)
    return cos

def reduced_1_row_norm(input, row_index, data_index):
    input[data_index, row_index, :] = torch.zeros(input.shape[-1])
    m = torch.norm(input[data_index, :, :], p = 'nuc').item()
    return m

def ci_score(path_conv):
    conv_output = torch.tensor(np.round(np.load(path_conv), 4))
    conv_reshape = conv_output.reshape(conv_output.shape[0], conv_output.shape[1], -1).cuda() #(B, C, H*W)

    r1_norm = torch.zeros([conv_reshape.shape[0], conv_reshape.shape[1]]) # (B, C)
    for i in range(conv_reshape.shape[0]):
        original_norm = conv_reshape[i, :, :].clone().detach() #(C, H*W)
        for j in range(conv_reshape.shape[1]):
            # r1_norm[i, j] = reduced_1_row_norm(conv_reshape.clone(), j, data_index = i)
            tmp = conv_reshape.clone()
            tmp[i, j, :] = torch.zeros(tmp.shape[-1])
            # r1_norm[i, j] = Coscorr(original_norm, tmp[i])
            r1_norm[i, j] = Personcorr(original_norm, tmp[i])
            # r1_norm[i, j] = Spearmancorr(original_norm, tmp[i])
        # print(r1_norm[i].shape)

    ci = np.zeros_like(r1_norm.cpu())

    for i in range(r1_norm.shape[0]):
        # original_norm = torch.norm(torch.tensor(conv_reshape[i, :, :]), p='nuc').item()
        # ci[i] = (original_norm - r1_norm[i]).cpu()
        ci[i] = (1.0 - r1_norm[i]).cpu()
        # ci[i] = r1_norm[i]
        # num = ci[i].shape[0]
        # dist = np.zeros((num, num))
        # for a in range(num):
        #     for b in range(a, num):
        #         distance = ci[i][a]-ci[i][b]
        # B = preprocessing.minmax_scale(ci[i].reshape(-1, 1))
        #+++++++++print("ci: ", ci[i])++++++++++++++
        #ttmp = ci[i].copy()
        #ttmp = (ttmp-ttmp.min())/(ttmp.max()-ttmp.min())
        #mn = np.sort(abs(ttmp-np.median(ttmp)))[1]
        #t = min(mn, 0.00001)
        #print("t: ",t)
        #mask = np.ones_like(ttmp)
        #clustering = Clustering(ttmp, thr=t)
        #for j in clustering:
        #    if type(j) == list:
        #        k = 0
        #        index = 0
        #        mk = ttmp[j[k]]
        #        while k+1 < len(j):
        #            if ttmp[j[k+1]] < mk :
        #                mask[j[k + 1]] = 0
        #            else:
        #                mask[j[index]] = 0
        #                mk = ttmp[j[k+1]]
        #                index = k+1
        #            k += 1
        #ci[i] *= mask
        #print(f"Numbers of zeroes in Mask: {mask.size-np.count_nonzero(mask)}")
    #B = preprocessing.minmax_scale(ci, axis=0)
    #julei = sch.linkage(B.T, metric='euclidean', method='single')
    #Z = sch.cut_tree(julei, height=0.1)
    #print(Z)
    return ci



def Clustering(ci, thr=0.01):
    #ci = (ci - ci.min())/(ci.max()-ci.min())
    label = list(range(len(ci)))
    D = 0
    def distance(a, b):
        if type(a) == int and type(b) == int:
            return abs(ci[a]- ci[b])
        elif type(a) == int and type(b) == list:
            return min(abs(ci[a]-ci[i]) for i in b)
        elif type(a) == list and type(b) == int:
            return min(abs(ci[i]-ci[b]) for i in a)
        elif type(a) == list and type(b) == list:
            return min(abs(ci[i]-ci[j]) for i in a for j in b)
        else:return 'Error! Type distance() must int or list!'

    while D < thr:
        dis = np.zeros(shape=(len(label),len(label)))
        for i in range(dis.shape[0]):
            for j in range(dis.shape[1]):
                if j > i:
                    dis[i, j] = distance(label[i], label[j])
                else:
                    dis[i, j] = float('inf')
        #print(dis)
        a = label[np.where(dis == np.min(dis))[0][0]]
        b = label[np.where(dis == np.min(dis))[1][0]]
        list2 = [a]+[b]
        #print(list2)
        for k in list2:
            label.remove(k)
        list3 = []
        if type(a)==int and type(b)==int:
            list3 = [a]+[b]
        elif type(a)==int and type(b)==list:
            list3 = [a]+b
        elif type(a)==list and type(b)==int:
            list3 = a+[b]
        elif type(a)==list and type(b)==list:
            list3 = a+b
        else:
            pass
        label.append(list3)
        D = np.min(dis)
        #print(label)
        #print(np.min(D))
    return label

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def z_score_normalize(data):
    mean = np.mean(data, axis=0) # 计算均值
    std_dev = np.std(data, axis=0) # 计算标准差
    normalized_data = (data - mean) / std_dev # 进行标准化
    return normalized_data
def softmax(x, T=1.0):
    """稳定版 softmax 带温度参数 T 控制平滑度"""
    x = np.asarray(x)
    x = x - np.max(x)  # 防止溢出
    exp_x = np.exp(x / T)
    return exp_x / np.sum(exp_x)

def mean_repeat_ci(repeat, num_layers):
    layer_ci_mean_total = []
    for j in range(num_layers):
        print("layers: ", j)
        #if j < 45 :
        #    continue
        repeat_ci_mean = []
        for i in range(repeat):
            ts = time.time()
            print("repeat ", i, " || ", ts)
            index = j * repeat + i + 1
            # add
            path_conv = "./100conv_feature_map/{0}_repeat5/conv_feature_map_tensor({1}).npy".format(str(args.arch), str(index))
            path_lxl = "./100conv_feature_map/{0}_repeat5/conv_sample_list_tensor({1}).npy".format(str(args.arch), str(index))
            # path_nuc = "./feature_conv_nuc/resnet_56_repeat5/feature_conv_nuctensor({0}).npy".format(str(index))
            # batch_ci = ci_score(path_conv, path_nuc)
            lxl = np.load(path_lxl)
            # midd = (lxl.max() + lxl.min())/2
            #midd = np.median(lxl)
            # mea = np.mean(lxl)
            # print(sample_lxl[:,None])
            # print(sample_lxl)
            sim = ci_score(path_conv)
            sim = z_score_normalize(sim)
            #print(time.time()-ts," s")
            # median normalization
            #sample_lxl = np.array([0 if j > midd else 1 for j in lxl])
            #sample_lxl = (abs(lxl - lxl.min()) / (lxl.max() - lxl.min()))
            # sample_lxl = 1.0 - (abs(lxl - midd) / (lxl.max() - lxl.min()))
            # Z normalization
            # sample_lxl = (lxl - mea)/np.std(lxl)
            sample_lxl = sigmoid(lxl)
            batch_ci = sim * sample_lxl[:, None]
            # select_index1 = np.argsort(np.mean(sim, axis=0))
            # select_index2 = np.argsort(np.mean(batch_ci, axis=0))
            # print(select_index1,"\n",select_index2)
            # plt.figure()
            # plt.plot(np.arange(batch_ci.shape[0]), lxl, label='globe', color='r')
            # plt.plot(np.arange(batch_ci.shape[0]), np.linalg.norm(batch_ci, ord=2, axis=1), label='score', color='g')
            # plt.plot(np.arange(batch_ci.shape[0]), np.linalg.norm(sim, ord=2, axis=1), label='local', color='b')
            # plt.legend()
            # plt.show()
            single_repeat_ci_mean = np.mean(batch_ci, axis=0)
            print(time.time()-ts," s")
            repeat_ci_mean.append(single_repeat_ci_mean)
        print(j," finish")

        layer_ci_mean = np.mean(repeat_ci_mean, axis=0)
        layer_ci_mean_total.append(layer_ci_mean)

    return np.array(layer_ci_mean_total,dtype=object)

def main():
    repeat = args.repeat
    num_layers = args.num_layers
    save_path = '100CI_' + args.arch
    ci = mean_repeat_ci(repeat, num_layers)
    if args.arch == 'resnet_50':
        num_layers = 53
    for i in range(num_layers):
        print(i)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save(save_path + "/ci_conv{0}.npy".format(str(i + 1)), ci[i])

if __name__ == '__main__':
    main()



