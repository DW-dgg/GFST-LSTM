import math
import os.path
import cv2
import numpy as np
from numpy import cov, iscomplexobj, trace
from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess
import torch
import codecs
import lpips
from scipy.linalg import sqrtm


def train(model, ims, real_input_flag, configs, itr):
    _, loss_l1, loss_l2 = model.train(ims, real_input_flag, itr)
    if itr % configs.display_interval == 0:
        print('itr: ' + str(itr),
              'training L1 loss: ' + str(loss_l1), 'training L2 loss: ' + str(loss_l2))


def test(model, test_input_handle, configs, itr):
    print('test...')
    loss_fn = lpips.LPIPS(net='alex', spatial=True).to(configs.device)
    res_path = configs.gen_frm_dir + '/' + str(itr)

    if not os.path.exists(res_path):
        os.mkdir(res_path)
    # f = codecs.open(res_path + '/performance.txt', 'w+')
    # f.truncate()
    save_sample = 5
    avg_mse = 0
    avg_mae = 0
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    batch_id = 0
    avg_hss10 = 0
    avg_hss20 = 0
    avg_hss40 = 0

    avg_csi10 = 0
    avg_csi20 = 0
    avg_csi40 = 0

    img_mse, img_mae, img_psnr, ssim, img_lpips, mse_list, mae_list, psnr_list, ssim_list, lpips_list = [], [], [], [], [], [], [], [], [], []
    img_hss10, img_csi10, HSS_list10, CSI_list10 = [], [], [], []
    img_hss20, img_csi20, HSS_list20, CSI_list20 = [], [], [], []
    img_hss40, img_csi40, HSS_list40, CSI_list40 = [], [], [], []
    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        img_mae.append(0)
        img_psnr.append(0)
        ssim.append(0)
        img_lpips.append(0)
        img_hss10.append(0)
        img_hss20.append(0)
        img_hss40.append(0)
        img_csi10.append(0)
        img_csi20.append(0)
        img_csi40.append(0)

        mse_list.append(0)
        mae_list.append(0)
        psnr_list.append(0)
        ssim_list.append(0)
        lpips_list.append(0)
        HSS_list10.append(0)
        HSS_list20.append(0)
        HSS_list40.append(0)
        CSI_list10.append(0)
        CSI_list20.append(0)
        CSI_list40.append(0)
    for epoch in range(1):  
        # if batch_id > configs.num_save_samples:
        #     break
        f = codecs.open(res_path + '/performance.txt', 'w+')
        f.truncate()

        for data in test_input_handle:
            # if batch_id > configs.num_save_samples:
            #     break
            print(batch_id)

            batch_size = data.shape[0]
            real_input_flag = np.zeros(
                (batch_size,
                 configs.total_length - configs.input_length - 1,
                 configs.img_height // configs.patch_size,
                 configs.img_width // configs.patch_size,
                 configs.patch_size ** 2 * configs.img_channel))

            img_gen = model.test(data, real_input_flag)
            img_gen = img_gen.transpose(0, 1, 3, 4, 2)  # * 0.5 + 0.5
            test_ims = data.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)  # * 0.5 + 0.5
            output_length = configs.total_length - configs.input_length
            output_length = min(output_length, configs.total_length - 1)
            test_ims = preprocess.reshape_patch_back(test_ims, configs.patch_size)
            img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
            img_out = img_gen[:, -output_length:, :]

            # MSE per frame
            for i in range(output_length):

                x = test_ims[:, i + configs.input_length, :]
                gx = img_out[:, i, :]
                gx = np.maximum(gx, 0)
                gx = np.minimum(gx, 1)

                mse = np.square(x - gx).sum() / batch_size
                mae = np.abs(x - gx).sum() / batch_size

                hss10, hss20, hss40 = 0, 0, 0
                csi10, csi20, csi40 = 0, 0, 0
                for id_batch in range(batch_size):
                    hssa_10, csia_10 = cal_hss_csi(x, gx, 10, 20)
                    hss10 += hssa_10
                    csi10 += csia_10

                    hssa_20, csia_20 = cal_hss_csi(x, gx, 20, 40)
                    hss20 += hssa_20
                    csi20 += csia_20

                    hssa_40, csia_40 = cal_hss_csi(x, gx, 40, 70)
                    hss40 += hssa_40
                    csi40 += csia_40

                hss10, hss20, hss40 = hss10 / batch_size, hss20 / batch_size, hss40 / batch_size
                csi10, csi20, csi40 = csi10 / batch_size, csi20 / batch_size, csi40 / batch_size

                psnr = 0
                t1 = torch.from_numpy((x - 0.5) / 0.5).to(configs.device)
                t1 = t1.permute((0, 3, 1, 2))
                t2 = torch.from_numpy((gx - 0.5) / 0.5).to(configs.device)
                t2 = t2.permute((0, 3, 1, 2))
                shape = t1.shape
                if not shape[1] == 3:
                    new_shape = (shape[0], 3, *shape[2:])
                    t1.expand(new_shape)
                    t2.expand(new_shape)
                d = loss_fn.forward(t1, t2)
                lpips_score = d.mean()
                lpips_score = lpips_score.detach().cpu().numpy() * 100
                for sample_id in range(batch_size):
                    mse_tmp = np.square(
                        x[sample_id, :] - gx[sample_id, :]).mean()
                    psnr += 10 * np.log10(1 / mse_tmp)
                psnr /= (batch_size)
                img_mse[i] += mse
                img_mae[i] += mae

                img_hss10[i] += hss10
                img_hss20[i] += hss20
                img_hss40[i] += hss40

                img_csi10[i] += csi10
                img_csi20[i] += csi20
                img_csi40[i] += csi40

                img_psnr[i] += psnr
                img_lpips[i] += lpips_score
                mse_list[i] = mse
                mae_list[i] = mae
                HSS_list10[i] = hss10
                HSS_list20[i] = hss20
                HSS_list40[i] = hss40
                CSI_list10[i] = csi10
                CSI_list20[i] = csi20
                CSI_list40[i] = csi40
                psnr_list[i] = psnr
                lpips_list[i] = lpips_score
                avg_mse += mse
                avg_mae += mae
                avg_hss10 += hss10
                avg_hss20 += hss20
                avg_hss40 += hss40
                avg_csi10 += csi10
                avg_csi20 += csi20
                avg_csi40 += csi40
                avg_psnr += psnr
                avg_lpips += lpips_score
                score = 0
                for b in range(batch_size):
                    score += compare_ssim(x[b, :], gx[b, :], multichannel=True)
                score /= batch_size
                ssim[i] += score
                ssim_list[i] = score
                avg_ssim += score

            if batch_id <= save_sample:  
                f.writelines(
                    str(batch_id) + '\n' + ',psnr' + str(psnr_list) + '\n' + ',mse' + str(
                        mse_list) + '\n' + ',mae' + ',HSS10' + str(HSS_list10) + '\n' + ',HSS20' + str(
                        HSS_list20) + '\n' + 'HSS40' + str(
                        HSS_list40) + '\n' + ',CSI10' + str(
                        CSI_list10) + '\n' + ',CSI20' + str(
                        CSI_list20) + '\n' + ',CSI40' + str(
                        CSI_list40) + '\n' + str(
                        mae_list) + '\n' + ',lpips' + str(
                        lpips_list) + '\n' + ',ssim' + str(ssim_list) + '\n')

                res_width = configs.img_width
                res_height = configs.img_height
                img = np.ones((2 * res_height,
                               configs.total_length * res_width,
                               configs.img_channel))
                name = str(batch_id) + '.png'
                file_name = os.path.join(res_path, name)
                for i in range(configs.total_length):
                    img[:res_height, i * res_width:(i + 1) * res_width, :] = test_ims[0, i, :]
                for i in range(output_length):
                    img[res_height:, (configs.input_length + i) * res_width:(configs.input_length + i + 1) * res_width,
                    :] = img_out[0, -output_length + i, :]
                img = np.maximum(img, 0)
                img = np.minimum(img, 1)
                cv2.imwrite(file_name, (img * 255).astype(np.uint8))
            else:
                f.close()

            batch_id = batch_id + 1
    # f.close()

    with codecs.open(res_path + '/data.txt', 'w+') as data_write:
        data_write.truncate()
        nums_sum = batch_id * output_length
        avg_mse = avg_mse / nums_sum
        print('mse per frame: ' + str(avg_mse))
        for i in range(configs.total_length - configs.input_length):
            print(img_mse[i] / batch_id)
            img_mse[i] = img_mse[i] / batch_id
        data_write.writelines('mse per frame: ' + str(avg_mse) + '\n')
        data_write.writelines(str(img_mse) + '\n')
        # --------------------------------------------------
        avg_hss10, avg_hss20, avg_hss40 = avg_hss10 / nums_sum, avg_hss20 / nums_sum, avg_hss40 / nums_sum
        print('hss10 per frame: ' + str(avg_hss10) + ' hss20 per frame: ' + str(avg_hss20) + ' hss40 per frame: ' + str(avg_hss40))
        for i in range(configs.total_length - configs.input_length):
            img_hss10[i], img_hss20[i], img_hss40[i] = img_hss10[i] / batch_id, img_hss20[i] / batch_id, img_hss40[i] / batch_id
        data_write.writelines('hss10 per frame: ' + str(avg_hss10) + '\n')
        data_write.writelines(str(img_hss10) + '\n')

        data_write.writelines('hss20 per frame: ' + str(avg_hss20) + '\n')
        data_write.writelines(str(img_hss20) + '\n')

        data_write.writelines('hss40 per frame: ' + str(avg_hss40) + '\n')
        data_write.writelines(str(img_hss40) + '\n')
        # --------------------------------------------------
        avg_csi10, avg_csi20, avg_csi40 = avg_csi10 / nums_sum, avg_csi20 / nums_sum, avg_csi40 / nums_sum
        print(
'csi10 per frame: ' + str(avg_csi10) + '  csi20 per frame: ' + str(avg_csi20) + '  csi40 per frame: ' + str(avg_csi40))
        for i in range(configs.total_length - configs.input_length):
            img_csi10[i], img_csi20[i], img_csi40[i] = img_csi10[i] / batch_id, img_csi20[i] / batch_id, img_csi40[i] / batch_id
        data_write.writelines('csi10 per frame: ' + str(avg_csi10) + '\n')
        data_write.writelines(str(img_csi10) + '\n')

        data_write.writelines('csi20 per frame: ' + str(avg_csi20) + '\n')
        data_write.writelines(str(img_csi20) + '\n')

        data_write.writelines('csi40 per frame: ' + str(avg_csi40) + '\n')
        data_write.writelines(str(img_csi40) + '\n')
        # --------------------------------------------------
        avg_mae = avg_mae / nums_sum
        print('mae per frame: ' + str(avg_mae))
        for i in range(configs.total_length - configs.input_length):
            print(img_mae[i] / batch_id)
            img_mae[i] = img_mae[i] / batch_id
        data_write.writelines('mae per frame: ' + str(avg_mae) + '\n')
        data_write.writelines(str(img_mae) + '\n')
        # --------------------------------------------------
        avg_psnr = avg_psnr / nums_sum
        print('psnr per frame: ' + str(avg_psnr))
        for i in range(configs.total_length - configs.input_length):
            print(img_psnr[i] / batch_id)
            img_psnr[i] = img_psnr[i] / batch_id
        data_write.writelines('psnr per frame: ' + str(avg_psnr) + '\n')
        data_write.writelines(str(img_psnr) + '\n')
        # --------------------------------------------------
        avg_ssim = avg_ssim / nums_sum
        print('ssim per frame: ' + str(avg_ssim))
        for i in range(configs.total_length - configs.input_length):
            print(ssim[i] / batch_id)
            ssim[i] = ssim[i] / batch_id
        data_write.writelines('ssim per frame: ' + str(avg_ssim) + '\n')
        data_write.writelines(str(ssim) + '\n')
        # --------------------------------------------------
        avg_lpips = avg_lpips / nums_sum
        print('lpips per frame: ' + str(avg_lpips))
        for i in range(configs.total_length - configs.input_length):
            print(img_lpips[i] / batch_id)
            img_lpips[i] = img_lpips[i] / batch_id
        data_write.writelines('lpips per frame: ' + str(avg_lpips) + '\n')
        data_write.writelines(str(img_lpips) + '\n')
        # --------------------------------------------------
        hss_csi_score = ((avg_hss10 + avg_csi10) * 0.5) * 0.25 + ((avg_hss20 + avg_csi20) * 0.5) * 0.35 + (
                (avg_hss40 + avg_csi40) * 0.5) * 0.4
        data_write.writelines('hss_csi_score per frame: ' + str(hss_csi_score) + '\n')


def prep_clf(obs, pre, low=0, high=70):
    '''
    func: 计算二分类结果-混淆矩阵的四个元素
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        hits, misses, falsealarms, correctnegatives
        #aliases: TP, FN, FP, TN
    '''
    # 根据阈值分类为 0, 1
    obs = (np.where(obs >= low, 1, 0)) & (np.where(obs < high, 1, 0))
    pre = (np.where(pre >= low, 1, 0)) & (np.where(pre < high, 1, 0))
    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1)).astype(np.float64)
    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0)).astype(np.float64)
    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1)).astype(np.float64)
    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0)).astype(np.float64)

    return hits, misses, falsealarms, correctnegatives


def HSS(hits, misses, falsealarms, correctnegatives):
    '''
    HSS - Heidke skill score
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): pre
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: HSS value
    '''

    HSS_num = 2 * (hits * correctnegatives - misses * falsealarms)
    HSS_den = (misses ** 2 + falsealarms ** 2 + 2 * hits * correctnegatives +
               (misses + falsealarms) * (hits + correctnegatives))
    if HSS_den == 0:
        return 0.0
    return HSS_num / HSS_den


def CSI(hits, misses, falsealarms, correctnegatives):
    denominator = (hits + falsealarms + misses)
    if (denominator == 0):
        return 0.0
    return hits / denominator


def BIAS(hits, misses, falsealarms, correctnegatives):
    '''
    func: 计算Bias评分: Bias =  (hits + falsealarms)/(hits + misses)
    	  alias: (TP + FP)/(TP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''

    if hits + misses == 0:
        return 0.0
    return (hits + falsealarms) / (hits + misses)


def calculate_fid(image1, image2):
    # calculate mean and covariance statistics
    mu1, sigma1 = image1.mean(axis=0), cov(image1, rowvar=False)
    mu2, sigma2 = image2.mean(axis=0), cov(image2, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) * 2.0)

    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    fid = 0
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def escore(obs, pre, low, high):
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           low=low, high=high)

    hss = HSS(hits, misses, falsealarms, correctnegatives)
    csi = CSI(hits, misses, falsealarms, correctnegatives)

    # bias = BIAS(hits, misses, falsealarms, correctnegatives)
    # fid = calculate_fid(obs, pre)
    # score = hss * (math.exp(-abs(1 - bias)) ** 0.2) * (math.exp(-(fid / 100)) ** 0.2)
    score = hss * csi
    return score


def cal_hss_csi(obs, pre, low, high):
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           low=low/70.0, high=high/70.0)
    hss = HSS(hits, misses, falsealarms, correctnegatives)
    csi = CSI(hits, misses, falsealarms, correctnegatives)
    return hss, csi


def radar_sc(image1, image2):
    return escore(image1, image2, 10, 20) * 0.3 + escore(image1, image2, 20, 40) * 0.3 + escore(image1, image2, 40,70) * 0.4