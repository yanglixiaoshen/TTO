import numpy as np
import os

import shutil

org_dir = r'/media/yl/yl_8t/traTransformer_experiments/imp_img' # 1080  original images
org_gt = r'/media/yl/yl_8t/traTransformer_experiments/data_1'  # 1080  gt

new_train_dir = r'/media/yl/yl_8t/traTransformer_experiments/imp_train_trans' # 900 * 12  copy images  resize to 512 * 512
new_test_dir = r'/media/yl/yl_8t/traTransformer_experiments/imp_test_trans'   # 180 * 12
hibayesTest = r'/media/yl/yl_8t/traTransformer_experiments/DTW_sal_eu' # 180

gt_train_dir = r'/media/yl/yl_8t/traTransformer_experiments/gt_train_trans'   # 900 * 12  different trajectories
gt_test_dir = r'/media/yl/yl_8t/traTransformer_experiments/gt_test_trans'     # 180 * 12

tra_train_id = r'/media/yl/yl_8t/traTransformer_experiments/trans_train_id.txt'
tra_test_id = r'/media/yl/yl_8t/traTransformer_experiments/trans_test_id.txt'

train_res = r'/media/yl/yl_8t/traTransformer_experiments/train_transformer_img_resize'
test_res = r'/media/yl/yl_8t/traTransformer_experiments/test_transformer_img_resize'

gt_train_norm = r'/media/yl/yl_8t/traTransformer_experiments/gt_train_norm'
gt_test_norm = r'/media/yl/yl_8t/traTransformer_experiments/gt_test_norm'



def make_gt_id():
    test_list = os.listdir(hibayesTest)
    test_id_list = []
    train_id_list = []
    for img_data in os.listdir(org_gt):
        if img_data in test_list:  # For test pre-processing
            img_data_splt = img_data.split('.')[0]
            img_test = img_data_splt + '.jpg'
            img_test_gt = np.loadtxt(os.path.join(org_gt, img_data))  # 17000 human data lon and lat
            for i in range(12):
                i += 1
                shutil.copyfile(os.path.join(org_dir, img_test), os.path.join(new_test_dir, img_data_splt + '_' + str(i) + '.jpg'))
                data_gt_sub = img_test_gt[(i-1)*1000: 1000*i]
                data_for_train_dub = data_gt_sub[range(0, 1000-4, 4), 2:4]  # 0 : 249
                data_for_gt_sub = data_gt_sub[range(4, 1000, 4) , 2:4]      # 1 : 250
                data_train_plus_gt_sub = np.hstack((data_for_train_dub, data_for_gt_sub))  # (249, 4)
                np.savetxt(os.path.join(gt_test_dir, img_data_splt + '_' + str(i) + '.txt'), data_train_plus_gt_sub)
                test_id_list.append(img_data_splt + '_' + str(i))


        else:  # For test pre-processing
            img_data_splt = img_data.split('.')[0]
            img_train = img_data_splt + '.jpg'
            img_train_gt = np.loadtxt(os.path.join(org_gt, img_data))  # 17000 human data lon and lat
            for i in range(12):
                i += 1
                shutil.copyfile(os.path.join(org_dir, img_train), os.path.join(new_train_dir, img_data_splt + '_' + str(i) + '.jpg'))
                data_gt_sub = img_train_gt[(i-1)*1000: 1000*i]
                print(data_gt_sub.shape)
                data_for_train_dub = data_gt_sub[range(0, 1000-4, 4), 2:4]  # 0 : 249
                print(data_for_train_dub.shape)
                data_for_gt_sub = data_gt_sub[range(4, 1000, 4) , 2:4]      # 1 : 250
                data_train_plus_gt_sub = np.hstack((data_for_train_dub, data_for_gt_sub))  # (249, 4)
                np.savetxt(os.path.join(gt_train_dir, img_data_splt + '_' + str(i) + '.txt'), data_train_plus_gt_sub)
                train_id_list.append(img_data_splt + '_' + str(i))

    np.savetxt(tra_train_id, np.array(train_id_list), fmt="%s")
    np.savetxt(tra_test_id, np.array(test_id_list), fmt="%s")

def make_gt_id1():
    test_list = os.listdir(hibayesTest)
    test_id_list = []
    train_id_list = []
    for img_data in os.listdir(org_gt):
        if img_data in test_list:  # For test pre-processing
            img_data_splt = img_data.split('.')[0]
            img_test = img_data_splt + '.jpg'
            # img_test_gt = np.loadtxt(os.path.join(org_gt, img_data))  # 17000 human data lon and lat
            for i in range(12):
                i += 1
                # shutil.copyfile(os.path.join(org_dir, img_test), os.path.join(new_test_dir, img_data_splt + '_' + str(i) + '.jpg'))
                # data_gt_sub = img_test_gt[(i-1)*1000: 1000*i]
                # data_for_train_dub = data_gt_sub[range(0, 1000-4, 4), 2:4]  # 0 : 249
                # data_for_gt_sub = data_gt_sub[range(4, 1000, 4) , 2:4]      # 1 : 250
                # data_train_plus_gt_sub = np.hstack((data_for_train_dub, data_for_gt_sub))  # (249, 4)
                # np.savetxt(os.path.join(gt_test_dir, img_data_splt + '_' + str(i) + '.txt'), data_train_plus_gt_sub)
                test_id_list.append(img_data_splt + '_' + str(i))


        else:  # For test pre-processing
            img_data_splt = img_data.split('.')[0]
            img_train = img_data_splt + '.jpg'
            #img_train_gt = np.loadtxt(os.path.join(org_gt, img_data))  # 17000 human data lon and lat
            for i in range(12):
                i += 1
                # shutil.copyfile(os.path.join(org_dir, img_train), os.path.join(new_train_dir, img_data_splt + '_' + str(i) + '.jpg'))
                # data_gt_sub = img_train_gt[(i-1)*1000: 1000*i]
                # print(data_gt_sub.shape)
                # data_for_train_dub = data_gt_sub[range(0, 1000-4, 4), 2:4]  # 0 : 249
                # print(data_for_train_dub.shape)
                # data_for_gt_sub = data_gt_sub[range(4, 1000, 4) , 2:4]      # 1 : 250
                # data_train_plus_gt_sub = np.hstack((data_for_train_dub, data_for_gt_sub))  # (249, 4)
                # np.savetxt(os.path.join(gt_train_dir, img_data_splt + '_' + str(i) + '.txt'), data_train_plus_gt_sub)
                train_id_list.append(img_data_splt + '_' + str(i))

    np.savetxt(tra_train_id, np.array(train_id_list), fmt="%s")
    np.savetxt(tra_test_id, np.array(test_id_list), fmt="%s")


def resize_image():
    from PIL import Image
    # for odi in os.listdir(new_train_dir):
    #
    #     im = Image.open(os.path.join(new_train_dir, odi))
    #     img = im.resize((512, 512), Image.BILINEAR)
    #     print(np.asarray(img).shape)
    #     ims = img.save(os.path.join(train_res, odi))
    for odi in os.listdir(new_test_dir):

        im = Image.open(os.path.join(new_test_dir, odi))
        img = im.resize((512, 512), Image.BILINEAR)
        print(np.asarray(img).shape)
        ims = img.save(os.path.join(test_res, odi))


# lon lat to 0, 1 range of fixation ((0,0) lcoates at upleft)
def normalize_fix():
    for gt_name in os.listdir(gt_train_dir):

        gt = np.loadtxt(os.path.join(gt_train_dir, gt_name))

        gt1 = gt

        gt1[:, 0], gt1[:, 1], gt1[:, 2], gt1[:, 3] = (-gt1[:, 0] + 180) / 360, (gt1[:, 1] + 90)/180, (-gt1[:, 2] + 180) / 360, (gt1[:, 3] + 90)/180

        np.savetxt(os.path.join(gt_train_norm, gt_name), gt1)

    for gt_name in os.listdir(gt_test_dir):

        gt = np.loadtxt(os.path.join(gt_test_dir, gt_name))

        gt1 = gt

        gt1[:, 0], gt1[:, 1], gt1[:, 2], gt1[:, 3] = (-gt1[:, 0] + 180) / 360, (gt1[:, 1] + 90)/180, (-gt1[:, 2] + 180) / 360, (gt1[:, 3] + 90)/180

        np.savetxt(os.path.join(gt_test_norm, gt_name), gt1)
    # plt.imshow(img)
    # plt.show()




if __name__ == '__main__':
    #make_gt_id()
    #make_gt_id1()
    resize_image()
    #normalize_fix()