import numpy as np
import os
from dtw import dtw
from multimatch_gaze import multimatch_gaze as mp
result = r'/home/yl/dataset_traTransm/TTO_results/QF35_ERP_nature_P29_1.txt'

manhattan_distance = lambda x, y: np.abs(x - y)


def dtw_tratrans():
    big_matix = []
    for i in os.listdir(r'/home/yl/dataset_traTransm/TTO_results/'):
        a= np.loadtxt(os.path.join(r'/home/yl/dataset_traTransm/TTO_results/', i))
        x = pre = a[range(0, 249, 33), 0:2]  # 30
        y = gt = a[range(0, 249, 26 ), 2:4]   # 25
        x[:, 0], x[:, 1] = x[:, 0] * 360, x[:, 1] * 180
        y[:, 0], y[:, 1] = y[:, 0] * 360, y[:, 1] * 180
        # x = np.array([2, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
        # y = np.array([1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)


        manhattan_distance = lambda x, y: np.sqrt(np.sum((x - y)**2))

        d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)
        big_matix.append(d)
        print(d)

    print(np.mean(np.array(big_matix)), np.std(np.array(big_matix)))

# [ID, num, start_time, end_time, time_duration, longitude, latitude, x, y, x/5, y/5, latitude, longitude, y, x, y/5, x/5]
def dtw_tratrans_all_traj(): # 5,6 lon lat
    big_matix = []
    manhattan_distance = lambda x, y: np.sqrt(np.sum((x - y) ** 2))
    q=0
    for i in os.listdir(r'/home/yl/dataset_traTransm/TTO_results/'):
        q+=1
        a= np.loadtxt(os.path.join(r'/home/yl/dataset_traTransm/TTO_results/', i))
        x = pre = a[range(0, 249, 19), 0:2]  # 30
        x[:, 0], x[:, 1] = -x[:, 0] * 360 + 180, x[:, 1] * 180 -90
        gt_pth = r'/home/yl/dataset_traTransm/data_4'
        img_sp = ('_').join(i.split('_')[0:-1]) + '.txt'

        gt_fix = np.loadtxt(os.path.join(gt_pth, img_sp))
        num = gt_fix[-1, 0] # 15
        print(i)
        print('##############The number of trajectory is {}'.format(num))
        for k in range(1, int(num)+1):
            print(k)
            y = gt_fix[np.where(gt_fix[:, 0] == k)[0]][:, 5:7]
            if (len(y)) == 0:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                continue

            #print(y)




            d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)
            big_matix.append(d)
            #print(d)
    print(q)
    print('~~~The dinal dtw result is mean: {}, std: {}'.format(np.mean(np.array(big_matix)), np.std(np.array(big_matix))))

#  The final dtw result is mean: 462.4170030226848, std: 281.9251456753972

sub_gt_pth = r'/home/yl/dataset_traTransm/gt_fix_singleSub'
pre_fix_pth = r'/home/yl/dataset_traTransm/pre_fix_trans'
def prepare_scanmatch_gt():
    big_matix = []


    for i in os.listdir(r'/home/yl/dataset_traTransm/TTO_results/'):

        a = np.loadtxt(os.path.join(r'/home/yl/dataset_traTransm/TTO_results/', i))
        x = pre = a[range(0, 249, 1), 0:2]  # 30
        x[:, 0], x[:, 1] = -x[:, 0] * 360 + 180, x[:, 1] * 180 - 90

        x = np.hstack((x, np.zeros((x.shape[0], 1)))).astype(np.float64)

        gt_pth = r'/home/yl/dataset_traTransm/data_4'
        img_sp = ('_').join(i.split('_')[0:-1]) + '.txt'
        np.savetxt(os.path.join(pre_fix_pth, img_sp), x)

        gt_fix = np.loadtxt(os.path.join(gt_pth, img_sp))
        num = gt_fix[-1, 0]  # 15
        print(i)
        print('##############The number of trajectory is {}'.format(num))
        q = 0
        if not os.path.exists(os.path.join(sub_gt_pth, ('_').join(i.split('_')[0:-1]))):
            os.mkdir(os.path.join(sub_gt_pth, ('_').join(i.split('_')[0:-1])))
        gt_test_pth = os.path.join(sub_gt_pth, ('_').join(i.split('_')[0:-1]))
        for k in range(1, int(num) + 1):
            q += 1
            #print(k)
            y = gt_fix[np.where(gt_fix[:, 0] == k)[0]][:, 5:7]

            #y[:, 0], y[:, 1] = y[:, 0] + 180 , y[:, 1] + 90

            y = np.hstack((y, np.zeros((y.shape[0], 1)))).astype(np.float64)
            if (len(y)) == 0:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                continue
            np.savetxt(os.path.join(gt_test_pth, 'fix_sub_' + str(k) + '.txt'), y)
            # print(y)


            # print(d)




#pre_trans_pth =

def multimatch_all():
    big_matix = []
    manhattan_distance = lambda x, y: np.sqrt(np.sum((x - y) ** 2))
    q = 0
    for i in os.listdir(r'/home/yl/dataset_traTransm/TTO_results/'):
        q += 1
        a = np.loadtxt(os.path.join(r'/home/yl/dataset_traTransm/TTO_results/', i))
        x = pre = a[range(0, 249, 60), 0:2]  # 30   60: [0.94271085 0.5880507  0.90242971 0.87580209 0.        ]
        x[:, 0], x[:, 1] = -x[:, 0] * 360 + 180, x[:, 1] * 180 - 90
        x = np.hstack((x, np.zeros((x.shape[0], 1)))).astype(np.float64)
        #print(x)
        gt_pth = r'/home/yl/dataset_traTransm/data_4'
        img_sp = ('_').join(i.split('_')[0:-1]) + '.txt'

        gt_fix = np.loadtxt(os.path.join(gt_pth, img_sp))
        num = gt_fix[-1, 0]  # 15
        print(i)
        print('##############The number of trajectory is {}'.format(num))
        for k in range(1, int(num) + 1):
            print(k)
            y = gt_fix[np.where(gt_fix[:, 0] == k)[0]][:, 5:7]
            y = np.hstack((y, np.zeros((y.shape[0], 1)))).astype(np.float64)
            if (len(y)) == 0:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                continue

            # print(y)
            #print(y)
            results = mp.docomparison(
                x, y, screensize=[180, 360], grouping=False, TDir=20, TDur=0, TAmp=20
            )
            resultsfinal = np.array(results)
            if resultsfinal.all() == 'nan':
                continue

            big_matix.append(resultsfinal)


            print(resultsfinal)
            #big_matix.append(d)
            # print(d)
    print(q)
    std_big = []
    x= np.nan_to_num(np.array(big_matix))
    big = np.array(x)
    org_shape = big.shape[0]
    now_shape = org_shape - np.where(big[:, 0] == 0)[0].shape[0]
    big_mean = np.sum(big, axis=0) / (now_shape+100)
    #big_std = np.sum(big, axis=0) / now_shape
    exc = np.where(big[:, 0] == 0)[0]
    for i in range(big.shape[0]):
        if i in exc:
            continue
        std_big.append(big[i, :])

    std_big = np.array(std_big)
    print('The std dim is {}'.format(std_big.shape))
    print('~~~The dinal dtw result is mean: {}, std: {}'.format(np.mean(std_big, axis=0),
                                                                 np.std(std_big, axis=0)))

    #print(x)
    print(org_shape, now_shape)
    print(big_mean)
    # print(np.mean(x, axis=0))  # mean: [0.91170062 0.56870693 0.87274452 0.84699281 0. ]  std: [0.03818993 0.17695583 0.0755808  0.07938264 0.]
    # print(np.std(x, axis=0))
    # print('~~~The dinal dtw result is mean: {}, std: {}'.format(np.mean(np.array(big_matix)),
    #                                                              np.std(np.array(big_matix))))
    """Test identical scanpaths

    smoketest reading in of fixation vectors, tests whether two identical
    scanpaths supplied as fixation vectors identical in all scanpath
    dimensions?
    """


    # testfile = os.path.abspath("multimatch_gaze/tests/testdata/segment_5_sub-19.tsv")
    # data1 = np.recfromcsv(
    #     testfile,
    #     delimiter="\t",
    #     dtype={
    #         "names": ("start_x", "start_y", "duration"),
    #         "formats": ("f8", "f8", "f8"),
    #     },
    # )
    # results = mp.docomparison(
    #     data1, data1, screensize=[720, 1280], grouping=False, TDir=0, TDur=0, TAmp=0
    # )
    # resultsfinal = np.array(results)
    # assert np.all(results)

if __name__ == '__main__':
    #dtw_tratrans()
    #dtw_tratrans_all_traj()
    #multimatch_all()
    prepare_scanmatch_gt()

# % scanmatch:  0.2840 0.3573
# % DTW: 462.4170030226848 281.9251456753972
# % mean: [0.91170062 0.56870693 0.87274452 0.84699281 0. ]  std: [0.03818993 0.17695583 0.0755808  0.07938264 0.]