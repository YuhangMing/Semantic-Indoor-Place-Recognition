from fileinput import filename
from multiprocessing.sharedctypes import Value
import numpy as np


if __name__ == '__main__':
    # file_name = '0-MyPRNet_Default_chkp60.txt'
    # file_name = '2-SIFT-BoW.txt'
    # file_name = '3-NetVLAD.txt'
    # file_name = '4-PointNetVLAD.txt'
    # file_name = '5-MinkLoc3D.txt'
    file_name = '6-IndoorDH3D.txt'
    
    # load the text file
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # print(lines)
    
    # loop through the lines and compute rates
    pcd_count = 0
    top1 = 0.
    succeeded = 0.
    for idx, line in enumerate(lines):
        # if line[0] in '0123456789':
        if line[0] in '0123456789/s':
            # valid pcd
            pcd_count += 1
            
            # check the next line:
            next_line = lines[idx+1]
            if next_line[0] not in ':-':
                raise ValueError('Should not happen.')
            next_line = next_line.strip().split(' ')
            print(next_line)
            # print(next_line[1], next_line[2])

            # # CGiS-Net
            # result = next_line[1]
            # query_name = line.strip().split(' ')[-1]
            # q_scene = query_name.split('_')
            # q_scene = q_scene[0]+'_'+q_scene[1]    
            # retri_name = next_line[2]
            # r_scene = retri_name.split('_')
            # r_scene = r_scene[0]+'_'+r_scene[1]

            # # SIFT+BoW
            # result = next_line[-1]
            # query_name = line.strip().split(' ')[-1]
            # q_scene = query_name.split('/')[-3]
            # retri_name = next_line[1]
            # r_scene = retri_name.split('/')[-3]

            # # NetVLAD
            # result = next_line[1][:-1]
            # query_name = line.strip().split(' ')[-1]
            # q_scene = query_name.split('/')[-3]
            # retri_name = next_line[2]
            # r_scene = retri_name.split('/')[-3]

            # # PointNetVLAD & MinkLoc3D
            # result = next_line[-1]
            # query_name = line.strip().split(' ')[0]
            # q_scene = query_name[:12]
            # retri_name = next_line[0]
            # r_scene = retri_name[2:14]

            # Indoor DH3D
            result = next_line[1]
            query_name = line.strip().split(' ')[0]
            q_scene = query_name[:12]
            retri_name = next_line[-2]
            r_scene = retri_name[:12]

            print(result, q_scene, r_scene)

            if result == 'SUCCESS':
                top1 += 1
                succeeded += 1
            else:
                # print(next_line[1], q_scene, r_scene)
                if q_scene == r_scene:
                    succeeded += 1
    print(top1/pcd_count)
    print(succeeded/pcd_count)



    print(pcd_count)
