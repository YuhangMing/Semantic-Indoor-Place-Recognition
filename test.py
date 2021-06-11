import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # ## plot legends ##
    # # Dict from labels to names
    # label_to_names = {0: 'ceiling',
    #                 1: 'floor',
    #                 2: 'wall',
    #                 3: 'beam',
    #                 4: 'column',
    #                 5: 'window',
    #                 6: 'door',
    #                 7: 'chair',
    #                 8: 'table',
    #                 9: 'bookcase',
    #                 10: 'sofa',
    #                 11: 'board',
    #                 12: 'clutter'
    #                 }
    # # Dict from labels to colours
    # label_to_colour = {0: [ 233, 229, 107], #'ceiling' .-> .yellow
    #                 1: [  95, 156, 196], #'floor' .-> . blue
    #                 2: [ 179, 116,  81], #'wall'  ->  brown
    #                 3: [ 241, 149, 131], #'beam'  ->  salmon
    #                 4: [  81, 163, 148], #'column'  ->  bluegreen
    #                 5: [  77, 174,  84], #'window'  ->  bright green
    #                 6: [ 108, 135,  75], #'door'   ->  dark green
    #                 7: [  41,  49, 101], #'chair'  ->  darkblue
    #                 8: [  79,  79,  76], #'table'  ->  dark grey
    #                 9: [223,  52,  52], #'bookcase'  ->  red
    #                 10: [ 89,  47,  95], #'sofa'  ->  purple
    #                 11: [ 81, 109, 114], #'board'   ->  grey
    #                 12: [233, 233, 229], #'clutter'  ->  light grey
    #                 13: [0   ,   0,   0], #unlabelled .->. black
    #                 }
    
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.axis([0, 4, 0, 20])
    # for i in range(13):
    #     y = 20 - (i*3+1)
    #     if y > 0:
    #         x = 0.5
    #     else:
    #         x = 2.5
    #         y = -y
    #     ax.plot([x], [y], '.', color=(label_to_colour[i][0]/255.0, 
    #                                 label_to_colour[i][1]/255.0, 
    #                                 label_to_colour[i][2]/255.0),  
    #             markersize=40) 
    #     ax.text(x+0.25, y-0.5, label_to_names[i], fontsize=15)
    # plt.show()

    ## plot loss ##
    steps = []
    loss = []
    count = 1
    avg = 600
    # with open('results/Recog_Log_2021-05-25_03-34-01/training.txt') as f:     # SGD 0.01, 5 features
    with open('results/Recog_Log_2021-05-26_09-28-44/training.txt') as f:     # Adam 0.0001, 3 features
        lines = f.readlines()
        tmp = 0
        for line in lines:
            line = line.rstrip().split(' ')
            epoch = int(line[0][1:4])
            # one_loss = float(line[1][5:])
            # one_loss = float(line[1][2:-1])
            one_loss = float(line[1][2:])
            tmp += one_loss
            if count % avg ==0:
                steps.append(count/avg)
                loss.append(tmp/avg)
                tmp = 0
            count += 1
    x1 = np.array(steps)
    y = np.array(loss)
    print(np.max(y))
    # plt.plot(x1, y)
    # plt.show()

    steps = []
    loss = []
    count = 1
    with open('results/Recog_Log_2021-05-26_11-51-58/training.txt') as f:     # Adam 0.0001, 5 features
        lines = f.readlines()
        tmp = 0
        for line in lines:
            line = line.rstrip().split(' ')
            epoch = int(line[0][1:4])
            one_loss = float(line[1][2:])
            tmp += one_loss
            if count % avg ==0:
                steps.append(count/avg)
                loss.append(tmp/avg)
                tmp = 0
            count += 1
    
    x2 = np.array(steps)
    z = np.array(loss)
    print(np.max(z))

    up_bd = min(len(x1), len(x2))
    plt.plot(x1[:up_bd], y[:up_bd], 'r', label='3 features')
    plt.plot(x2[:up_bd], z[:up_bd], 'b', label='5 features')
    plt.legend()
    plt.title('Loss per epoch')
    plt.xlabel('Epoches')
    plt.xlim([0, 50])
    plt.ylabel('Triplet Loss')
    plt.show()

