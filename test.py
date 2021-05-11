import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Dict from labels to names
    label_to_names = {0: 'ceiling',
                    1: 'floor',
                    2: 'wall',
                    3: 'beam',
                    4: 'column',
                    5: 'window',
                    6: 'door',
                    7: 'chair',
                    8: 'table',
                    9: 'bookcase',
                    10: 'sofa',
                    11: 'board',
                    12: 'clutter'
                    }
    # Dict from labels to colours
    colour_to_label = {0: [ 233, 229, 107], #'ceiling' .-> .yellow
                    1: [  95, 156, 196], #'floor' .-> . blue
                    2: [ 179, 116,  81], #'wall'  ->  brown
                    3: [ 241, 149, 131], #'beam'  ->  salmon
                    4: [  81, 163, 148], #'column'  ->  bluegreen
                    5: [  77, 174,  84], #'window'  ->  bright green
                    6: [ 108, 135,  75], #'door'   ->  dark green
                    7: [  41,  49, 101], #'chair'  ->  darkblue
                    8: [  79,  79,  76], #'table'  ->  dark grey
                    9: [223,  52,  52], #'bookcase'  ->  red
                    10: [ 89,  47,  95], #'sofa'  ->  purple
                    11: [ 81, 109, 114], #'board'   ->  grey
                    12: [233, 233, 229], #'clutter'  ->  light grey
                    13: [0   ,   0,   0], #unlabelled .->. black
                    }
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.axis([0, 4, 0, 20])
    for i in range(13):
        y = 20 - (i*3+1)
        if y > 0:
            x = 0.5
        else:
            x = 2.5
            y = -y
        ax.plot([x], [y], '.', color=(colour_to_label[i][0]/255.0, 
                                    colour_to_label[i][1]/255.0, 
                                    colour_to_label[i][2]/255.0),  
                markersize=40) 
        ax.text(x+0.25, y-0.5, label_to_names[i], fontsize=15)
    plt.show()
