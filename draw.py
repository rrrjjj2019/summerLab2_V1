import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
def draw_loss(EPOCH, lst_loss, title):
    
    iter_num = list(range(1, EPOCH+1))
    
    plt.figure(0)
    plt.plot(iter_num, lst_loss, '-b', label='loss')

    plt.xlabel("EPOCH")
    plt.legend(loc='lower right')
    plt.title(title)

    # save image
    plt.savefig(title+".png")  # should before show method

    # show
    plt.show()

def draw_accu(EPOCH, lst_loss, title):
    
    iter_num = list(range(1, EPOCH+1))
    
    
    plt.figure(1)
    plt.plot(iter_num, lst_loss, '-b', label='accuracy')

    plt.xlabel("EPOCH")
    plt.legend(loc='lower right')
    plt.title(title)

    # save image
    plt.savefig(title+".png")  # should before show method

    # show
    plt.show()


if __name__ == '__main__':
    draw_loss(7, [1, 2, 3, 4, 5, 50, 60], 'loss')