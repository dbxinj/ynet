from singa import tensor
from PIL import image
import numpy as np

import matplotlib.pyplot as plt

def vis_attention(args, data, net, nbatch=1):
    # data.shopid_pid
    data.start(0, 1)
    count = 0
    for i in range(nbatch):
        origin_img, pid = data.next()
        img = net.put_input_to_gpu(origin_img)
        fea = net.forward_layers(True, img, net.shared[0:-1] + net.shop[0:-2])
        _, w = net.shop[-2].forward(True, [fea, data.tag2vec(pid)], output_weight=True)
        s = w.shape
        pos = np.argmax(w)
        grad = np.zeros(s[0], fea.shape[1], s[1])
        for i in range(s[0]):
            grad[i, :, pos[i]] = 1
        grad=grad.reshape(fea.shape)
        tgrad = tensor.from_numpy(grad)
        tgrad.to_device(net.device)
        params = []
        dx = net.backward_layers(tgrad, net.shop[-3::-1] + net.shared[-2::-1], params)
        dx = tensor.to_numpy(dx)
        for i in range(s[0]):
            save_img(origin_img[i], dx[i], os.path.join(args.vis_dir, '%d.jpg' % count))
            count += 1
    data.stop()


def save_img(img_original, saliency, title):
    # convert saliency from c01 to 01c
    saliency = saliency.transpose(1, 2, 0)
    # plot the original image and the three saliency map variants
    plt.figure(figsize=(10, 10), facecolor='w')
    # plt.suptitle("Class: " + classes[max_class] + ". Saliency: " + title)
    plt.subplot(2, 2, 1)
    plt.title('input')
    plt.imshow(img_original)
    plt.subplot(2, 2, 2)
    plt.title('abs. saliency')
    plt.imshow(np.abs(saliency).max(axis=-1), cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title('pos. saliency')
    plt.imshow((np.maximum(0, saliency) / saliency.max()))
    plt.subplot(2, 2, 4)
    plt.title('neg. saliency')
    #plt.imshow((np.maximum(0, -saliency) / -saliency.min()))
    #plt.show()
    plt.savefig(title, bbox_inches='tight')
