from singa import tensor
import numpy as np
import os
import matplotlib as mlt
mlt.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.axis('off')
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

def vis_attention(args, data, net, nbatch=2):
    # data.shopid_pid
    data.start(0, 1)
    count = 0
    meanstd = np.load(os.path.join(args.data_dir, args.dataset, 'mean-std.npy'))
    tags = None
    for i in range(nbatch):
        origin_img, pid = data.next()
        #origin_img = img + meanstd[2][np.newaxis, :, np.newaxis, np.newaxis]
        #origin_img *= meanstd[3][np.newaxis, :, np.newaxis, np.newaxis]
        img = origin_img - meanstd[2][np.newaxis, :, np.newaxis, np.newaxis]
        img /= meanstd[3][np.newaxis, :, np.newaxis, np.newaxis]
        img = net.put_input_to_gpu(img)
        fea = net.forward_layers(True, img, net.shared[0:-1] + net.shop[0:-2])
        tag = data.tag2vec(pid)
        if tags is None:
            tags = np.empty((nbatch*data.batchsize, tag.shape[1]))
        tags[i*data.batchsize: (i+1)*data.batchsize] = tag
        _, w = net.shop[-2].forward(True, [fea, tag], output_weight=True)
        s = w.shape
        pos = np.argmin(w, axis=1)
        grad = np.zeros((s[0], fea.shape[1], s[1]))
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
    np.savetxt(os.path.join(args.vis_dir, 'tag.txt'), tags.astype(int), fmt='%d')


def save_img(img_original, saliency, title):
    # convert saliency from c01 to 01c
    saliency = saliency.transpose(1, 2, 0)
    img_original = img_original.transpose(1, 2, 0).astype(np.uint8)
    # plot the original image and the three saliency map variants
    plt.figure(figsize=(10, 10)) # facecolor='w')
    # plt.suptitle("Class: " + classes[max_class] + ". Saliency: " + title)
    plt.subplot(1, 2, 1)
    plt.title('Input')
    plt.imshow(img_original)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Attention')
    plt.imshow(np.abs(saliency).max(axis=-1), cmap='gray')
    '''
    plt.subplot(2, 2, 3)
    plt.title('pos. saliency')
    plt.imshow((np.maximum(0, saliency) / saliency.max()))
    plt.subplot(2, 2, 4)
    plt.title('neg. saliency')
    plt.imshow((np.maximum(0, -saliency) / -saliency.min()))
    #plt.show()
    '''
    plt.axis('off')
    plt.savefig(title, bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()
