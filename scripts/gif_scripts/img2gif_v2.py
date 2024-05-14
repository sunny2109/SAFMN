import cv2  
import imageio

def img2gif(lq_path, hq_path, save_path, duration=5, num_frames=20, num_extra_frames=2):
    '''
    Save a series of images as a GIF.  [LQ, ..., LQ->HQ, ..., HQ]

    Arguments:
        lq_path (str): Path for low-quality (LQ) images.
        hq_path (str): Path for high-quality (HQ) images.
        save_path (str): Path for output GIF.
        duration (float): Duration of each frame in GIF.
        num_frames (int): Number of frames for mixing HQ with LQ images.
        num_extra_frames (int):  Number of frames for single HQ or LQ image.
    '''
    lq = cv2.imread(lq_path)
    hq = cv2.imread(hq_path)
    w, h = hq.shape[:2]

    lq = cv2.resize(lq, (h, w), interpolation=cv2.INTER_CUBIC)
    hq = cv2.resize(hq, (h, w), interpolation=cv2.INTER_CUBIC)

    step = w//num_frames

    images = list()
    for i in range(num_extra_frames):
        images.append(cv2.cvtColor(lq, cv2.COLOR_BGR2RGB))

    for i in range(step, w, step):
        img = cv2.hconcat([hq[:,:i], lq[:,i:]])
        cv2.line(img, (i,0), (i,h), color=(238, 104, 123), thickness=w//200)
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for i in range(num_extra_frames):
        images.append(cv2.cvtColor(hq, cv2.COLOR_BGR2RGB))

    imageio.mimsave(save_path, images, "GIF", duration=duration)


if __name__ == '__main__':
    lq_path = './results/00021.png'
    hq_path = './results/00021_out.png'
    save_path = 'results/gif/00021.gif'
    img2gif(lq_path, hq_path, save_path)
