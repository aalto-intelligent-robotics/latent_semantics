# VLMaps code for getting image embeddings. Adapted to return embeddings for single image.

from lseg.modules.models.lseg_net import LSegEncNet
from lseg.additional_utils.models import resize_image, pad_image, crop_image
import cv2
import os, math, argparse
import numpy as np
from PIL import Image
import torch
from embeddings_from_images.pixelEmbeddingCreator import PixelEmbeddingCreator

def get_lseg_feat(model: LSegEncNet, image: np.array, labels, transform, crop_size=480, \
                 base_size=520, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]):

    # shape before 1, 3, h, w
    # shape to h, w, 3
    image = transform(image).unsqueeze(0).cuda()
    # shape to h, w, 3
    img = image[0].permute(1,2,0)
    img = img * 0.5 + 0.5

    batch, _, h, w = image.size()
    stride_rate = 2.0/3.0
    stride = int(crop_size * stride_rate)

    long_size = base_size
    if h > w:
        height = long_size
        width = int(1.0 * w * long_size / h + 0.5)
        short_size = width
    else:
        width = long_size
        height = int(1.0 * h * long_size / w + 0.5)
        short_size = height

    cur_img = resize_image(image, height, width, **{'mode': 'bilinear', 'align_corners': True})


    if long_size <= crop_size:
        pad_img = pad_image(cur_img, norm_mean,
                            norm_std, crop_size)
        with torch.no_grad():
            outputs, _ = model(pad_img, labels)
        outputs = crop_image(outputs, 0, height, 0, width)
    else:
        if short_size < crop_size:
            # pad if needed
            pad_img = pad_image(cur_img, norm_mean,
                                norm_std, crop_size)
        else:
            pad_img = cur_img
        _,_,ph,pw = pad_img.shape #.size()
        assert(ph >= height and pw >= width)
        h_grids = int(math.ceil(1.0 * (ph-crop_size)/stride)) + 1
        w_grids = int(math.ceil(1.0 * (pw-crop_size)/stride)) + 1

        with torch.cuda.device_of(image):
            with torch.no_grad():
                outputs = image.new().resize_(batch, model.out_c,ph,pw).zero_().cuda()
                # logits_outputs = image.new().resize_(batch, len(labels),ph,pw).zero_()
            count_norm = image.new().resize_(batch,1,ph,pw).zero_().cuda()
        # grid evaluation
        for idh in range(h_grids):
            for idw in range(w_grids):
                h0 = idh * stride
                w0 = idw * stride
                h1 = min(h0 + crop_size, ph)
                w1 = min(w0 + crop_size, pw)
                crop_img = crop_image(pad_img, h0, h1, w0, w1)
                # pad if needed
                pad_crop_img = pad_image(crop_img, norm_mean,
                                            norm_std, crop_size)
                with torch.no_grad():
                    output, _ = model(pad_crop_img, labels)
                cropped = crop_image(output, 0, h1-h0, 0, w1-w0)
                # cropped_logits = crop_image(logits, 0, h1-h0, 0, w1-w0)
                outputs[:,:,h0:h1,w0:w1] += cropped
                # logits_outputs[:,:,h0:h1,w0:w1] += cropped_logits
                count_norm[:,:,h0:h1,w0:w1] += 1
        assert((count_norm==0).sum()==0)
        outputs = outputs / count_norm
        # logits_outputs = logits_outputs / count_norm
        outputs = outputs[:,:,:height,:width]
        # logits_outputs = logits_outputs[:,:,:height,:width]
    outputs = outputs.cpu()
    outputs = outputs.numpy() # B, D, H, W
    # predicts = [torch.max(logit, 0)[1].cpu().numpy() for logit in logits_outputs]
    # pred = predicts[0]

    return outputs


def read_image(image_path):
    return cv2.imread(image_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--image_path", type=str, help="Path to an RGB image.")
  parser.add_argument("--clip_path", type=str, help="Path to saved CLIP weights.")
  parser.add_argument("--weights_path", type=str, help="Path to the model weights.")
  args = parser.parse_args()

  creator = PixelEmbeddingCreator(args.weights_path, 480)
  image = read_image(args.image_path)
  image_embeddings = creator.get_embeddings(image, args.clip_path, args.weights_path)

  print('image_embeddings:', image_embeddings.shape)
