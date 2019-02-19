
# ref: https://github.com/yueruchen/sppnet-pytorch/blob/master/spp_layer.py

import math

def spatial_pyramid_pool(prev_conv, num_sample, prev_conv_size, out_pool_size):
    for i in range(len(out_pool_size)):
        
        # Size
        h_w = int(math.ceil(prev_conv_size[0] / out_pool_size[i]))
        w_w = int(math.ceil(prev_conv_size[1] / out_pool_size[i]))
        

        h_pad = (h_w * out_pool_size[i] - prev_conv_size[0] + 1)/2
        w_pad = (w_w * out_pool_size[i] - prev_conv_size[1] + 1)/2
        
        maxpool = nn.MaxPool2d((h_w, w_w), stride=(h,w, w_w), padding=(h_pad, w_pad))
        x = maxpool(prev_conv)

        if i is 0:
            spp = x.view(num_sample,-1)

        else:
            spp = torch.cat((spp,x.view(num_sample,-1)),1)

    return spp



