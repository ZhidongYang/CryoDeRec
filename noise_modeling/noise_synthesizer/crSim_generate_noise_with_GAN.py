from utils.config import parse_args
from utils.data_loader_256 import get_data_loader

from models.wgan_gradient_penalty_savemrc_2D_256 import WGAN_GP

import os
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(args):
    
    model = WGAN_GP(args)

    # Start model training
    if args.is_train == 'True':
        # 训练时才需要加载数据集
        if args.dataroot is None:
            raise ValueError("Training requires --dataroot to be specified.")
        train_loader, test_loader = get_data_loader(args)
        model.train(train_loader)
    else:
        original_image_size = args.image_size
        adjusted_image_size = args.image_size

        def is_power_of_two(n):
            return n > 0 and (n & (n - 1)) == 0
        
        if not is_power_of_two(args.image_size):
            adjusted_image_size = 2 ** math.ceil(math.log2(args.image_size))
            print(f"原始image_size {args.image_size} 不是2的幂数，调整为 {adjusted_image_size}")
        
        model.evaluate(args.load_D, args.load_G, tilt_angles=args.tilt_angles, patch_size=args.patch_size,
                       image_size=adjusted_image_size, save_path=args.save_path, original_image_size=original_image_size)


if __name__ == '__main__':
    args = parse_args()
    main(args)
