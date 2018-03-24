import argparse
import os
import time
import numpy as np
from sliced_wasserstein import API as SWD
from data.swd import ImageFolder, CreateDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--datarootR', required=True, help='path to colored dataset')
parser.add_argument('--datarootVBC', required=True, help='path to colored dataset')
parser.add_argument('--datarootVC', required=True, help='path to colored dataset')
parser.add_argument('--datarootVGG', required=True, help='path to colored dataset')
parser.add_argument('--datarootCAN', required=True, help='path to colored dataset')
parser.add_argument('--datarootSAT', required=True, help='path to colored dataset')
parser.add_argument('--datarootTAN', required=True, help='path to colored dataset')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--imageSize', type=int, default=512, help='the height / width of the input image to network')

opt = parser.parse_args()
print(opt)

met = 'swd'
im_shape = [512, 512]


def format_time(seconds):
    s = int(np.rint(seconds))
    if s < 60:
        return '%ds' % (s)
    elif s < 60 * 60:
        return '%dm %02ds' % (s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return '%dh %02dm %02ds' % (s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return '%dd %02dh %02dm' % (s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)


def evaluate_metrics(datasets):
    metric_class_names = {
        'swd': 'metrics.sliced_wasserstein.API',
        'fid': 'metrics.frechet_inception_distance.API',
        'is': 'metrics.inception_score.API',
        'msssim': 'metrics.ms_ssim.API',
    }

    # Initialize metrics.

    print(f'Initializing ... {met}')
    image_shape = [3] + im_shape
    swd = SWD(image_shape=image_shape)

    mode = 'warmup'
    swd.begin(mode)
    swd.feed(mode, np.random.randint(0, 256, size=[8] + image_shape, dtype=np.uint8))
    swd.end(mode)

    # Print table header.
    print()
    print('%-10s%-12s' % ('Snapshot', 'Time_eval'), end='')
    for name, fmt in zip(swd.get_metric_names(), swd.get_metric_formatting()):
        print('%-*s' % (len(fmt % 0), name), end='')
    print()
    print('%-10s%-12s' % ('---', '---'), end='')
    for fmt in swd.get_metric_formatting():
        print('%-*s' % (len(fmt % 0), '---'), end='')
    print()

    # Feed in reals.
    for title, mode, dataset in [('Reals', 'reals', datasets[0]), ('Reals2', 'fakes', datasets[0]),
                                 ('canna', 'fakes', datasets[4]),
                                 ('satsuki', 'fakes', datasets[5]),
                                 ('tanpopo', 'fakes', datasets[6]),
                                 ('VANBCE_C', 'fakes', datasets[1]),
                                 ('VAN_C', 'fakes', datasets[2]),
                                 ('VAN_VGG', 'fakes', datasets[3])
                                 ]:
        print('%-10s\n' % title, end='')
        time_begin = time.time()
        data_iter = iter(dataset)

        swd.begin(mode)
        for _ in range(len(dataset)):
            print(f'\rbiu {_}/{len(dataset)}', end='', flush=True)
            batch = data_iter.next()
            swd.feed(mode, batch.numpy())

        results = swd.end(mode)
        print('%-12s' % format_time(time.time() - time_begin), end='')
        for val, fmt in zip(results, swd.get_metric_formatting()):
            print(fmt % val, end='')
        print()

        # # Evaluate each network snapshot.
        # for snapshot_idx, snapshot_pkl in enumerate(reversed(snapshot_pkls)):
        #     prefix = 'network-snapshot-'
        #     postfix = '.pkl'
        #     snapshot_name = os.path.basename(snapshot_pkl)
        #     assert snapshot_name.startswith(prefix) and snapshot_name.endswith(postfix)
        #     snapshot_kimg = int(snapshot_name[len(prefix): -len(postfix)])
        #
        #     print('%-10d' % snapshot_kimg, end='')
        #     mode = 'fakes'
        #     [swd.begin(mode) for swd in metric_objs]
        #     time_begin = time.time()
        #     with tf.Graph().as_default(), tfutil.create_session(config.tf_config).as_default():
        #         G, D, Gs = misc.load_pkl(snapshot_pkl)
        #         for begin in range(0, num_images, minibatch_size):
        #             end = min(begin + minibatch_size, num_images)
        #             latents = misc.random_latents(end - begin, Gs)
        #             images = Gs.run(latents, labels[begin:end], num_gpus=config.num_gpus, out_mul=127.5, out_add=127.5,
        #                             out_dtype=np.uint8)
        #             if images.shape[1] == 1:
        #                 images = np.tile(images, [1, 3, 1, 1])  # grayscale => RGB
        #             [swd.feed(mode, images) for swd in metric_objs]
        #     results = [swd.end(mode) for swd in metric_objs]
        #     print('%-12s' % misc.format_time(time.time() - time_begin), end='')
        #     for swd, vals in zip(metric_objs, results):
        #         for val, fmt in zip(vals, swd.get_metric_formatting()):
        #             print(fmt % val, end='')
        #     print()
        # print()


# ----------------------------------------------------------------------------

if __name__ == '__main__':
    datasets = [CreateDataLoader(opt.datarootR, opt.batchSize),
                CreateDataLoader(opt.datarootVBC, opt.batchSize),
                CreateDataLoader(opt.datarootVC, opt.batchSize),
                CreateDataLoader(opt.datarootVGG, opt.batchSize),
                CreateDataLoader(opt.datarootCAN, opt.batchSize),
                CreateDataLoader(opt.datarootSAT, opt.batchSize),
                CreateDataLoader(opt.datarootTAN, opt.batchSize),
                ]

    evaluate_metrics(datasets)
