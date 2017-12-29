import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
epoch_length = 100

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
# visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model = create_model(opt)
    model.set_input(data)
    visualizer = Visualizer(opt)
    img_path = model.get_image_paths()
    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for j in range(opt.TEST_ITERS):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.optimize_noise()
            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/epoch_length, opt, errors)

            if total_steps % opt.save_model_freq == 0:
                print('saving the at (total_steps %d)' %
                      (total_steps))
                model.save(webpage, img_path, total_steps)
            if total_steps % epoch_length == 0:
                break

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
    visuals = model.get_current_visuals()
    print('%04d: process image... %s' % (i, img_path))
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
