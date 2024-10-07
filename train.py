from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
import os
import sys
import tensorboardX
import shutil

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
    opts = parser.parse_args()

    # Enable cuDNN auto-tuning
    cudnn.benchmark = True

    # Load experiment settings
    config = get_config(opts.config)
    max_iter = config['max_iter']
    display_size = config['display_size']
    config['vgg_model_path'] = opts.output_path

    # Setup model based on the trainer type
    if opts.trainer == 'MUNIT':
        trainer = MUNIT_Trainer(config)
    elif opts.trainer == 'UNIT':
        trainer = UNIT_Trainer(config)
    else:
        sys.exit("Only support MUNIT|UNIT")
    trainer.cuda()

    # # Load data loaders
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)

    # Debug: Check dataset size
    print(f"Length of train_loader_a dataset: {len(train_loader_a.dataset)}")
    print(f"Length of train_loader_b dataset: {len(train_loader_b.dataset)}")

    # Select images for display during training
    train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda()
    train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()
    test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda()
    test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda()


    # Setup logger and output directories
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

    # Start training
    iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
    while True:
        for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
            trainer.update_learning_rate()
            images_a, images_b = images_a.squeeze(1), images_b.squeeze(1)  # Squeeze out any singleton dimensions
            images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

            with Timer("Elapsed time in update: %f"):
                # Main training code
                trainer.dis_update(images_a, images_b, config)
                trainer.gen_update(images_a, images_b, config)
                torch.cuda.synchronize()

            # Log training statistics
            if (iterations + 1) % config['log_iter'] == 0:
                print(f"Iteration: {iterations + 1:08d}/{max_iter:08d}")
                write_loss(iterations, trainer, train_writer)
                print(f"Generator Loss: {trainer.loss_gen_total.item()}, Discriminator Loss: {trainer.loss_dis_total.item()}")

            # Save and display images during training
            if (iterations + 1) % config['image_save_iter'] == 0:
                print(f"Saving images at iteration {iterations + 1}")

                with torch.no_grad():
                    test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                    train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)

                # Check if image directory exists
                if not os.path.exists(image_directory):
                    print(f"Image directory {image_directory} does not exist. Creating it.")
                    os.makedirs(image_directory)
                else:
                    print(f"Image directory {image_directory} already exists.")

                # Log saving images
                print(f"Saving test images for iteration {iterations + 1} in {image_directory}")
                write_2images(test_image_outputs, display_size, image_directory, f'test_{iterations + 1:08d}')
                print(f"Saving train images for iteration {iterations + 1} in {image_directory}")
                write_2images(train_image_outputs, display_size, image_directory, f'train_{iterations + 1:08d}')

                # HTML output
                print(f"Writing HTML output for iteration {iterations + 1}")
                write_html(f"{output_directory}/index.html", iterations + 1, config['image_save_iter'], 'images')

            if (iterations + 1) % config['image_display_iter'] == 0:
                with torch.no_grad():
                    image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                print(f"Displaying train images for iteration {iterations + 1}")
                write_2images(image_outputs, display_size, image_directory, 'train_current')

            # Save network weights periodically
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                print(f"Saving model checkpoint at iteration {iterations + 1}")
                trainer.save(checkpoint_directory, iterations)

            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')

if __name__ == '__main__':
    main()

