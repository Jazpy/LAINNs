import torch
import argparse
import time
import os
import torch.nn as nn
import torch.optim as optim
import src.smoother.data_management as data_management
from src.smoother.model import CNNSmoother


def main():
    start_t = time.time()
    print(f'Found device: {"cuda" if torch.cuda.is_available() else "cpu"}')
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Argument handling
    args        = handle_args()
    data_dir    = args['data_directory']
    num_classes = args['classes']
    batch_size  = args['batch']
    learning_r  = args['learn']
    epochs      = args['epochs']
    optim_type  = args['optimizer']
    save_dir    = args['save_directory']

    # Data loaders
    train_dl, valid_dl = data_management.create_training_loaders(data_dir, batch_size=batch_size)
    elapsed_t = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_t))
    print(f'Done creating data loaders, elapsed time = {elapsed_t}.')

    # Set model to train
    model_type = 'cnn_smoother'
    model_str = f'{model_type}_{learning_r:.0e}_{optim_type}'
    model = CNNSmoother(num_classes=num_classes)

    model = model.to(device)
    if optim_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_r)
    elif optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_r, nesterov=True, momentum=0.9)

    # Run training loop
    print(f'Training... ({model_str=})')
    start_t = time.time()
    train(model, optimizer, train_dl, valid_dl, len(train_dl) // 2, save_dir, device,
          epochs, start_t=start_t, model_id=model_str)

def train(model, optimizer, train_i, valid_i, eval_step, save_dir, device, epochs=20,
          criterion=nn.CrossEntropyLoss(), best_valid_loss=float('Inf'), start_t=time.time(), model_id='x'):
    running_loss       = 0.0
    valid_running_loss = 0.0
    global_step        = 0
    train_loss_list    = []
    valid_loss_list    = []
    global_steps_list  = []

    model.train()
    for epoch in range(epochs):
        for (labels, probs, cms) in train_i:
            labels = labels.to(device)
            probs  = probs.to(device)
            cms    = cms.to(device)

            optimizer.zero_grad()
            output = model(probs, cms)
            loss   = criterion(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            running_loss += loss.item()
            global_step  += 1

            # Evaluation
            if global_step % eval_step == 0:
                model.eval()
                with torch.no_grad():
                    # Validation
                    for (labels, probs, cms) in valid_i:
                        labels = labels.to(device)
                        probs  = probs.to(device)
                        cms    = cms.to(device)

                        output = model(probs, cms)
                        loss   = criterion(output, labels)
                        valid_running_loss += loss.item()

                avg_train_loss = running_loss / eval_step
                avg_valid_loss = valid_running_loss / len(valid_i)
                train_loss_list.append(avg_train_loss)
                valid_loss_list.append(avg_valid_loss)
                global_steps_list.append(global_step)

                running_loss       = 0.0
                valid_running_loss = 0.0
                model.train()

                # Save checkpoint
                if best_valid_loss > avg_valid_loss:
                    elapsed_t = time.strftime('%H:%M:%S',
                        time.gmtime(time.time() - start_t))
                print((f'epoch [{epoch + 1}/{epochs}],\t'
                    f'step [{global_step}/{epochs * len(train_i)}],\t'
                    f'training loss = {avg_train_loss:.3f},\t'
                    f'validation loss = {avg_valid_loss:.3f},\t'
                    f'elapsed time = {elapsed_t}'))
                best_valid_loss = avg_valid_loss
                data_management.save_checkpoint(f'{save_dir}/{model_id}_model.pt',
                    model, optimizer, best_valid_loss)
                data_management.save_metrics(f'{save_dir}/{model_id}_metrics.pt',
                    train_loss_list, valid_loss_list, global_steps_list)


def handle_args():
    parser = argparse.ArgumentParser(description='Model training.')
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('-d','--data-directory',
        help='Directory with preprocessed CSV files', type=str)
    parser.add_argument('-c','--classes',
        help='Number of classes to predict', required=True, type=int)
    parser.add_argument('-b','--batch',
        help='Batch size', default=32, type=int)
    parser.add_argument('-e','--epochs',
        help='Epochs to train for', default=70, type=int)
    parser.add_argument('-l','--learn',
        help='Learning rate', default=1e-5, type=float)
    parser.add_argument('-o','--optimizer',
        help='Optimizer type', required=True, type=str, choices=['adamw', 'sgd'])
    parser.add_argument('-s','--save-directory',
        help='Location to save trained model', default='.', type=str)

    return vars(parser.parse_args())


def welcome():
    print(r'''
       _____ __  __  ____   ____ _______ _    _ _____ _   _       _______ ____  _____
      / ____|  \/  |/ __ \ / __ \__   __| |  | |_   _| \ | |   /\|__   __/ __ \|  __ \
     | (___ | \  / | |  | | |  | | | |  | |__| | | | |  \| |  /  \  | | | |  | | |__) |
      \___ \| |\/| | |  | | |  | | | |  |  __  | | | | . ` | / /\ \ | | | |  | |  _  /
      ____) | |  | | |__| | |__| | | |  | |  | |_| |_| |\  |/ ____ \| | | |__| | | \ \
     |_____/|_|  |_|\____/ \____/  |_|  |_|  |_|_____|_| \_/_/    \_\_|  \____/|_|  \_\
    ''')


if __name__ == '__main__':
    welcome()
    main()
