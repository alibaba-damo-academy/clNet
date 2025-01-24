#   Author @Dazhou Guo
#   Data: 08.04.2023

def combined_lr_lambda(epoch, warmup_epochs, max_epochs, initial_lr):
    warmup_lr = initial_lr / 20
    if max_epochs < 1000:
        end_lr = max_epochs / 1000 * initial_lr
    else:
        end_lr = 0
    if epoch < warmup_epochs:
        # Linear warmup
        lr = (initial_lr / 10 - warmup_lr) / warmup_epochs * epoch + warmup_lr
    else:
        # Polynomial decay
        lr = (initial_lr - end_lr) * ((1 - (epoch - warmup_epochs) / (max_epochs - warmup_epochs)) ** 0.9) + end_lr
    return lr
