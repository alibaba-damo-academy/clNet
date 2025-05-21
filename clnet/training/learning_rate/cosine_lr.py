from math import cos, pi, log


def cosine_restarts_lr(epoch, T_0, initial_lr, eta_min=0, T_mult=1):
    T_i = T_0
    if epoch >= T_0:
        if T_mult == 1:
            T_cur = epoch % T_0
        else:
            n = int(log((epoch / T_0 * (T_mult - 1) + 1), T_mult))
            T_cur = epoch - T_0 * (T_mult ** n - 1) / (T_mult - 1)
            T_i = T_0 * T_mult ** (n)
    else:
        T_i = T_0
        T_cur = epoch
    # cosine_decay = 0.5 * (1 + cos(pi * current_step / current_decay_steps))
    # decayed = (1 - alpha) * cosine_decay + alpha
    lr = eta_min + (initial_lr - eta_min) * (1 + cos(pi * T_cur / T_i)) / 2
    return lr