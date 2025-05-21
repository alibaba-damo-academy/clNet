from torch import nn

from clnet.training.loss_functions.dice_loss import DC_and_CE_loss


class MultipleOutputLossEnsemble(nn.Module):
    def __init__(self, task_classes, batch_dice, weights_for_side=None, is_ddp=False):
        super(MultipleOutputLossEnsemble, self).__init__()
        self.weights_for_side = weights_for_side
        self.task_classes = task_classes
        self.loss = DC_and_CE_loss({'batch_dice': batch_dice, 'smooth': 1e-5, 'do_bg': False, "ddp": is_ddp}, {})
        self.is_ddp = is_ddp

    def forward(self, x, y, weights_for_head=None):
        assert isinstance(x, dict), "x must be a dict"
        assert isinstance(y, dict), "y must be either tuple or list"
        l_ret = None
        for head in x:

            if head in y:
                current_x = x[head]
                current_y = y[head]
                assert isinstance(current_x, (tuple, list)), "x must be either tuple or list"
                assert isinstance(current_y, (tuple, list)), "y must be either tuple or list"
                if self.weights_for_side is None:
                    weights_for_side = [1] * len(current_x)
                else:
                    weights_for_side = self.weights_for_side

                l_head = 0
                # weights_for_side[0] * self.loss(current_x[0], current_y[0])
                for i in range(len(current_x)):
                    if weights_for_side[i] != 0:
                        if isinstance(current_x[i], tuple):
                            for j in range(len(current_x[i])):
                                l_head += weights_for_side[i] * self.loss(current_x[i][j], current_y[i])
                        else:
                            l_head += weights_for_side[i] * self.loss(current_x[i], current_y[i])
            else:
                txt_msg = "Prediction head %s is not found in target label." % head
                raise RuntimeError(txt_msg)
            if weights_for_head is not None:
                if l_ret is None:
                    l_ret = weights_for_head[head] * l_head
                else:
                    l_ret += weights_for_head[head] * l_head
            else:
                if l_ret is None:
                    l_ret = l_head
                else:
                    l_ret += l_head
        return l_ret
