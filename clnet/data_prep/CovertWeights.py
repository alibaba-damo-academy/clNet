import torch
import copy
import numpy as np

pth_model_source = "/nas/dazhou.guo/Data_Partial/results_yanzhou/Task016_StructSeg_OAR22/Task016_StructSeg_OAR22/clNetTrainerV2_SelectiveChannelDA_Continual_Decoding_Ensemble__clNetPlans/all/Task016_StructSeg_OAR22_from_scratch_model_final_checkpoint.model"
# pth_model_target = "/nas/dazhou.guo/Data_Partial/results/clNet/3d_fullres/Task100_GeneralEncoder/" \
#                    "clNetTrainerV2_SelectiveChannelDA_Continual_Decoding_Ensemble__clNetPlans/all/" \
#                    "GeneralEncoder_model_final_checkpoint.model"

model_source = torch.load(pth_model_source, weights_only=False)
model_target = copy.deepcopy(model_source)
state_dict_in = model_source["state_dict"]
state_dict_out = {}

encoder_key = "context"
td_key = "td"

decoder_key = "localization"
tu_key = "tu"


for key in state_dict_in:
    if encoder_key in key:
        state_dict_out["encoder." + key] = copy.deepcopy(state_dict_in[key])
    elif decoder_key in key:
        state_dict_out["decoder_dict.ge_decoder." + key] = copy.deepcopy(state_dict_in[key])
    elif tu_key in key:
        state_dict_out["decoder_dict.ge_decoder." + key] = copy.deepcopy(state_dict_in[key])
model_target["state_dict"] = state_dict_out
model_target["epoch"] = 0
model_target["optimizer_state_dict"] = None
model_target["amp_grad_scaler"] = None
model_target["plot_stuff"] = []
model_target["plot_stuff"] = []
model_target["best_stuff"] = []
a = 1
# torch.save(model_target, pth_model_target)
