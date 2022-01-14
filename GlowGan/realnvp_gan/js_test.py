
import torch 

def calc_js_div(p_imagegpt, p_realnvp, q_imagegpt, q_realnvp):
        
        p_imagegpt = torch.tensor(p_imagegpt)
        q_imagegpt = torch.tensor(q_imagegpt)
        
        
        
        log_p_imagegpt = torch.log(p_imagegpt)
        log_p_realnvp = torch.log(p_realnvp)
#        log_q_imagegpt = torch.log(q_imagegpt)
#        log_q_realnvp = torch.log(q_realnvp)
        
#        log_q_imagegpt = log_q_imagegpt.to("cuda:0")
#        log_p_imagegpt = log_p_imagegpt.to("cuda:0")
        logM_x = torch.log((p_imagegpt + q_realnvp)/2)
        logM_z = torch.log((q_imagegpt + p_realnvp)/2)
        
#        logM_z = torch.log(torch.tensor(0.5)) + torch.logsumexp(torch.stack((log_p_realnvp.squeeze(), log_q_imagegpt.squeeze())), dim=0)
#        logM_x = torch.log(torch.tensor(0.5)) + torch.logsumexp(torch.stack((log_q_realnvp.squeeze(), log_p_imagegpt.squeeze())), dim=0)

        return 0.5 * ((log_p_imagegpt - logM_x).sum() + (log_p_realnvp - logM_z).sum())
    

def calc_gradient(model, loss):
    # model.zero_grad()
    g = list(
        torch.autograd.grad(
            loss,
            model.parameters(),
            retain_graph=True,
            allow_unused=True,
            )
        )
    grad = torch.cat([torch.flatten(grad) for grad in g])
    model.zero_grad()
    return grad

def calc_cosine_similarity(vec1, vec2):
    cos_sim=0
    return cos_sim

