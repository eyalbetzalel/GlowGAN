
import torch 

def calc_js_div(p_imagegpt, p_realnvp, q_imagegpt, q_realnvp):
        
        log_p_imagegpt = torch.log(p_imagegpt)
        log_p_realnvp = torch.log(p_realnvp)
        log_q_imagegpt = torch.log(q_imagegpt)
        log_q_realnvp = torch.log(q_realnvp)

        logM_z = torch.log(0.5) + torch.logsumexp(torch.stack((log_p_realnvp.squeeze(), log_q_imagegpt.squeeze())), dim=0)
        logM_x = torch.log(0.5) + torch.logsumexp(torch.stack((log_q_realnvp.squeeze(), log_p_imagegpt.squeeze())), dim=0)

        return 0.5 * ((log_p_imagegpt - logM_x).sum() + (log_p_realnvp - logM_z).sum())
    

def calc_gradient(model, loss):
    g = list(
        torch.autograd.grad(
            loss,
            model.parameters(),
            retain_graph=True,
            )
        )
    grad = torch.cat([torch.flatten(grad) for grad in g])
    model.zero_grad()
    return grad

def calc_cosine_similarity(vec1, vec2):
    cos_sim=0
    return cos_sim

