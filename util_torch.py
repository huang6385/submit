import torch
import torch.nn.functional as F


def pier_XY(x, y):
    """
    node's piersen distance
    x: B*N*k

    return
    adj:B*N*N -1~1 1 is more related
    """
    B, N, k = x.shape
    x_std = torch.std(x, dim=-1)
    y_std = torch.std(y, dim=-1)
    xy_std = torch.matmul(x_std.view(B, N, 1), y_std.view(B, 1, N))

    x_mean = torch.mean(x, dim=-1, keepdim=True)
    y_mean = torch.mean(y, dim=-1, keepdim=True)
    xy = torch.matmul((x - x_mean), (y - y_mean).transpose(1, 2))

    adj = torch.abs(xy / (xy_std + 1e-12) * (1.0 / (k - 1)))

    #     idx = torch.arange(adj.shape[1],device = adj.device)
    #     adj[:,idx,idx] = 1
    return adj


def graphs_threshold(adjs,threshold):
    adjs = adjs * (adjs>=threshold)
    return adjs


def graphs_re(adjs,sparse = 10,axis = -1):
    if sparse>adjs.shape[axis]:
        assert False
    val, idx = torch.topk(adjs, k=sparse,axis = axis)
    val = torch.index_select(val, axis, torch.LongTensor([sparse-1]).to(val.device))
    r = torch.zeros_like(adjs)
    r[adjs>=val] = adjs[adjs>=val]
    return r



def choice_by_prob(matrix,sparse):
    '''
    matrix: b*N*N , b can be multi graph or batch!!
    '''
    if sparse>adjs.shape[-1]:assert False
    
    b,N,N = matrix.shape
    
    matrix = matrix.reshape(b*N,N)
    out_matrix = torch.zeros_like(matrix)
    
    samples_idx = torch.multinomial(torch.exp(matrix),sparse)
    
    idx = torch.arange(0,b*N).reshape(-1,1).repeat((1,sparse))
    
    out_matrix[idx,samples_idx] = matrix[idx,samples_idx]
    
    out_matrix = out_matrix.reshape(b,N,N)
    return out_matrix
                                                   

def sample_gumbel(shape,device, eps=1e-20):
    U = torch.Tensor(shape).uniform_(0,1).to(device) #.cuda()
    return -(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size(),device = logits.device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1, hard=False):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y = (y_hard - y).detach() + y
    return y

    