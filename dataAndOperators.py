import torch.autograd
import torch
import torch.nn.functional as F
import numpy as np

class dop:
    def __init__(self, params):
        self.visocsity = 1
        self.density = 1
        self.nFrames = params['nFramesDesired']
        t = torch.tensor(range(1, np.floor(params['nFramesDesired']).astype(np.int) + 1)).cuda()
        self.t0 = torch.max(t)
        t = t / torch.max(t)
        x = torch.tensor(range(0, 340)).cuda()
        x = x / torch.max(x)
        y = torch.tensor(range(0, 340)).cuda()
        y = y / torch.max(y)
        self.t, self.x, self.y = torch.meshgrid(t, x, y)
        self.t = self.t.unsqueeze(1)
        self.sobelX = torch.Tensor([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]).cuda()
        self.sobelY = torch.Tensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]]).cuda()
        self.sobelX = torch.reshape(self.sobelX, [1, 1, 3, 3])
        self.sobelY = torch.reshape(self.sobelY, [1, 1, 3, 3])
        self.sobelT = self.sobelX
        self.speed = 1
        self.mask_in = torch.Tensor(np.ceil(np.load('data/in_mask.npy'))).cuda()
        self.mask_out = torch.Tensor(np.ceil(np.load('data/out_mask.npy'))).cuda()
        self.t_step = 1
        self.U_bound = torch.Tensor(np.load('data/polarX.npy')).cuda() * self.mask_in
        self.U_bound = self.U_bound / torch.max(self.U_bound)
        self.V_bound = torch.Tensor(np.load('data/polarY.npy')).cuda() * self.mask_in
        self.V_bound = self.V_bound / torch.max(self.V_bound)

    def changeNumFrames(self, params):
        nFramesNew = params['nFramesDesired']
        x = np.arange(0, nFramesNew)
        nf = self.nFrames
        t = self.t.cpu().numpy()
        xp = np.arange(0, nf) * nFramesNew / nf
        tpnew = np.zeros((nFramesNew, 1))
        tpnew[:, 0] = np.interp(x, xp, t[:, 0, 0, 0])

        t_new = torch.FloatTensor(tpnew).unsqueeze(1).unsqueeze(1) * torch.ones([nFramesNew, 1, params['im_size'][0], params['im_size'][1]])
        self.nFrames = nFramesNew
        self.t = t_new.cuda()

    def partial_derivative(self, f, x_vector):

        if(x_vector == 'x'):
            return F.conv2d(f, self.sobelX, stride=1, padding=1)
        elif(x_vector == 'y'):
            return F.conv2d(f, self.sobelY, stride=1, padding=1)
        elif(x_vector == 't'):
            f = f.permute([2, 1, 0, 3])
            f = F.conv2d(f, self.sobelT, stride=1, padding=1)
            f = f.permute([2, 1, 0, 3])
            return f
        else:
            raise Exception("Valid axis not defined.")
            return f

    def boundary_conditions(self, U, V, P):
        loss = torch.mean(torch.mean(torch.square(U * self.mask_out)))
        loss += torch.mean(torch.mean(torch.square(V * self.mask_out)))
        loss += torch.mean(torch.mean(torch.square(P * self.mask_out)))
        loss += torch.mean(torch.mean(torch.square(U * self.mask_in - self.t * self.speed * self.t_step * self.U_bound)))
        loss += torch.mean(torch.mean(torch.square(V * self.mask_in - self.t * self.speed * self.t_step * self.V_bound)))
        loss += torch.mean(torch.mean(torch.square(P * self.mask_in)))
        return loss


    def loss(self, U, V, P, lambda1):

        density = self.density
        viscosity = self.visocsity

        du_dx = self.partial_derivative(U, 'x')
        du_dy = self.partial_derivative(U, 'y')
        du_dt = self.partial_derivative(U, 't')
        d2u_dx2 = self.partial_derivative(du_dx, 'x')
        d2u_dy2 = self.partial_derivative(du_dy, 'y')

        dv_dx = self.partial_derivative(V, 'x')
        dv_dy = self.partial_derivative(V, 'y')
        dv_dt = self.partial_derivative(V, 't')
        d2v_dx2 = self.partial_derivative(dv_dx, 'x')
        d2v_dy2 = self.partial_derivative(dv_dy, 'y')

        dp_dx = self.partial_derivative(P, 'x')
        dp_dy = self.partial_derivative(P, 'y')
        d2p_dx2 = self.partial_derivative(dp_dx, 'x')
        d2p_dy2 = self.partial_derivative(dp_dy, 'y')

        loss = torch.mean(torch.mean(torch.square(du_dt + U * du_dx + V * du_dy - (-(1/density) * dp_dx + viscosity * (d2u_dx2 + d2u_dy2)))[1:(self.nFrames - 2), :, :, :]
                                     * torch.logical_and(self.mask_in - 1, self.mask_out - 1)))
        loss += torch.mean(torch.mean(torch.square(dv_dt + U * dv_dx + V * dv_dy - (-(1/density) * dp_dy + viscosity * (d2v_dx2 + d2v_dy2)))[1:(self.nFrames - 2), :, :, :]
                                      * torch.logical_and(self.mask_in - 1, self.mask_out - 1)))
        loss += torch.mean(torch.mean(torch.square(d2p_dx2 + d2p_dy2 + density * (du_dx * du_dx + 2 * du_dy * dv_dx + dv_dy * dv_dy))[1:(self.nFrames - 2), :, :, :]
                                      * torch.logical_and(self.mask_in - 1, self.mask_out - 1)))
        return loss + lambda1 * self.boundary_conditions(U, V, P)