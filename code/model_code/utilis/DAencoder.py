import numpy as np
from scipy.optimize import brentq
from scipy import special as sp
import torch
from math import sqrt, pi as PI
import sympy as sym
from basis_utils import bessel_basis, real_sph_harm


#==========Distance==========
class Envelope(torch.nn.Module):
    """
    Envelope function that ensures a smooth cutoff
    实现上述公式，默认p=6，但是返回的是 u(d)/d
    """
    def __init__(self, exponent=5):
        super(Envelope, self).__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        # Envelope function divided by x
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2

class Distance(torch.nn.Module):
    def __init__(self, num_radial = 16, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.Tensor(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)  # mul_表示乘上PI
        self.freq.requires_grad_()

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff    # x = d/c
        return self.envelope(dist) * (self.freq * dist).sin()
#         return self.envelope(dist) * (self.freq * dist).sin() * 2**0.5 * (1/self.cutoff**3)**0.5


#==========Angle==========
def Jn(r, n):
    """
    numerical spherical bessel functions of order n
    计算球贝塞尔函数的数值，公式如上图所示，其中n为阶数，是唯一的超参，决定贝塞尔函数的形式，r为z是输入
    """
#     return np.sqrt(np.pi/(2*r)) * sp.jv(n+0.5, r)
    z = r
    return sp.spherical_jn(n, z)

def Jn_zeros(n, k):
    """
    Compute the first k zeros of the spherical bessel functions up to order n (excluded)
    # 计算球0至（n-1）阶贝塞尔函数的解，k表示第几个解，解就是使得贝塞尔函数为0的时候r的数值
    # 返回的是是一个矩阵，每一行表示一个阶数即n，每一行中的第一个数字就是pi-2pi之间使得Jn为0的r值，
    # 二个就是2pi-3pi之间使得Jn=0的r值，以此类推
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,)) # 使用2分法，求解最小值，最后的i对应的是Jn中的n
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj

# def real_sph_harm(n, m, theta, phi):
#     """
#     计算球谐函数的实数部分，n为阶数，m，theta为极角，phi为方位角
#     公式如上
#     """
#     return sp.sph_harm(m,n,phi,theta).real

def spherical_bessel_formulas(n):
    """
    Computes the sympy formulas for the spherical bessel functions up to order n (excluded)
    计算0至（n-1）阶下球贝塞尔函数的形式form
    """
    x = sym.symbols('x')

    f = [sym.sin(x)/x]
    a = sym.sin(x)/x
    for i in range(1, n):
        b = sym.diff(a, x)/x
        f += [sym.simplify(b*(-x)**i)]
        a = sym.simplify(b)
    return f

def bessel_basis(n, k):
    """
    Compute the sympy formulas for the normalized and rescaled spherical bessel functions up to
    order n (excluded) and maximum frequency k (excluded).
    # 计算的是上述的aSBF的前半个部分，n依然是阶数，k是公式中的n，
    # 输出0至（n-1）阶，前k个矫正过的aSBF前半部分的形式，注意其中没计算c，x表示d/c
    """

    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5*Jn(zeros[order, i], order+1)**2]
        normalizer_tmp = 1/np.array(normalizer_tmp)**0.5    # 计算第一项（不包含c，是常数）
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [sym.simplify(normalizer[order]
                                            [i]*f[order].subs(x, zeros[order, i]*x))] # 计算第2项（x=D/c，是关于d的函数）
        bess_basis += [bess_basis_tmp]
    return bess_basis

class Angle(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super(Angle, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

                
    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf
        
        cbf = []
        n, k = self.num_spherical, self.num_radial
        
        for lenth in range(len(angle)):
            cbf_ = []
            for order in range(n):
                cbf_.append(real_sph_harm(order, 0, angle[lenth], 0))
            cbf.append(torch.tensor(cbf_))
        
        cbf = torch.stack(cbf, dim=0)
        
#         cbf = []
#         n, k = self.num_spherical, self.num_radial
#         for order in range(n):
#             cbf.append(torch.tensor([real_sph_harm(order, 0, angle, 0)]))
        
#         cbf = torch.stack(cbf, dim=1)

        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
#         out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k) * dist.view(-1, 1) * (1/self.cutoff**3)**0.5
        return out

#==========torsion==========
class Torsion(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super(Torsion, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical #
        self.num_radial = num_radial
        self.cutoff = cutoff
        # self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        self.bessel_funcs = []

        x = sym.symbols('x')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(self.num_spherical):
            for j in range(self.num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, phi, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        
        cbf = []
        n, k = self.num_spherical, self.num_radial
        for lenth in range(len(angle)):
            cbf_ = []
            for order in range(n):
                if order == 0:
                    cbf_.append(real_sph_harm(order, 0, angle[lenth], phi[lenth]))
                else:
                    for m in range(-1*order, order+1):
                        cbf_.append(real_sph_harm(order, m, angle[lenth], phi[lenth]))      
            cbf.append(torch.tensor(cbf_))
        
        cbf = torch.stack(cbf, dim=0)
        

        out = (rbf[idx_kj].view(-1, 1, n, k) * cbf.view(-1, n, n, 1)).view(-1, n * n * k)
        return out



class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.Tensor(num_radial),requires_grad=False)
        # self.freq = torch.arange(1., 17.).mul_(PI).to(self.envelope.device)

        self.reset_parameters()

    def reset_parameters(self):
        torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)


    def forward(self, dist):
        dist = (dist.unsqueeze(-1) / self.cutoff)
        return self.envelope(dist) * (self.freq * dist).sin()


class SphericalBasisLayer(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super().__init__()

        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        # self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        # rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out