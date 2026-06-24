import math
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from e3nn.math import soft_unit_step
from e3nn.nn import BatchNorm, Gate
from e3nn.o3 import (
    FullyConnectedTensorProduct,
    Irreps,
    Linear,
    spherical_harmonics,
)


class EquivUNet(nn.Module):
    """3D equivariant UNet-like backbone.

    Input and output tensor ordering is the same as the existing UNet:
    [B, C, D, H, W].
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        steps,
        n_levels=2,
        feat_mult=2,
        kernel_size=5,
        activation="relu",
        last_activation="softmax",
        normalization="instance",
        lmax=2,
        scale=2,
        return_fmaps=False,
    ):
        """Initialize the equivariant U-Net.

        Parameters
        ----------
        irreps_in : str or e3nn.o3.Irreps
            Input irreducible representations.
        irreps_out : str or e3nn.o3.Irreps
            Output irreducible representations.
        steps : tuple of float
            Voxel spacing used to build convolution kernels.
        n_levels : int, optional
            Number of U-Net levels.
        feat_mult : int, optional
            Feature multiplier controlling channel width.
        kernel_size : int, optional
            Base convolution diameter.
        activation : str, optional
            Nonlinearity name.
        last_activation : str, optional
            Final output activation.
        normalization : str, optional
            Normalization mode.
        lmax : int, optional
            Maximum spherical harmonic order.
        scale : int, optional
            Downsampling scale factor.
        return_fmaps : bool, optional
            Whether to return intermediate feature maps.
        """
        super().__init__()

        self.n_classes_scalar = Irreps(irreps_out).count("0e")
        self.num_classes = self.n_classes_scalar
        self.n_downsample = n_levels - 1
        self.return_fmaps = return_fmaps

        if normalization not in ["None", "batch", "instance"]:
            raise ValueError("normalization must be one of: 'None', 'batch', 'instance'")

        if activation == "relu":
            activation = [torch.relu]
        else:
            raise NotImplementedError("Only relu activation is currently supported")

        self.last_activation_name = last_activation
        if last_activation == "relu":
            self.last_activation = torch.nn.ReLU()
        elif last_activation == "sigmoid":
            self.last_activation = torch.nn.Sigmoid()
        elif last_activation == "softmax":
            # Matched to the existing UNet behavior used by keypoint heatmaps:
            # spatial softmax per channel.
            self.last_activation = None
        else:
            self.last_activation = None

        irreps_sh = Irreps.spherical_harmonics(lmax, p=1)
        diameters = [kernel_size * 2**i for i in range(self.n_downsample + 1)]
        scales = [scale * 2**i for i in range(self.n_downsample)]

        steps_array = [steps]
        for i in range(self.n_downsample):
            output_steps = []
            for step in steps:
                if step < scales[i]:
                    kernel_dim = math.floor(scales[i] / step)
                    output_steps.append(kernel_dim * step)
                else:
                    output_steps.append(step)
            steps_array.append(tuple(output_steps))

        self.down = EquivDown(
            n_downsample=self.n_downsample,
            activation=activation,
            irreps_sh=irreps_sh,
            ne=feat_mult,
            no=0,
            normalization=normalization,
            irreps_in=irreps_in,
            diameters=diameters,
            steps=steps_array,
            scale=scales,
        )

        self.up = EquivUp(
            n_blocks_up=self.n_downsample,
            activation=activation,
            irreps_sh=irreps_sh,
            ne=feat_mult * 2 ** (self.n_downsample - 1),
            no=0,
            normalization=normalization,
            irreps_downblock=self.down.down_irreps_out,
            diameters=diameters[::-1][1:],
            steps=steps_array[::-1][1:],
            scale=scales[::-1],
            return_fmaps=return_fmaps,
        )

        self.out = EquivConvolutionBlock(
            irreps_in=self.up.up_blocks[-1].irreps_out,
            irreps_hidden=Irreps(irreps_out),
            activation=activation,
            irreps_sh=irreps_sh,
            normalization=normalization,
            diameter=kernel_size,
            steps=steps,
            transpose=False,
        )

    def _apply_last_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.last_activation_name == "softmax":
            b, c, *spatial = x.shape
            x_flat = x.reshape(b, c, -1)
            x_prob = torch.softmax(x_flat, dim=-1)
            return x_prob.reshape(b, c, *spatial)
        if self.last_activation is not None:
            return self.last_activation(x)
        return x

    def forward(self, x):
        """Run the equivariant U-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``[B, C, D, H, W]``.

        Returns
        -------
        torch.Tensor or list of torch.Tensor
            Output tensor or feature maps when ``return_fmaps`` is enabled.
        """
        if self.return_fmaps:
            return self.forward_fmaps(x)

        pad = self.pad_size(x.shape[-3:])
        x = torch.nn.functional.pad(x, (pad[-1], 0, pad[-2], 0, pad[-3], 0))

        down_ftrs = self.down(x)
        x = self.up(down_ftrs[-1], down_ftrs)
        x = self.out(x)

        x = x[..., pad[0]:, pad[1]:, pad[2]:]
        x = self._apply_last_activation(x)
        return x

    def forward_fmaps(self, x):
        """Return intermediate feature maps and output heatmaps.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``[B, C, D, H, W]``.

        Returns
        -------
        list of torch.Tensor
            Downstream feature maps, upstream feature maps, and final output.
        """
        pad = self.pad_size(x.shape[-3:])
        x = torch.nn.functional.pad(x, (pad[-1], 0, pad[-2], 0, pad[-3], 0))

        down_ftrs = self.down(x)
        up_ftrs = self.up(down_ftrs[-1], down_ftrs)
        x = self.out(up_ftrs[-1])
        x = self._apply_last_activation(x)

        fmaps = down_ftrs + up_ftrs + [x]
        fmaps = [f[..., pad[0]:, pad[1]:, pad[2]:] for f in fmaps]
        return fmaps

    def pad_size(self, image_shape, odd=False):
        """Compute padding needed for pooling-compatible image sizes.

        Parameters
        ----------
        image_shape : tuple of int
            Spatial image shape ``(D, H, W)``.
        odd : bool, optional
            Whether to pad for odd-sized pooling compatibility.

        Returns
        -------
        list of int
            Padding values for each spatial axis.
        """
        pooling_factor = np.ones(3, dtype="int")
        for pool in self.down.down_pool:
            pooling_factor *= np.array(pool.kernel_size)

        pad = []
        for f, s in zip(pooling_factor, image_shape):
            p = 0
            if odd:
                t = (s - 1) % f
            else:
                t = s % f

            if t != 0:
                p = f - t
            pad.append(p)

        return pad


class EquivDown(nn.Module):
    """Downsampling path for ``EquivUNet``."""

    def __init__(
        self,
        n_downsample,
        activation,
        irreps_sh,
        ne,
        no,
        normalization,
        irreps_in,
        diameters,
        steps,
        scale,
    ):
        """Initialize equivariant downsampling blocks.

        Parameters
        ----------
        n_downsample : int
            Number of downsampling operations.
        activation : list
            Scalar activation functions.
        irreps_sh : e3nn.o3.Irreps
            Spherical harmonic irreducible representations.
        ne, no : int
            Initial even and odd representation multipliers.
        normalization : str
            Normalization mode.
        irreps_in : str or e3nn.o3.Irreps
            Input irreducible representations.
        diameters : sequence of int
            Convolution diameters per level.
        steps : sequence of tuple
            Voxel steps per level.
        scale : sequence of int
            Pooling scale per level.
        """
        super().__init__()

        blocks = []
        self.down_irreps_out = []
        for n in range(n_downsample + 1):
            irreps_hidden = Irreps(
                f"{4 * ne}x0e + {4 * no}x0o + {2 * ne}x1e + {2 * no}x1o + {ne}x2e + {no}x2o"
            ).simplify()
            block = EquivConvolutionBlock(
                irreps_in=irreps_in,
                irreps_hidden=irreps_hidden,
                activation=activation,
                irreps_sh=irreps_sh,
                normalization=normalization,
                diameter=diameters[n],
                steps=steps[n],
                transpose=False,
            )
            blocks.append(block)
            self.down_irreps_out.append(block.irreps_out)
            irreps_in = block.irreps_out
            ne *= 2
            no *= 2
        self.down_blocks = nn.ModuleList(blocks)

        pooling = []
        for n in range(n_downsample):
            pooling.append(
                EquivDynamicPool3d(
                    scale=scale[n],
                    steps=steps[n],
                    mode="maxpool3d",
                    irreps=self.down_irreps_out[n],
                )
            )
        self.down_pool = nn.ModuleList(pooling)

    def forward(self, x):
        """Run the downsampling path.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        list of torch.Tensor
            Feature tensors at each downsampling level.
        """
        features = []
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            features.append(x)
            if i < len(self.down_blocks) - 1:
                x = self.down_pool[i](x)
        return features


class EquivUp(nn.Module):
    """Upsampling path for ``EquivUNet``."""

    def __init__(
        self,
        n_blocks_up,
        activation,
        irreps_sh,
        ne,
        no,
        normalization,
        irreps_downblock,
        diameters,
        steps,
        scale,
        return_fmaps,
    ):
        """Initialize equivariant upsampling blocks.

        Parameters
        ----------
        n_blocks_up : int
            Number of upsampling blocks.
        activation : list
            Scalar activation functions.
        irreps_sh : e3nn.o3.Irreps
            Spherical harmonic irreducible representations.
        ne, no : int
            Initial even and odd representation multipliers.
        normalization : str
            Normalization mode.
        irreps_downblock : sequence
            Irreps produced by the downsampling path.
        diameters : sequence of int
            Convolution diameters per level.
        steps : sequence of tuple
            Voxel steps per level.
        scale : sequence of int
            Upsampling scales.
        return_fmaps : bool
            Whether to return feature maps instead of the final tensor.
        """
        super().__init__()

        self.n_blocks_up = n_blocks_up
        self.return_fmaps = return_fmaps

        irreps_in = irreps_downblock[-1]
        blocks = []
        upsample_op = []
        for n in range(n_blocks_up):
            irreps_hidden = Irreps(
                f"{4 * ne}x0e + {4 * no}x0o + {2 * ne}x1e + {ne}x2e + {2 * no}x1o + {no}x2o"
            ).simplify()
            block = EquivConvolutionBlock(
                irreps_in=irreps_in + irreps_downblock[::-1][n + 1],
                irreps_hidden=irreps_hidden,
                activation=activation,
                irreps_sh=irreps_sh,
                normalization=normalization,
                diameter=diameters[n],
                steps=steps[n],
                transpose=True,
            )
            blocks.append(block)
            irreps_in = block.irreps_out
            ne //= 2
            no //= 2

            upsample_scale_factor = tuple(
                [math.floor(scale[n] / step) if step < scale[n] else 1 for step in steps[n]]
            )
            upsample_op.append(
                nn.Upsample(
                    scale_factor=upsample_scale_factor,
                    mode="trilinear",
                    align_corners=True,
                )
            )

        self.up_blocks = nn.ModuleList(blocks)
        self.upsample_ops = nn.ModuleList(upsample_op)

    def forward(self, x, down_features):
        """Run the upsampling path with skip connections.

        Parameters
        ----------
        x : torch.Tensor
            Deepest feature tensor.
        down_features : list of torch.Tensor
            Skip features from the downsampling path.

        Returns
        -------
        torch.Tensor or list of torch.Tensor
            Final tensor or intermediate upsampling feature maps.
        """
        if self.return_fmaps:
            fmaps = []
            for i in range(self.n_blocks_up):
                x = self.upsample_ops[i](x)
                x = torch.cat([x, down_features[::-1][i + 1]], dim=1)
                x = self.up_blocks[i](x)
                fmaps.append(x)
            return fmaps

        for i in range(self.n_blocks_up):
            x = self.upsample_ops[i](x)
            x = torch.cat([x, down_features[::-1][i + 1]], dim=1)
            x = self.up_blocks[i](x)
        return x


class EquivDynamicPool3d(torch.nn.Module):
    """Pooling layer that respects scalar and higher-order equivariant channels."""

    def __init__(self, scale, steps, mode, irreps):
        """Initialize equivariant pooling.

        Parameters
        ----------
        scale : int
            Target pooling scale.
        steps : tuple of float
            Voxel spacing.
        mode : str
            Pooling mode.
        irreps : e3nn.o3.Irreps
            Channel irreducible representations.
        """
        super().__init__()

        self.scale = scale
        self.steps = steps
        self.mode = mode
        self.kernel_size = tuple(
            [math.floor(self.scale / step) if step < self.scale else 1 for step in self.steps]
        )
        self.irreps = irreps

    def forward(self, x):
        """Pool an equivariant feature tensor.

        Parameters
        ----------
        x : torch.Tensor
            Feature tensor with shape ``[B, C, D, H, W]``.

        Returns
        -------
        torch.Tensor
            Pooled feature tensor.
        """
        if self.mode == "average":
            out = F.avg_pool3d(x, self.kernel_size, stride=self.kernel_size)

        elif self.mode == "maxpool3d":
            if x.shape[1] != self.irreps.dim:
                raise ValueError(f"Shape mismatch: expected channel dim {self.irreps.dim}, got {x.shape[1]}")

            cat_list = []
            start = 0
            for i in self.irreps.ls:
                end = start + 2 * i + 1
                temp = x[:, start:end, ...]
                if i == 0:
                    pooled, _ = F.max_pool3d_with_indices(
                        temp[:, 0, ...],
                        self.kernel_size,
                        stride=self.kernel_size,
                        return_indices=True,
                    )
                    cat_list.append(pooled)
                else:
                    _, indices = F.max_pool3d_with_indices(
                        temp.norm(dim=1),
                        self.kernel_size,
                        stride=self.kernel_size,
                        return_indices=True,
                    )
                    for tensor_slice in range(2 * i + 1):
                        pooled = temp[:, tensor_slice, ...].flatten()[indices]
                        cat_list.append(pooled)
                start = end
            out = torch.stack(tuple(cat_list), dim=1)

        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

        return out


class EquivConvolutionBlock(nn.Module):
    """Two equivariant convolutions with gated nonlinearities."""

    def __init__(
        self,
        irreps_in,
        irreps_hidden,
        activation,
        irreps_sh,
        normalization,
        diameter,
        steps,
        transpose,
    ):
        """Initialize an equivariant convolution block.

        Parameters
        ----------
        irreps_in : e3nn.o3.Irreps
            Input irreducible representations.
        irreps_hidden : e3nn.o3.Irreps
            Hidden irreducible representations.
        activation : list
            Scalar activation functions.
        irreps_sh : e3nn.o3.Irreps
            Spherical harmonic irreducible representations.
        normalization : str
            Normalization mode.
        diameter : int
            Kernel diameter.
        steps : tuple of float
            Voxel spacing.
        transpose : bool
            Whether to use transposed convolution.
        """
        super().__init__()

        if normalization == "batch":
            bn_cls = BatchNorm
        elif normalization == "instance":
            bn_cls = partial(BatchNorm, instance=True)
        else:
            bn_cls = None

        irreps_scalars = Irreps([(mul, ir) for mul, ir in irreps_hidden if ir.l == 0])
        irreps_gated = Irreps([(mul, ir) for mul, ir in irreps_hidden if ir.l > 0])
        irreps_gates = Irreps(f"{irreps_gated.num_irreps}x0e")

        if irreps_gates.dim == 0:
            irreps_gates = irreps_gates.simplify()
            activation_gate = []
        else:
            activation_gate = [torch.sigmoid]

        self.gate1 = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=activation,
            irreps_gates=irreps_gates,
            act_gates=activation_gate,
            irreps_gated=irreps_gated,
        )
        self.conv1 = Convolution(
            irreps_in=irreps_in,
            irreps_out=self.gate1.irreps_in,
            irreps_sh=irreps_sh,
            diameter=diameter,
            num_radial_basis=diameter,
            steps=steps,
            transpose=transpose,
        )
        self.batchnorm1 = bn_cls(self.gate1.irreps_in) if bn_cls is not None else None

        self.gate2 = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=activation,
            irreps_gates=irreps_gates,
            act_gates=activation_gate,
            irreps_gated=irreps_gated,
        )
        self.conv2 = Convolution(
            irreps_in=self.gate1.irreps_out,
            irreps_out=self.gate2.irreps_in,
            irreps_sh=irreps_sh,
            diameter=diameter,
            num_radial_basis=diameter,
            steps=steps,
            transpose=transpose,
        )
        self.batchnorm2 = bn_cls(self.gate2.irreps_in) if bn_cls is not None else None

        self.irreps_out = self.gate2.irreps_out

    def forward(self, x):
        """Apply the equivariant convolution block.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Output feature tensor.
        """
        x = self.conv1(x)
        if self.batchnorm1 is not None:
            x = self.batchnorm1(x.transpose(1, 4)).transpose(1, 4)
        x = self.gate1(x.transpose(1, 4)).transpose(1, 4)

        x = self.conv2(x)
        if self.batchnorm2 is not None:
            x = self.batchnorm2(x.transpose(1, 4)).transpose(1, 4)
        x = self.gate2(x.transpose(1, 4)).transpose(1, 4)

        return x


class Convolution(torch.nn.Module):
    """Convolution on voxel grids."""

    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_sh,
        diameter,
        num_radial_basis,
        steps=(1.0, 1.0, 1.0),
        cutoff=False,
        transpose=False,
        **kwargs,
    ):
        """Initialize an equivariant convolution on a voxel grid.

        Parameters
        ----------
        irreps_in : e3nn.o3.Irreps
            Input irreducible representations.
        irreps_out : e3nn.o3.Irreps
            Output irreducible representations.
        irreps_sh : e3nn.o3.Irreps
            Spherical harmonic irreducible representations.
        diameter : int
            Kernel diameter.
        num_radial_basis : int
            Number of radial basis functions.
        steps : tuple of float, optional
            Voxel spacing.
        cutoff : bool or str, optional
            Radial basis cutoff mode.
        transpose : bool, optional
            Whether to use transposed convolution.
        **kwargs
            Extra arguments passed to PyTorch convolution.
        """
        super().__init__()

        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.irreps_sh = Irreps(irreps_sh)

        self.num_radial_basis = num_radial_basis
        self.transpose = transpose

        self.sc = Linear(self.irreps_in, self.irreps_out)

        r = diameter / 2

        s = math.floor(r / steps[0])
        x = torch.arange(-s, s + 1.0) * steps[0]

        s = math.floor(r / steps[1])
        y = torch.arange(-s, s + 1.0) * steps[1]

        s = math.floor(r / steps[2])
        z = torch.arange(-s, s + 1.0) * steps[2]

        lattice = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)
        self.register_buffer("lattice", lattice)

        if "padding" not in kwargs:
            kwargs["padding"] = tuple(s // 2 for s in lattice.shape[:3])
        self.kwargs = kwargs

        emb = soft_one_hot_linspace(
            x=lattice.norm(dim=-1),
            start=0.0,
            end=r,
            number=self.num_radial_basis,
            basis="smooth_finite",
            cutoff=cutoff,
        )
        self.register_buffer("emb", emb)

        sh = spherical_harmonics(
            l=self.irreps_sh,
            x=lattice,
            normalize=True,
            normalization="component",
        )
        self.register_buffer("sh", sh)

        self.tp = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            shared_weights=False,
            compile_right=True,
        )

        self.weight = torch.nn.Parameter(torch.randn(self.num_radial_basis, self.tp.weight_numel))

    def kernel(self):
        """Construct the convolution kernel from learned radial weights.

        Returns
        -------
        torch.Tensor
            Convolution kernel tensor.
        """
        weight = self.emb @ self.weight
        weight = weight / (self.sh.shape[0] * self.sh.shape[1] * self.sh.shape[2])
        kernel = self.tp.right(self.sh, weight)
        kernel = torch.einsum("xyzio->oixyz", kernel)
        return kernel

    def forward(self, x):
        """Apply equivariant convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Output feature tensor.
        """
        sc = self.sc(x.transpose(1, 4)).transpose(1, 4)

        if self.transpose:
            out = sc + torch.nn.functional.conv_transpose3d(
                x, self.kernel().transpose(0, 1), **self.kwargs
            )
        else:
            out = sc + torch.nn.functional.conv3d(x, self.kernel(), **self.kwargs)

        return out


def soft_one_hot_linspace(x: torch.Tensor, start, end, number, basis=None, cutoff=None):
    """Embed distances with smooth one-hot radial basis functions.

    Parameters
    ----------
    x : torch.Tensor
        Input distances.
    start : float
        Start of the embedding interval.
    end : float
        End of the embedding interval.
    number : int
        Number of basis functions.
    basis : str, optional
        Basis type.
    cutoff : bool or str, optional
        Cutoff mode.

    Returns
    -------
    torch.Tensor
        Radial basis embedding.
    """
    if cutoff not in [True, False, "left", "right"]:
        raise ValueError("cutoff must be specified: True, False, 'left', 'right'")

    if not cutoff:
        values = torch.linspace(start, end, number, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
    elif cutoff == "left":
        values = torch.linspace(start, end, number + 1, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
        values = values[1:]
    elif cutoff == "right":
        values = torch.linspace(start, end, number + 1, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
        values = values[:-1]
    else:
        values = torch.linspace(start, end, number + 2, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
        values = values[1:-1]

    diff = (x[..., None] - values) / step

    if basis == "gaussian":
        return diff.pow(2).neg().exp().div(1.12)

    if basis == "cosine":
        return torch.cos(math.pi / 2 * diff) * (diff < 1) * (-1 < diff)

    if basis == "smooth_finite":
        output = 1.14136 * math.exp(2.0) * soft_unit_step(diff + 1) * soft_unit_step(1 - diff)
        return output

    if basis == "fourier":
        x = (x[..., None] - start) / (end - start)
        if not cutoff:
            i = torch.arange(0, number, dtype=x.dtype, device=x.device)
            return torch.cos(math.pi * i * x) / math.sqrt(0.25 + number / 2)
        if cutoff == "left":
            i = torch.arange(1, number + 1, dtype=x.dtype, device=x.device)
            return torch.sin(math.pi * i * x) / math.sqrt(0.25 + number / 2) * (0 < x)
        if cutoff == "right":
            i = torch.arange(1, number + 1, dtype=x.dtype, device=x.device)
            return torch.sin(math.pi * i * x) / math.sqrt(0.25 + number / 2) * (x < 1)

        i = torch.arange(1, number + 1, dtype=x.dtype, device=x.device)
        return torch.sin(math.pi * i * x) / math.sqrt(0.25 + number / 2) * (0 < x) * (x < 1)

    if basis == "bessel":
        x = x[..., None] - start
        c = end - start
        bessel_roots = torch.arange(1, number + 1, dtype=x.dtype, device=x.device) * math.pi
        out = math.sqrt(2 / c) * torch.sin(bessel_roots * x / c) / x

        if not cutoff:
            return out
        if cutoff == "left":
            return out * (0 < x)
        if cutoff == "right":
            return out * ((x / c) < 1)
        return out * ((x / c) < 1) * (0 < x)

    raise ValueError(f'basis="{basis}" is not a valid entry')
