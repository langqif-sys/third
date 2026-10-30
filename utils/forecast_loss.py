"""The original TimeEmb objectives, shared by FP32 and AMP training."""
import torch
from utils.polynomial import chebyshev_torch, hermite_torch, laguerre_torch, leg_torch


def forecast_loss(outputs, target, args, mask=None):
    if args.rec_lambda < 0 or args.auxi_lambda < 0 or args.rec_lambda + args.auxi_lambda <= 0:
        raise ValueError('Loss weights must be nonnegative and at least one must be positive')
    # FFT loss also needs FP32 for non-power-of-two horizons under CUDA AMP.
    with torch.autocast(device_type=outputs.device.type, enabled=False):
        outputs, target = outputs.float(), target.float()
        loss = args.rec_lambda * (outputs - target).square().mean()
        if not args.auxi_lambda:
            return loss
        mode = args.auxi_mode
        if mode == 'fft':
            error = torch.fft.fft(outputs, dim=1) - torch.fft.fft(target, dim=1)
        elif mode == 'rfft':
            a, b = torch.fft.rfft(outputs, dim=1), torch.fft.rfft(target, dim=1)
            kind = args.auxi_type
            if kind == 'complex':
                error = a - b
            elif kind == 'complex-phase':
                error = (a - b).angle()
            elif kind == 'complex-mag-phase':
                error = torch.stack([(a - b).abs(), (a - b).angle()])
            elif kind == 'phase':
                error = a.angle() - b.angle()
            elif kind == 'mag':
                error = a.abs() - b.abs()
            elif kind == 'mag-phase':
                error = torch.stack([a.abs() - b.abs(), a.angle() - b.angle()])
            else:
                raise ValueError('Unknown auxi_type: ' + kind)
        elif mode == 'rfft-D':
            error = torch.fft.rfft(outputs, dim=-1) - torch.fft.rfft(target, dim=-1)
        elif mode == 'rfft-2D':
            error = torch.fft.rfft2(outputs) - torch.fft.rfft2(target)
        else:
            transforms = {'legendre': leg_torch, 'chebyshev': chebyshev_torch,
                          'hermite': hermite_torch, 'laguerre': laguerre_torch}
            if mode not in transforms:
                raise ValueError('Unknown auxi_mode: ' + mode)
            transform = transforms[mode]
            error = (transform(outputs, args.leg_degree, device=outputs.device)
                     - transform(target, args.leg_degree, device=outputs.device))
        if mask is not None:
            error = error * mask
        if args.auxi_loss == 'MAE':
            auxiliary = error.abs().mean() if args.module_first else error.mean().abs()
        elif args.auxi_loss == 'MSE':
            auxiliary = error.abs().square().mean() if args.module_first else error.square().mean().abs()
        else:
            raise ValueError('Unknown auxi_loss: ' + args.auxi_loss)
        return loss + args.auxi_lambda * auxiliary
