#!/usr/bin/env python
"""
Export trained models for production use.

Exports models in various formats:
- PyTorch (.pt): Standard PyTorch checkpoint
- TorchScript (.torchscript): JIT-compiled for production
- ONNX (.onnx): Cross-platform inference

Usage:
    # Export to PyTorch format
    python -m imspy_predictors.scripts.export_model \
        --checkpoint ./checkpoints/ccs/best_model.pt \
        --output ./pretrained/ccs_timstof_v1.pt \
        --format pytorch

    # Export to TorchScript
    python -m imspy_predictors.scripts.export_model \
        --checkpoint ./checkpoints/ccs/best_model.pt \
        --output ./pretrained/ccs_timstof_v1.torchscript \
        --format torchscript \
        --task ccs

    # Export encoder only
    python -m imspy_predictors.scripts.export_model \
        --checkpoint ./checkpoints/pretrain_intensity/best_model.pt \
        --output ./pretrained/encoder_v1.pt \
        --format pytorch \
        --encoder-only
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export trained models for production",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--checkpoint", type=str, required=True, help="Input checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument(
        "--format",
        choices=["pytorch", "torchscript", "onnx"],
        default="pytorch",
        help="Export format",
    )
    parser.add_argument(
        "--task",
        choices=["ccs", "rt", "charge", "intensity", "all"],
        default="all",
        help="Task(s) to include in export",
    )
    parser.add_argument(
        "--encoder-only",
        action="store_true",
        help="Export encoder only (for transfer learning)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for tracing")
    parser.add_argument("--seq-length", type=int, default=50, help="Sequence length for tracing")
    parser.add_argument("--device", type=str, default="cpu", help="Device for export")
    parser.add_argument("--optimize", action="store_true", help="Apply optimizations")
    parser.add_argument("--half", action="store_true", help="Convert to half precision")

    return parser.parse_args()


def export_pytorch(model, output_path, encoder_only=False, half=False):
    """Export as PyTorch checkpoint."""
    if encoder_only:
        model.encoder.save_pretrained(str(output_path))
    else:
        if half:
            model = model.half()
        model.save_pretrained(str(output_path))

    logger.info(f"Exported PyTorch checkpoint to {output_path}")


def export_torchscript(model, output_path, task, batch_size, seq_length, device, half=False):
    """Export as TorchScript."""
    model.eval()
    if half:
        model = model.half()

    # Create dummy inputs
    tokens = torch.randint(0, 100, (batch_size, seq_length), device=device)
    padding_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)

    # Trace based on task
    if task == "ccs":
        mz = torch.rand(batch_size, device=device) * 2000 + 500
        charge = torch.randint(1, 5, (batch_size,), device=device)

        class CCSWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, tokens, mz, charge, padding_mask):
                return self.model.predict_ccs(tokens, mz, charge, padding_mask)

        wrapper = CCSWrapper(model)
        traced = torch.jit.trace(wrapper, (tokens, mz, charge, padding_mask))

    elif task == "rt":
        class RTWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, tokens, padding_mask):
                return self.model.predict_rt(tokens, padding_mask)

        wrapper = RTWrapper(model)
        traced = torch.jit.trace(wrapper, (tokens, padding_mask))

    elif task == "charge":
        class ChargeWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, tokens, padding_mask):
                return self.model.predict_charge(tokens, padding_mask)

        wrapper = ChargeWrapper(model)
        traced = torch.jit.trace(wrapper, (tokens, padding_mask))

    elif task == "intensity":
        charge = torch.randint(1, 5, (batch_size,), device=device)
        ce = torch.rand(batch_size, device=device) * 0.4 + 0.2

        class IntensityWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, tokens, charge, collision_energy, padding_mask):
                return self.model.predict_intensity(tokens, charge, collision_energy, padding_mask)

        wrapper = IntensityWrapper(model)
        traced = torch.jit.trace(wrapper, (tokens, charge, ce, padding_mask))

    else:
        raise ValueError(f"Unknown task: {task}")

    torch.jit.save(traced, str(output_path))
    logger.info(f"Exported TorchScript to {output_path}")


def export_onnx(model, output_path, task, batch_size, seq_length, device, half=False):
    """Export as ONNX."""
    try:
        import onnx
    except ImportError:
        raise ImportError("Please install onnx: pip install onnx")

    model.eval()
    if half:
        model = model.half()
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Create dummy inputs
    tokens = torch.randint(0, 100, (batch_size, seq_length), device=device)
    padding_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)

    if task == "ccs":
        mz = torch.rand(batch_size, device=device, dtype=dtype) * 2000 + 500
        charge = torch.randint(1, 5, (batch_size,), device=device)

        class CCSWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, tokens, mz, charge, padding_mask):
                mean, std = self.model.predict_ccs(tokens, mz, charge, padding_mask)
                return mean, std

        wrapper = CCSWrapper(model)
        dummy_inputs = (tokens, mz, charge, padding_mask)
        input_names = ["tokens", "mz", "charge", "padding_mask"]
        output_names = ["ccs_mean", "ccs_std"]
        dynamic_axes = {
            "tokens": {0: "batch_size", 1: "seq_length"},
            "mz": {0: "batch_size"},
            "charge": {0: "batch_size"},
            "padding_mask": {0: "batch_size", 1: "seq_length"},
            "ccs_mean": {0: "batch_size"},
            "ccs_std": {0: "batch_size"},
        }

    elif task == "rt":
        class RTWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, tokens, padding_mask):
                return self.model.predict_rt(tokens, padding_mask)

        wrapper = RTWrapper(model)
        dummy_inputs = (tokens, padding_mask)
        input_names = ["tokens", "padding_mask"]
        output_names = ["rt"]
        dynamic_axes = {
            "tokens": {0: "batch_size", 1: "seq_length"},
            "padding_mask": {0: "batch_size", 1: "seq_length"},
            "rt": {0: "batch_size"},
        }

    elif task == "charge":
        class ChargeWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, tokens, padding_mask):
                return self.model.predict_charge(tokens, padding_mask)

        wrapper = ChargeWrapper(model)
        dummy_inputs = (tokens, padding_mask)
        input_names = ["tokens", "padding_mask"]
        output_names = ["charge_probs"]
        dynamic_axes = {
            "tokens": {0: "batch_size", 1: "seq_length"},
            "padding_mask": {0: "batch_size", 1: "seq_length"},
            "charge_probs": {0: "batch_size"},
        }

    elif task == "intensity":
        charge = torch.randint(1, 5, (batch_size,), device=device)
        ce = torch.rand(batch_size, device=device, dtype=dtype) * 0.4 + 0.2

        class IntensityWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, tokens, charge, collision_energy, padding_mask):
                return self.model.predict_intensity(tokens, charge, collision_energy, padding_mask)

        wrapper = IntensityWrapper(model)
        dummy_inputs = (tokens, charge, ce, padding_mask)
        input_names = ["tokens", "charge", "collision_energy", "padding_mask"]
        output_names = ["intensities"]
        dynamic_axes = {
            "tokens": {0: "batch_size", 1: "seq_length"},
            "charge": {0: "batch_size"},
            "collision_energy": {0: "batch_size"},
            "padding_mask": {0: "batch_size", 1: "seq_length"},
            "intensities": {0: "batch_size"},
        }

    else:
        raise ValueError(f"Unknown task: {task}")

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )

    # Validate
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    logger.info(f"Exported ONNX to {output_path}")


def main():
    """Main export function."""
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Import after parsing
    from imspy_predictors.models.unified import UnifiedPeptideModel

    # Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    tasks = None if args.task == "all" else [args.task]
    model = UnifiedPeptideModel.from_pretrained(
        args.checkpoint,
        map_location=args.device,
        tasks=tasks,
    )
    model = model.to(args.device)
    model.eval()

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Enabled tasks: {model.enabled_tasks}")

    # Export
    if args.format == "pytorch":
        export_pytorch(model, output_path, args.encoder_only, args.half)

    elif args.format == "torchscript":
        if args.task == "all":
            raise ValueError("TorchScript export requires a specific task")
        export_torchscript(
            model, output_path, args.task,
            args.batch_size, args.seq_length, args.device, args.half
        )

    elif args.format == "onnx":
        if args.task == "all":
            raise ValueError("ONNX export requires a specific task")
        export_onnx(
            model, output_path, args.task,
            args.batch_size, args.seq_length, args.device, args.half
        )

    logger.info("Export complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
