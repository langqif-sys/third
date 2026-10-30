"""Run matched ETTh1 comparisons. Use --smoke for plumbing checks, not results."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True, help='Directory containing ETTh1.csv')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2024, 2025, 2026])
    parser.add_argument('--variants', nargs='+', default=['baseline', 'plain', 'residual', 'complement'],
                        choices=['baseline', 'plain', 'residual', 'complement', 'moving_average', 'no_stopgrad', 'no_inverted'])
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--train_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--use_gpu', type=int, choices=[0, 1], default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = Path(args.root_path).resolve()
    if not (data / 'ETTh1.csv').is_file():
        parser.error('ETTh1.csv was not found under root_path')
    # Same losses, optimizer settings, patch geometry, heads and dropout for all
    # temporal variants. This is a controlled starting point, not tuned results.
    common = ['--is_training', '1', '--data', 'ETTh1', '--features', 'M',
              '--root_path', str(data), '--data_path', 'ETTh1.csv', '--enc_in', '7',
              '--seq_len', str(args.seq_len), '--pred_len', str(args.pred_len),
              '--d_model', '512', '--use_hour_index', '1', '--use_day_index', '0',
              '--rec_lambda', '0.25', '--auxi_lambda', '0.75', '--auxi_mode', 'fft',
              '--train_epochs', str(args.train_epochs), '--patience', '5',
              '--batch_size', str(args.batch_size), '--learning_rate', str(args.learning_rate),
              '--use_gpu', str(args.use_gpu), '--num_workers', str(args.num_workers),
              '--frca_dim', '64', '--frca_heads', '4', '--frca_patch_len', '16',
              '--frca_stride', '8', '--frca_beta_init', '0.1', '--frca_dropout', '0.1']
    if args.smoke:
        common += ['--max_train_batches', '4', '--max_eval_batches', '2']
    extra = {
        'baseline': ['--model', 'TimeEmb'],
        'plain': ['--model', 'TimeEmb_FRCA', '--frca_mode', 'plain'],
        'residual': ['--model', 'TimeEmb_FRCA', '--frca_mode', 'residual'],
        'complement': ['--model', 'TimeEmb_FRCA', '--frca_mode', 'complement'],
        'moving_average': ['--model', 'TimeEmb_FRCA', '--frca_reference', 'moving_average'],
        'no_stopgrad': ['--model', 'TimeEmb_FRCA', '--frca_detach_reference', '0'],
        'no_inverted': ['--model', 'TimeEmb_FRCA', '--frca_inverted', '0'],
    }
    for seed in args.seeds:
        for variant in args.variants:
            command = [sys.executable, '-u', 'run.py'] + common + extra[variant]
            command += ['--model_id', 'ETTh1_' + variant, '--random_seed', str(seed)]
            print('Running:', ' '.join(command), flush=True)
            subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
