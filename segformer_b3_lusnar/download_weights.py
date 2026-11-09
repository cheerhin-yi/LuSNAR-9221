"""Optional: save NVIDIA MiT-B3 files locally for offline training."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='pretrained/mit-b3')
    parser.add_argument('--revision', default=None)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    snapshot_download(repo_id='nvidia/mit-b3', local_dir=str(output), revision=args.revision,
                      allow_patterns=['config.json', 'pytorch_model.bin', '*.safetensors', '*.index.json',
                                      'preprocessor_config.json', 'README.md', '*LICENSE*'])
    if not (output/'config.json').is_file(): raise RuntimeError('Downloaded files missing config.json')
    if not any((output/name).is_file() for name in ('model.safetensors', 'pytorch_model.bin')):
        raise RuntimeError('Downloaded files missing model weights')
    print('Weights ready:', output)
    print(f'Run: python train_segformer_b3.py train --pretrained "{output}" --local-files-only')
