"""Optional: download an ImageNet Swin-T checkpoint on a connected computer."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='pretrained/swin-tiny')
    parser.add_argument('--revision', default=None)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    snapshot_download(repo_id='microsoft/swin-tiny-patch4-window7-224', local_dir=str(output), revision=args.revision,
                      allow_patterns=['config.json', 'model.safetensors', 'preprocessor_config.json', 'README.md', '*LICENSE*'])
    if not (output/'config.json').is_file() or not (output/'model.safetensors').is_file():
        raise RuntimeError('Download incomplete: expected config.json and model.safetensors')
    print('Weights ready:', output)
    print(f'Run: python train_upernet_swin_t.py train --pretrained "{output}" --local-files-only --output runs/upernet_swin_t_pretrained')
