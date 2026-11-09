"""Regression checks on real script functions without importing PyTorch.

Epoch reporting and CUDA routing use test doubles, not a real GPU.
"""
import ast
import argparse
import contextlib
import csv
import io
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
SOURCE = (ROOT / 'train_segformer_b3.py').read_text(encoding='utf-8-sig')
TREE = ast.parse(SOURCE)


def get_function(name, scope=None):
    node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {} if scope is None else scope
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual ' + name + '>', 'exec'), namespace)
    return namespace[name]


def epoch_tail(has_validation, preview_count=0):
    train = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    epoch = next(n for n in ast.walk(train) if isinstance(n, ast.For)
                 and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
    start = next(i for i, n in enumerate(epoch.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'train_loss' for t in n.targets))
    code = compile(ast.Module(body=epoch.body[start:], type_ignores=[]), '<actual epoch reporting>', 'exec')
    files, documents, evaluations, previews = {}, {}, [], []
    def evaluate(*args):
        evaluations.append(True)
        if not has_validation:
            raise AssertionError('All-data mode must not evaluate training data')
        return dict(miou=0.7, pixel_accuracy=0.9, iou=[0.7, 0.7])
    stream = io.StringIO()
    fields = ['epoch', 'train_loss', 'miou', 'pixel_accuracy', 'seconds'] if has_validation else ['epoch', 'train_loss', 'seconds']
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    stateful = SimpleNamespace(state_dict=lambda: {'tensor': 'stand-in'}, config=SimpleNamespace(to_dict=lambda: {'architecture': 'B3'}), initialization={'type': 'random'})
    scope = dict(has_validation=has_validation, total_loss=5., sample_count=10, best=-1.,
                 best_train_loss=float('inf'), model=stateful, optimizer=stateful,
                 scaler=stateful, epoch=0, global_step=10, schema={}, out=Path('virtual_run'),
                 time=time, begin=time.time(), random=random, device=SimpleNamespace(type='cpu'),
                 numpy_rng_state=lambda: {'name': 'test-state'},
                 torch=SimpleNamespace(get_rng_state=lambda: b'cpu-state'),
                 generator=SimpleNamespace(get_state=lambda: b'loader-state'),
                 writer=writer, stream=stream, evaluate=evaluate, val_loader=object(),
                 args=SimpleNamespace(size=512, preview_count=preview_count), amp=True,
                 splits=dict(train=[{'scene': 'Moon_1'}], val=[{'scene': 'Moon_8'}]),
                 save_previews=lambda *args: previews.append((args[1], args[-2].name, args[-1])),
                 save_checkpoint=lambda path, obj: files.update({path.name: obj}),
                 dump=lambda path, obj: documents.update({path.name: obj}))
    scope['schema'] = {'names': ['a', 'b']}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, scope)
    return files, documents, evaluations, stream.getvalue(), previews


class ChangesTests(unittest.TestCase):
    def test_default_command_selects_gpu0_and_requested_scene_split(self):
        captured = []
        main = get_function('main', dict(argparse=argparse, train=captured.append, inspect=captured.append))
        with patch.object(sys, 'argv', ['train_segformer_b3.py', 'train']):
            main()
        args = captured[0]
        self.assertEqual(args.device, 'cuda:0')
        self.assertFalse(args.train_all)
        self.assertEqual(args.camera, 'image0')
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.output, 'runs/segformer_b3_gpu0_scene_split_aug')
        self.assertEqual(args.augmentation, 'standard')
        self.assertTrue(args.no_amp)
        pairs = [dict(scene=f'Moon_{index}') for index in range(1, 11)]
        with contextlib.redirect_stdout(io.StringIO()):
            split = get_function('split_data')(pairs, args)
        self.assertEqual([p['scene'] for p in split['train']],
                         ['Moon_1', 'Moon_3', 'Moon_5', 'Moon_6', 'Moon_7', 'Moon_8'])
        self.assertEqual([p['scene'] for p in split['val']], ['Moon_2'])
        self.assertEqual([p['scene'] for p in split['test']], ['Moon_4', 'Moon_9'])

    def test_fixed_split_rejects_missing_requested_scene(self):
        args = SimpleNamespace(split_manifest=None, train_all=False, val_scenes=None,
                               test_scenes=None, camera='image0')
        pairs = [dict(scene=f'Moon_{index}') for index in range(1, 9)]
        with self.assertRaisesRegex(ValueError, 'Missing paired data.*Moon_9'):
            get_function('split_data')(pairs, args)

    def test_all_samples_used_without_holdout(self):
        pairs = [dict(scene=f'Moon_{s}', image=str(i)) for s in (1, 3, 7, 8) for i in range(3)]
        args = SimpleNamespace(split_manifest=None, train_all=True, val_scenes=None, test_scenes=None)
        with contextlib.redirect_stdout(io.StringIO()):
            result = get_function('split_data')(pairs, args)
        self.assertEqual(result, dict(train=pairs, val=[], test=[]))

    def test_single_scene_full_training_allowed(self):
        args = SimpleNamespace(split_manifest=None, train_all=True, val_scenes=None, test_scenes=None)
        with contextlib.redirect_stdout(io.StringIO()):
            result = get_function('split_data')([dict(scene='Moon_1')], args)
        self.assertEqual(len(result['train']), 1)

    def test_explicit_holdout_rejected_in_full_mode(self):
        args = SimpleNamespace(split_manifest=None, train_all=True, val_scenes=['Moon_8'], test_scenes=None)
        with self.assertRaisesRegex(ValueError, 'scene-split'):
            get_function('split_data')([dict(scene='Moon_1')], args)

    def test_explicit_scene_mode_preserved(self):
        args = SimpleNamespace(split_manifest=None, train_all=False, val_scenes=['Moon_8'], test_scenes=['Moon_7'])
        with contextlib.redirect_stdout(io.StringIO()):
            result = get_function('split_data')([dict(scene=f'Moon_{s}') for s in (1, 7, 8)], args)
        self.assertEqual(result['val'], [dict(scene='Moon_8')])
        self.assertEqual(result['test'], [dict(scene='Moon_7')])

    def test_full_mode_reports_loss_and_saves_correct_checkpoints(self):
        files, docs, evaluations, history, _ = epoch_tail(False)
        self.assertEqual(evaluations, [])
        self.assertEqual(set(files), {'last.pth', 'best_train_loss.pth'})
        self.assertEqual(docs['best_metrics.json']['selection_criterion'], 'training_loss')
        self.assertEqual(docs['best_metrics.json']['train_loss'], 0.5)
        self.assertNotIn('miou', docs['best_metrics.json'])
        self.assertNotIn('miou', history)
        self.assertEqual(files['last.pth']['best_train_loss'], 0.5)

    def test_scene_mode_still_selects_validation_best(self):
        files, docs, evaluations, history, _ = epoch_tail(True)
        self.assertEqual(evaluations, [True])
        self.assertEqual(set(files), {'last.pth', 'best.pth'})
        self.assertEqual(docs['best_metrics.json']['miou'], 0.7)
        self.assertIn('miou', history)

    def test_full_mode_preview_is_explicitly_training_data(self):
        files, _, _, _, previews = epoch_tail(False, preview_count=3)
        self.assertEqual(previews, [([{'scene': 'Moon_1'}], 'training_previews', 3)])
        self.assertEqual(files['best_train_loss.pth']['model_config']['architecture'], 'B3')

    def test_scene_mode_preview_remains_validation_data(self):
        _, _, _, _, previews = epoch_tail(True, preview_count=3)
        self.assertEqual(previews, [([{'scene': 'Moon_8'}], 'previews', 3)])

    def test_second_gpu_selected_and_missing_gpu_rejected(self):
        selected = []
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
            current_device=lambda: 0, set_device=lambda dev: selected.append(dev.index),
            get_device_name=lambda dev: 'test GPU', get_device_properties=lambda dev: SimpleNamespace(total_memory=1024))
        # device_for's arithmetic smoke check is represented by a minimal tensor.
        class Tensor:
            def __matmul__(self, other): return self
            def sum(self): return self
            def item(self): return 1
        torch = SimpleNamespace(cuda=cuda, device=lambda value: SimpleNamespace(type='cuda', index=int(value.split(':')[1])),
                                randn=lambda *a, **kw: Tensor())
        device_for = get_function('device_for', {'torch': torch})
        with contextlib.redirect_stdout(io.StringIO()):
            device_for('cuda:1')
        self.assertEqual(selected, [1])
        cuda.device_count = lambda: 1
        with self.assertRaisesRegex(RuntimeError, 'only 1 CUDA'):
            device_for('cuda:1')


if __name__ == '__main__':
    unittest.main(verbosity=2)
