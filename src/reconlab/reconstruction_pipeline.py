"""Normal configured orchestration; backend owns GPU memory and optimization."""
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys


def _source_evidence(root):
    files = sorted((root / 'src/reconlab').rglob('*.py'))
    files += [root / 'scripts/run_reconstruction.py']
    sources = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in files if path.is_file()}
    environment = {'python': sys.version, 'executable': sys.executable,
                   'platform': platform.platform(),
                   'loaded_reconlab_files': {name: str(Path(module.__file__).resolve())
                       for name, module in list(sys.modules.items())
                       if (name == 'reconlab' or name.startswith('reconlab.'))
                       and getattr(module, '__file__', None)}}
    git = {'status': 'unavailable', 'reason': 'No Git working tree at project_root'}
    if (root / '.git').exists():
        env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
        def read(*args):
            return subprocess.check_output(['git', *args], cwd=root, env=env,
                                           stderr=subprocess.PIPE).decode().strip()
        git = {'status': 'available', 'head': read('rev-parse', 'HEAD'),
               'branch': read('branch', '--show-current'),
               'working_tree_status': read('status', '--porcelain', '--untracked-files=normal')}
    return {'implementation_sources': sources, 'environment': environment, 'git': git}


def _evidence_files(directory):
    files = {}
    # Mutable final manifests are not hashed into themselves.
    for path in sorted(Path(directory).rglob('*')):
        if not path.is_file() or path.name in ('run_manifest.json', 'run_status.json'):
            continue
        relative = path.relative_to(directory).as_posix()
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        files[relative] = {'sha256': digest, 'bytes': path.stat().st_size}
    return files


class ReconstructionPipeline:
    """run(ResolvedConfig) -> RunManifest; fresh directory and staged failures.

dry_run validates assets/settings only. It does not decode images or instantiate
a backend. A successful execution is distinct from external GPU/quality acceptance.
"""
    def __init__(self, project_root, *, dry_run=False):
        self.root = Path(project_root).resolve()
        if type(dry_run) is not bool:
            raise ValueError('dry_run must be a boolean')
        self.dry_run = dry_run

    def run(self, resolved_config):
        from . import entry
        from .config import ResolvedConfig
        from .pipeline import PipelineRun
        if not isinstance(resolved_config, ResolvedConfig):
            raise ValueError('Expected ResolvedConfig')
        config = resolved_config
        output_root = Path(config.values['output']['root']).resolve()
        if not output_root.is_relative_to(self.root):
            raise ValueError('Output directory must be inside project_root')
        phases = ['configuration', 'asset_validation']
        if not self.dry_run:
            phases += ['image_loading', 'pairing', 'backend_prepare', 'backend_reconstruct',
                       'backend_close', 'normalize', 'export', 'report']
        run = PipelineRun(output_root, config.values['scene']['id'], phases,
            mode='reconstruction', provenance={'config_semantic_hash': config.semantic_hash,
                'dry_run': self.dry_run, 'normal_entry_bound': True,
                'verification': {'real_gpu_pipeline_verified': False,
                                 'M2_complete': False, 'geometry_quality_accepted': False}})
        print('本次输出目录：', run.run_dir, flush=True)
        with run:
            with run.stage('configuration'):
                config.write_evidence(run.run_dir)
                run.update_provenance(_source_evidence(self.root))
            with run.stage('asset_validation'):
                prepared = entry.validate_assets(config, project_root=self.root,
                                                 output=run.run_dir, run_id=run.run_id)
                run.update_provenance({'preflight': prepared.evidence})
            if not self.dry_run:
                with run.stage('image_loading'):
                    batch = entry._load_run_images(prepared)
                with run.stage('pairing'):
                    graph, inputs, options = entry._build_backend_inputs(prepared, batch)
                    run.update_provenance({'processed_pixel_hash': batch.processed_pixel_hash,
                        'pair_graph_hash': graph.graph_hash,
                        'preprocessing_source': batch.upstream_source})
                with run.stage('backend_prepare'):
                    from .backends.mast3r import BackendWorkspace, MASt3RSparseBackend
                    weights, runtime = entry._backend_specs(prepared)
                    backend = MASt3RSparseBackend(self.root)
                    def close_backend():
                        backend.close()
                        entry._append_backend_event(prepared, {'event': 'backend_closed'})
                    run.register_resource('backend', close_backend)
                    workspace = BackendWorkspace(run.run_dir, run.run_dir, run.run_id,
                        lambda event: entry._append_backend_event(prepared, event))
                    fingerprint = backend.prepare(weights, runtime)
                    entry._write_json(run.run_dir / 'backend_fingerprint.json', fingerprint.metadata())
                    entry._append_backend_event(prepared, {'event': 'backend_prepared',
                        'fingerprint_hash': fingerprint.fingerprint_hash})
                    run.update_provenance({'backend_fingerprint_hash': fingerprint.fingerprint_hash})
                with run.stage('backend_reconstruct'):
                    result = backend.reconstruct(batch, graph, options, workspace)
                    entry._write_json(run.run_dir / 'backend_result.json', result.metadata())
                with run.stage('backend_close'):
                    run.close_resource('backend')
                with run.stage('normalize'):
                    from .results import normalize_sparse_scene
                    standard = normalize_sparse_scene(result.scene,
                        pixels_xy=result.scene.pixels_xy,
                        processed_hw=[record.processed_hw for record in batch.records])
                    if standard.image_ids != tuple(record.image_id for record in batch.records):
                        raise entry.EntryError('Backend result identities differ from the validated batch')
                    entry._write_json(run.run_dir / 'standard_result.json', standard.metadata())
                with run.stage('export'):
                    from .exporter import export_result
                    artifacts = export_result(standard, {}, run.run_dir)
                    run.set_artifacts(artifacts.metadata())
                with run.stage('report'):
                    report = {**prepared.evidence, 'check': 'normal_reconstruction_pipeline',
                        'result': standard.metadata(), 'artifacts': artifacts.metadata(),
                        'backend_result_file': 'backend_result.json',
                        'backend_events_file': 'backend_events.jsonl',
                        'pair_graph_hash': graph.graph_hash, 'logical_edge_count': len(graph.edges),
                        'directed_pair_count': len(inputs.pairs), 'processed_pixel_hash': batch.processed_pixel_hash,
                        'per_image_observation_counts': [len(view.points_world) for view in result.scene.views],
                        'final_losses': list(result.final_losses),
                        'reconstruction_seconds_excluding_model_load': result.reconstruction_seconds,
                        'peak_allocated_gib': result.peak_allocated_gib,
                        'backend_measurement_scope': result.metadata()['measurement_scope'],
                        'verification': {'real_gpu_pipeline_verified': False, 'M2_complete': False,
                                         'geometry_quality_accepted': False},
                        'quality_review': 'not_yet_reviewed',
                        'source_pixels_xy': result.scene.metadata()['source_pixels_xy'],
                        'limitations': ['Observation rows are not unique physical points',
                            ('Source pixels were not supplied' if standard.source_pixels_xy is None
                             else 'Source pixels are processed-image coordinates; duplicates are retained'),
                            'Reconstruction confidence remains unavailable',
                            'Different coarse/fine objective values are not a quality comparison']}
                    entry._write_json(run.run_dir / 'reconstruction.json', report)
                    run.update_provenance({'evidence_files': _evidence_files(run.run_dir)})
        if self.dry_run:
            print('资源与参数校验通过；未加载模型、未处理图像。')
        else:
            print(f'标准导出：{len(standard.image_ids)} 台相机 / {len(standard.points_world)} 个观测点')
            print('点云文件：', run.run_dir / 'artifacts/scene.ply')
            print('入口执行与文件读回通过；真实 GPU pipeline 验收、M2 和几何质量仍待完成。')
        print('最终清单：', run.run_dir / 'run_manifest.json')
        return run.manifest
