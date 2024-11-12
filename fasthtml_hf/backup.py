import os, shutil
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import time
from fastcore.utils import *
from datetime import datetime
from huggingface_hub import snapshot_download, upload_folder, create_repo, repo_exists, whoami

__all__ = ['download', 'upload', 'setup_hf_backup']
def _token(): return os.getenv("HF_TOKEN")

def get_cfg():
    return Config('.', 'config.ini',
                  types=dict(dataset_id=str, db_dir=str, private_backup=bool, interval=int),
                  create=dict(dataset_id='space-backup', db_dir='data', private_backup=True, interval=15))

def get_dataset_id(cfg):
    did = cfg.dataset_id
    if "/" in did or _token() is None: return did
    return f"{whoami(_token())['name']}/{did}"

def download():
    print('Searching for database backup...')
    cfg = get_cfg()
    did = get_dataset_id(cfg)
    upload_on_schedule()
    if os.getenv("SPACE_ID") and repo_exists(did, repo_type="dataset", token=_token()):
        print('Found existing backup, copying...')
        cache_path = snapshot_download(repo_id=did, repo_type='dataset', token=_token())
        shutil.copytree(cache_path, cfg.db_dir, dirs_exist_ok=True)
        print(f'copied db from {cache_path} to {cfg.db_dir}...done.')
        print('\n'.join(map(str, Path(cfg.db_dir).iterdir())))
    else:
        print(f'no db found at {did}! (or the SPACE_ID environment variable isn\'t set)')

def upload():
    try:
        cfg = get_cfg()
        if not os.getenv("SPACE_ID"): return
        did = get_dataset_id(cfg)
        create_repo(did, token=_token(), private=cfg.private_backup, repo_type='dataset', exist_ok=True)
        upload_folder(folder_path=cfg.db_dir, token=_token(), repo_id=did,
                    repo_type='dataset', commit_message=f"backup {datetime.now()}")
        print('='*60)
        print(f'NOTE: backed up {cfg.db_dir} to {did}')
        print('='*60)
    except Exception as e:
        print('-'*80)
        print('WARNING!!!!! THERE MIGHT BE A PROBLEM WITH BACKUPS!!!! CHECK BACKUPS')
        print(e)
        

@threaded
def upload_on_schedule():
    cfg = get_cfg()
    while True:
        time.sleep(cfg.interval*60)
        upload()


def setup_hf_backup(app):
    app.on_event("startup")(download)
    app.on_event("shutdown")(upload)

