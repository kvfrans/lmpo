from lmpo.models.qwen3 import create_model_from_hf
from argparse import ArgumentParser
from lmpo.utils.checkpoint import Checkpoint
import shutil

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf_dir", required=True, default="/nfs/hf/Qwen--Qwen3-1.7B/")
    parser.add_argument("--ckpt_dir", required=True, default='/nfs/gcs/jaxconverted/Qwen3-1.7B/')
    args = parser.parse_args()
    hf_dir = args.hf_dir
    ckpt_dir = args.ckpt_dir

    model, params = create_model_from_hf(hf_dir)
    ckpt = Checkpoint(ckpt_dir+'params.pkl', parallel=False)
    ckpt.params = params
    ckpt.save()

    # copy config.json to new dir.
    shutil.copy(hf_dir + 'config.json', ckpt_dir + 'config.json')
    shutil.copy(hf_dir + 'tokenizer_config.json', ckpt_dir + 'tokenizer_config.json')
    shutil.copy(hf_dir + 'tokenizer.json', ckpt_dir + 'tokenizer.json')