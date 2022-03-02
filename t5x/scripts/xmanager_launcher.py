# Copyright 2022 The T5X Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""XManager launcher for t5x.

See XManager:
https://github.com/deepmind/xmanager
"""

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_local

_MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    None,
    'Model dir to save logs, ckpts, etc. in "gs://model_dir" format.',
)
_TFDS_DATA_DIR = flags.DEFINE_string(
    'tfds_data_dir',
    None,
    'Data dir to save the processed dataset in "gs://data_dir" format.',
)


@xm.run_in_asyncio_loop
async def main(_):
  name = 't5x'
  async with xm_local.create_experiment(experiment_title=name) as experiment:
    tensorboard = await xm_local.vertex_client().get_or_create_tensorboard(name)

    executor = xm_local.Vertex(
        requirements=xm.JobRequirements(tpu_v3=8, cpu=64, ram=240*xm.GiB),
        tensorboard=xm_local.TensorboardCapability(
            name=tensorboard,
            base_output_directory=_MODEL_DIR.value,
        ),)
    [executable] = experiment.package([
        xm.python_container(
            executor.Spec(),
            path='..',
            base_image='gcr.io/deeplearning-platform-release/base-cpu',
            docker_instructions=[
                'RUN git clone --branch=main https://github.com/google-research/t5x',  # comment to test local repo
                # 'COPY t5x/ t5x'  # uncomment to test local repo
                'WORKDIR t5x',
                'RUN python3 -m pip install -e ".[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html',
            ],
            entrypoint=xm.CommandList([
                f'export MODEL_DIR=\'"{_MODEL_DIR.value}"\'',
                f'export TFDS_DATA_DIR={_TFDS_DATA_DIR.value}',
                'export T5X_DIR=.',

                # Uncomment if download fails, following these directions,
                # https://www.tensorflow.org/datasets/overview#manual_download_if_download_fails
                # File list:
                # https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/wmt_t2t_translate.txt
                # 'mkdir -p ~/tensorflow_datasets/downloads/manual/',
                # 'curl -L -o ~/tensorflow_datasets/downloads/manual/training-parallel-nc-v13.tgz http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz',
                # 'curl -L -o ~/tensorflow_datasets/downloads/manual/dev.tgz http://data.statmt.org/wmt19/translation-task/dev.tgz',
                # 'curl -L -o ~/tensorflow_datasets/downloads/manual/training-parallel-commoncrawl.tgz http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz',
                # 'curl -L -o ~/tensorflow_datasets/downloads/manual/training-parallel-europarl-v7.tgz http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz',
                # 'gsutil cp ~/tensorflow_datasets/downloads/manual/* ${TFDS_DATA_DIR}/downloads/manual/',

                ('python3 ${T5X_DIR}/t5x/train.py '
                 '--gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin" '
                 '--gin.MODEL_DIR=${MODEL_DIR} '
                 '--tfds_data_dir=${TFDS_DATA_DIR}'),
            ]),
        ),
    ])
    experiment.add(xm.Job(executable=executable, executor=executor))


if __name__ == '__main__':
  app.run(main)
