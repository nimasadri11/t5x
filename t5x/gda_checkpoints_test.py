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

"""Tests for gda_checkpoints."""

from typing import Any, Mapping

from absl.testing import absltest
from absl.testing import parameterized
from flax import optim
import jax
from jax._src import test_util as jtu
from jax._src.util import prod
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.pjit import pjit
import jax.numpy as jnp
import numpy as np
from t5x import checkpoints
from t5x import partitioning
from t5x import train_state as train_state_lib
from t5x.partitioning import PartitionSpec as P

jax.config.update('jax_parallel_functions_output_gda', True)


def make_train_state(
    *,
    step: int,
    params: Mapping[str, Any],
    param_states: Mapping[str, Any],
    flax_optimizer_def: optim.OptimizerDef = optim.GradientDescent()
) -> train_state_lib.TrainState:
  """Helper to construct a train state for testing."""
  optimizer = optim.Optimizer(
      flax_optimizer_def,
      state=optim.OptimizerState(step=step, param_states=param_states),
      target=params)
  return train_state_lib.FlaxOptimTrainState(optimizer)


def all_gda_shards(gda):
  global_array = np.zeros(gda.shape)
  for shard in gda.global_shards:
    global_array[shard.index] = shard.data
  return global_array


class FakePartitioner(partitioning.BasePartitioner):

  def __init__(self, mesh, mesh_axes):
    super().__init__(num_partitions=1)
    self._mesh = mesh
    self._mesh_axes = make_train_state(
        step=None,
        params={
            'bias': mesh_axes,
            'kernel': mesh_axes
        },
        param_states={
            'bias': mesh_axes,
            'kernel': mesh_axes
        })
    self._local_chunker = partitioning.LocalChunker(self._mesh)

  def get_data_layout(self):
    return partitioning.DataLayout(
        batch_size=None,
        shard_id=1,
        num_shards=1,
        is_first_host_in_replica_set=True)

  @property
  def params_on_devices(self):
    return self._params_on_devices

  def move_params_to_devices(self, train_state, train_state_axes):
    return train_state

  def get_mesh_axes(self, train_state):
    return self._mesh_axes

  def _local_chunker(self):
    return self._local_chunker

  def partition(self,
                fn,
                in_axis_resources,
                out_axis_resources,
                static_argnums=(),
                donate_argnums=()):
    pjitted = pjit(
        fn,
        in_axis_resources=in_axis_resources,
        out_axis_resources=out_axis_resources,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums)
    return partitioning.PjittedFnWithContext(pjitted, self._mesh)

  def compile(self, partitioned_fn, *args):
    return None


class GdaCheckpointsTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.tmp_dir = self.create_tempdir().full_path

  def make_train_state_non_gda(self,
                               global_mesh,
                               global_input_shape,
                               mesh_axes,
                               step=42,
                               dtype=np.float32,
                               shard=False):
    bias = np.ones(global_input_shape, dtype=dtype)
    kernel = np.arange(
        prod(global_input_shape), dtype=dtype).reshape(global_input_shape)

    if shard:
      jax.config.update('jax_parallel_functions_output_gda', False)
      partition_array = pjit(
          lambda x: x, in_axis_resources=(None,), out_axis_resources=mesh_axes)
      with global_mesh:
        kernel = partition_array(kernel)
      jax.config.update('jax_parallel_functions_output_gda', True)

    train_state = make_train_state(
        step=np.int32(step),
        params={
            'bias': bias * 2,
            'kernel': kernel * 2
        },
        param_states={  # only target gets cast
            'bias': bias.astype(np.float32),
            'kernel': kernel.astype(np.float32)
        })
    return train_state

  def make_train_state(self,
                       global_mesh,
                       global_input_shape,
                       mesh_axes,
                       step=42,
                       dtype=np.float32):
    train_state = self.make_train_state_non_gda(
        global_mesh, global_input_shape, mesh_axes, step=step, dtype=dtype)

    def create_gda(elem):

      def cb(index):
        return elem[index]

      if np.isscalar(elem):
        return elem
      return GlobalDeviceArray.from_callback(global_input_shape, global_mesh,
                                             mesh_axes, cb)

    return jax.tree_map(
        create_gda, train_state, is_leaf=lambda x: isinstance(x, np.ndarray))

  def assert_equal(self, a, b):
    assert isinstance(
        a, type(b)), f'Found incompatible types: {type(a)}, {type(b)}'
    if not isinstance(a, GlobalDeviceArray):
      self.assertArraysEqual(a, b)
    else:
      for s1, s2 in zip(a.local_shards, b.local_shards):
        self.assertArraysEqual(s1.data, s2.data)

  def validate_save_restore(self,
                            train_state,
                            global_mesh,
                            mesh_axes,
                            step=42,
                            save_dtype=np.float32,
                            restore_dtype=np.float32,
                            multi_optimizer=False):
    step = np.int32(step)

    checkpointer = checkpoints.Checkpointer(
        train_state,
        FakePartitioner(global_mesh, mesh_axes),
        self.tmp_dir,
        save_dtype=save_dtype,
        restore_dtype=restore_dtype,
        use_gda=True)
    checkpointer.save(train_state)

    restored_train_state = checkpointer.restore(step=step)
    jax.tree_multimap(self.assert_equal, train_state, restored_train_state)

  @jtu.with_mesh([('x', 4), ('y', 2)])
  def test_basic(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')
    global_input_shape = (8, 2)

    train_state = self.make_train_state(global_mesh, global_input_shape,
                                        mesh_axes)
    self.validate_save_restore(train_state, global_mesh, mesh_axes)

  @parameterized.named_parameters(
      (
          'bfloat16',
          jnp.bfloat16,
      ),
      (
          'float32',
          np.float32,
      ),
      (
          'int32',
          np.int32,
      ),
  )
  @jtu.with_mesh([('x', 4), ('y', 2)])
  def test_params_restore_as_type(self, dtype):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')
    global_input_shape = (8, 2)

    train_state = self.make_train_state(
        global_mesh, global_input_shape, mesh_axes, dtype=dtype)
    self.validate_save_restore(
        train_state,
        global_mesh,
        mesh_axes,
        save_dtype=np.float32,
        restore_dtype=dtype)

  @parameterized.named_parameters(
      (
          'bfloat16',
          jnp.bfloat16,
      ),
      (
          'float32',
          np.float32,
      ),
      (
          'int32',
          np.int32,
      ),
  )
  @jtu.with_mesh([('x', 2), ('y', 2)])
  def test_restore_old_format_as_type(self, dtype):
    global_mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')
    global_input_shape = (2, 2)

    step = np.int32(12)
    old_train_state = self.make_train_state_non_gda(
        global_mesh,
        global_input_shape,
        mesh_axes,
        step=step,
        shard=True,
        dtype=np.float32)
    old_checkpointer = checkpoints.Checkpointer(
        old_train_state,
        FakePartitioner(global_mesh, mesh_axes),
        self.tmp_dir,
        save_dtype=np.uint16,
        restore_dtype=dtype,
        use_gda=False)
    old_checkpointer.save(old_train_state)

    new_train_state = self.make_train_state(
        global_mesh, global_input_shape, mesh_axes, step=step, dtype=dtype)
    new_checkpointer = checkpoints.Checkpointer(
        new_train_state,
        FakePartitioner(global_mesh, mesh_axes),
        self.tmp_dir,
        save_dtype=np.uint16,
        restore_dtype=dtype,
        use_gda=True)
    restored_train_state = new_checkpointer.restore(step=step)

    jax.tree_multimap(self.assert_equal, new_train_state, restored_train_state)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
