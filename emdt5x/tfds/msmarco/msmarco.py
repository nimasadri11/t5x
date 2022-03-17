"""msmarco dataset."""

import tensorflow_datasets as tfds

# TODO(msmarco): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(msmarco): BibTeX citation
_CITATION = """
"""


class Msmarco(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for msmarco dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(msmarco): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'query': tfds.features.Text(),
            'doc': tfds.features.Text(),
            'label': tfds.features.ClassLabel(names=['0', '1']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('query', 'doc', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(msmarco): Downloads the data and defines the splits
    url = 'https://nimasadri11.github.io/data/'
    train_path = dl_manager.download_and_extract(url + 'train.tsv')
    test_path = dl_manager.download_and_extract(url + 'test.tsv')

    # TODO(msmarco): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(train_path),
        'test': self._generate_examples(test_path),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(msmarco): Yields (key, example) tuples from the dataset
    with open(path) as f:
      lines = f.readlines()
      for i, line in enumerate(lines):
        query, doc, label = line.split('\t')
        label = label.strip()
        yield f'line_{i}', dict(query=query, doc=doc, label=label)
    # for f in path.glob('*.jpeg'):
    #   yield 'key', {
    #       'image': f,
    #       'label': 'yes',
    #   }
