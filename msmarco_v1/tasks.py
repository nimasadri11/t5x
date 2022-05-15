
import functools
import seqio
import tensorflow_datasets as tfds
from t5.evaluation import metrics
from t5.data import preprocessors
import tensorflow as tf
from typing import Mapping

def relevance(dataset: tf.data.Dataset) -> tf.data.Dataset:
  def _relevance(ex: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    """Convert a msmarco relevance example to a text2text pair.
    """
    if ex['label'] > 0:
      target= "true"
    else:
      target="false"
    to_return = {
        'inputs': tf.strings.join(['Query: ', ex['query'], ' Document: ', ex['doc'], ' Relevant:']),
        'targets': target,
        'doc_id': ex['doc_id'],
        'query_id': ex['query_id']
    }
    
    return to_return

  return dataset.map(_relevance,
                     num_parallel_calls=tf.data.experimental.AUTOTUNE)

vocabulary = seqio.SentencePieceVocabulary(
    'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model', extra_ids=100)
print("NIMA_VOCAB: ", vocabulary)
output_features = {
    'inputs': seqio.Feature(vocabulary=vocabulary),
    'targets': seqio.Feature(vocabulary=vocabulary),
#    'doc_id': int

}

seqio.TaskRegistry.add(
    'msmarco_v1',
    source=seqio.TfdsDataSource(tfds_name='msmarco:1.0.0'),
    preprocessors=[
        functools.partial(
           relevance,
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.bleu],
    output_features=output_features)
