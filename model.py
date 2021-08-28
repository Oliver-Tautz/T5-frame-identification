import tensorflow as tf
from transformers import TFT5ForConditionalGeneration,AutoTokenizer
import defaults




class identFrameT5(TFT5ForConditionalGeneration):

    """
    simple wrapper class for T5 finetuning. Can be loaded with .from_pretrained("t5-small"). 


    this is adapted from
    https://colab.research.google.com/github/snapthat/TF-T5-text-to-text/blob/master/snapthatT5/notebooks/TF-T5-Datasets%20Training.ipynb#scrollTo=2xcGqd9qDXOF

    """

    # this decoration is used to surpress annoying warnings ...
    @tf.autograph.experimental.do_not_convert
    def __init__(self, *args, log_dir=None, cache_dir= None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker= tf.keras.metrics.Mean(name='loss')

    @tf.autograph.experimental.do_not_convert
    @tf.function
    def train_step(self, data):

        x = data[0]
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            loss = outputs[0]
            logits = outputs[1]
            loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        lr = self.optimizer._decayed_lr(tf.float32)

        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.update({'lr': lr})

        return metrics

    def test_step(self, data):
        x = data[0]
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        output = self(x, training=False)
        loss = output[0]
        loss = tf.reduce_mean(loss)
        logits = output[1]

        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}

def get_tokenizer(pretrained_name=defaults.pretrained_name):
    return AutoTokenizer.from_pretrained(pretrained_name)
def get_model(pretrained_name=defaults.pretrained_name):
    return identFrameT5.from_pretrained(pretrained_name)
