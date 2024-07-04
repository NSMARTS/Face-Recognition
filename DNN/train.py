
import tensorflow as tf

from networks.recognition.models import ArcFaceModel, ArcHead
from networks.recognition.losses import SoftmaxLoss
import data.dataloader as dataset
from utils_train.utils import load_yaml, get_ckpt_inf


def main(ongoing=False):
    print('main')
    cfg = load_yaml("configs/arc_res50_kface_finetune.yaml")

    # load pretrained model
    num_classes = cfg['num_classes'] if ongoing else 85742
    model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         num_classes=num_classes,
                         head_type=cfg['head_type'],
                         embd_shape=cfg['embd_shape'],
                         w_decay=cfg['w_decay'],
                         training=True)
    model.summary(line_length=80)

    # compile
    learning_rate = tf.constant(cfg['base_lr'])
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    loss_fn = SoftmaxLoss()

    model_name = cfg['sub_name'] + f"-lr{cfg['base_lr']}-bs{cfg['batch_size']}-trainable2"

    # 진행 중인 학습을 이어서 할지 새로 할지 설정합니다
    if ongoing:
        # 진행 중인 학습
        ckpt_path = tf.train.latest_checkpoint("weights/" + model_name)
        dataset_len = cfg['num_samples']
        steps_per_epoch = dataset_len // cfg['batch_size']
    else:
        # 처음부터 학습
        ckpt_path = tf.train.latest_checkpoint('weights/arc_res50_ccrop')
        dataset_len = cfg['num_samples']
        steps_per_epoch = dataset_len // cfg['batch_size']

    if ckpt_path:
        # 체크포인트 파일이 있는 경우
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)  # 체크포인트에서 가중치를 불러옵니다
        epochs, steps = get_ckpt_inf(ckpt_path, steps_per_epoch)  # 체크포인트 정보에서 에포크와 스텝을 가져옵니다
    else:
        # 체크포인트 파일이 없는 경우
        print(f"No checkpoint found at the specified path. Starting training from scratch.")
        epochs, steps = 1, 1  # 처음부터 학습하는 경우 에포크와 스텝을 1로 설정합니다

    # newly define model
    n_label = tf.keras.layers.Input([])
    n_model = model.layers[1](model.layers[0].output)
    n_model = model.layers[2](n_model)
    n_model = ArcHead(num_classes=cfg['num_classes'], margin=0.5, logist_scale=64.0)(n_model, n_label)
    n_model = tf.keras.Model(inputs=(model.input[0], n_label), outputs=n_model)
    model = n_model
    for layer in model.layers[:2]:
        layer.trainable = False
    model.summary()

    # load dataset
    train_dataset = dataset.load_tfrecord_dataset(cfg['train_dataset'],
                                                  cfg['batch_size'],
                                                  cfg['binary_img'],
                                                  is_ccrop=cfg['is_ccrop'])
    train_dataset = iter(train_dataset)
    summary_writer = tf.summary.create_file_writer("logs/" + model_name)

    # training loop
    while epochs <= cfg['epochs']:
        inputs, labels = next(train_dataset)

        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            reg_loss = tf.reduce_sum(model.losses)
            pred_loss = loss_fn(labels, logits)
            total_loss = pred_loss + reg_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if steps % (cfg['save_steps'] // 10) == 0:
            verb_str = "Epoch {}/{}: {}/{}, loss={:.2f}, lr={:.4f}"
            print(verb_str.format(epochs, cfg['epochs'],
                                  steps % steps_per_epoch,
                                  steps_per_epoch,
                                  total_loss.numpy(),
                                  learning_rate.numpy()))

            with summary_writer.as_default():
                tf.summary.scalar('total loss', total_loss, step=steps)
                tf.summary.scalar('pred loss', pred_loss, step=steps)
                tf.summary.scalar('reg loss', reg_loss, step=steps)
                tf.summary.scalar('learning rate', optimizer.lr, step=steps)

        if steps % cfg['save_steps'] == 0:
            print(f"Saved checkpoint for step {int(steps)}")
            model.save_weights('checkpoints/' + model_name + f"/e_{epochs}_b_{steps % steps_per_epoch}.ckpt")

        steps += 1
        epochs = steps // steps_per_epoch + 1


if __name__ == "__main__":
    main(ongoing=False)