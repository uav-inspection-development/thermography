import os
import timeit
from datetime import datetime

import numpy as np
import tensorflow as tf

from thermography.classification.dataset import ThermoDataset, ThermoClass, create_directory_list
from thermography.classification.models import ThermoNet3x3


def main():
    ########################### Input and output paths ###########################

    dataset_path = "Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/padded_dataset"
    dataset_directories = create_directory_list(dataset_path)

    print("Input dataset directories:")
    for path_index, path in enumerate(dataset_directories):
        print("  ({}) {}".format(path_index, path))
    print()

    output_data_path = "Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/output"
    print("Output data path:\n      {}".format(output_data_path))
    print()

    # Path for tf.summary.FileWriter and to store model checkpoints
    summary_path = os.path.join(output_data_path, "tensorboard")
    checkpoint_path = os.path.join(output_data_path, "checkpoints")
    print("Checkpoint directory: {}\nSummary directory: {}".format(checkpoint_path, summary_path))
    print()

    ############################ Thermography classes ############################

    working_class = ThermoClass("working", 0)
    broken_class = ThermoClass("broken", 1)
    misdetected_class = ThermoClass("misdetected", 2)
    thermo_class_list = [working_class, broken_class, misdetected_class]

    ############################# Runtime parameters #############################

    # Dataset params
    load_all_data = True
    normalize_images = True

    # Learning params
    num_epochs = 100000
    batch_size = 128
    global_step = tf.Variable(0, name="global_step")
    learning_rate = 0.00025


    # Network params
    image_shape = np.array([96, 120, 1])
    keep_probability = 0.5

    # Summary params
    write_train_summaries_every_n_steps = 501
    write_histograms_every_n_steps = 1001
    write_kernel_images_every_n_steps = 1001
    write_test_summaries_every_n_epochs = 20
    save_model_every_n_epochs = 20

    ############################# Loading the dataset ############################

    # Place data loading and preprocessing on the cpu.
    with tf.device('/cpu:0'):
        with tf.name_scope("dataset"):
            with tf.name_scope("loading"):
                dataset = ThermoDataset(batch_size=batch_size, balance_data=True, img_shape=image_shape,
                                        normalize_images=normalize_images)
                dataset.set_train_test_validation_fraction(train_fraction=0.8, test_fraction=0.2,
                                                           validation_fraction=0.0)

                dataset.load_dataset(root_directory_list=dataset_directories, class_list=thermo_class_list,
                                     load_all_data=load_all_data)
                dataset.print_info()

            with tf.name_scope("iterator"):
                train_dataset = dataset.get_train_dataset()
                test_dataset = dataset.get_test_dataset()
                train_size = dataset.train_size
                train_steps_per_epoch = int(np.ceil(train_size / batch_size))

    ############################### Net construction #############################

    with tf.name_scope("placeholders"):
        # TF placeholder for graph input and output
        input_images = tf.keras.Input(shape=image_shape, dtype=tf.float32, name="input_image")
        input_one_hot_labels = tf.keras.Input(shape=(dataset.num_classes,), dtype=tf.int32, name="input_one_hot_labels")
        input_labels = tf.argmax(input_one_hot_labels, axis=1, name="input_labels")
        keep_prob = tf.placeholder(tf.float32, name="keep_probability")

    # Initialize model
    model = ThermoNet3x3(x=input_images, image_shape=image_shape, num_classes=dataset.num_classes, keep_prob=keep_prob)

    # Operation for calculating the loss
    with tf.name_scope("cross_ent"):
        # Link variable to model output
        logits = model.logits
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Add the loss to summary
    with tf.summary.create_file_writer(summary_path).as_default():
        tf.summary.scalar('train/cross_entropy', loss, step=global_step)
    with tf.summary.create_file_writer(summary_path).as_default():
        tf.summary.scalar('test/cross_entropy', loss, step=global_step)

    # Train operation
    with tf.name_scope("train"):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, name="adam")

    # Predict operation
    with tf.name_scope("prediction"):
        class_prediction_op = tf.argmax(logits, axis=1, name="class_predictions")
        correct_pred = tf.equal(class_prediction_op, input_labels, name="correct_predictions")
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    # Add the accuracy to the summary
    with tf.summary.create_file_writer(summary_path).as_default():
        tf.summary.scalar('train/accuracy', accuracy, step=global_step)
    with tf.summary.create_file_writer(summary_path).as_default():
        tf.summary.scalar('test/accuracy', accuracy, step=global_step)


    # Merge all summaries together
    train_summaries = tf.summary.create_file_writer(summary_path + "train")
    test_summaries = tf.summary.create_file_writer(summary_path + "test")
    histogram_summaries = tf.summary.create_file_writer(summary_path + "histograms")
    kernel_summaries = tf.summary.create_file_writer(summary_path + "kernels")

    # Initialize the FileWriter
    writer = tf.summary.create_file_writer(summary_path)

    # Initialize a saver for store model checkpoints
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, global_step=global_step)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_path, max_to_keep=3)

    # 初始化全局步数
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)

    # Start Tensorflow session
    for epoch in range(num_epochs):
        print("=" * 60)
        print(f"{datetime.now()} Starting epoch {epoch}")

        # 训练阶段 ---------------------------------------------------------
        train_preds = []
        train_labels = []
        wrong_samples = []

        for step, (img_batch, label_batch) in enumerate(dataset.train_dataset):
            start_time = timeit.default_timer()

            # 前向传播 + 计算梯度
            with tf.GradientTape() as tape:
                logits = model(img_batch, training=True)  # 自动处理 dropout
                loss = loss(label_batch, logits)
                predictions = tf.argmax(logits, axis=1)

            # 记录训练预测
            train_preds.extend(predictions.numpy())
            train_labels.extend(tf.argmax(label_batch, axis=1).numpy())

            # 反向传播
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # 更新全局步数
            global_step.assign_add(1)
            current_step = global_step.numpy()

            # 记录训练摘要
            if current_step % write_train_summaries_every_n_steps == 0:
                with writer.as_default():
                    tf.summary.scalar("train_loss", loss, step=current_step)
                    # 添加其他指标...

            # 记录直方图
            if current_step % write_histograms_every_n_steps == 0:
                with writer.as_default():
                    for layer in model.layers:
                        for weight in layer.weights:
                            tf.summary.histogram(f"{layer.name}/{weight.name}", weight, step=current_step)

            # 记录卷积核可视化
            if current_step % write_kernel_images_every_n_steps == 0:
                with writer.as_default():
                    for layer in model.layers:
                        if isinstance(layer, tf.keras.layers.Conv2D):
                            kernels = layer.weights[0]
                            # 调整维度格式 [kH, kW, inC, outC] -> [outC, kH, kW, inC]
                            kernels_transposed = tf.transpose(kernels, [3, 0, 1, 2])
                            tf.summary.image(f"{layer.name}/kernels", kernels_transposed[..., :4],
                                             step=current_step, max_outputs=8)

            # 打印进度
            duration = timeit.default_timer() - start_time
            print(f"{datetime.now()} Step {current_step} | Epoch {epoch} ({step + 1}/{train_steps_per_epoch}) | "
                  f"Duration: {duration:.3f}s | Loss: {loss.numpy():.4f}")

        # 训练集混淆矩阵
        train_cm = tf.math.confusion_matrix(train_labels, train_preds, dataset.num_classes)
        print(f"\nTraining Confusion Matrix (Epoch {epoch}):\n{train_cm.numpy()}")

        # 测试阶段 ---------------------------------------------------------
        test_preds = []
        test_labels = []
        test_loss = tf.keras.metrics.Mean()
        wrong_samples = []

        for step, (img_batch, label_batch) in enumerate(dataset.test_dataset):
            # 前向传播
            logits = model(img_batch, training=False)  # 关闭 dropout
            loss = loss(label_batch, logits)
            test_loss.update_state(loss)

            predictions = tf.argmax(logits, axis=1)
            test_preds.extend(predictions.numpy())
            test_labels.extend(tf.argmax(label_batch, axis=1).numpy())

            # 记录错误样本
            correct = tf.equal(predictions, tf.argmax(label_batch, axis=1))
            wrong_indices = tf.where(~correct)
            for idx in wrong_indices:
                idx = idx.numpy()[0]
                wrong_samples.append({
                    "image": img_batch[idx],
                    "pred": predictions[idx].numpy(),
                    "true": tf.argmax(label_batch[idx]).numpy()
                })

        # 记录测试摘要
        if epoch % write_test_summaries_every_n_epochs == 0:
            with writer.as_default():
                tf.summary.scalar("test_loss", test_loss.result(), step=current_step)
                # 添加其他测试指标...

            # 记录错误分类图像（最多10个）
            if len(wrong_samples) > 0:
                with writer.as_default():
                    for i, sample in enumerate(wrong_samples[:10]):
                        img = tf.expand_dims(sample["image"], 0)  # 添加batch维度
                        tf.summary.image(
                            f"wrong_predictions/{i}_true{sample['true']}_pred{sample['pred']}",
                            img,
                            step=current_step
                        )

        # 测试集混淆矩阵
        test_cm = tf.math.confusion_matrix(test_labels, test_preds, dataset.num_classes)
        print(f"\nTest Confusion Matrix (Epoch {epoch}):\n{test_cm.numpy()}")

        # 保存模型
        if epoch % save_model_every_n_epochs == 0:
            save_path = checkpoint_manager.save(checkpoint_number=global_step)
            print(f"{datetime.now()} Saved checkpoint at {save_path}")

        print(f"{datetime.now()} Finished epoch {epoch}\n")
if __name__ == '__main__':
    main()
