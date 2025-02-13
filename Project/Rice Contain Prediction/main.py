import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

rice_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv")

feature_mean = rice_dataset.mean(numeric_only=True)
feature_std = rice_dataset.std(numeric_only=True)
numerical_features = rice_dataset.select_dtypes('number').columns
normalized_dataset = (rice_dataset[numerical_features] - feature_mean) / feature_std

label_encoder = LabelEncoder()
rice_dataset['Class'] = label_encoder.fit_transform(rice_dataset['Class'])

normalized_dataset['Class'] = rice_dataset['Class']

keras.utils.set_random_seed(42)

number_samples = len(normalized_dataset)
index_80th = round(number_samples * 0.8)
index_90th = index_80th + round(number_samples * 0.1)

shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100)
train_data = shuffled_dataset.iloc[0:index_80th]
validation_data = shuffled_dataset.iloc[index_80th:index_90th]
test_data = shuffled_dataset.iloc[index_90th:]

train_features = train_data.drop(columns=['Class'])
train_labels = train_data['Class'].to_numpy()
validation_features = validation_data.drop(columns=['Class'])
validation_labels = validation_data['Class'].to_numpy()
test_features = test_data.drop(columns=['Class'])
test_labels = test_data['Class'].to_numpy()

input_features = [
    'Eccentricity',
    'Major_Axis_Length',
    'Area',
]

import dataclasses


@dataclasses.dataclass()
class ExperimentSettings:
    learning_rate: float
    number_epochs: int
    batch_size: int
    classification_threshold: float
    input_features: list[str]


@dataclasses.dataclass()
class Experiment:
    name: str
    settings: ExperimentSettings
    model: keras.Model
    epochs: np.ndarray
    metrics_history: keras.callbacks.History

    def get_final_metric_value(self, metric_name: str) -> float:
        if metric_name not in self.metrics_history:
            raise ValueError(
                f'Unknown metric {metric_name}: available metrics are'
                f' {list(self.metrics_history.columns)}'
            )
        return self.metrics_history[metric_name].iloc[-1]


def create_model(settings: ExperimentSettings, metrics: list[keras.metrics.Metric]) -> keras.Model:
    model_inputs = [keras.Input(name=feature, shape=(1,)) for feature in settings.input_features]

    concatenated_inputs = keras.layers.Concatenate()(model_inputs)
    dense = keras.layers.Dense(units=1, input_shape=(1,), name='dense_layer', activation=keras.activations.sigmoid)
    model_output = dense(concatenated_inputs)
    model = keras.Model(inputs=model_inputs, outputs=model_output)
    model.compile(
        optimizer=keras.optimizers.RMSprop(
            settings.learning_rate
        ),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )
    return model


def train_model(
        experiment_name: str,
        model: keras.Model,
        dataset: pd.DataFrame,
        labels: np.ndarray,
        settings: ExperimentSettings,
) -> Experiment:
    features = {
        feature_name: np.array(dataset[feature_name])
        for feature_name in settings.input_features
    }

    history = model.fit(
        x=features,
        y=labels,
        batch_size=settings.batch_size,
        epochs=settings.number_epochs,
    )

    return Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history),
    )


print('Defined the create_model and train_model functions.')


def plot_experiment_metrics(experiment: Experiment, metrics: list[str]):
    plt.figure(figsize=(12, 8))

    for metric in metrics:
        plt.plot(
            experiment.epochs, experiment.metrics_history[metric], label=metric
        )

    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.grid()
    plt.legend()


print("Defined the plot_curve function.")

# Let's define our first experiment settings.
settings = ExperimentSettings(
    learning_rate=0.001,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.35,
    input_features=input_features,
)

metrics = [
    keras.metrics.BinaryAccuracy(
        name='accuracy', threshold=settings.classification_threshold
    ),
    keras.metrics.Precision(
        name='precision', thresholds=settings.classification_threshold
    ),
    keras.metrics.Recall(
        name='recall', thresholds=settings.classification_threshold
    ),
    keras.metrics.AUC(num_thresholds=100, name='auc'),
]

# Establish the model's topography.
model = create_model(settings, metrics)

# Train the model on the training set.
experiment = train_model(
    'baseline', model, train_features, train_labels, settings
)

# Plot metrics vs. epochs
plot_experiment_metrics(experiment, ['accuracy', 'precision', 'recall'])
plot_experiment_metrics(experiment, ['auc'])


def evaluate_experiment(
        experiment: Experiment, test_dataset: pd.DataFrame, test_labels: np.array
) -> dict[str, float]:
    features = {
        feature_name: np.array(test_dataset[feature_name])
        for feature_name in experiment.settings.input_features
    }
    return experiment.model.evaluate(
        x=features,
        y=test_labels,
        batch_size=settings.batch_size,
        verbose=0,
        return_dict=True,
    )


def compare_train_test(experiment: Experiment, test_metrics: dict[str, float]):
    print('Comparing metrics between train and test:')
    for metric, test_value in test_metrics.items():
        print('------')
        print(f'Train {metric}: {experiment.get_final_metric_value(metric):.4f}')
        print(f'Test {metric}:  {test_value:.4f}')


# Evaluate test metrics
test_metrics = evaluate_experiment(experiment, test_features, test_labels)
compare_train_test(experiment, test_metrics)

all_input_features = [
    'Eccentricity',
    'Major_Axis_Length',
    'Minor_Axis_Length',
    'Area',
    'Convex_Area',
    'Perimeter',
    'Extent',
]

settings_all_features = ExperimentSettings(
    learning_rate=0.001,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.5,
    input_features=all_input_features,
)

# Modify the following definition of METRICS to generate
# not only accuracy and precision, but also recall:
metrics = [
    keras.metrics.BinaryAccuracy(
        name='accuracy',
        threshold=settings_all_features.classification_threshold,
    ),
    keras.metrics.Precision(
        name='precision',
        thresholds=settings_all_features.classification_threshold,
    ),
    keras.metrics.Recall(
        name='recall', thresholds=settings_all_features.classification_threshold
    ),
    keras.metrics.AUC(num_thresholds=100, name='auc'),
]

# Establish the model's topography.
model_all_features = create_model(settings_all_features, metrics)

# Train the model on the training set.
experiment_all_features = train_model(
    'all features',
    model_all_features,
    train_features,
    train_labels,
    settings_all_features,
)

# Plot metrics vs. epochs
plot_experiment_metrics(
    experiment_all_features, ['accuracy', 'precision', 'recall']
)
plot_experiment_metrics(experiment_all_features, ['auc'])
